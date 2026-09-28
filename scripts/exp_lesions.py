#!/usr/bin/env python3
"""Experiment 2, form 2 - the main result.

Compares, on the same filter-strength scale:

  * the RIQE score,
  * the signal fidelity of inserted low-contrast lesions (matched-filter
    retention and peak retention),
  * the d' of the NPW matched filter.

The question: **does RIQE keep improving in the region where signal fidelity
has already collapsed?** If so, the metric rewards filtering that has already
erased low-contrast lesions, and this must be stated plainly.

Why three measures and not one. The d' of the matched filter with known
template and location is almost insensitive to smoothing -- a known property
of the near-optimal observer, verified here -- and used alone it would
suggest that overfiltering does no harm. Loss of signal amplitude is what
erases a lesion to the eye. The two diverge, and the divergence is
information, not noise: it is reported.

Substrate: full-dose slices (treated as low-noise truth), plus a realisation
of FBP-like noise at the level measured from the real full-dose / reduced-dose
contrast of the same protocol cell. The variance of d' is taken **over noise
realisations at the same site**, so the anatomical term cancels exactly.

Everything is reported per denoiser as well as pooled: pooling over denoisers
averages a filter the metric rejects (Gaussian) with one it prefers
(bilateral), and the pooled verdict hides the difference.

    .venv/bin/python scripts/exp_lesions.py --n-slices 24 --realizations 16
"""

from __future__ import annotations

import os

# One thread per process: parallelism comes from processes, and letting every
# worker open its own BLAS threads causes oversubscription (load 90 on 32
# cores, measured) instead of speed. Must be set before importing numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import concurrent.futures as cf
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe import degrade as dg  # noqa: E402
from riqe.cache import FeatureCache  # noqa: E402
from riqe.dicomio import read_hu  # noqa: E402
from riqe.extract import MIN_PATCHES_FOR_SCORE, Spec, features_from_hu, masks_for  # noqa: E402
from riqe.model import RCOND, mahalanobis_mixed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"

#: lesions: (diameter mm, contrast HU). They range from clearly detectable to
#: below threshold, to see where filtering does harm first.
LESIONS = ((10.0, 25.0), (6.0, 15.0), (4.0, 10.0))
N_SITES = 16

_G = {}


def _init(cfg):
    _G.update(cfg)


def _score(hu, pad, masks, spec, model):
    pf = features_from_hu(hu, pad, spec, fitting=False, masks=masks)
    f = pf.feat[pf.valid]
    if f.shape[0] < MIN_PATCHES_FOR_SCORE:
        return np.nan
    return mahalanobis_mixed(model.nu, model.sigma, f.mean(axis=0), np.cov(f, rowvar=False), RCOND)


def _one(job):
    path, cell, noise_sigma, seed = job
    spec, model = _G["spec"], _G["model"]
    K = _G["K"]
    hu, pad, meta = read_hu(str(ROOT / path))
    fov, body = masks_for(hu, pad)
    masks = (fov, body)
    ps = meta["pixel_spacing"]
    rng = np.random.default_rng(seed)

    # disjoint sites for each lesion type
    groups, used = [], []
    for dmm, contrast in LESIONS:
        r = dmm / 2.0 / ps
        sites = dg.find_homogeneous_sites(hu, body, r, N_SITES * 3, np.random.default_rng(seed + 1))
        keep = []
        for (cy, cx) in sites:
            if all((cy - y) ** 2 + (cx - x) ** 2 >= (4 * r + 8) ** 2 for y, x in used):
                keep.append((cy, cx))
                used.append((cy, cx))
            if len(keep) >= N_SITES:
                break
        if len(keep) >= 4:
            groups.append({"dmm": dmm, "contrast": contrast, "r": r, "sites": keep})
    if not groups:
        return []

    lesion_field = np.zeros_like(hu, dtype=np.float32)
    for g in groups:
        for cy, cx in g["sites"]:
            lesion_field += (g["contrast"] * dg._disc(hu.shape, cy, cx, g["r"])).astype(np.float32)

    noise_seeds = [seed + 100 + k for k in range(K)]
    noisy0 = dg.add_fbp_noise(hu, noise_sigma, np.random.default_rng(noise_seeds[0]))

    conditions = [("none", None, np.nan)]
    for name in dg.DENOISERS:
        for lvl in dg.RESIDUAL_LEVELS_HU:
            st, res, _ = dg.calibrate_strength(noisy0, name, float(lvl), body)
            conditions.append((name, functools.partial(dg.apply_denoiser, name=name, strength=st), lvl))

    rows = []
    for cname, filt, lvl in conditions:
        # RIQE score of the filtered noisy image, without lesions
        img = filt(noisy0) if filt is not None else noisy0
        sc = _score(img, pad, masks, spec, model)
        res_hu = dg.residual_std(noisy0, img, body) if filt is not None else 0.0

        # template responses over K realisations, with and without lesions
        resp_abs = {i: [] for i in range(len(groups))}
        resp_sig = {i: [] for i in range(len(groups))}
        peak = {i: [] for i in range(len(groups))}
        for k in range(K):
            n = dg.add_fbp_noise(np.zeros_like(hu), noise_sigma, np.random.default_rng(noise_seeds[k]))
            base = hu + n
            wles = base + lesion_field
            bf = filt(base) if filt is not None else base
            wf = filt(wles) if filt is not None else wles
            diff = wf.astype(np.float64) - bf.astype(np.float64)
            for i, g in enumerate(groups):
                resp_abs[i].append(dg.npw_response(bf, g["sites"], g["r"]))
                resp_sig[i].append(dg.npw_response(wf, g["sites"], g["r"]))
                if k < 4:
                    half = int(np.ceil(g["r"] * 2.5))
                    pk = []
                    for cy, cx in g["sites"]:
                        y0, y1 = max(cy - half, 0), min(cy + half + 1, diff.shape[0])
                        x0, x1 = max(cx - half, 0), min(cx + half + 1, diff.shape[1])
                        pk.append(float(diff[y0:y1, x0:x1].max()) / g["contrast"])
                    peak[i].append(np.nanmean(pk))

        for i, g in enumerate(groups):
            A = np.asarray(resp_sig[i])
            B = np.asarray(resp_abs[i])
            delta = np.nanmean(A - B, axis=0)
            sd = np.nanstd(B, axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dp = np.where(sd > 0, delta / sd, np.nan)
            half = int(np.ceil(g["r"] * 2.5))
            t = dg._template(g["r"], half)
            ref = float(np.nansum(t * dg._disc((2 * half + 1,) * 2, half, half, g["r"]) * g["contrast"]))
            rows.append({
                "slice_path": path, "cell": cell, "pixel_spacing": ps,
                "noise_sigma_hu": noise_sigma,
                "denoiser": cname, "target_residual_hu": lvl, "residual_hu": res_hu,
                "riqe": sc,
                "diameter_mm": g["dmm"], "contrast_hu": g["contrast"], "n_sites": len(g["sites"]),
                "dprime": float(np.nanmean(dp)),
                "retention_matched": float(np.nanmean(delta) / ref) if ref else np.nan,
                "retention_peak": float(np.nanmean(peak[i])) if peak[i] else np.nan,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-slices", type=int, default=24)
    ap.add_argument("--realizations", type=int, default=16)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    ch_file = ROOT / "experiments" / "hparam_choice.json"
    ch = json.loads(ch_file.read_text()) if ch_file.exists() else {}
    P = args.P if args.P is not None else ch.get("P")
    C = args.C if args.C is not None else ch.get("C")
    p = args.p if args.p is not None else ch.get("p")
    if P is None:
        ap.error("need --P --C --p, or experiments/hparam_choice.json")

    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit_pids, test_pids = set(split["fit"]), set(split["test"])
    cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")
    fit = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(fit_pids)]
    model = cache.fit(list(fit["path"]), p, n_patients=fit.patient_id.nunique())

    # noise level to add: measured from the real full-dose / reduced-dose
    # contrast of the same cell
    slices = pd.read_parquet(ROOT / "corpus" / "slices.parquet")
    noise_by_cell = {}
    for cell, g in slices.groupby("cell"):
        fu = g[g.kind == "full"]["body_sigma_med"].median()
        lo = g[g.kind == "low"]["body_sigma_med"].median()
        if np.isfinite(lo):
            noise_by_cell[cell] = float(np.sqrt(max(lo ** 2 - fu ** 2, 1.0)))
    default_noise = float(np.median(list(noise_by_cell.values()))) if noise_by_cell else 25.0
    print("added noise per cell (sigma HU, from real full vs reduced dose):")
    for k, v in noise_by_cell.items():
        print(f"  {k:26s} {v:6.1f}")
    print(f"  cells without a real pair: {default_noise:.1f} (median of the others)")

    # test slices: abdomen (homogeneous liver parenchyma), from the TEST split
    cand = corpus[(corpus["kind"] == "full") & corpus["keep"]
                  & corpus.patient_id.isin(test_pids)
                  & corpus["body_part"].eq("ABDOMEN")]
    # one central slice per patient. Explicit selection instead of
    # groupby.apply: with include_groups=False pandas drops the grouping
    # column from the result, and downstream code looks for it.
    rng = np.random.default_rng(args.seed)
    idx = [g.sort_values("z").index[len(g) // 2] for _, g in cand.groupby("patient_id")]
    pick = cand.loc[idx]
    if len(pick) > args.n_slices:
        pick = pick.iloc[np.sort(rng.choice(len(pick), args.n_slices, replace=False))]
    print(f"\ntest slices: {len(pick)} (abdomen, TEST split, {pick.patient_id.nunique()} patients)")

    jobs = [(r.path, r.cell, noise_by_cell.get(r.cell, default_noise), args.seed + 977 * i)
            for i, r in enumerate(pick.itertuples())]
    spec = Spec(P=P, C=C, p=None)
    t0 = time.time()
    rows = []
    with cf.ProcessPoolExecutor(
        args.workers, initializer=_init,
        initargs=({"spec": spec, "model": model, "K": args.realizations},)
    ) as ex:
        for i, rr in enumerate(ex.map(_one, jobs, chunksize=1), 1):
            rows += rr
            print(f"  {i}/{len(jobs)} slices, {len(rows)} rows, {(time.time()-t0)/60:.1f} min",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp2_form2_lesions.csv", index=False)
    print(f"\nwrote {OUT}/exp2_form2_lesions.csv  ({(time.time()-t0)/60:.1f} min)")

    # ---- the central question ---------------------------------------------
    base = df[df.denoiser == "none"].set_index(["slice_path", "diameter_mm"])
    f = df[df.denoiser != "none"].copy()
    f["riqe_base"] = [base.riqe.get((r.slice_path, r.diameter_mm), np.nan) for r in f.itertuples()]
    f["riqe_improves"] = f["riqe"] < f["riqe_base"]

    print("\n" + "=" * 78)
    print("MAIN RESULT - RIQE versus signal fidelity")
    print("=" * 78)
    agg = (f.groupby(["target_residual_hu", "diameter_mm"])
             .agg(riqe=("riqe", "median"), riqe_improves=("riqe_improves", "mean"),
                  retention_matched=("retention_matched", "median"),
                  retention_peak=("retention_peak", "median"),
                  dprime=("dprime", "median")).reset_index())
    print("pooled over denoisers:")
    print(agg.round(3).to_string(index=False))

    piv = agg.pivot(index="target_residual_hu", columns="diameter_mm", values="retention_matched")
    riqe_med = f.groupby("target_residual_hu")["riqe"].median()
    base_med = base.riqe.median()
    print(f"\nmedian RIQE score without filtering: {base_med:.4f}")
    best_lvl = riqe_med.idxmin()
    print(f"strength that optimises RIQE (pooled): residual {best_lvl:g} HU "
          f"(score {riqe_med.min():.4f})")
    for dmm in sorted(piv.columns):
        col = piv[dmm]
        collapsed = col[col < 0.5]
        first = collapsed.index.min() if len(collapsed) else None
        print(f"  lesion {dmm:4.0f} mm: retention at that strength = "
              f"{col.get(best_lvl, np.nan):.3f}"
              + (f"; falls below 0.5 at residual {first:g} HU" if first is not None else
                 "; never falls below 0.5"))
    verdict = (
        "DANGEROUS: the RIQE optimum falls where signal fidelity has already collapsed"
        if any(piv[c].get(best_lvl, 1.0) < 0.5 for c in piv.columns)
        else "the pooled RIQE optimum does not fall in the region of signal collapse"
    )
    print(f"\nPOOLED VERDICT: {verdict}")

    # ---- per denoiser: the pooled verdict hides the difference -------------
    per = (f.groupby(["denoiser", "target_residual_hu", "diameter_mm"])
             .agg(riqe_improves=("riqe_improves", "mean"),
                  retention_matched=("retention_matched", "median"),
                  retention_peak=("retention_peak", "median"),
                  dprime=("dprime", "median")).reset_index())
    print("\nper denoiser, fraction of images in which filtering improves RIQE:")
    print((100 * per.pivot_table(index="denoiser", columns="target_residual_hu",
                                 values="riqe_improves", aggfunc="mean")).round(1).to_string())
    by_denoiser = {}
    for name, g in per.groupby("denoiser"):
        by_denoiser[name] = {
            "riqe_improves_by_strength": g.groupby("target_residual_hu")["riqe_improves"].mean().to_dict(),
            "retention_matched": g.pivot(index="target_residual_hu", columns="diameter_mm",
                                         values="retention_matched").to_dict(),
            "retention_peak": g.pivot(index="target_residual_hu", columns="diameter_mm",
                                      values="retention_peak").to_dict(),
            "dprime": g.pivot(index="target_residual_hu", columns="diameter_mm",
                              values="dprime").to_dict(),
        }

    (OUT / "exp2_form2_summary.json").write_text(json.dumps({
        "riqe_base_median": float(base_med),
        "riqe_optimal_strength_hu": float(best_lvl),
        "riqe_optimum": float(riqe_med.min()),
        "riqe_by_strength": riqe_med.to_dict(),
        "retention_matched": piv.to_dict(),
        "fraction_cases_riqe_improves": float(f["riqe_improves"].mean()),
        "pooled_verdict": verdict,
        "by_denoiser": by_denoiser,
        "n_slices": int(df.slice_path.nunique()),
        "realizations": args.realizations,
    }, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
