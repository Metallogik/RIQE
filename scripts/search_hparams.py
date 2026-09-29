#!/usr/bin/env python3
"""Search of the hyperparameters (P, C, p) with a lexicographic criterion.

ORIGINAL CRITERION, declared **before** running, in this order:

  1. OVERFILTERING SAFETY. Number of cases in which a filtered image scores
     better than the full-dose original. Must be 0: settings that fail are
     **eliminated**, not penalised.
  2. MONOTONICITY. Correct ordering at every step of the noise (white and
     FBP-like) and blur ladders.
  3. STABILITY. Median bootstrap divergence over patients.
  4. PARSIMONY AND CONDITIONING. At equal performance, the setting with more
     patches per image and a better-conditioned Sigma.

This is not the criterion of Gunawan et al. (Computers 2025), who maximise the
fraction of images on which the deepest network is ranked the best denoiser:
that assumes the answer and uses it as the target.

REVISED CRITERION, declared as such. The criterion above was blind to the most
important property -- the ordering of real reduced-dose images -- and selected
a setting that ranks it correctly in fewer than half of the abdominal
validation pairs.
After inspecting validation data, and before touching the test set, the
criterion becomes:

  1. REAL DOSE. Fraction of (full dose, real reduced dose, same slice) pairs in
     which the reduced-dose image scores worse, taken as the minimum over chest
     and abdomen. Pass threshold 0.95.
  2. OVERFILTERING. Fraction of filtered images scoring better than the
     original: lowest.
  3. RELATIVE NOISE. Fraction of images with added noise >= 20% of native noise
     that score worse: highest.
  4. STABILITY, then patches per image.
If no setting passes point 1, settings are ranked by point 1, then point 2.
The choice of the original criterion is computed and reported alongside, not
silently replaced.

Everything on the inner split of FIT only. TEST is not touched.

    .venv/bin/python scripts/search_hparams.py \
        --moments experiments/bank_val_inner_moments.npz
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
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.cache import FeatureCache  # noqa: E402
from riqe.dosetest import dose_ordering, dose_pairs, image_moments, summarize  # noqa: E402
from riqe.model import RCOND, mahalanobis_mixed, model_divergence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P_GRID = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 0.90)
DOSE_PASS = 0.95
REL_NOISE_MIN = 0.20


def load_moments(paths):
    """One or more moment files, concatenated; the index is the row."""
    NUs, SGs, metas = [], [], []
    for path in paths:
        z = np.load(path, allow_pickle=False)
        meta = pd.DataFrame(json.loads(str(z["meta"])))
        meta["slice_path"] = json.loads(str(z["slice_path"]))
        meta["patient_id"] = json.loads(str(z["patient_id"]))
        meta["cell"] = json.loads(str(z["cell"]))
        NUs.append(z["nu"])
        SGs.append(z["sigma"])
        metas.append(meta)
    meta = pd.concat(metas, ignore_index=True)
    return np.concatenate(NUs), np.concatenate(SGs), meta


def score_rows(model, NU, SG, idx) -> np.ndarray:
    """Scores of a set of moment rows against a model."""
    out = np.full(len(idx), np.nan)
    for k, i in enumerate(idx):
        if np.isnan(NU[i, 0]):
            continue
        out[k] = mahalanobis_mixed(model.nu, model.sigma, NU[i], SG[i], RCOND)
    return out


def bootstrap_divergence(cache, paths_by_patient, p, B, seed) -> np.ndarray:
    """D(full model, model on a patient bootstrap sample)."""
    rng = np.random.default_rng(seed)
    pids = sorted(paths_by_patient)
    allp = [q for pid in pids for q in paths_by_patient[pid]]
    full = cache.fit_fast(allp, p, n_patients=len(pids))
    ds = []
    for _ in range(B):
        take = rng.choice(pids, size=len(pids), replace=True)
        paths = [q for pid in take for q in paths_by_patient[pid]]
        try:
            ds.append(model_divergence(full, cache.fit_fast(paths, p, n_patients=len(pids))))
        except ValueError:
            ds.append(np.nan)
    return np.asarray(ds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moments", required=True, nargs="+")
    ap.add_argument("--bootstrap", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default="experiments/hparam_search.csv")
    args = ap.parse_args()

    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit_inner = set(split["fit_inner"])

    kept = corpus[(corpus["kind"] == "full") & corpus["keep"]]
    fit_slices = kept[kept.patient_id.isin(fit_inner)]
    by_patient = defaultdict(list)
    for r in fit_slices.itertuples():
        by_patient[r.patient_id].append(r.path)
    print(f"inner fitting set: {len(by_patient)} patients, {len(fit_slices)} slices")

    NU, SG, meta = load_moments(args.moments)
    configs = sorted({(int(r.P), float(r.C)) for r in meta.itertuples()})
    print(f"bank: {len(meta)} rows, {meta.slice_path.nunique()} slices, "
          f"{len(configs)} (P, C) settings")

    # native noise per slice, to express added noise in relative terms
    slices_tab = pd.read_parquet(ROOT / "corpus" / "slices.parquet").set_index("path")
    sigma_nat = slices_tab["body_sigma_med"]

    # real-dose pairs on the inner validation split
    pairs = dose_pairs(corpus, split["val_inner"])
    print(f"real-dose pairs (validation): {len(pairs)} "
          f"(chest {int((pairs.region=='chest').sum())}, abdomen {int((pairs.region=='abdomen').sum())})")

    # rows of interest, per setting
    full_rows = meta[meta["source"] == "full"]
    results = []
    t0 = time.time()

    for P, C in configs:
        cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")
        sub = full_rows[(full_rows.P == P) & (full_rows.C == C)]
        orig = {r.slice_path: r.Index for r in sub[sub.kind == "original"].itertuples()}
        pair_moments = image_moments(cache, list(pairs.path_full) + list(pairs.path_low))
        noise = sub[sub.kind.isin(["noise_white", "noise_fbp", "noise_rel"])].copy()
        sn = noise["slice_path"].map(sigma_nat).to_numpy(dtype=float)
        noise["rel"] = np.sqrt(sn ** 2 + noise["sigma_hu"].to_numpy(dtype=float) ** 2) / sn - 1.0
        noise = noise[(noise["rel"] >= REL_NOISE_MIN) & noise["slice_path"].isin(orig)]

        for p in P_GRID:
            try:
                model = cache.fit_fast([q for v in by_patient.values() for q in v], p,
                                       n_patients=len(by_patient))
            except ValueError as e:
                results.append({"P": P, "C": C, "p": p, "error": str(e)})
                continue

            # --- criterion 1: overfiltering safety ----------------------------
            den = sub[sub.kind == "denoise"]
            s_orig = score_rows(model, NU, SG, [orig[sp] for sp in den.slice_path])
            s_den = score_rows(model, NU, SG, list(den.index))
            ok = np.isfinite(s_orig) & np.isfinite(s_den)
            fail_mask = ok & (s_den < s_orig)
            n_fail = int(fail_mask.sum())
            frac_fail = float(fail_mask.sum() / max(ok.sum(), 1))

            # blur: also an overfiltering of the pristine image
            bl = sub[sub.kind == "blur"]
            sb_o = score_rows(model, NU, SG, [orig[sp] for sp in bl.slice_path])
            sb = score_rows(model, NU, SG, list(bl.index))
            okb = np.isfinite(sb_o) & np.isfinite(sb)
            n_fail_blur = int((okb & (sb < sb_o)).sum())

            # --- criterion 2: monotonicity ------------------------------------
            mono = {}
            for kind, param in (("noise_white", "sigma_hu"), ("noise_fbp", "sigma_hu"),
                                ("blur", "sigma_px")):
                g = sub[sub.kind == kind]
                perfect = total = 0
                for sp, gg in g.groupby("slice_path"):
                    gg = gg.sort_values(param)
                    sc = score_rows(model, NU, SG, list(gg.index))
                    s0 = score_rows(model, NU, SG, [orig[sp]])[0]
                    seq = np.concatenate([[s0], sc])
                    if np.isfinite(seq).all():
                        total += 1
                        if np.all(np.diff(seq) > 0):
                            perfect += 1
                mono[kind] = perfect / total if total else np.nan
            mono_mean = float(np.nanmean(list(mono.values())))
            mono_min = float(np.nanmin(list(mono.values())))

            # --- real dose ----------------------------------------------------
            dsum = summarize(dose_ordering(model, pairs, pair_moments))

            # --- relative noise -----------------------------------------------
            s_n = score_rows(model, NU, SG, list(noise.index))
            # the original must exist: a fallback index such as -1 would read
            # the last row of the array, i.e. the score of another image
            s_n0 = score_rows(model, NU, SG, [orig[sp] for sp in noise.slice_path])
            okn = np.isfinite(s_n) & np.isfinite(s_n0)
            rel_detect = float((s_n[okn] > s_n0[okn]).mean()) if okn.any() else np.nan

            # --- criterion 3: stability ---------------------------------------
            d_boot = bootstrap_divergence(cache, by_patient, p, args.bootstrap, args.seed)
            d_med = float(np.nanmedian(d_boot))
            d_p95 = float(np.nanpercentile(d_boot, 95))

            # --- criterion 4: patches and conditioning ------------------------
            counts = cache.patch_counts([q for v in by_patient.values() for q in v], p)
            n_score_patches = np.asarray([
                r.n_patches for r in sub[sub.kind == "original"].itertuples()
            ])

            results.append({
                "P": P, "C": C, "p": p,
                "dose_correct_min": dsum["correct_min"],
                "dose_correct_chest": dsum["correct_chest"],
                "dose_correct_abdomen": dsum["correct_abdomen"],
                "dose_n_chest": dsum["n_chest"],
                "dose_n_abdomen": dsum["n_abdomen"],
                "rel_noise_detected": rel_detect,
                "n_overfilter_fail": n_fail,
                "frac_overfilter_fail": frac_fail,
                "n_blur_fail": n_fail_blur,
                "mono_mean": mono_mean,
                "mono_min": mono_min,
                **{f"mono_{k}": v for k, v in mono.items()},
                "d_boot_median": d_med,
                "d_boot_p95": d_p95,
                "fit_patches_per_slice": float(counts.mean()),
                "fit_patches_total": int(counts.sum()),
                "score_patches_median": float(np.median(n_score_patches)),
                "cond_sigma": model.cond(),
                "score_orig_median": float(np.nanmedian(
                    score_rows(model, NU, SG, list(sub[sub.kind == "original"].index)))),
            })
            r = results[-1]
            print(f"  P={P:3d} C={C:<5g} p={p:<5.2f} | dose c/a={dsum['correct_chest']:.2f}/"
                  f"{dsum['correct_abdomen']:.2f} | rel.noise={rel_detect:.2f} | overfilt.={n_fail:4d} "
                  f"({100*frac_fail:5.1f}%) blur_fail={n_fail_blur:3d} | mono={mono_mean:.3f} "
                  f"(min {mono_min:.3f}) | D_boot={d_med:.4f} | patch/slice={r['fit_patches_per_slice']:6.1f}"
                  f" | cond={model.cond():.1e}", flush=True)

    df = pd.DataFrame(results)
    out = ROOT / args.out
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({(time.time()-t0)/60:.1f} min)")

    cols = ["P", "C", "p", "dose_correct_min", "dose_correct_chest", "dose_correct_abdomen",
            "rel_noise_detected", "frac_overfilter_fail", "mono_min", "d_boot_median",
            "fit_patches_per_slice", "score_patches_median", "cond_sigma"]

    # --- 1. ORIGINAL criterion, declared before running --------------------
    print("\n=== original criterion (declared before running) ===")
    safe = df[df["n_overfilter_fail"] == 0]
    if len(safe) == 0:
        print("NO setting is safe against overfiltering (criterion 1 = 0).")
        ranked_o = df.sort_values(
            ["frac_overfilter_fail", "mono_min", "mono_mean", "d_boot_median"],
            ascending=[True, False, False, True])
    else:
        ranked_o = safe.sort_values(
            ["mono_min", "mono_mean", "d_boot_median", "fit_patches_per_slice"],
            ascending=[False, False, True, False])
    best_o = ranked_o.iloc[0]
    print(f"original choice: P={int(best_o.P)} C={best_o.C:g} p={best_o.p:g}  "
          f"(dose c/a {best_o.dose_correct_chest:.3f}/{best_o.dose_correct_abdomen:.3f}, "
          f"overfiltering {100*best_o.frac_overfilter_fail:.1f}%)")

    # --- 2. REVISED criterion -------------------------------------------------
    print("\n=== revised criterion (real dose first) ===")
    passing = df[df["dose_correct_min"] >= DOSE_PASS]
    if len(passing):
        print(f"settings ranking real dose correctly >= {DOSE_PASS:.0%} in both regions: "
              f"{len(passing)}/{len(df)}")
        ranked = passing.sort_values(
            ["frac_overfilter_fail", "rel_noise_detected", "d_boot_median", "fit_patches_per_slice"],
            ascending=[True, False, True, False])
    else:
        print(f"NO setting ranks real dose correctly >= {DOSE_PASS:.0%} in both regions: "
              f"ranking by dose, then overfiltering.")
        ranked = df.sort_values(
            ["dose_correct_min", "frac_overfilter_fail", "rel_noise_detected", "d_boot_median"],
            ascending=[False, True, False, True])
    print(ranked[cols].head(15).to_string(index=False))
    best = ranked.iloc[0]

    (ROOT / "experiments" / "hparam_choice.json").write_text(json.dumps({
        "P": int(best.P), "C": float(best.C), "p": float(best.p),
        "criterion": ("revised: real-dose ordering (min over chest/abdomen, pass >= 0.95), "
                      "then overfiltering failure fraction, then relative-noise detection, "
                      "then bootstrap stability, then patches per slice"),
        "declared_before_run": False,
        "revision_reason": ("the pre-declared criterion ignored real-dose ordering and selected "
                            "a configuration ordering reduced-dose abdomen correctly in "
                            f"{100 * best_o.dose_correct_abdomen:.1f}% of validation pairs; "
                            "revised after seeing validation data and before using the test set"),
        "any_config_passes_dose": bool(len(passing) > 0),
        "any_safe_config_overfiltering": bool(len(safe) > 0),
        "metrics": {k: (float(best[k]) if k in best else None) for k in cols},
        "original_criterion_choice": {
            "P": int(best_o.P), "C": float(best_o.C), "p": float(best_o.p),
            "criterion": "lexicographic: overfilter safety, then step monotonicity, "
                         "then bootstrap stability, then patches per slice",
            "declared_before_run": True,
            "metrics": {k: (float(best_o[k]) if k in best_o else None) for k in cols},
        },
    }, indent=1))
    print(f"\nchoice (revised criterion): P={int(best.P)} C={best.C:g} p={best.p:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
