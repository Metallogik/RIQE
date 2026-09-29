#!/usr/bin/env python3
"""Validation battery, experiments 0-4, on the TEST split only.

  0. REDUCED DOSE. A reduced-dose image must score worse than the full-dose
     image of the same slice. The reduced-dose reconstructions are simulated
     by projection-domain noise insertion (Moen et al., Med Phys 2021).
  1. MONOTONICITY. The score worsens at every step of increasing noise (white
     and FBP-like) and blur; 1b repeats this with noise added relative to the
     native noise of each image.
  2. PREFERENCE FOR FILTERED FULL-DOSE IMAGES, forms 1 and 3.
     Form 1: how often a filtered full-dose image scores better than its
     source, a preference rate (a full-dose image still contains noise).
     Form 3: noise added at fixed levels, against the noise floor -- the
     behaviour by which a NIQE variant ranked a noisy ultrasound image better
     than the original (MTAP 2024).
     Form 2, which compares the metric's optimum with the signal fidelity of
     inserted lesions, is in scripts/exp_lesions.py.
  3. DISCRIMINATION. Five denoisers at matched strength on the same
     reduced-dose input: ordering, agreement across images.
  4. STABILITY. Bootstrap and leave-one-patient-out on the fitting corpus,
     learning curve, confidence intervals on the scores.

Outcomes are reported **whichever way they go**: failure definitions were
declared before running and are not renegotiated after seeing the numbers.

    .venv/bin/python scripts/run_battery.py --moments experiments/bank_test_moments.npz
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

from riqe import degrade as dg  # noqa: E402
from riqe.cache import FeatureCache  # noqa: E402
from riqe.evaluate import (  # noqa: E402
    attach_scores, holm, kendall_w, load_moments, sign_test, spearman, step_monotone,
)
from riqe.model import model_divergence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"


# ---------------------------------------------------------------------------
# 0. Reduced dose  (added after validation)
# ---------------------------------------------------------------------------

def exp0_real_dose(P, C, p, model, d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reduced dose (projection-domain simulated) must worsen the score."""
    from riqe.dosetest import dose_ordering, dose_pairs, low_cache_dir, pair_moments, summarize

    print("\n" + "=" * 78)
    print("EXPERIMENT 0 - REDUCED DOSE VS FULL DOSE, SAME SLICE")
    print("=" * 78)
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")
    pairs = dose_pairs(corpus, split["test"])
    res = dose_ordering(model, pairs, pair_moments(cache, FeatureCache(low_cache_dir(P, C)), pairs))
    sm = summarize(res)
    print(f"  TEST pairs: {sm['n_pairs']} scoreable "
          f"(chest {sm['n_chest']}, abdomen {sm['n_abdomen']}; unscoreable {sm['n_unscoreable']})")
    print(f"  reduced dose scores worse (correct):  chest {100*sm['correct_chest']:.1f}%   "
          f"abdomen {100*sm['correct_abdomen']:.1f}%   all {100*sm['correct_all']:.1f}%")
    ok = res.dropna(subset=["score_full", "score_low"])
    for reg in ("chest", "abdomen"):
        g = ok[ok.region == reg]
        if len(g):
            print(f"    {reg}: median delta {np.median(g.score_low - g.score_full):+.4f}")
    res.to_csv(OUT / "exp0_real_dose.csv", index=False)
    return res, sm


def exp0_original_criterion() -> dict | None:
    """The same test for the setting chosen by the criterion declared before the
    search, fitted on the same patients: how the discarded choice fares on TEST."""
    from riqe.dosetest import dose_ordering, dose_pairs, low_cache_dir, pair_moments, summarize

    ch = json.loads((OUT / "hparam_choice.json").read_text()).get("original_criterion_choice")
    if not ch:
        return None
    P, C, p = int(ch["P"]), float(ch["C"]), float(ch["p"])
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(set(split["fit"]))]
    cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")
    model = cache.fit(list(fit["path"]), p, n_patients=fit.patient_id.nunique())
    pairs = dose_pairs(corpus, split["test"])
    res = dose_ordering(model, pairs, pair_moments(cache, FeatureCache(low_cache_dir(P, C)), pairs))
    sm = summarize(res)
    sm.update({"P": P, "C": C, "p": p})
    print(f"  original-criterion setting P={P} C={C:g} p={p:g}: reduced dose scores worse in "
          f"chest {100*sm['correct_chest']:.1f}%  abdomen {100*sm['correct_abdomen']:.1f}% "
          f"({sm['n_chest']}/{sm['n_abdomen']} pairs)")
    res.to_csv(OUT / "exp0_original_criterion.csv", index=False)
    return sm


# ---------------------------------------------------------------------------
# 1b. Noise relative to native noise  (added after validation)
# ---------------------------------------------------------------------------

def exp1b_relative_noise(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 78)
    print("EXPERIMENT 1b - ADDED NOISE RELATIVE TO NATIVE NOISE")
    print("=" * 78)
    g = d[(d.kind == "noise_rel") & (d.source == "full")].copy()
    if g.empty:
        print("  no noise_rel entries in the bank")
        return pd.DataFrame(), {}
    orig = {r.slice_path: r.score for r in d[(d.kind == "original") & (d.source == "full")].itertuples()}
    g["base"] = g.slice_path.map(orig)
    g = g.dropna(subset=["score", "base"])
    g["worse"] = g.score > g.base
    g["region"] = np.where(g.cell.str.contains("CHEST"), "chest", "abdomen")
    t = g.pivot_table(index="region", columns="rel_increase", values="worse", aggfunc="mean")
    print("  fraction in which the score worsens, by relative noise increase:")
    print((100 * t).round(1).to_string())
    perfect = total = 0
    for sp, gg in g.groupby("slice_path"):
        seq = np.concatenate([[orig[sp]], gg.sort_values("rel_increase").score.to_numpy()])
        if np.isfinite(seq).all():
            total += 1
            perfect += int(step_monotone(seq))
    print(f"  monotone at every step: {perfect}/{total} ({100*perfect/max(total,1):.1f}%)")
    g.to_csv(OUT / "exp1b_relative_noise.csv", index=False)
    return g, {"worse_by_level": {f"{k}": {f"{c:g}": float(v) for c, v in row.items()}
                                  for k, row in t.iterrows()},
               "fraction_monotone": perfect / max(total, 1)}


# ---------------------------------------------------------------------------
# 1. Monotonicity
# ---------------------------------------------------------------------------

def exp1_monotonicity(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 78)
    print("EXPERIMENT 1 - MONOTONICITY")
    print("=" * 78)
    rows, detail = [], []
    orig = {r.slice_path: r.score for r in d[(d.kind == "original") & (d.source == "full")].itertuples()}
    for kind, param, unit in (("noise_white", "sigma_hu", "HU"),
                              ("noise_fbp", "sigma_hu", "HU"),
                              ("blur", "sigma_px", "px")):
        g = d[(d.kind == kind) & (d.source == "full")]
        if g.empty:
            continue
        # On the TEST bank the fine noise ladder (form 3) repeats some levels
        # of the standard ladder with different realisations: only the
        # standard ladder is used here, averaging realisations of the same
        # level, otherwise a slice would have two scores per step.
        if param == "sigma_hu":
            g = g[g[param].isin(dg.NOISE_SIGMAS_HU)]
        g = g.groupby(["slice_path", param], as_index=False).agg(
            score=("score", "mean"), cell=("cell", "first"))
        levels = sorted(g[param].unique())
        perfect = total = 0
        rhos = []
        for sp, gg in g.groupby("slice_path"):
            gg = gg.sort_values(param)
            seq = np.concatenate([[orig.get(sp, np.nan)], gg["score"].to_numpy()])
            xs = np.concatenate([[0.0], gg[param].to_numpy()])
            if not np.isfinite(seq).all():
                continue
            total += 1
            ok = step_monotone(seq, increasing=True)
            perfect += int(ok)
            rhos.append(spearman(xs, seq))
            detail.append({"kind": kind, "slice_path": sp, "monotone": ok,
                           "spearman": rhos[-1], **{f"s{j}": v for j, v in enumerate(seq)}})
        # sign test between consecutive levels
        pvals = {}
        prev_lab, prev = "original", np.array([orig.get(sp, np.nan) for sp in sorted(orig)])
        prev_idx = sorted(orig)
        for lv in levels:
            cur = g[g[param] == lv].set_index("slice_path")["score"]
            cur = np.array([cur.get(sp, np.nan) for sp in prev_idx])
            k, n, p = sign_test(cur, prev)
            # the key includes the ladder: "original -> 5HU" appears in both
            # noise_white and noise_fbp, and a key on the step alone would let
            # the second overwrite the correction of the first
            pvals[(kind, f"{prev_lab} -> {lv:g}{unit}")] = p
            rows.append({"ladder": kind, "step": f"{prev_lab} -> {lv:g}{unit}",
                         "n_worse": k, "n_valid": n,
                         "fraction_worse": k / n if n else np.nan, "p_sign": p})
            prev, prev_lab = cur, f"{lv:g}{unit}"
        adj = holm(pvals)
        for r in rows:
            key = (r["ladder"], r["step"])
            if key in adj:
                r["p_holm"] = adj[key]
        print(f"\n  {kind}: {perfect}/{total} images monotone at every step "
              f"({100*perfect/max(total,1):.1f}%), median Spearman {np.nanmedian(rhos):+.3f}")
        for r in [x for x in rows if x["ladder"] == kind]:
            print(f"    {r['step']:28s} worse in {r['n_worse']:3d}/{r['n_valid']:3d} "
                  f"({100*r['fraction_worse']:5.1f}%)  p_Holm={r.get('p_holm', np.nan):.3g}")
    df = pd.DataFrame(rows)
    det = pd.DataFrame(detail)
    summary = {
        "fraction_monotone": det.groupby("kind")["monotone"].mean().to_dict() if len(det) else {},
        "median_spearman": det.groupby("kind")["spearman"].median().to_dict() if len(det) else {},
        "every_step_worse": bool((df["fraction_worse"] > 0.99).all()) if len(df) else False,
    }
    det.to_csv(OUT / "exp1_monotonicity_detail.csv", index=False)
    return df, summary


# ---------------------------------------------------------------------------
# 2. Preference for filtered full-dose images, and for added noise
# ---------------------------------------------------------------------------

#: tolerances on the score change, to separate preferences from near ties
EPSILONS = (0.0, 0.01, 0.05, 0.1)


def exp2_overfiltering(d: pd.DataFrame, model, NU) -> tuple[pd.DataFrame, dict]:
    """Form 1: how often a filtered full-dose image scores better than its
    source. A full-dose image still contains noise, so a better score after
    filtering is not by itself an error of the metric: this is a preference
    rate, and whether the preferred filtering destroys signal is tested with
    inserted lesions (scripts/exp_lesions.py).

    Form 3: whether adding FBP-like noise to a full-dose image improves the
    score, at each fixed level, against the difference between two noise
    realisations of the same level (the noise floor of the test)."""
    from riqe.evaluate import score_all_refcov

    print("\n" + "=" * 78)
    print("EXPERIMENT 2 - PREFERENCE FOR FILTERED FULL-DOSE IMAGES (forms 1 and 3)")
    print("=" * 78)
    full = d[d.source == "full"].copy()
    full["score_refcov"] = score_all_refcov(model, NU, full["row"].to_numpy())
    orig = full[full.kind == "original"].set_index("slice_path")
    full["region"] = np.where(full.cell.str.contains("CHEST"), "chest", "abdomen")

    # --- form 1 ------------------------------------------------------------
    den = full[full.kind == "denoise"].copy()
    den["score_orig"] = den["slice_path"].map(orig["score"])
    den["delta"] = den["score"] - den["score_orig"]
    den["preferred"] = (den["delta"] < 0).astype(float)
    den["delta_refcov"] = den["score_refcov"] - den["slice_path"].map(orig["score_refcov"])
    den["preferred_refcov"] = (den["delta_refcov"] < 0).astype(float)
    den = den[den["delta"].notna()]
    print("\n  FORM 1 - filtering a full-dose image")
    print(f"  filtered images: {len(den)} from {den.slice_path.nunique()} slices, "
          f"{den.patient_id.nunique()} patients")
    eps = {f"{e:g}": float((den["delta"] < -e).mean()) for e in EPSILONS}
    print("  preferred to the source (score lower by more than eps): "
          + ", ".join(f"eps={k}: {100*v:.1f}%" for k, v in eps.items()))
    tab = den.pivot_table(index="denoiser", columns="target_residual_hu", values="preferred", aggfunc="mean")
    print("\n  preference rate by denoiser x strength (residual HU):")
    print((100 * tab).round(1).to_string())
    tab2 = den.pivot_table(index="denoiser", columns="target_residual_hu", values="delta", aggfunc="median")
    print("\n  median score change (positive = worse):")
    print(tab2.round(3).to_string())
    tab_ref = den.pivot_table(index="denoiser", columns="target_residual_hu",
                              values="preferred_refcov", aggfunc="mean")
    print("\n  ablation, image covariance left out of the distance: preference rate")
    print((100 * tab_ref).round(1).to_string())

    # --- form 3 ------------------------------------------------------------
    fine = full[full.kind == "noise_fbp"].copy()
    fine["delta"] = fine["score"] - fine["slice_path"].map(orig["score"])
    fine["delta_refcov"] = fine["score_refcov"] - fine["slice_path"].map(orig["score_refcov"])
    # noise floor: levels drawn twice with different realisations
    dup = fine.groupby(["slice_path", "sigma_hu"])["score"].agg(["count", lambda x: x.max() - x.min()])
    dup.columns = ["count", "range"]
    floor = dup[dup["count"] == 2].groupby(level=1)["range"].median()
    rows3 = []
    from scipy.stats import binomtest
    for (reg, sig), g in fine.groupby(["region", "sigma_hu"]):
        g = g.groupby("slice_path").agg(delta=("delta", "mean"), delta_refcov=("delta_refcov", "mean"),
                                        patient_id=("patient_id", "first"))
        g = g.dropna()
        k, n = int((g.delta < 0).sum()), int(len(g))
        rows3.append({"region": reg, "sigma_hu": float(sig), "n_images": n,
                      "n_patients": int(g.patient_id.nunique()),
                      "fraction_better": k / n if n else np.nan,
                      "p_sign": binomtest(k, n, 0.5).pvalue if n else np.nan,
                      "median_delta": float(g.delta.median()),
                      "fraction_better_refcov": float((g.delta_refcov < 0).mean()),
                      "noise_floor": float(floor.get(sig, np.nan))})
    r3 = pd.DataFrame(rows3)
    print("\n  FORM 3 - adding FBP-like noise to a full-dose image, per fixed level")
    print("  noise floor = median |difference| between two realisations of the same level")
    print(r3.round(4).to_string(index=False))

    # the endpoint used earlier, kept for comparison: minimum over the ladder
    amin = []
    for sp, gg in fine.groupby("slice_path"):
        gg = gg.groupby("sigma_hu")["score"].mean()
        sc = np.concatenate([[orig["score"].get(sp, np.nan)], gg.to_numpy()])
        if np.isfinite(sc).all():
            amin.append(int(np.argmin(sc)) > 0)
    frac_argmin = float(np.mean(amin)) if amin else np.nan
    print(f"\n  images whose minimum over the {fine.sigma_hu.nunique()} levels is not at zero: "
          f"{100*frac_argmin:.1f}% (inflated by taking a minimum over many near ties)")

    den.to_csv(OUT / "exp2_form1_overfiltering.csv", index=False)
    r3.to_csv(OUT / "exp2_form3_fixed_levels.csv", index=False)
    summary = {
        "form1_n_evaluated": int(len(den)),
        "form1_n_patients": int(den.patient_id.nunique()),
        "form1_preference_rate": float(den["preferred"].mean()),
        "form1_preference_rate_by_eps": eps,
        "form1_preference_rate_refcov": float(den["preferred_refcov"].mean()),
        "form1_by_denoiser": (100 * tab).round(2).to_dict(),
        "form1_by_denoiser_refcov": (100 * tab_ref).round(2).to_dict(),
        "form3_fixed_levels": r3.to_dict("records"),
        "form3_fraction_argmin_not_at_zero": frac_argmin,
    }
    return den, summary


# ---------------------------------------------------------------------------
# 3. Discrimination
# ---------------------------------------------------------------------------

def exp3_discrimination(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 78)
    print("EXPERIMENT 3 - DISCRIMINATION BETWEEN DENOISERS")
    print("=" * 78)
    low = d[(d.kind == "denoise") & (d.source == "low")]
    if low.empty:
        print("  no reduced-dose images in the bank: experiment not runnable")
        return pd.DataFrame(), {"runnable": False}
    rows, W = [], {}
    for lvl, g in low.groupby("target_residual_hu"):
        piv = g.pivot_table(index="slice_path", columns="denoiser", values="score")
        piv = piv.dropna()
        if len(piv) < 3:
            continue
        ranks = piv.rank(axis=1)
        w = kendall_w(ranks.to_numpy())
        W[float(lvl)] = w
        mean_rank = ranks.mean().sort_values()
        rows.append({"residual_hu": float(lvl), "n_images": len(piv), "kendall_W": w,
                     "order": " < ".join(mean_rank.index),
                     **{f"rank_{k}": v for k, v in mean_rank.items()},
                     **{f"score_{k}": v for k, v in piv.mean().items()}})
        print(f"\n  residual {lvl:g} HU  ({len(piv)} images)  Kendall's W = {w:.3f}")
        print("    order (best -> worst according to RIQE): " + " < ".join(mean_rank.index))
        print("    mean score: " + "  ".join(f"{k}={v:.3f}" for k, v in piv.mean().items()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp3_discrimination.csv", index=False)
    print(f"\n  agreement across images (Kendall's W): "
          f"median {np.nanmedian(list(W.values())):.3f} over {len(W)} levels")
    print("  High W = the model orders the denoisers consistently; it does not say")
    print("  that the order is the right one. Agreement with signal fidelity is")
    print("  in scripts/exp_lesions.py.")
    return df, {"runnable": True, "kendall_W_by_level": W,
                "kendall_W_median": float(np.nanmedian(list(W.values()))) if W else None}


# ---------------------------------------------------------------------------
# 4. Stability
# ---------------------------------------------------------------------------

def exp4_stability(P, C, p, B, seed, d_test, NU, SG) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 78)
    print("EXPERIMENT 4 - STABILITY")
    print("=" * 78)
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit_pids = sorted(set(split["fit"]))
    full = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(fit_pids)]
    by_patient = defaultdict(list)
    for r in full.itertuples():
        by_patient[r.patient_id].append(r.path)
    cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")

    allp = [q for pid in fit_pids for q in by_patient[pid]]
    full_model = cache.fit_fast(allp, p, n_patients=len(fit_pids))
    print(f"  full model: {len(fit_pids)} patients, {full_model.n_patches} patches, "
          f"cond={full_model.cond():.3e}")

    rng = np.random.default_rng(seed)
    d_boot, boot_models = [], []
    for _ in range(B):
        take = rng.choice(fit_pids, size=len(fit_pids), replace=True)
        m = cache.fit_fast([q for pid in take for q in by_patient[pid]], p, n_patients=len(fit_pids))
        d_boot.append(model_divergence(full_model, m))
        boot_models.append(m)
    d_boot = np.asarray(d_boot)
    print(f"  patient bootstrap (B={B}): median D {np.median(d_boot):.4f}, "
          f"p95 {np.percentile(d_boot,95):.4f}, max {d_boot.max():.4f}")

    d_lopo = []
    for pid in fit_pids:
        m = cache.fit_fast([q for q2 in fit_pids if q2 != pid for q in by_patient[q2]], p,
                           n_patients=len(fit_pids) - 1)
        d_lopo.append({"patient_id": pid, "D": model_divergence(full_model, m)})
    lopo = pd.DataFrame(d_lopo).sort_values("D", ascending=False)
    print(f"  leave-one-patient-out: median D {lopo.D.median():.5f}, "
          f"max {lopo.D.max():.5f} ({lopo.patient_id.iloc[0]})")

    cv = np.sqrt(np.diag(full_model.sigma)) / np.abs(full_model.nu)
    print(f"  coefficient of variation of the 36 components of nu: "
          f"median {np.median(cv):.3f}, max {cv.max():.3f}")

    curve = []
    for n in (10, 20, 40, 80, 120, len(fit_pids)):
        if n > len(fit_pids):
            continue
        ds = []
        for rep in range(10):
            take = np.random.default_rng(seed + rep).choice(fit_pids, size=n, replace=False)
            m = cache.fit_fast([q for pid in take for q in by_patient[pid]], p, n_patients=n)
            ds.append(model_divergence(full_model, m))
        curve.append({"n_patients": n, "D_median": float(np.median(ds)),
                      "D_p95": float(np.percentile(ds, 95))})
        print(f"    learning curve: {n:3d} patients -> D = {np.median(ds):.4f}")

    # confidence intervals on the scores of a fixed bank
    orig_rows = d_test[(d_test.kind == "original") & (d_test.source == "full")]["row"].to_numpy()
    from riqe.evaluate import score_all
    S = np.stack([score_all(m, NU, SG, orig_rows) for m in boot_models[: min(B, 60)]])
    base = score_all(full_model, NU, SG, orig_rows)
    lo, hi = np.nanpercentile(S, [2.5, 97.5], axis=0)
    width = np.nanmedian(hi - lo)
    rel = np.nanmedian((hi - lo) / base)
    rho = [spearman(base, S[i]) for i in range(S.shape[0])]
    print(f"  scores on {len(orig_rows)} TEST images: median 95% CI width = "
          f"{width:.4f} ({100*rel:.1f}% of the score)")
    print(f"  Spearman between rankings from different bootstrap models: "
          f"median {np.nanmedian(rho):.4f}, min {np.nanmin(rho):.4f}")

    pd.DataFrame(curve).to_csv(OUT / "exp4_learning_curve.csv", index=False)
    lopo.to_csv(OUT / "exp4_lopo.csv", index=False)
    summary = {
        "bootstrap_B": B,
        "D_boot_median": float(np.median(d_boot)),
        "D_boot_p95": float(np.percentile(d_boot, 95)),
        "D_lopo_median": float(lopo.D.median()),
        "D_lopo_max": float(lopo.D.max()),
        "most_influential_patient": lopo.patient_id.iloc[0],
        "cv_nu_median": float(np.median(cv)),
        "cv_nu_max": float(cv.max()),
        "learning_curve": curve,
        "ci95_width_median": float(width),
        "ci95_relative_median": float(rel),
        "spearman_bootstrap_median": float(np.nanmedian(rho)),
        "spearman_bootstrap_min": float(np.nanmin(rho)),
        "cond_sigma": full_model.cond(),
    }
    return lopo, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moments", default="experiments/bank_test_moments.npz")
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--skip", default="", help="experiments to skip, e.g. '4'")
    args = ap.parse_args()

    ch_file = ROOT / "experiments" / "hparam_choice.json"
    ch = json.loads(ch_file.read_text()) if ch_file.exists() else {}
    P = args.P if args.P is not None else ch.get("P")
    C = args.C if args.C is not None else ch.get("C")
    p = args.p if args.p is not None else ch.get("p")
    if P is None:
        ap.error("need --P --C --p, or experiments/hparam_choice.json")

    split = json.load(open(ROOT / "corpus" / "split.json"))
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    fit_pids = set(split["fit"])
    fit = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(fit_pids)]
    cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")
    model = cache.fit(list(fit["path"]), p, n_patients=fit.patient_id.nunique())

    meta, NU, SG = load_moments(args.moments)
    meta = meta[(meta.P == P) & (meta.C == C)].copy()
    d = attach_scores(meta, model, NU, SG)
    test_pids = set(split["test"])
    intruders = set(d.patient_id) - test_pids
    print(f"model: P={P} C={C:g} p={p:g}, {model.n_patches} patches from "
          f"{model.n_patients} FIT patients")
    print(f"bank: {len(d)} images, {d.slice_path.nunique()} slices, "
          f"{d.patient_id.nunique()} patients"
          + (f"  WARNING: {len(intruders)} patients outside TEST" if intruders else " (all TEST)"))
    print(f"undefined scores: {int(d.score.isna().sum())}/{len(d)}")

    t0 = time.time()
    summaries = {}
    skip = set(args.skip.split(","))
    if "0" not in skip:
        _, summaries["exp0_real_dose"] = exp0_real_dose(P, C, p, model, d)
        summaries["exp0_original_criterion"] = exp0_original_criterion()
    if "1" not in skip:
        _, summaries["exp1_monotonicity"] = exp1_monotonicity(d)
        _, summaries["exp1b_relative_noise"] = exp1b_relative_noise(d)
    if "2" not in skip:
        _, summaries["exp2_overfiltering"] = exp2_overfiltering(d, model, NU)
    if "3" not in skip:
        _, summaries["exp3_discrimination"] = exp3_discrimination(d)
    if "4" not in skip:
        _, summaries["exp4_stability"] = exp4_stability(P, C, p, args.bootstrap, args.seed, d, NU, SG)

    summaries["_config"] = {"P": P, "C": C, "p": p, "moments": args.moments,
                            "n_test_patients": int(d.patient_id.nunique())}
    (OUT / "battery_summary.json").write_text(json.dumps(summaries, indent=1, default=float))
    d.to_csv(OUT / "battery_scores.csv", index=False)
    print(f"\nwrote {OUT}/battery_summary.json and battery_scores.csv "
          f"({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
