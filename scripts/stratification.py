#!/usr/bin/env python3
"""Central experiment: is a single model enough, or is stratification needed?

For every contrast between two groups of patients, the divergence between the
two models is computed and compared with two references measured on the same
corpus:

  LOWER ANCHOR, permutation null. The patients of the two groups are pooled
  and reassigned at random, **by patient**, to two groups of the same sizes.
  The resulting distribution of D is what one would observe if the factor did
  not matter at all. It is exact, it is matched in sample size **by
  construction** (which removes the artefact whereby a small stratum looks
  divergent only because its Sigma is estimated worse), and it needs no
  distributional assumption.

  UPPER ANCHOR, a real physical difference. D between the full-dose model and
  the reduced-dose model on the **same** patients: a true acquisition
  difference, of the size the metric must detect. Its null is the paired
  permutation of the dose label within each patient.

The reported index is

    eta = (D_obs - median(null)) / (D_dose - median(null_dose))

i.e. the excess over sampling noise, in units of the excess produced by a
known physical difference.

DECISION RULE, declared before running: a stratum needs its own sub-model if
and only if
  (a) permutation p < 0.05, AND
  (b) eta >= 0.25, AND
  (c) the operational criterion fails (Spearman < 0.95 on the common bank,
      or a verdict reversal).
Conditions (a) and (b) are evaluated here; (c) in run_battery.py.

    .venv/bin/python scripts/stratification.py --P 16 --C 0.25 --p 0.2
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
from riqe.model import feature_shift, model_divergence, symmetric_kl  # noqa: E402
from riqe.nss import FEATURE_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def slices_by_patient(df: pd.DataFrame) -> dict[str, list[str]]:
    out = defaultdict(list)
    for r in df.itertuples():
        out[r.patient_id].append(r.path)
    return dict(out)


def fit_group(cache, by_patient, pids, p):
    paths = [q for pid in pids for q in by_patient.get(pid, [])]
    return cache.fit_fast(paths, p, n_patients=len(pids))


def permutation_null(cache, by_patient, pids_a, pids_b, p, B, seed):
    """Null: random reassignment **by patient** of the pool A+B."""
    rng = np.random.default_rng(seed)
    pool = list(pids_a) + list(pids_b)
    na = len(pids_a)
    out = []
    for _ in range(B):
        perm = rng.permutation(pool)
        try:
            ma = fit_group(cache, by_patient, perm[:na], p)
            mb = fit_group(cache, by_patient, perm[na:], p)
            out.append(model_divergence(ma, mb))
        except ValueError:
            out.append(np.nan)
    return np.asarray(out)


def paired_dose_anchor(cache, full_by_patient, low_by_patient, pids, p, B, seed):
    """Upper anchor and its paired null.

    Null: for every patient the full-dose / reduced-dose label is swapped at
    random. If dose did not matter, the two resulting models would be
    equivalent.
    """
    rng = np.random.default_rng(seed)
    m_full = cache.fit_fast([q for pid in pids for q in full_by_patient[pid]], p, n_patients=len(pids))
    m_low = cache.fit_fast([q for pid in pids for q in low_by_patient[pid]], p, n_patients=len(pids))
    d_obs = model_divergence(m_full, m_low)
    null = []
    for _ in range(B):
        a, b = [], []
        for pid in pids:
            if rng.random() < 0.5:
                a += full_by_patient[pid]
                b += low_by_patient[pid]
            else:
                a += low_by_patient[pid]
                b += full_by_patient[pid]
        try:
            null.append(model_divergence(
                cache.fit_fast(a, p, n_patients=len(pids)), cache.fit_fast(b, p, n_patients=len(pids))))
        except ValueError:
            null.append(np.nan)
    return d_obs, np.asarray(null), m_full, m_low


def contrast_row(name, axis, cache, by_patient, pids_a, pids_b, p, B, seed, anchor_excess):
    ma = fit_group(cache, by_patient, pids_a, p)
    mb = fit_group(cache, by_patient, pids_b, p)
    d_obs = model_divergence(ma, mb)
    null = permutation_null(cache, by_patient, pids_a, pids_b, p, B, seed)
    med = float(np.nanmedian(null))
    pval = float((np.nansum(null >= d_obs) + 1) / (np.sum(np.isfinite(null)) + 1))
    excess = d_obs - med
    eta = excess / anchor_excess if anchor_excess and np.isfinite(anchor_excess) else np.nan
    shift = feature_shift(ma, mb)
    top = np.argsort(-np.abs(np.nan_to_num(shift)))[:4]
    return {
        "contrast": name,
        "axis": axis,
        "n_A": len(pids_a),
        "n_B": len(pids_b),
        "D_obs": d_obs,
        "null_median": med,
        "null_p95": float(np.nanpercentile(null, 95)),
        "p_perm": pval,
        "eta": eta,
        "KL_sym": symmetric_kl(ma, mb),
        "cond_A": ma.cond(),
        "cond_B": mb.cond(),
        "patch_A": ma.n_patches,
        "patch_B": mb.n_patches,
        "dominant_features": ", ".join(
            f"{FEATURE_NAMES[i]}={shift[i]:+.2f}" for i in top if np.isfinite(shift[i])
        ),
        "significant_a": bool(pval < 0.05),
        "relevant_b": bool(np.isfinite(eta) and eta >= 0.25),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--B", type=int, default=200, help="permutation repetitions")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default="experiments/stratification.csv")
    args = ap.parse_args()

    choice_file = ROOT / "experiments" / "hparam_choice.json"
    if args.P is None and choice_file.exists():
        ch = json.loads(choice_file.read_text())
        args.P, args.C, args.p = ch["P"], ch["C"], ch["p"]
    if args.P is None:
        ap.error("need --P --C --p, or experiments/hparam_choice.json")
    print(f"hyperparameters: P={args.P} C={args.C:g} p={args.p:g}, B={args.B} permutations")

    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit_pids = set(split["fit"])
    cache = FeatureCache(ROOT / "data" / "features" / f"P{args.P}_C{args.C:g}")

    full = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(fit_pids)]
    low = corpus[(corpus["kind"] == "low") & corpus["keep"] & corpus.patient_id.isin(fit_pids)]
    by_full = slices_by_patient(full)
    by_low = slices_by_patient(low)

    meta = full.groupby("patient_id").agg(
        cell=("cell", "first"), manufacturer=("manufacturer", "first"),
        body_part=("body_part", "first"), kernel=("kernel", "first"),
        thickness=("slice_thickness", "first"), ps=("pixel_spacing", "median"))
    print(f"FIT patients: {len(meta)}; with reduced dose: {len(by_low)}")
    print(meta.groupby("cell").size().to_string())

    t0 = time.time()
    # --- upper anchor: paired dose contrast ---------------------------------
    dose_pids = sorted(set(by_low) & set(by_full))
    d_dose, null_dose, m_full_d, m_low_d = paired_dose_anchor(
        cache, by_full, by_low, dose_pids, args.p, args.B, args.seed)
    anchor_excess = d_dose - float(np.nanmedian(null_dose))
    print(f"\nUPPER ANCHOR (full vs reduced dose, {len(dose_pids)} paired patients):")
    print(f"  D = {d_dose:.4f}, paired null median {np.nanmedian(null_dose):.4f}, "
          f"p95 {np.nanpercentile(null_dose,95):.4f}, excess {anchor_excess:.4f}")

    # --- contrasts -----------------------------------------------------------
    g = lambda **kw: sorted(meta[np.logical_and.reduce(
        [meta[k] == v for k, v in kw.items()])].index)

    contrasts = []
    ge_abd = g(manufacturer="GE", body_part="ABDOMEN")
    si_abd = g(manufacturer="SIEMENS", body_part="ABDOMEN")
    ge_chest = sorted(meta[(meta.manufacturer == "GE") & (meta.body_part == "CHEST")].index)
    si_chest = g(manufacturer="SIEMENS", body_part="CHEST")
    contrasts.append(("A: abdomen 5 mm, GE STANDARD vs Siemens B30f", "kernel/vendor", ge_abd, si_abd))
    contrasts.append(("B: chest, GE STANDARD 1.25 vs Siemens B50f 1.5", "kernel/vendor", ge_chest, si_chest))
    contrasts.append(("C: GE, chest 1.25 vs abdomen 5", "thickness+anatomy (confounded)", ge_chest, ge_abd))
    contrasts.append(("D: Siemens, chest B50f 1.5 vs abdomen B30f 5", "kernel+thickness+anatomy", si_chest, si_abd))
    # WARNING: this is NOT a negative control on anatomy, however tempting it
    # is to call it one. In this collection the chest is *always* a sharp
    # kernel with thin slices and the abdomen *always* a soft kernel with
    # thick slices: the contrast mixes anatomy, kernel and thickness
    # inseparably. A clean negative control on anatomy **does not exist** in
    # these data, and must be declared as a limitation rather than simulated
    # with this contrast.
    contrasts.append(("G: chest vs abdomen (anatomy CONFOUNDED with kernel and thickness)",
                      "anatomy+kernel+thickness (NOT a negative control)",
                      sorted(set(ge_chest) | set(si_chest)), sorted(set(ge_abd) | set(si_abd))))
    contrasts.append(("H: vendor, all GE vs all Siemens", "vendor",
                      sorted(meta[meta.manufacturer == "GE"].index),
                      sorted(meta[meta.manufacturer == "SIEMENS"].index)))
    # E: pixel spacing, tertiles within the same cell so as not to confound with protocol
    for cell in sorted(meta.cell.unique()):
        sub = meta[meta.cell == cell].sort_values("ps")
        if len(sub) < 20:
            continue
        k = len(sub) // 3
        contrasts.append((f"E: pixel spacing, low vs high tertile — {cell}",
                          "spatial sampling",
                          sorted(sub.index[:k]), sorted(sub.index[-k:])))

    rows = []
    for i, (name, axis, a, b) in enumerate(contrasts):
        if len(a) < 5 or len(b) < 5:
            print(f"  skipping {name}: groups too small ({len(a)}, {len(b)})")
            continue
        r = contrast_row(name, axis, cache, by_full, a, b, args.p, args.B,
                         args.seed + 31 * i, anchor_excess)
        rows.append(r)
        print(f"  {name[:52]:52s} D={r['D_obs']:.4f} null={r['null_median']:.4f} "
              f"p={r['p_perm']:.3f} eta={r['eta']:+.3f}", flush=True)

    rows.append({
        "contrast": "ANCHOR: full vs reduced dose (same patients)",
        "axis": "dose", "n_A": len(dose_pids), "n_B": len(dose_pids),
        "D_obs": d_dose, "null_median": float(np.nanmedian(null_dose)),
        "null_p95": float(np.nanpercentile(null_dose, 95)),
        "p_perm": float((np.nansum(null_dose >= d_dose) + 1) / (np.sum(np.isfinite(null_dose)) + 1)),
        "eta": 1.0, "KL_sym": symmetric_kl(m_full_d, m_low_d),
        "cond_A": m_full_d.cond(), "cond_B": m_low_d.cond(),
        "patch_A": m_full_d.n_patches, "patch_B": m_low_d.n_patches,
        "dominant_features": "", "significant_a": True, "relevant_b": True,
    })

    df = pd.DataFrame(rows)
    df["decision_ab"] = np.where(
        df.significant_a & df.relevant_b,
        "(a) and (b) met: check operational (c)",
        np.where(df.significant_a, "distinguishable but below the relevance threshold",
                 "indistinguishable from sampling noise"))
    out = ROOT / args.out
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({(time.time()-t0)/60:.1f} min)")
    print(df[["contrast", "axis", "n_A", "n_B", "D_obs", "null_median",
              "p_perm", "eta", "decision_ab"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
