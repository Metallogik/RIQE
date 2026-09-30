#!/usr/bin/env python3
"""Confidence intervals for every endpoint, resampling patients.

Slices, dose pairs, degraded versions and filtered versions of one patient
are not independent observations. Every interval here is a percentile
bootstrap over **patients** (B = 2000): a resample draws patients with
replacement and keeps all their rows together. Point estimates are the same
as in the experiment outputs; only the intervals are added.

Reads the outputs of run_battery.py, exp_lesions.py and exp5_photo_baseline.py
and writes experiments/uncertainty.json. LDCTIQAC intervals, clustered by
inferred source slice, are computed in exp6_ldctiqac.py.

    .venv/bin/python scripts/uncertainty.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
B = 2000
SEED = 20260917


def boot(frame: pd.DataFrame, stat, cluster: str = "patient_id") -> dict:
    """Percentile interval of stat(frame) over patient resamples."""
    frame = frame.reset_index(drop=True)
    idx = [g.index.to_numpy() for _, g in frame.groupby(cluster)]
    rng = np.random.default_rng(SEED)
    vals = np.empty(B)
    for b in range(B):
        take = rng.integers(0, len(idx), len(idx))
        vals[b] = stat(frame.iloc[np.concatenate([idx[i] for i in take])])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"estimate": float(stat(frame)), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(len(frame)), "n_patients": int(len(idx))}


def all_correct_patients(frame: pd.DataFrame, col: str) -> dict:
    """Patients in which every unit is correct, with the exact (Clopper-Pearson)
    95% interval. When every patient is at 100%, the percentile bootstrap of the
    pooled rate collapses to [100, 100], which expresses no uncertainty; this
    endpoint does. It is a different quantity: the probability that all units
    of a patient are correct, not the rate of correct units."""
    from scipy.stats import binomtest

    per = frame.groupby("patient_id")[col].min()
    k, n = int((per == 1).sum()), int(len(per))
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return {"k": k, "n": n, "ci_low": float(ci.low), "ci_high": float(ci.high)}


def prop(col):
    return lambda f: float(f[col].mean())


def med(col):
    return lambda f: float(f[col].median())


def patient_of(path: str) -> str:
    return path.split("/")[2]


def main() -> int:
    out: dict = {"method": f"percentile bootstrap over patients, B={B}, seed {SEED}"}

    # --- reduced dose (projection-domain simulated) --------------------------
    d0 = pd.read_csv(EXP / "exp0_real_dose.csv").dropna(subset=["correct"])
    out["dose"] = {reg: boot(g, prop("correct")) for reg, g in d0.groupby("region")}
    out["dose_patients_all_correct"] = {reg: all_correct_patients(g, "correct")
                                        for reg, g in d0.groupby("region")}
    if (EXP / "exp0_original_criterion.csv").exists():
        d0o = pd.read_csv(EXP / "exp0_original_criterion.csv").dropna(subset=["correct"])
        out["dose_original_criterion"] = {reg: boot(g, prop("correct")) for reg, g in d0o.groupby("region")}

    # --- relative noise and blur -------------------------------------------------
    rel = pd.read_csv(EXP / "exp1b_relative_noise.csv")
    out["relative_noise"] = {f"{reg}|{lv:g}": boot(g, prop("worse"))
                             for (reg, lv), g in rel.groupby(["region", "rel_increase"])}
    out["relative_noise_patients_all_correct"] = {
        f"{reg}|{lv:g}": all_correct_patients(g, "worse")
        for (reg, lv), g in rel.groupby(["region", "rel_increase"])}
    bs = pd.read_csv(EXP / "battery_scores.csv")
    full = bs[bs.source == "full"].copy()
    orig = full[full.kind == "original"].set_index("slice_path")["score"]
    full["region"] = np.where(full.cell.str.contains("CHEST"), "chest", "abdomen")
    bl = full[full.kind == "blur"].copy()
    bl["worse"] = (bl.score > bl.slice_path.map(orig)).astype(float)
    bl = bl.dropna(subset=["score"])
    out["blur"] = {f"{reg}|{s:g}": boot(g, prop("worse"))
                   for (reg, s), g in bl.groupby(["region", "sigma_px"])}
    out["blur_patients_all_correct"] = {f"{reg}|{s:g}": all_correct_patients(g, "worse")
                                        for (reg, s), g in bl.groupby(["region", "sigma_px"])}

    # --- preference for filtered full-dose images -------------------------------
    f1 = pd.read_csv(EXP / "exp2_form1_overfiltering.csv")
    out["filtered_preference"] = {"all": boot(f1, prop("preferred")),
                                  "all_refcov": boot(f1, prop("preferred_refcov"))}
    for e in (0.01, 0.05, 0.1):
        f1[f"pref_eps{e:g}"] = (f1["delta"] < -e).astype(float)
        out["filtered_preference"][f"eps{e:g}"] = boot(f1, prop(f"pref_eps{e:g}"))
    out["filtered_preference_by_denoiser"] = {
        f"{n}|{lv:g}": boot(g, prop("preferred"))
        for (n, lv), g in f1.groupby(["denoiser", "target_residual_hu"])}

    # --- added noise at fixed levels --------------------------------------------
    nz = full[full.kind == "noise_fbp"].copy()
    nz["delta"] = nz.score - nz.slice_path.map(orig)
    nz = nz.groupby(["slice_path", "sigma_hu"], as_index=False).agg(
        delta=("delta", "mean"), patient_id=("patient_id", "first"), region=("region", "first"))
    nz["better"] = (nz["delta"] < 0).astype(float)
    out["added_noise_better"] = {f"{reg}|{s:g}": boot(g.dropna(subset=["delta"]), prop("better"))
                                 for (reg, s), g in nz.groupby(["region", "sigma_hu"])}

    # --- lesions -------------------------------------------------------------------
    les = pd.read_csv(EXP / "exp2_form2_lesions.csv")
    les["patient_id"] = les.slice_path.map(patient_of)
    x = les[les.diameter_mm == 4.0].copy()
    base = x[x.denoiser == "none"].set_index("slice_path")
    x = x[x.denoiser != "none"].copy()
    x["preferred"] = (x.riqe < x.slice_path.map(base.riqe)).astype(float)
    x["preferred_refcov"] = (x.riqe_refcov < x.slice_path.map(base.riqe_refcov)).astype(float)
    lesions = {}
    for (n, lv), g in x.groupby(["denoiser", "target_residual_hu"]):
        lesions[f"{n}|{lv:g}"] = {"preferred": boot(g, prop("preferred")),
                                  "preferred_refcov": boot(g, prop("preferred_refcov")),
                                  "retention": boot(g, med("retention_matched")),
                                  "dprime": boot(g, med("dprime"))}
    lesions["none"] = {"dprime": boot(base.reset_index(), med("dprime"))}
    # paired comparisons at matched strength, per slice
    for lv in (16.0, 32.0):
        a = x[(x.denoiser == "bilateral") & (x.target_residual_hu == lv)].set_index("slice_path")
        g = x[(x.denoiser == "gaussian") & (x.target_residual_hu == lv)].set_index("slice_path")
        pair = pd.DataFrame({"d_dprime": a.dprime - g.dprime,
                             "d_retention": a.retention_matched - g.retention_matched,
                             "patient_id": a.patient_id}).dropna()
        lesions[f"bilateral_minus_gaussian|{lv:g}"] = {
            "dprime": boot(pair, med("d_dprime")), "retention": boot(pair, med("d_retention"))}
    out["lesions_4mm"] = lesions

    # --- photographic baselines ------------------------------------------------------
    if (EXP / "exp5_dose_pairs.csv").exists():
        dp = pd.read_csv(EXP / "exp5_dose_pairs.csv").dropna(subset=["correct"])
        out["baseline_dose"] = {f"{m}|{reg}": boot(g, prop("correct"))
                                for (m, reg), g in dp.groupby(["model", "region"])}
        sc = pd.read_csv(EXP / "exp5_scores_three_models.csv")
        sc = sc[sc.source == "full"]
        res = {}
        for col in ("riqe", "photo", "niqe_params"):
            if col not in sc:
                continue
            o = sc[sc.kind == "original"].set_index("slice_path")[col]
            r = sc[sc.kind == "noise_rel"].copy()
            r["worse"] = (r[col] > r.slice_path.map(o)).astype(float)
            for lv in (0.2, 1.0):
                g = r[(r.rel_increase == lv)].dropna(subset=[col])
                res[f"{col}|noise+{int(100*lv)}%"] = boot(g, prop("worse"))
            dn = sc[sc.kind == "denoise"].copy()
            dn["pref"] = (dn[col] < dn.slice_path.map(o)).astype(float)
            res[f"{col}|filtered_preference"] = boot(dn.dropna(subset=[col]), prop("pref"))
        out["baseline_detection"] = res

    (EXP / "uncertainty.json").write_text(json.dumps(out, indent=1))
    d = out["dose"]
    print("reduced dose, correct: " + ", ".join(
        f"{k} {100*v['estimate']:.1f}% [{100*v['ci_low']:.1f}, {100*v['ci_high']:.1f}] "
        f"({v['n']} pairs, {v['n_patients']} patients)" for k, v in d.items()))
    fp = out["filtered_preference"]["all"]
    print(f"filtered preference: {100*fp['estimate']:.1f}% [{100*fp['ci_low']:.1f}, {100*fp['ci_high']:.1f}]")
    print(f"wrote {EXP / 'uncertainty.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
