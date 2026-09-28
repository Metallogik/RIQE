#!/usr/bin/env python3
"""Apply the inclusion criteria, define the split and write the corpus
manifest.

Input  : corpus/slices.parquet (from build_slice_table.py)
Output : corpus/corpus.parquet        full table with S1-S6 outcomes
         corpus/split.json            patient-level split, frozen
         corpus/exclusion_report.csv  counts per criterion
         corpus/corpus_manifest.json  identity of the corpus for the artefact

The split is **by patient**, never by slice: slices of one patient are
strongly correlated, and splitting them would inflate every stability
estimate. It is stratified by protocol cell.

    .venv/bin/python scripts/make_corpus.py [--test-frac 0.2] [--seed 20260917]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe import inclusion  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def code_hash() -> str:
    """Hash of the source that determines the numbers: identity of the code."""
    h = hashlib.sha256()
    for f in sorted((ROOT / "riqe").glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def stratified_patient_split(df: pd.DataFrame, test_frac: float, seed: int):
    """Patient-level split, stratified by protocol cell."""
    rng = np.random.default_rng(seed)
    per_patient = df.groupby("patient_id")["cell"].agg(lambda s: s.mode().iloc[0])
    fit, test = [], []
    for cell, grp in per_patient.groupby(per_patient):
        pids = sorted(grp.index)
        rng.shuffle(pids)
        n_test = int(round(len(pids) * test_frac))
        test += pids[:n_test]
        fit += pids[n_test:]
    return sorted(fit), sorted(test)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / "corpus" / "slices.parquet")
    print(f"slices read: {len(df)}  patients: {df.patient_id.nunique()}")

    full = df[df["kind"] == "full"].copy()
    low = df[df["kind"] == "low"].copy()
    print(f"  full dose: {len(full)} slices, {full.patient_id.nunique()} patients")
    print(f"  reduced dose: {len(low)} slices, {low.patient_id.nunique()} patients")

    # The criteria define the pristine (full-dose) corpus. Reduced-dose slices
    # go through the same geometric criteria but are not pristine: they serve
    # as test images and as the upper anchor of model divergence.
    full_c = inclusion.apply_criteria(full)
    low_c = inclusion.apply_criteria(low)

    rep = inclusion.exclusion_report(full_c)
    print("\nexclusions on the full-dose corpus (in cascade):")
    print(rep.to_string(index=False))
    rep.to_csv(ROOT / "corpus" / "exclusion_report.csv", index=False)

    kept = full_c[full_c["keep"]]
    print(f"\nkept: {len(kept)} slices from {kept.patient_id.nunique()} patients")
    print(kept.groupby("cell").agg(slices=("sop_uid", "size"), patients=("patient_id", "nunique")).to_string())

    fit_ids, test_ids = stratified_patient_split(kept, args.test_frac, args.seed)
    # inner split of FIT for the hyperparameter search
    inner = kept[kept.patient_id.isin(fit_ids)]
    fit_inner, val_inner = stratified_patient_split(inner, 0.25, args.seed + 1)

    split = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "unit": "patient",
        "fit": fit_ids,
        "test": test_ids,
        "fit_inner": fit_inner,
        "val_inner": val_inner,
    }
    (ROOT / "corpus" / "split.json").write_text(json.dumps(split, indent=1))
    print(f"\nsplit: FIT {len(fit_ids)} patients (inner {len(fit_inner)}/"
          f"{len(val_inner)}), TEST {len(test_ids)} patients")

    out = pd.concat([full_c, low_c], ignore_index=True)
    out["split"] = np.where(
        out.patient_id.isin(test_ids), "test",
        np.where(out.patient_id.isin(fit_ids), "fit", "unused"),
    )
    out.to_parquet(ROOT / "corpus" / "corpus.parquet", index=False)

    manifest = {
        "collection": "LDCT-and-Projection-data",
        "collection_doi": "https://doi.org/10.7937/9npb-2637",
        "collection_version": 7,
        "license": "Creative Commons Attribution 4.0 International",
        "license_uri": "https://creativecommons.org/licenses/by/4.0/",
        "subsets_used": ["Chest", "Liver/Abdomen"],
        "subsets_excluded": {
            "Head": "NIH Controlled Data Access; not exposed by the public NBIA API"
        },
        "required_citation": (
            "McCollough, C., Chen, B., Holmes III, D., Duan, X., Yu, Z., Yu, L., "
            "Leng, S., Fletcher, J. (2020). Low Dose CT Image and Projection Data "
            "(LDCT-and-Projection-data) (Version 7) [dataset]. The Cancer Imaging "
            "Archive. https://doi.org/10.7937/9npb-2637"
        ),
        "required_acknowledgement": "NIBIB grants EB017095 and EB017185 (PI: Cynthia McCollough)",
        "inclusion_thresholds": {
            "S1_trim": inclusion.S1_TRIM,
            "S2_body_frac_fov": list(inclusion.S2_BODY_FRAC),
            "S3_ring_px": inclusion.S3_RING_PX,
            "S3_max_on_ring": inclusion.S3_MAX_ON_RING,
            "S4_metal_hu": inclusion.S4_METAL_HU,
            "S4_max_frac": inclusion.S4_MAX_FRAC,
            "S5_enabled": inclusion.S5_ENABLED,
            "S5_max_robust_z": inclusion.S5_MAX_Z,
            "S6_per_patient": inclusion.S6_PER_PATIENT,
        },
        "exclusion_report": rep.to_dict("records"),
        "split": split,
        "n_slices_kept_full_dose": int(len(kept)),
        "n_patients_kept": int(kept.patient_id.nunique()),
        "cells": kept.groupby("cell").size().to_dict(),
        "code": {"git_commit": git_commit(), "riqe_sha256": code_hash()},
        "slices": [
            {
                "patient_id": r.patient_id,
                "series_uid": r.series_uid,
                "sop_uid": r.sop_uid,
                "z": float(r.z),
                "cell": r.cell,
                "split": "test" if r.patient_id in set(test_ids) else "fit",
            }
            for r in kept.itertuples()
        ],
    }
    (ROOT / "corpus" / "corpus_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote corpus.parquet, split.json, corpus_manifest.json, exclusion_report.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
