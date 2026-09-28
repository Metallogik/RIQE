#!/usr/bin/env python3
"""Fit the generic model and write the artifact.

The artifact is not the file of numbers: it is the file of numbers **plus
everything needed to redo them**. That is the precise defect of existing work,
which publishes a result and not the means to reproduce it. It contains:

  * nu, Sigma and the precomputed pseudo-inverse with the threshold used;
  * the complete intensity mapping (HU window, scale, C, kernel, mask rules),
    which *defines* the model;
  * the hyperparameters and the criterion that chose them;
  * the identity of the corpus: collection, DOI, version, every PatientID,
    SeriesInstanceUID and SOPInstanceUID used, with the SHA-256 of every
    slice, and the exclusion counts per criterion;
  * the identity of the code: git commit, SHA-256 of the source, library
    versions, generator seeds;
  * licence and required attributions.

    .venv/bin/python scripts/fit_model.py --P 16 --C 0.25 --p 0.2
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
import hashlib
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import riqe  # noqa: E402
from riqe.cache import FeatureCache  # noqa: E402
from riqe.extract import Spec  # noqa: E402
from riqe.model import RCOND  # noqa: E402
from riqe.nss import FEATURE_NAMES, _ALPHA_MAX, _ALPHA_MIN, _ALPHA_N  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "riqe-v1.0"


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
        return out + ("-dirty" if dirty.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def code_hash() -> dict:
    per_file, h = {}, hashlib.sha256()
    for f in sorted((ROOT / "riqe").glob("*.py")):
        b = f.read_bytes()
        per_file[f.name] = hashlib.sha256(b).hexdigest()
        h.update(f.name.encode())
        h.update(b)
    return {"package_sha256": h.hexdigest(), "files": per_file}


def slice_hashes() -> dict[str, str]:
    """Relative slice path -> SHA-256, from the download manifests.

    The link is by path, not by SOPInstanceUID: the download manifests index
    by file name within the series, and the file name is not the SOP UID.
    Linking directly would give zero matches, i.e. an artifact that *claims*
    to carry the hashes without carrying any.
    """
    out = {}
    for m in sorted((ROOT / "data" / "manifest").glob("*.json")):
        d = json.loads(m.read_text())
        base = f"data/dicom/{d['patient_id']}/{d['kind']}"
        for f in d["files"]:
            out[f"{base}/{f['file']}"] = f["sha256"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    choice_file = ROOT / "experiments" / "hparam_choice.json"
    choice = json.loads(choice_file.read_text()) if choice_file.exists() else {}
    if args.P is None:
        args.P, args.C, args.p = choice.get("P"), choice.get("C"), choice.get("p")
    if args.P is None:
        ap.error("need --P --C --p, or experiments/hparam_choice.json")

    spec = Spec(P=args.P, C=args.C, p=args.p)
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    manifest = json.loads((ROOT / "corpus" / "corpus_manifest.json").read_text())

    fit_pids = set(split["fit"])
    fit_slices = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(fit_pids)]
    cache = FeatureCache(ROOT / "data" / "features" / f"P{args.P}_C{args.C:g}")

    print(f"fit: {len(fit_slices)} slices, {fit_slices.patient_id.nunique()} patients, "
          f"P={args.P} C={args.C:g} p={args.p:g}")
    model = cache.fit(list(fit_slices["path"]), args.p, n_patients=fit_slices.patient_id.nunique())
    print(f"selected patches: {model.n_patches} ({model.meta['patches_per_slice']:.1f}/slice)")
    print(f"cond(Sigma) = {model.cond():.3e}")

    pinv = np.linalg.pinv(model.sigma, rcond=RCOND)
    sha = slice_hashes()
    path_of = {r.sop_uid: r.path for r in corpus.itertuples()}
    used = []
    for s in manifest["slices"]:
        if s["patient_id"] not in fit_pids:
            continue
        pth = path_of.get(s["sop_uid"], "")
        used.append({**s, "path": pth, "sha256": sha.get(pth, "")})
    missing = sum(1 for s in used if not s["sha256"])
    if missing:
        raise SystemExit(
            f"ERROR: {missing} of {len(used)} slices without SHA-256. The artifact "
            f"is not written: a model that claims the identity of its corpus "
            f"without carrying it is exactly the defect this work addresses."
        )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{MODEL_ID}.npz", nu=model.nu, sigma=model.sigma, pinv_sigma=pinv)

    art = {
        "model_id": MODEL_ID,
        "created": date.today().isoformat(),
        "description": (
            "NIQE-style no-reference reference model, fitted on full-dose CT. "
            "The score is a distance from this model, not an absolute judgement "
            "of diagnostic quality: the model does not know what a lesion is, "
            "and its values are not comparable with NIQE values in the literature."
        ),
        "license": {
            "model": "CC BY 4.0",
            "code": "MIT",
            "holder": "Metallogik",
            "inherited_attribution": manifest["required_citation"],
            "inherited_acknowledgement": manifest["required_acknowledgement"],
            "algorithm_source": (
                "Mittal A., Soundararajan R., Bovik A.C., Making a 'Completely Blind' "
                "Image Quality Analyzer, IEEE SPL 20(3):209-212, 2013. Implemented "
                "from the paper; no third-party code incorporated."
            ),
        },
        "dimensions": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "arrays": {
            "file": f"{MODEL_ID}.npz",
            "nu": "mean vector (36,)",
            "sigma": "covariance (36, 36)",
            "pinv_sigma": f"pseudo-inverse of Sigma, rcond={RCOND:g}",
            "sha256": hashlib.sha256((out / f"{MODEL_ID}.npz").read_bytes()).hexdigest(),
        },
        "intensity_mapping": {
            **spec.as_dict(),
            "formula": "L = 255 * (clip(HU, -1000, 1000) + 1000) / 2000, float32, not rounded",
            "note": (
                "The mapping is part of the model specification: different "
                "windows produce mutually incompatible models."
            ),
        },
        "masking": {
            "fov_required_fraction": 1.0,
            "fov_rule": "pixel > PixelPaddingValue, and >= -1024 HU when the tag is missing",
            "body_min_fraction_per_patch": spec.body_min,
            "body_rule": "HU > -300, 5x5 opening, largest connected component, holes filled",
            "why": (
                "GE images in this collection carry 55,772 pixels per slice (21.3%) "
                "of -3024 HU padding outside the field of view; Siemens images do "
                "not. Without a FOV mask the vendor difference one would measure is "
                "a DICOM convention, not the physics of the acquisition."
            ),
        },
        "estimator": {
            "gaussian_window": "7x7, sigma 1.0, unit volume",
            "ggd": "moments (Sharifi & Leon-Garcia 1995)",
            "aggd": "moments (Lasmar, Stitou, Berthoumieu 2009)",
            "alpha_inversion_grid": {"min": _ALPHA_MIN, "max": _ALPHA_MAX, "n": _ALPHA_N},
            "scales": 2,
            "second_scale": "same Gaussian kernel, 2:1 decimation, patch P/2",
            "score_formula": "sqrt((nu1-nu2)^T ((S1+S2)/2)^-1 (nu1-nu2)), pinv rcond=%g" % RCOND,
        },
        "hyperparameters": {
            "P": args.P, "C": args.C, "p": args.p,
            "selection_criterion": choice.get("criterion", "not recorded"),
            "declared_before_run": choice.get("declared_before_run"),
            "search_metrics": choice.get("metrics"),
        },
        "fit": {
            "n_patches": model.n_patches,
            "n_slices": model.n_images,
            "n_patients": model.n_patients,
            "patches_per_slice": model.meta["patches_per_slice"],
            "cond_sigma": model.cond(),
            "split": "FIT (the TEST partition was not used for fitting)",
        },
        "corpus": {
            k: manifest[k] for k in (
                "collection", "collection_doi", "collection_version", "license",
                "license_uri", "subsets_used", "subsets_excluded",
                "inclusion_thresholds", "exclusion_report", "cells")
        },
        "corpus_split": split,
        "corpus_slices_used": used,
        "code": {
            **code_hash(),
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "pydicom": __import__("pydicom").__version__,
            "package": getattr(riqe, "__name__", "riqe"),
        },
        "reproduce": (
            "scripts/download_corpus.py -> scripts/build_slice_table.py -> "
            "scripts/make_corpus.py -> scripts/cache_features.py -> "
            "scripts/fit_model.py ; then scripts/verify_model.py for the comparison"
        ),
    }
    (out / f"{MODEL_ID}.json").write_text(json.dumps(art, indent=1))
    print(f"\nwrote {out}/{MODEL_ID}.npz and {MODEL_ID}.json "
          f"({len(used)} slices recorded with SHA-256)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
