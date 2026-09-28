#!/usr/bin/env python3
"""Fit del modello generico e scrittura dell'artefatto.

L'artefatto non e' il file dei numeri: e' il file dei numeri **piu' tutto cio'
che serve a rifarli**.  E' il difetto preciso del lavoro esistente, che
pubblica un risultato e non i mezzi per riprodurlo.  Qui dentro finiscono:

  * nu, Sigma e la pseudo-inversa precalcolata con la soglia usata;
  * la mappatura di intensita' completa (finestra HU, scala, C, kernel,
    regole di maschera), che *definisce* il modello;
  * gli iperparametri e il criterio che li ha scelti;
  * l'identita' del corpus: collezione, DOI, versione, ogni PatientID,
    SeriesInstanceUID e SOPInstanceUID usato, con lo SHA-256 di ogni slice,
    e i conteggi di esclusione per criterio;
  * l'identita' del codice: commit git, SHA-256 del sorgente, versioni delle
    librerie, semi dei generatori;
  * licenza e attribuzioni dovute.

    .venv/bin/python scripts/fit_model.py --P 16 --C 0.25 --p 0.2
"""

from __future__ import annotations

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
    """SOPInstanceUID -> SHA-256, dai manifest di scaricamento."""
    out = {}
    for m in (ROOT / "data" / "manifest").glob("*.json"):
        d = json.loads(m.read_text())
        for f in d["files"]:
            out[Path(f["file"]).stem] = f["sha256"]
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
        ap.error("servono --P --C --p, oppure experiments/hparam_choice.json")

    spec = Spec(P=args.P, C=args.C, p=args.p)
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    manifest = json.loads((ROOT / "corpus" / "corpus_manifest.json").read_text())

    fit_pids = set(split["fit"])
    fit_slices = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(fit_pids)]
    cache = FeatureCache(ROOT / "data" / "features" / f"P{args.P}_C{args.C:g}")

    print(f"fit: {len(fit_slices)} slice, {fit_slices.patient_id.nunique()} pazienti, "
          f"P={args.P} C={args.C:g} p={args.p:g}")
    model = cache.fit(list(fit_slices["path"]), args.p, n_patients=fit_slices.patient_id.nunique())
    print(f"patch selezionate: {model.n_patches} ({model.meta['patches_per_slice']:.1f}/slice)")
    print(f"cond(Sigma) = {model.cond():.3e}")

    pinv = np.linalg.pinv(model.sigma, rcond=RCOND)
    sha = slice_hashes()
    used = [
        {**s, "sha256": sha.get(s["sop_uid"], "")}
        for s in manifest["slices"] if s["patient_id"] in fit_pids
    ]
    missing = sum(1 for s in used if not s["sha256"])
    if missing:
        print(f"ATTENZIONE: {missing} slice senza SHA-256 nel manifest di scaricamento")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{MODEL_ID}.npz", nu=model.nu, sigma=model.sigma, pinv_sigma=pinv)

    art = {
        "model_id": MODEL_ID,
        "created": date.today().isoformat(),
        "description": (
            "Modello di riferimento no-reference in stile NIQE, fittato su TC a "
            "dose piena. Il punteggio e' una distanza rispetto a questo modello, "
            "non un giudizio assoluto di qualita' diagnostica: il modello non sa "
            "cosa sia una lesione, e i valori non sono confrontabili con valori "
            "NIQE della letteratura."
        ),
        "license": {
            "model": "CC BY 4.0",
            "code": "MIT",
            "holder": "Metallogik",
            "inherited_attribution": manifest["required_citation"],
            "inherited_acknowledgement": manifest["required_acknowledgement"],
            "algorithm_source": (
                "Mittal A., Soundararajan R., Bovik A.C., Making a 'Completely Blind' "
                "Image Quality Analyzer, IEEE SPL 20(3):209-212, 2013. Implementato "
                "dall'articolo; nessun codice di terzi incorporato."
            ),
        },
        "dimensions": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "arrays": {
            "file": f"{MODEL_ID}.npz",
            "nu": "vettore medio (36,)",
            "sigma": "covarianza (36, 36)",
            "pinv_sigma": f"pseudo-inversa di Sigma, rcond={RCOND:g}",
            "sha256": hashlib.sha256((out / f"{MODEL_ID}.npz").read_bytes()).hexdigest(),
        },
        "intensity_mapping": {
            **spec.as_dict(),
            "formula": "L = 255 * (clip(HU, -1000, 1000) + 1000) / 2000, float32 non arrotondato",
            "note": (
                "La mappatura fa parte della specifica del modello: finestre "
                "diverse producono modelli fra loro incompatibili."
            ),
        },
        "masking": {
            "fov_required_fraction": 1.0,
            "fov_rule": "pixel > PixelPaddingValue, e >= -1024 HU quando il tag manca",
            "body_min_fraction_per_patch": spec.body_min,
            "body_rule": "HU > -300, apertura 5x5, componente connessa maggiore, riempimento buchi",
            "why": (
                "Le immagini GE di questa collezione portano 55.772 pixel per slice "
                "(21,3%) a -3024 HU di riempimento fuori campo, le Siemens no. Senza "
                "maschera FOV la differenza fra costruttori che si misurerebbe e' una "
                "convenzione DICOM, non la fisica dell'acquisizione."
            ),
        },
        "estimator": {
            "gaussian_window": "7x7, sigma 1.0, volume unitario",
            "ggd": "momenti (Sharifi & Leon-Garcia 1995)",
            "aggd": "momenti (Lasmar, Stitou, Berthoumieu 2009)",
            "alpha_inversion_grid": {"min": _ALPHA_MIN, "max": _ALPHA_MAX, "n": _ALPHA_N},
            "scales": 2,
            "second_scale": "stesso kernel gaussiano, decimazione 2:1, patch P/2",
            "score_formula": "sqrt((nu1-nu2)^T ((S1+S2)/2)^-1 (nu1-nu2)), pinv rcond=%g" % RCOND,
        },
        "hyperparameters": {
            "P": args.P, "C": args.C, "p": args.p,
            "selection_criterion": choice.get("criterion", "non registrato"),
            "declared_before_run": choice.get("declared_before_run"),
            "search_metrics": choice.get("metrics"),
        },
        "fit": {
            "n_patches": model.n_patches,
            "n_slices": model.n_images,
            "n_patients": model.n_patients,
            "patches_per_slice": model.meta["patches_per_slice"],
            "cond_sigma": model.cond(),
            "split": "FIT (la partizione TEST non e' stata usata per il fitting)",
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
            "scripts/fit_model.py ; poi scripts/verify_model.py per il confronto"
        ),
    }
    (out / f"{MODEL_ID}.json").write_text(json.dumps(art, indent=1))
    print(f"\nscritti {out}/{MODEL_ID}.npz e {MODEL_ID}.json "
          f"({len(used)} slice registrate con SHA-256)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
