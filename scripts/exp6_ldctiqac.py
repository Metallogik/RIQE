#!/usr/bin/env python3
"""Esperimento 6 - accordo con i giudizi dei radiologi (LDCTIQAC 2023).

Dati: LDCTIQAC 2023 training set (Zenodo 10.5281/zenodo.7833096, CC BY 4.0),
1000 immagini TC addominali a bassa dose, ciascuna con il punteggio medio di
cinque radiologi su scala 0 (peggiore) - 4 (migliore).

ANALISI ESPLORATIVA, con tre adattamenti dichiarati:

1. Le immagini sono distribuite normalizzate a [0, 1] con una finestra di
   tessuti molli, non in HU, e la finestra non e' dichiarata in modo univoco
   (W350/L40 sulla pagina della challenge, W400/L50 in un lavoro successivo).
   Le si riporta in HU con entrambe e si confrontano i risultati.  Con C
   piccolo i coefficienti MSCN sono quasi invarianti a una trasformazione
   affine dell'intensita', quindi l'effetto atteso della finestra e' piccolo:
   lo si misura invece di assumerlo.
2. Fuori dalla finestra le immagini sono saturate (0 o 1): aria, polmone e
   osso sono piatti.  Il dominio ammesso per le patch diventa l'insieme dei
   pixel non saturi, con lo stesso requisito al 100% del FOV nella
   specifica del modello.
3. Sorgente parzialmente comune: le immagini Mayo di LDCTIQAC vengono dallo
   stesso archivio della nostra collezione.  L'unico paziente identificabile
   in comune (L143) e' nel nostro split TEST, mai usato per il fit.

Cosa misura e cosa no: accordo con la qualita' **percepita** da radiologi,
non con la performance diagnostica.  I metodi della challenge sono addestrati
su questi punteggi; RIQE no, quindi il confronto con loro non e' alla pari.

    .venv/bin/python scripts/exp6_ldctiqac.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.cache import FeatureCache  # noqa: E402
from riqe.extract import MIN_PATCHES_FOR_SCORE, Spec, features_from_hu  # noqa: E402
from riqe.model import RCOND, MVGModel, mahalanobis_mixed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ldctiqac" / "LDCTIQAG2023_train"
OUT = ROOT / "experiments"

#: finestre candidate (larghezza, livello) in HU
WINDOWS = {"W350/L40": (350.0, 40.0), "W400/L50": (400.0, 50.0)}
#: tolleranza per considerare un pixel saturo
SAT_EPS = 1e-6

_G: dict = {}


def _init(spec_dict):
    _G["spec"] = Spec(**spec_dict)


def _moments(job):
    name, W, L = job
    x = np.asarray(Image.open(DATA / "image" / name), dtype=np.float32)
    hu = (L - W / 2.0) + x * W
    unsat = (x > SAT_EPS) & (x < 1.0 - SAT_EPS)
    pf = features_from_hu(hu, None, _G["spec"], fitting=False, masks=(unsat, unsat))
    f = pf.feat[pf.valid]
    if f.shape[0] < MIN_PATCHES_FOR_SCORE:
        return name, None, None, int(f.shape[0]), float(unsat.mean())
    return name, f.mean(axis=0), np.cov(f, rowvar=False), int(f.shape[0]), float(unsat.mean())


def bootstrap_ci(a, b, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(fn(a[i], b[i])[0])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()

    ch = json.loads((OUT / "hparam_choice.json").read_text())
    P, C, p = ch["P"], ch["C"], ch["p"]
    scores = json.loads((DATA / "train.json").read_text())
    names = sorted(scores)
    y = np.array([scores[n] for n in names], dtype=float)
    print(f"LDCTIQAC: {len(names)} immagini, punteggi radiologici {y.min():g}-{y.max():g}")
    print(f"modello: P={P} C={C:g} p={p:g}")

    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(set(split["fit"]))]
    model = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}").fit(
        list(fit["path"]), p, n_patients=fit.patient_id.nunique())
    models = {"RIQE": model}
    photo_f = OUT / "exp5_photo_model.npz"
    if photo_f.exists():
        z = np.load(photo_f)
        models["fotografico"] = MVGModel(nu=z["nu"], sigma=z["sigma"], n_patches=0,
                                         n_images=0, n_patients=0)

    rows, summary = [], {"P": P, "C": C, "p": p, "n_images": len(names), "windows": {}}
    for wname, (W, L) in WINDOWS.items():
        with cf.ProcessPoolExecutor(args.workers, initializer=_init,
                                    initargs=({"P": P, "C": C, "p": None},)) as ex:
            res = list(ex.map(_moments, [(n, W, L) for n in names], chunksize=8))
        summary["windows"][wname] = {}
        for mname, m in models.items():
            s = np.array([np.nan if r[1] is None else
                          mahalanobis_mixed(m.nu, m.sigma, r[1], r[2], RCOND) for r in res])
            ok = np.isfinite(s)
            rho = spearmanr(s[ok], y[ok])[0]
            r_p = pearsonr(s[ok], y[ok])[0]
            lo, hi = bootstrap_ci(s[ok], y[ok], spearmanr)
            summary["windows"][wname][mname] = {
                "n_scored": int(ok.sum()), "spearman": float(rho), "spearman_ci95": [lo, hi],
                "pearson": float(r_p)}
            print(f"  {wname:9s} {mname:12s}: {ok.sum():4d} punteggiabili  "
                  f"Spearman {rho:+.3f} [IC95 {lo:+.3f}, {hi:+.3f}]  Pearson {r_p:+.3f}")
            for (n, *_rest), sc in zip(res, s):
                rows.append({"image": n, "window": wname, "model": mname, "score": sc,
                             "radiologist": scores[n]})
        summary["windows"][wname]["unsaturated_fraction_median"] = float(
            np.median([r[4] for r in res]))

    pd.DataFrame(rows).to_csv(OUT / "exp6_ldctiqac_scores.csv", index=False)
    (OUT / "exp6_ldctiqac_summary.json").write_text(json.dumps(summary, indent=1))
    print("\nsegno atteso: negativo (RIQE piu' basso = piu' vicino al modello; "
          "punteggio radiologico piu' alto = migliore)")
    print(f"scritti {OUT}/exp6_ldctiqac_{{scores.csv,summary.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
