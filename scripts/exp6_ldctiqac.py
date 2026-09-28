#!/usr/bin/env python3
"""Experiment 6 - agreement with radiologists' ratings (LDCTIQAC 2023).

Data: LDCTIQAC 2023 training set (Zenodo 10.5281/zenodo.7833096, CC BY 4.0),
1000 low-dose abdominal CT images, each with the mean score of five
radiologists on a scale from 0 (worst) to 4 (best).

Download the training archive from the Zenodo record and unpack it into
data/ldctiqac/, so that data/ldctiqac/LDCTIQAG2023_train/train.json and
data/ldctiqac/LDCTIQAG2023_train/image/ exist.

EXPLORATORY ANALYSIS, with three declared adaptations:

1. The images are distributed normalised to [0, 1] with a soft-tissue window,
   not in HU, and the window is not stated unambiguously (W350/L40 on the
   challenge page, W400/L50 in a later paper). They are mapped back to HU
   with both, and the results compared. With small C the MSCN coefficients
   are nearly invariant to an affine intensity transform, so the expected
   effect of the window is small: it is measured rather than assumed.
2. Outside the window the images are saturated (0 or 1): air, lung and bone
   are flat. The admissible patch domain becomes the set of unsaturated
   pixels, with the same 100% requirement as the FOV in the model
   specification.
3. Partly shared source: the Mayo images in LDCTIQAC come from the same
   archive as our collection. The only identifiable shared patient (L143) is
   in our TEST split, never used for fitting.

What it measures and what it does not: agreement with the quality
**perceived** by radiologists, not with diagnostic performance. The challenge
methods are trained on these scores; RIQE is not, so a comparison with them
is not like for like.

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

#: candidate windows (width, level) in HU
WINDOWS = {"W350/L40": (350.0, 40.0), "W400/L50": (400.0, 50.0)}
#: tolerance for considering a pixel saturated
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
    print(f"LDCTIQAC: {len(names)} images, radiologist scores {y.min():g}-{y.max():g}")
    print(f"model: P={P} C={C:g} p={p:g}")

    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(set(split["fit"]))]
    model = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}").fit(
        list(fit["path"]), p, n_patients=fit.patient_id.nunique())
    models = {"RIQE": model}
    photo_f = OUT / "exp5_photo_model.npz"
    if photo_f.exists():
        z = np.load(photo_f)
        models["photographic"] = MVGModel(nu=z["nu"], sigma=z["sigma"], n_patches=0,
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
            print(f"  {wname:9s} {mname:12s}: {ok.sum():4d} scoreable  "
                  f"Spearman {rho:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]  Pearson {r_p:+.3f}")
            for (n, *_rest), sc in zip(res, s):
                rows.append({"image": n, "window": wname, "model": mname, "score": sc,
                             "radiologist": scores[n]})
        summary["windows"][wname]["unsaturated_fraction_median"] = float(
            np.median([r[4] for r in res]))

    pd.DataFrame(rows).to_csv(OUT / "exp6_ldctiqac_scores.csv", index=False)
    (OUT / "exp6_ldctiqac_summary.json").write_text(json.dumps(summary, indent=1))
    print("\nexpected sign: negative (lower RIQE = closer to the model; "
          "higher radiologist score = better)")
    print(f"wrote {OUT}/exp6_ldctiqac_{{scores.csv,summary.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
