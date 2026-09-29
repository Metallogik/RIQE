#!/usr/bin/env python3
"""The photographic baselines on their own domain: held-out photographs.

On CT the model fitted on photographs with the parameters published for NIQE
prefers noisier images. That could be a property of CT seen through a
photographic reference, or an error in this implementation. The two are
separated here without the reference NIQE model (which is not used): the
photographs are split in two halves, the models are refitted on one half,
and the other half is degraded with white Gaussian noise and Gaussian blur.
On photographs a NIQE-style model must score both worse.

Writes experiments/photo_sanity.json.

    .venv/bin/python scripts/photo_sanity.py
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
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp5_photo_baseline import luminance_of, photo_features, photo_features_niqe  # noqa: E402
from riqe import niqecfg  # noqa: E402
from riqe.extract import MIN_PATCHES_FOR_SCORE  # noqa: E402
from riqe.model import RCOND, fit_mvg, mahalanobis_mixed  # noqa: E402
from riqe.nss import patch_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"
NOISE = (5.0, 10.0, 20.0)   # white Gaussian noise, code values on the 0-255 scale
BLUR = (1.0, 2.0, 4.0)      # Gaussian blur, pixels

_G: dict = {}


def _init(g):
    _G.update(g)


def _scores(img):
    """(score with the RIQE settings, score with the NIQE parameters)."""
    P, C = _G["P"], _G["C"]
    pf = patch_features(img, P=P, C=C, fov=None, body=None, p=None)
    f = pf.feat[pf.valid]
    m = _G["riqe_settings"]
    s1 = (mahalanobis_mixed(m.nu, m.sigma, f.mean(axis=0), np.cov(f, rowvar=False), RCOND)
          if f.shape[0] >= MIN_PATCHES_FOR_SCORE else np.nan)
    mu, cov, _ = niqecfg.image_moments(img)
    return s1, niqecfg.score(_G["niqe_params"], mu, cov)


def _one(job):
    path, seed = job
    lum = luminance_of(ROOT / path)
    if lum is None or min(lum.shape) < 2 * niqecfg.P:
        return None
    rng = np.random.default_rng(seed)
    rows = [("original", 0.0, lum)]
    rows += [("noise", s, lum + rng.normal(0.0, s, lum.shape).astype(np.float32)) for s in NOISE]
    rows += [("blur", s, gaussian_filter(lum, s)) for s in BLUR]
    return [{"path": path, "kind": k, "level": lv, **dict(zip(("riqe_settings", "niqe_params"),
                                                              _scores(img)))}
            for k, lv, img in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    ch = json.loads((OUT / "hparam_choice.json").read_text())
    P, C, p = int(ch["P"]), float(ch["C"]), float(ch["p"])
    photos = json.loads((ROOT / "corpus" / "photo_corpus_manifest.json").read_text())["photos"]
    order = np.random.default_rng(args.seed).permutation(len(photos))
    fit_idx, test_idx = order[: len(photos) // 2], order[len(photos) // 2:]

    with cf.ProcessPoolExecutor(args.workers) as ex:
        a = [f for f in ex.map(photo_features, [(photos[i]["path"], P, C, p, args.seed + i)
                                                for i in fit_idx]) if f is not None and f.shape[0]]
        b = [f for f in ex.map(photo_features_niqe, [(photos[i]["path"], args.seed + i)
                                                     for i in fit_idx]) if f is not None]
    riqe_settings = fit_mvg(np.concatenate(a), n_images=len(a), n_patients=len(a))
    niqe_params = fit_mvg(np.concatenate(b), n_images=len(b), n_patients=len(b))
    print(f"fitted on {len(a)} / {len(b)} photographs; testing on {len(test_idx)}")

    rows = []
    with cf.ProcessPoolExecutor(args.workers, initializer=_init,
                                initargs=({"P": P, "C": C, "riqe_settings": riqe_settings,
                                           "niqe_params": niqe_params},)) as ex:
        for r in ex.map(_one, [(photos[i]["path"], args.seed + 7 * int(i)) for i in test_idx]):
            if r:
                rows += r

    import pandas as pd
    d = pd.DataFrame(rows)
    out = {"n_fit_photos": len(b), "n_test_photos": int(d.path.nunique()), "worse": {}}
    for col in ("riqe_settings", "niqe_params"):
        o = d[d.kind == "original"].set_index("path")[col]
        for (k, lv), g in d[d.kind != "original"].groupby(["kind", "level"]):
            delta = (g[col] - g.path.map(o)).dropna()
            out["worse"][f"{col}|{k}|{lv:g}"] = {"fraction": float((delta > 0).mean()),
                                                 "n": int(len(delta)),
                                                 "median_delta": float(delta.median())}
    (OUT / "photo_sanity.json").write_text(json.dumps(out, indent=1))
    print("fraction of held-out photographs scored worse after the degradation:")
    for k, v in out["worse"].items():
        print(f"  {k:28s} {100 * v['fraction']:5.1f}%  (n={v['n']}, median change {v['median_delta']:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
