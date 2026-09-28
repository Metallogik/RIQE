#!/usr/bin/env python3
"""Estrae e mette in cache le feature per patch di tutte le slice del corpus.

Una cache per combinazione (P, C): sono le sole componenti che cambiano le
feature per patch.  La soglia di nitidezza `p` agisce **a valle**, sulla
selezione, quindi si puo' spazzare gratis a partire da `delta` salvata.

Vengono salvate solo le patch nel dominio (FOV al 100%, corpo alla frazione
dichiarata): le altre non sono usabili ne' in fitting ne' in punteggio.

    .venv/bin/python scripts/cache_features.py [--workers 24] \
        [--P 16,24,32] [--C 1.0,0.5,0.25,0.1] [--kind full,low]
"""

from __future__ import annotations

import os

# Un thread per processo: il parallelismo lo diamo con i processi, e lasciare
# che ogni worker apra i propri thread BLAS porta a oversubscription (load 90
# su 32 core, misurato) invece che a velocita'. Va fatto prima di numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.dicomio import read_hu  # noqa: E402
from riqe.extract import Spec, features_from_hu, masks_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "features"


def cache_name(P: int, C: float) -> str:
    return f"P{P}_C{C:g}"


_SPEC: Spec | None = None


def _init(spec_dict: dict) -> None:
    global _SPEC
    _SPEC = Spec(**spec_dict)


def _one(path: str):
    hu, pad, _ = read_hu(path)
    fov, body = masks_for(hu, pad)
    pf = features_from_hu(hu, pad, _SPEC, fitting=False, masks=(fov, body))
    v = pf.valid
    return (
        path,
        pf.feat[v].astype(np.float32),
        pf.delta[v].astype(np.float32),
        int(pf.feat.shape[0]),
    )


def build(P: int, C: float, paths: list[str], workers: int) -> dict:
    spec = Spec(P=P, C=C, p=None)  # p=None: selezione applicata a valle
    out = CACHE / cache_name(P, C)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    feats, deltas, index = [], [], []
    row = 0
    with cf.ProcessPoolExecutor(
        workers, initializer=_init, initargs=({"P": P, "C": C, "p": None},)
    ) as ex:
        for i, (path, f, d, ntot) in enumerate(ex.map(_one, paths, chunksize=8), 1):
            feats.append(f)
            deltas.append(d)
            index.append({"path": str(Path(path).relative_to(ROOT)), "start": row,
                          "n": int(f.shape[0]), "n_patch_total": ntot})
            row += int(f.shape[0])
            if i % 2000 == 0:
                print(f"    {i}/{len(paths)}  {(time.time()-t0)/60:.1f} min", flush=True)

    F = np.concatenate(feats) if feats else np.zeros((0, 36), np.float32)
    D = np.concatenate(deltas) if deltas else np.zeros((0,), np.float32)
    np.save(out / "features.npy", F)
    np.save(out / "delta.npy", D)
    (out / "index.json").write_text(json.dumps(index))
    (out / "spec.json").write_text(json.dumps(spec.as_dict()))
    info = {
        "P": P,
        "C": C,
        "n_slices": len(paths),
        "n_patches": int(F.shape[0]),
        "mean_patches_per_slice": float(F.shape[0] / max(len(paths), 1)),
        "bytes": int(F.nbytes + D.nbytes),
        "minutes": (time.time() - t0) / 60,
    }
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(f"  {cache_name(P,C)}: {info['n_patches']} patch, "
          f"{info['mean_patches_per_slice']:.0f}/slice, {info['bytes']/1e6:.0f} MB, "
          f"{info['minutes']:.1f} min", flush=True)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--P", default="16,24,32")
    ap.add_argument("--C", default="1.0,0.5,0.25,0.1")
    ap.add_argument("--kind", default="full,low")
    args = ap.parse_args()

    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    kinds = args.kind.split(",")
    sel = corpus[(corpus["kind"].isin(kinds)) & (corpus["keep"])]
    paths = [str(ROOT / p) for p in sel["path"]]
    print(f"slice da estrarre: {len(paths)}  "
          f"(pazienti {sel.patient_id.nunique()}, celle {sel.cell.nunique()})")

    infos = []
    for P in [int(x) for x in args.P.split(",")]:
        for C in [float(x) for x in args.C.split(",")]:
            infos.append(build(P, C, paths, args.workers))
    (CACHE / "cache_summary.json").write_text(json.dumps(infos, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
