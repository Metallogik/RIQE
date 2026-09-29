#!/usr/bin/env python3
"""Extract and cache the per-patch features of every slice of the corpus.

One cache per (P, C) setting: these are the only components that change the
per-patch features. The sharpness threshold `p` acts **downstream**, on
selection, so it can be swept for free from the stored `delta`.

With --paired-low, the cache holds instead the reduced-dose slice paired to
every kept full-dose slice (same patient, identical z), extracted on the
masks of the full-dose slice (see riqe/dosetest.py).

Only patches inside the domain are stored (field of view at 100%, body at the
declared fraction): the others are usable neither for fitting nor for
scoring.

    .venv/bin/python scripts/cache_features.py [--workers 24] \
        [--P 16,24,32] [--C 1,0.5,0.25,0.1,0.05,0.025,0.01] [--paired-low]
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
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.dicomio import read_hu  # noqa: E402
from riqe.dosetest import dose_pairs, low_cache_dir  # noqa: E402
from riqe.extract import Spec, features_from_hu, masks_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "features"


def cache_name(P: int, C: float) -> str:
    return f"P{P}_C{C:g}"


_SPEC: Spec | None = None


def _init(spec_dict: dict) -> None:
    global _SPEC
    _SPEC = Spec(**spec_dict)


def _one(job):
    # a path, or (reduced-dose path, full-dose path whose masks are used)
    path, mask_path = (job, job) if isinstance(job, str) else job
    hu, pad, _ = read_hu(path)
    if mask_path == path:
        fov, body = masks_for(hu, pad)
    else:
        huf, padf, _ = read_hu(mask_path)
        fov, body = masks_for(huf, padf)
    pf = features_from_hu(hu, pad, _SPEC, fitting=False, masks=(fov, body))
    v = pf.valid
    return (
        path,
        pf.feat[v].astype(np.float32),
        pf.delta[v].astype(np.float32),
        int(pf.feat.shape[0]),
    )


def build(P: int, C: float, paths: list, workers: int, out: Path | None = None) -> dict:
    spec = Spec(P=P, C=C, p=None)  # p=None: selection applied downstream
    out = out or CACHE / cache_name(P, C)
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
    print(f"  {cache_name(P,C)}: {info['n_patches']} patches, "
          f"{info['mean_patches_per_slice']:.0f}/slice, {info['bytes']/1e6:.0f} MB, "
          f"{info['minutes']:.1f} min", flush=True)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--P", default="16,24,32")
    ap.add_argument("--C", default="1.0,0.5,0.25,0.1,0.05,0.025,0.01")
    ap.add_argument("--kind", default="full,low")
    ap.add_argument("--paired-low", action="store_true",
                    help="cache the reduced-dose slices paired to kept full-dose slices, "
                         "on the full-dose masks")
    args = ap.parse_args()

    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    if args.paired_low:
        pairs = dose_pairs(corpus)
        jobs = [(str(ROOT / lo), str(ROOT / fu)) for lo, fu in zip(pairs.path_low, pairs.path_full)]
        print(f"reduced-dose slices paired to kept full-dose slices: {len(jobs)} "
              f"(patients {pairs.patient_id.nunique()})")
    else:
        kinds = args.kind.split(",")
        sel = corpus[(corpus["kind"].isin(kinds)) & (corpus["keep"])]
        jobs = [str(ROOT / p) for p in sel["path"]]
        print(f"slices to extract: {len(jobs)}  "
              f"(patients {sel.patient_id.nunique()}, protocol cells {sel.cell.nunique()})")

    infos = []
    for P in [int(x) for x in args.P.split(",")]:
        for C in [float(x) for x in args.C.split(",")]:
            out = low_cache_dir(P, C) if args.paired_low else None
            infos.append(build(P, C, jobs, args.workers, out))
    summary = (low_cache_dir(0, 0).parent if args.paired_low else CACHE) / "cache_summary.json"
    summary.write_text(json.dumps(infos, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
