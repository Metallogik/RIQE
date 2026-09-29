#!/usr/bin/env python3
"""Compute the MVG moments (nu, Sigma) of every bank image, for every (P, C)
setting.

Why moments and not scores: the score also depends on the model, which
depends on the threshold `p`. Storing (nu_2, Sigma_2) per image turns the
score under any model into a matrix product, and the sweep over `p` costs
nothing. That is 36 + 36x36 numbers per image: 5 kB, against 1 MB for the
image.

Structure of the pass: iterate **over slices**, not over settings. The
dominant cost is applying filters to regenerate the images, and that does not
depend on (P, C): each image is regenerated once and its features extracted
for every setting.

    .venv/bin/python scripts/score_bank.py --bank experiments/bank_val_inner.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.bank import from_dict, render  # noqa: E402
from riqe.dicomio import read_hu  # noqa: E402
from riqe import niqecfg  # noqa: E402
from riqe.extract import MIN_PATCHES_FOR_SCORE, Spec, features_from_hu, luminance, masks_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

_CONFIGS: list[tuple[int, float]] = []
_NIQE = False

#: the parameters published for NIQE, see riqe/niqecfg.py
NIQE_CONFIG = (niqecfg.P, niqecfg.C)


def _init(configs, niqe: bool = False) -> None:
    global _CONFIGS, _NIQE
    _CONFIGS = [tuple(c) for c in configs]
    _NIQE = niqe


def _moments(hu, pad, masks, P, C):
    if _NIQE:
        return niqecfg.image_moments(luminance(hu, Spec(P=P, C=C, p=None, use_masks=False)))
    pf = features_from_hu(hu, pad, Spec(P=P, C=C, p=None), fitting=False, masks=masks)
    f = pf.feat[pf.valid]
    if f.shape[0] < MIN_PATCHES_FOR_SCORE:
        return None, None, int(f.shape[0])
    return f.mean(axis=0), np.cov(f, rowvar=False), int(f.shape[0])


def _one(rec: dict):
    out = {"path": rec["path"], "patient_id": rec["patient_id"], "cell": rec["cell"],
           "pixel_spacing": rec["pixel_spacing"], "rows": []}
    t0 = time.time()

    sources = [("full", rec["path"], rec["entries"])]
    if "low" in rec:
        sources.append(("low", rec["low"]["path"], rec["low"]["entries"]))

    masks = None
    for src, path, entries in sources:
        hu, pad, _ = read_hu(str(ROOT / path))
        if masks is None:
            # masks of the full-dose slice, used for its reduced-dose
            # counterpart as well (same anatomy, same position)
            masks = masks_for(hu, pad)
        for ed in entries:
            e = from_dict(ed)
            img = render(e, hu)
            for P, C in _CONFIGS:
                nu, sg, n = _moments(img, pad, masks, P, C)
                out["rows"].append({
                    "source": src,
                    "label": e.label(),
                    "kind": e.kind,
                    "denoiser": e.denoiser,
                    "target_residual_hu": e.target_residual_hu,
                    "residual_hu": e.residual_hu,
                    "sigma_hu": e.sigma_hu,
                    "sigma_px": e.sigma_px,
                    "rel_increase": e.rel_increase,
                    "P": P,
                    "C": C,
                    "n_patches": n,
                    # numpy arrays, not Python lists: a list of 1296 floats costs
                    # ~30 bytes per element, and with 288,000 rows on the test
                    # bank that would be over 10 GB in the parent process
                    "nu": None if nu is None else nu.astype(np.float32),
                    "sigma": None if sg is None else sg.astype(np.float32),
                })
    out["seconds"] = time.time() - t0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--P", default="16,24,32")
    ap.add_argument("--C", default="1.0,0.5,0.25,0.1")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=None)
    ap.add_argument("--niqe-config", action="store_true",
                    help="score with the published NIQE parameters (P=96, C=1, no masks)")
    args = ap.parse_args()

    bank = json.load(open(args.bank))
    configs = [(int(P), float(C)) for P in args.P.split(",") for C in args.C.split(",")]
    if args.niqe_config:
        configs = [NIQE_CONFIG]
    print(f"bank: {bank['n_slices']} slices, {bank['n_entries']} images; "
          f"{len(configs)} (P, C) settings")

    t0 = time.time()
    recs = []
    with cf.ProcessPoolExecutor(args.workers, initializer=_init,
                                initargs=(configs, args.niqe_config)) as ex:
        for i, r in enumerate(ex.map(_one, bank["slices"], chunksize=1), 1):
            recs.append(r)
            if i % 10 == 0:
                done = sum(len(x["rows"]) for x in recs)
                print(f"  {i}/{bank['n_slices']} slices, {done} rows, "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)

    # compact serialisation: numpy arrays instead of JSON for the moments
    rows = [r2 for r in recs for r2 in r["rows"]]
    meta_keys = ["source", "label", "kind", "denoiser", "target_residual_hu",
                 "residual_hu", "sigma_hu", "sigma_px", "rel_increase", "P", "C", "n_patches"]
    slice_of = []
    for r in recs:
        slice_of += [r["path"]] * len(r["rows"])
    NU = np.full((len(rows), 36), np.nan, np.float32)
    SG = np.full((len(rows), 36, 36), np.nan, np.float32)
    for i, r in enumerate(rows):
        if r["nu"] is not None:
            NU[i] = r["nu"]
            SG[i] = r["sigma"]

    suffix = "_moments_niqecfg.npz" if args.niqe_config else "_moments.npz"
    out = Path(args.out or (Path(args.bank).with_suffix("").as_posix() + suffix))
    np.savez_compressed(
        out,
        nu=NU,
        sigma=SG,
        meta=json.dumps([{k: r[k] for k in meta_keys} for r in rows]),
        slice_path=json.dumps(slice_of),
        patient_id=json.dumps([r["patient_id"] for r in recs for _ in r["rows"]]),
        cell=json.dumps([r["cell"] for r in recs for _ in r["rows"]]),
        pixel_spacing=json.dumps([r["pixel_spacing"] for r in recs for _ in r["rows"]]),
        configs=json.dumps(configs),
    )
    n_bad = int(np.isnan(NU[:, 0]).sum())
    print(f"wrote {out}: {len(rows)} rows, {n_bad} without a defined score "
          f"(< {MIN_PATCHES_FOR_SCORE} patches), {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
