#!/usr/bin/env python3
"""Single pass: for every downloaded slice, metadata + objective statistics.

Produces `corpus/slices.parquet`. The inclusion criteria S1-S6 are then pure
operations on this table (riqe.inclusion.apply_criteria), so they can be
re-run and re-thresholded without re-reading the DICOM files.

    .venv/bin/python scripts/build_slice_table.py [--workers 24] [--kind full]
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
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.dicomio import cell_of, read_hu  # noqa: E402
from riqe.inclusion import slice_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def one(path_str: str) -> dict | None:
    try:
        hu, pad, meta = read_hu(path_str)
        st = slice_stats(hu, pad)
        row = {**meta, **st, "path": str(Path(path_str).relative_to(ROOT)),
               "kind": Path(path_str).parent.name}
        row["cell"] = cell_of(meta)
        return row
    except Exception as e:  # noqa: BLE001
        return {"path": path_str, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--kind", choices=["full", "low", "both"], default="both")
    ap.add_argument("--out", default="corpus/slices.parquet")
    args = ap.parse_args()

    pat = "*" if args.kind == "both" else args.kind
    files = sorted(str(p) for p in (ROOT / "data" / "dicom").glob(f"*/{pat}/*.dcm"))
    print(f"slices to analyse: {len(files)}  workers={args.workers}", flush=True)

    rows, t0 = [], time.time()
    with cf.ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one, files, chunksize=32), 1):
            if r is not None:
                rows.append(r)
            if i % 5000 == 0:
                print(f"  {i}/{len(files)}  {(time.time()-t0)/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    bad = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    if len(bad):
        print(f"WARNING: {len(bad)} unreadable slices")
        print(bad[["path", "error"]].head(10).to_string())
        df = df[df["error"].isna()].drop(columns=["error"])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} rows to {out}  ({(time.time()-t0)/60:.1f} min)")
    print("\nslices per protocol cell and dose:")
    print(df.groupby(["cell", "kind"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
