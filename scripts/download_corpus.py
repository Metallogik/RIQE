#!/usr/bin/env python3
"""Download the reconstructed image series of LDCT-and-Projection-Data.

Only the `Full Dose Images` and `Low Dose Images` series are downloaded
(38 GB): the projection data (1.25 TB) are not needed. Uses `getImage`, which
delivers a series as a zip: per byte it is faster than `getSingleImage` slice
by slice, and it gives the whole series, hence the exact anatomical ordering
and the exact pairing of full and reduced dose.

The list of series is read from corpus/series.json, the exact list used for
the released model. If that file is missing, it is fetched from the public
NBIA API (the collection may have changed since).

Idempotent and resumable: a series already extracted is skipped. Every slice
is recorded with its SHA-256 in data/manifest/<uid>.json, which ends up in the
model artefact as the identity of the corpus.

    .venv/bin/python scripts/download_corpus.py [--workers 8] [--only full|low]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.tcia import BASE, _get, get_series  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DICOM = ROOT / "data" / "dicom"
MANIFEST = ROOT / "data" / "manifest"

WANTED = {"Full Dose Images": "full", "Low Dose Images": "low"}


def download_series(rec: dict) -> dict:
    uid = rec["SeriesInstanceUID"]
    pid = rec["PatientID"]
    kind = WANTED[rec["SeriesDescription"]]
    out = DICOM / pid / kind
    man = MANIFEST / f"{uid}.json"
    if man.exists():
        return {"uid": uid, "pid": pid, "kind": kind, "status": "cached"}

    t0 = time.time()
    raw = _get(f"{BASE}/getImage?SeriesInstanceUID={uid}", timeout=3600)
    if not raw[:2] == b"PK":
        raise RuntimeError(f"{pid}/{kind}: response is not a zip ({raw[:80]!r})")

    out.mkdir(parents=True, exist_ok=True)
    files = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            data = z.read(name)
            if data[128:132] != b"DICM":
                continue
            base = Path(name).name
            (out / base).write_bytes(data)
            files.append({"file": base, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})

    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(
        json.dumps(
            {
                "series_instance_uid": uid,
                "patient_id": pid,
                "series_description": rec["SeriesDescription"],
                "kind": kind,
                "collection": rec["Collection"],
                "license": rec.get("LicenseName"),
                "n_files": len(files),
                "zip_bytes": len(raw),
                "files": files,
            },
            indent=1,
        )
    )
    return {
        "uid": uid,
        "pid": pid,
        "kind": kind,
        "status": "ok",
        "n": len(files),
        "mb": len(raw) / 1e6,
        "s": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", choices=["full", "low", "both"], default="both")
    args = ap.parse_args()

    series_file = ROOT / "corpus" / "series.json"
    if not series_file.exists():
        print("corpus/series.json not found: fetching the series list from NBIA", flush=True)
        series_file.parent.mkdir(parents=True, exist_ok=True)
        series_file.write_text(json.dumps(get_series()))
    series = json.load(open(series_file))
    todo = [s for s in series if s.get("SeriesDescription") in WANTED]
    if args.only != "both":
        todo = [s for s in todo if WANTED[s["SeriesDescription"]] == args.only]
    # smallest first: problems show up immediately
    todo.sort(key=lambda s: int(s.get("FileSize") or 0))

    print(f"series to download: {len(todo)}  "
          f"({sum(int(s['FileSize']) for s in todo)/1e9:.1f} GB)  workers={args.workers}",
          flush=True)

    done = errors = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download_series, s): s for s in todo}
        for fut in cf.as_completed(futs):
            s = futs[fut]
            try:
                r = fut.result()
                done += 1
                if r["status"] == "ok":
                    print(f"[{done}/{len(todo)}] {r['pid']}/{r['kind']} "
                          f"{r['n']} slices {r['mb']:.0f} MB in {r['s']:.0f}s "
                          f"| total {(time.time()-t0)/60:.1f} min", flush=True)
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"ERROR {s['PatientID']} {s['SeriesDescription']}: {e}", flush=True)

    used = shutil.disk_usage(DICOM.parent).used
    print(f"DONE: {done} series, {errors} errors, {(time.time()-t0)/60:.1f} min, "
          f"disk used {used/1e9:.0f} GB", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
