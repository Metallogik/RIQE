#!/usr/bin/env python3
"""Scarica le serie di immagini ricostruite di LDCT-and-Projection-Data.

Scarica solo le serie `Full Dose Images` e `Low Dose Images` (38 GB): i dati
di proiezione (1,25 TB) non servono.  Usa `getImage`, che consegna la serie
come zip: a parita' di byte e' piu' veloce di `getSingleImage` slice per
slice, e ci da' l'intera serie, quindi l'ordinamento anatomico esatto e
l'appaiamento esatto fra dose piena e dose ridotta.

Idempotente e riprendibile: una serie gia' estratta viene saltata.  Ogni
slice e' registrata con il suo SHA-256 in data/manifest/<uid>.json, che
finisce nell'artefatto modello come identita' del corpus.

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

from riqe.tcia import BASE, _get  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DICOM = ROOT / "data" / "dicom"
MANIFEST = ROOT / "data" / "manifest"

WANTED = {"Full Dose Images": "full", "Low Dose Images": "low"}


def slug(s: str) -> str:
    return s.lower().replace(" ", "_")


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
        raise RuntimeError(f"{pid}/{kind}: risposta non zip ({raw[:80]!r})")

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

    series = json.load(open(ROOT / "corpus" / "series.json"))
    todo = [s for s in series if s.get("SeriesDescription") in WANTED]
    if args.only != "both":
        todo = [s for s in todo if WANTED[s["SeriesDescription"]] == args.only]
    # le piu' piccole per prime: si vede subito se qualcosa non va
    todo.sort(key=lambda s: int(s.get("FileSize") or 0))

    print(f"serie da scaricare: {len(todo)}  "
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
                          f"{r['n']} slice {r['mb']:.0f} MB in {r['s']:.0f}s "
                          f"| totale {(time.time()-t0)/60:.1f} min", flush=True)
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"ERRORE {s['PatientID']} {s['SeriesDescription']}: {e}", flush=True)

    used = shutil.disk_usage(DICOM.parent).used
    print(f"FINITO: {done} serie, {errors} errori, {(time.time()-t0)/60:.1f} min, "
          f"disco usato {used/1e9:.0f} GB", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
