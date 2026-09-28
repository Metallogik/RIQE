#!/usr/bin/env python3
"""Costruisce il banco di immagini di prova (ricette, non pixel).

La calibrazione della forza dei filtri e' costosa e non dipende da (P, C):
si fa qui, una volta, e il risultato serve sia alla ricerca degli
iperparametri sia alla batteria di validazione.

    .venv/bin/python scripts/prepare_bank.py --split val_inner --per-patient 4
    .venv/bin/python scripts/prepare_bank.py --split test --per-patient 6 --fine-noise
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

from riqe import degrade as dg  # noqa: E402
from riqe.bank import build_bank  # noqa: E402
from riqe.dicomio import read_hu  # noqa: E402
from riqe.extract import masks_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"

_FINE = False


def _init(fine: bool) -> None:
    global _FINE
    _FINE = fine


def _one(job: tuple):
    path, patient_id, sop_uid, cell, seed, source, low_path, low_sop = job
    hu, pad, meta = read_hu(str(ROOT / path))
    fov, body = masks_for(hu, pad)
    t0 = time.time()
    entries = build_bank(
        hu, body, source="full", seed=seed,
        noise_fine=dg.NOISE_FINE_HU if _FINE else (),
    )
    rec = {
        "path": path,
        "patient_id": patient_id,
        "sop_uid": sop_uid,
        "cell": cell,
        "pixel_spacing": meta["pixel_spacing"],
        "kernel": meta["kernel"],
        "slice_thickness": meta["slice_thickness"],
        "body_frac": float(body.mean()),
        "entries": [e.as_dict() for e in entries],
        "seconds": time.time() - t0,
    }
    if low_path:
        hul, padl, _ = read_hu(str(ROOT / low_path))
        fovl, bodyl = masks_for(hul, padl)
        low_entries = build_bank(
            hul, bodyl, source="low", seed=seed + 5000,
            noise_sigmas=(), blur_sigmas=(),  # sulla dose ridotta interessano i denoiser
        )
        rec["low"] = {
            "path": low_path,
            "sop_uid": low_sop,
            "entries": [e.as_dict() for e in low_entries],
        }
    return rec


def match_low_dose(corpus: pd.DataFrame, kept: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per ciascun paziente con dose ridotta, la tabella delle sue slice a
    dose ridotta, indicizzata per z: serve per appaiare per posizione."""
    low = corpus[(corpus["kind"] == "low")]
    return {pid: g.sort_values("z") for pid, g in low.groupby("patient_id")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_inner", choices=["val_inner", "fit_inner", "test", "fit"])
    ap.add_argument("--per-patient", type=int, default=4)
    ap.add_argument("--fine-noise", action="store_true",
                    help="aggiunge la scala fine di rumore da zero (forma 3 del test)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    pids = set(split[args.split])
    kept = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(pids)]

    rng = np.random.default_rng(args.seed)
    chosen = []
    for pid, g in kept.groupby("patient_id"):
        g = g.sort_values("z")
        k = min(args.per_patient, len(g))
        idx = np.linspace(0, len(g) - 1, k).round().astype(int)
        chosen.append(g.iloc[np.unique(idx)])
    chosen = pd.concat(chosen, ignore_index=True)
    print(f"split={args.split}: {chosen.patient_id.nunique()} pazienti, {len(chosen)} slice")

    lows = match_low_dose(corpus, kept)
    jobs = []
    for i, r in enumerate(chosen.itertuples()):
        low_path = low_sop = None
        if r.patient_id in lows:
            gl = lows[r.patient_id]
            j = int(np.abs(gl["z"].to_numpy() - r.z).argmin())
            if abs(float(gl["z"].iloc[j]) - r.z) <= max(float(r.slice_thickness), 1.0):
                low_path, low_sop = gl["path"].iloc[j], gl["sop_uid"].iloc[j]
        jobs.append((r.path, r.patient_id, r.sop_uid, r.cell,
                     args.seed + 97 * i, "full", low_path, low_sop))
    n_paired = sum(1 for j in jobs if j[6])
    print(f"slice con dose ridotta appaiata per z: {n_paired}/{len(jobs)}")

    t0 = time.time()
    recs = []
    with cf.ProcessPoolExecutor(args.workers, initializer=_init,
                                initargs=(args.fine_noise,)) as ex:
        for i, rec in enumerate(ex.map(_one, jobs, chunksize=1), 1):
            recs.append(rec)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    n_entries = sum(len(r["entries"]) + len(r.get("low", {}).get("entries", [])) for r in recs)
    miss = sum(
        1
        for r in recs
        for e in r["entries"] + r.get("low", {}).get("entries", [])
        if e["kind"] == "denoise" and not e["target_reached"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (args.out or f"bank_{args.split}.json")
    out.write_text(json.dumps({
        "split": args.split,
        "per_patient": args.per_patient,
        "fine_noise": args.fine_noise,
        "seed": args.seed,
        "n_slices": len(recs),
        "n_entries": n_entries,
        "n_denoise_target_missed": miss,
        "residual_levels_hu": list(dg.RESIDUAL_LEVELS_HU),
        "noise_sigmas_hu": list(dg.NOISE_SIGMAS_HU),
        "blur_sigmas_px": list(dg.BLUR_SIGMAS_PX),
        "denoisers": list(dg.DENOISERS),
        "slices": recs,
    }, indent=1))
    print(f"scritto {out}: {len(recs)} slice, {n_entries} immagini di prova, "
          f"{miss} bersagli di residuo non raggiunti, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
