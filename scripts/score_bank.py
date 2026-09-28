#!/usr/bin/env python3
"""Calcola i momenti MVG (nu, Sigma) di ogni immagine del banco, per ogni
combinazione (P, C).

Perche' i momenti e non il punteggio: il punteggio dipende anche dal modello,
che dipende dalla soglia `p`.  Salvando (nu_2, Sigma_2) per immagine il
punteggio sotto qualunque modello diventa un prodotto matriciale, e la
spazzata su `p` costa zero.  Sono 36 + 36x36 numeri per immagine: 5 kB,
contro 1 MB per l'immagine.

Struttura della passata: si itera **sulle slice**, non sulle configurazioni.
Il costo dominante e' applicare i filtri per rigenerare le immagini, e quello
non dipende da (P, C): si rigenera una volta e si estrae per tutte le
configurazioni.

    .venv/bin/python scripts/score_bank.py --bank experiments/bank_val_inner.json
"""

from __future__ import annotations

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
from riqe.extract import MIN_PATCHES_FOR_SCORE, Spec, features_from_hu, masks_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

_CONFIGS: list[tuple[int, float]] = []


def _init(configs) -> None:
    global _CONFIGS
    _CONFIGS = [tuple(c) for c in configs]


def _moments(hu, pad, masks, P, C):
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

    for src, path, entries in sources:
        hu, pad, _ = read_hu(str(ROOT / path))
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
                    "P": P,
                    "C": C,
                    "n_patches": n,
                    "nu": None if nu is None else nu.astype(np.float32).tolist(),
                    "sigma": None if sg is None else sg.astype(np.float32).ravel().tolist(),
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
    args = ap.parse_args()

    bank = json.load(open(args.bank))
    configs = [(int(P), float(C)) for P in args.P.split(",") for C in args.C.split(",")]
    print(f"banco: {bank['n_slices']} slice, {bank['n_entries']} immagini; "
          f"{len(configs)} configurazioni (P, C)")

    t0 = time.time()
    recs = []
    with cf.ProcessPoolExecutor(args.workers, initializer=_init, initargs=(configs,)) as ex:
        for i, r in enumerate(ex.map(_one, bank["slices"], chunksize=1), 1):
            recs.append(r)
            if i % 10 == 0:
                done = sum(len(x["rows"]) for x in recs)
                print(f"  {i}/{bank['n_slices']} slice, {done} righe, "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)

    # serializzazione compatta: array numpy invece di JSON per i momenti
    rows = [r2 for r in recs for r2 in r["rows"]]
    meta_keys = ["source", "label", "kind", "denoiser", "target_residual_hu",
                 "residual_hu", "sigma_hu", "sigma_px", "P", "C", "n_patches"]
    slice_of = []
    for r in recs:
        slice_of += [r["path"]] * len(r["rows"])
    NU = np.full((len(rows), 36), np.nan, np.float32)
    SG = np.full((len(rows), 36, 36), np.nan, np.float32)
    for i, r in enumerate(rows):
        if r["nu"] is not None:
            NU[i] = r["nu"]
            SG[i] = np.asarray(r["sigma"], np.float32).reshape(36, 36)

    out = Path(args.out or (Path(args.bank).with_suffix("").as_posix() + "_moments.npz"))
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
    print(f"scritto {out}: {len(rows)} righe, {n_bad} senza punteggio definibile "
          f"(< {MIN_PATCHES_FOR_SCORE} patch), {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
