#!/usr/bin/env python3
"""Esperimento centrale: serve un modello unico o stratificato?

Per ogni contrasto fra due gruppi di pazienti si calcola la divergenza fra i
due modelli e la si confronta con due riferimenti misurati sullo stesso
corpus:

  ANCORA BASSA, nulla di permutazione.  Si mettono in comune i pazienti dei
  due gruppi e si ripartiscono a caso, **per paziente**, in due gruppi delle
  stesse dimensioni.  La distribuzione di D che ne esce e' cio' che si
  osserverebbe se il fattore non contasse nulla.  E' esatta, e' appaiata per
  numerosita' **per costruzione** (il che elimina l'artefatto per cui uno
  strato piccolo sembra divergente solo perche' Sigma e' stimata peggio), e
  non richiede assunzioni distributive.

  ANCORA ALTA, differenza fisica reale.  D fra il modello a dose piena e
  quello a dose ridotta sugli **stessi** pazienti: una differenza di
  acquisizione vera, della grandezza che la metrica deve rilevare.  Il suo
  nullo e' la permutazione appaiata dell'etichetta dose dentro ciascun
  paziente.

L'indice riportato e'

    eta = (D_oss - mediana(nullo)) / (D_dose - mediana(nullo_dose))

cioe' l'eccesso sul rumore di campionamento, in unita' dell'eccesso prodotto
da una differenza fisica nota.

REGOLA DI DECISIONE, dichiarata prima di eseguire (docs/01-proposta.md 5.3):
lo strato richiede un sotto-modello proprio se e solo se
  (a) p di permutazione < 0,05, E
  (b) eta >= 0,25, E
  (c) il criterio operativo fallisce (Spearman < 0,95 sul banco comune,
      oppure un ribaltamento di verdetto).
Le condizioni (a) e (b) si valutano qui; la (c) in run_battery.py.

    .venv/bin/python scripts/stratification.py --P 16 --C 0.25 --p 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.cache import FeatureCache  # noqa: E402
from riqe.model import feature_shift, model_divergence, symmetric_kl  # noqa: E402
from riqe.nss import FEATURE_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def slices_by_patient(df: pd.DataFrame) -> dict[str, list[str]]:
    out = defaultdict(list)
    for r in df.itertuples():
        out[r.patient_id].append(r.path)
    return dict(out)


def fit_group(cache, by_patient, pids, p):
    paths = [q for pid in pids for q in by_patient.get(pid, [])]
    return cache.fit(paths, p, n_patients=len(pids))


def permutation_null(cache, by_patient, pids_a, pids_b, p, B, seed):
    """Nullo: ripartizione casuale **per paziente** del pool A+B."""
    rng = np.random.default_rng(seed)
    pool = list(pids_a) + list(pids_b)
    na = len(pids_a)
    out = []
    for _ in range(B):
        perm = rng.permutation(pool)
        try:
            ma = fit_group(cache, by_patient, perm[:na], p)
            mb = fit_group(cache, by_patient, perm[na:], p)
            out.append(model_divergence(ma, mb))
        except ValueError:
            out.append(np.nan)
    return np.asarray(out)


def paired_dose_anchor(cache, full_by_patient, low_by_patient, pids, p, B, seed):
    """Ancora alta e il suo nullo appaiato.

    Nullo: per ogni paziente si scambia a caso l'etichetta dose piena / dose
    ridotta.  Se la dose non contasse, i due modelli risultanti sarebbero
    equivalenti.
    """
    rng = np.random.default_rng(seed)
    m_full = cache.fit([q for pid in pids for q in full_by_patient[pid]], p, n_patients=len(pids))
    m_low = cache.fit([q for pid in pids for q in low_by_patient[pid]], p, n_patients=len(pids))
    d_obs = model_divergence(m_full, m_low)
    null = []
    for _ in range(B):
        a, b = [], []
        for pid in pids:
            if rng.random() < 0.5:
                a += full_by_patient[pid]
                b += low_by_patient[pid]
            else:
                a += low_by_patient[pid]
                b += full_by_patient[pid]
        try:
            null.append(model_divergence(
                cache.fit(a, p, n_patients=len(pids)), cache.fit(b, p, n_patients=len(pids))))
        except ValueError:
            null.append(np.nan)
    return d_obs, np.asarray(null), m_full, m_low


def contrast_row(name, axis, cache, by_patient, pids_a, pids_b, p, B, seed, anchor_excess):
    ma = fit_group(cache, by_patient, pids_a, p)
    mb = fit_group(cache, by_patient, pids_b, p)
    d_obs = model_divergence(ma, mb)
    null = permutation_null(cache, by_patient, pids_a, pids_b, p, B, seed)
    med = float(np.nanmedian(null))
    pval = float((np.nansum(null >= d_obs) + 1) / (np.sum(np.isfinite(null)) + 1))
    excess = d_obs - med
    eta = excess / anchor_excess if anchor_excess and np.isfinite(anchor_excess) else np.nan
    shift = feature_shift(ma, mb)
    top = np.argsort(-np.abs(np.nan_to_num(shift)))[:4]
    return {
        "contrasto": name,
        "asse": axis,
        "n_A": len(pids_a),
        "n_B": len(pids_b),
        "D_oss": d_obs,
        "nullo_mediana": med,
        "nullo_p95": float(np.nanpercentile(null, 95)),
        "p_perm": pval,
        "eta": eta,
        "KL_sim": symmetric_kl(ma, mb),
        "cond_A": ma.cond(),
        "cond_B": mb.cond(),
        "patch_A": ma.n_patches,
        "patch_B": mb.n_patches,
        "feature_dominanti": ", ".join(
            f"{FEATURE_NAMES[i]}={shift[i]:+.2f}" for i in top if np.isfinite(shift[i])
        ),
        "significativo_a": bool(pval < 0.05),
        "rilevante_b": bool(np.isfinite(eta) and eta >= 0.25),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--B", type=int, default=200, help="ripetizioni di permutazione")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default="experiments/stratification.csv")
    args = ap.parse_args()

    choice_file = ROOT / "experiments" / "hparam_choice.json"
    if args.P is None and choice_file.exists():
        ch = json.loads(choice_file.read_text())
        args.P, args.C, args.p = ch["P"], ch["C"], ch["p"]
    if args.P is None:
        ap.error("servono --P --C --p, oppure experiments/hparam_choice.json")
    print(f"iperparametri: P={args.P} C={args.C:g} p={args.p:g}, B={args.B} permutazioni")

    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit_pids = set(split["fit"])
    cache = FeatureCache(ROOT / "data" / "features" / f"P{args.P}_C{args.C:g}")

    full = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(fit_pids)]
    low = corpus[(corpus["kind"] == "low") & corpus["keep"] & corpus.patient_id.isin(fit_pids)]
    by_full = slices_by_patient(full)
    by_low = slices_by_patient(low)

    meta = full.groupby("patient_id").agg(
        cell=("cell", "first"), manufacturer=("manufacturer", "first"),
        body_part=("body_part", "first"), kernel=("kernel", "first"),
        thickness=("slice_thickness", "first"), ps=("pixel_spacing", "median"))
    print(f"pazienti FIT: {len(meta)}; con dose ridotta: {len(by_low)}")
    print(meta.groupby("cell").size().to_string())

    t0 = time.time()
    # --- ancora alta: contrasto di dose, appaiato --------------------------
    dose_pids = sorted(set(by_low) & set(by_full))
    d_dose, null_dose, m_full_d, m_low_d = paired_dose_anchor(
        cache, by_full, by_low, dose_pids, args.p, args.B, args.seed)
    anchor_excess = d_dose - float(np.nanmedian(null_dose))
    print(f"\nANCORA ALTA (dose piena vs ridotta, {len(dose_pids)} pazienti appaiati):")
    print(f"  D = {d_dose:.4f}, nullo appaiato mediana {np.nanmedian(null_dose):.4f}, "
          f"p95 {np.nanpercentile(null_dose,95):.4f}, eccesso {anchor_excess:.4f}")

    # --- contrasti ---------------------------------------------------------
    g = lambda **kw: sorted(meta[np.logical_and.reduce(
        [meta[k] == v for k, v in kw.items()])].index)

    contrasts = []
    ge_abd = g(manufacturer="GE", body_part="ABDOMEN")
    si_abd = g(manufacturer="SIEMENS", body_part="ABDOMEN")
    ge_chest = sorted(meta[(meta.manufacturer == "GE") & (meta.body_part == "CHEST")].index)
    si_chest = g(manufacturer="SIEMENS", body_part="CHEST")
    contrasts.append(("A: addome 5mm, GE STANDARD vs Siemens B30f", "kernel/costruttore", ge_abd, si_abd))
    contrasts.append(("B: torace, GE STANDARD 1,25 vs Siemens B50f 1,5", "kernel/costruttore", ge_chest, si_chest))
    contrasts.append(("C: GE, torace 1,25 vs addome 5", "spessore+anatomia (confusi)", ge_chest, ge_abd))
    contrasts.append(("D: Siemens, torace B50f 1,5 vs addome B30f 5", "kernel+spessore+anatomia", si_chest, si_abd))
    contrasts.append(("G: anatomia, tutti torace vs tutti addome", "anatomia (controllo negativo)",
                      sorted(set(ge_chest) | set(si_chest)), sorted(set(ge_abd) | set(si_abd))))
    contrasts.append(("H: costruttore, tutti GE vs tutti Siemens", "costruttore",
                      sorted(meta[meta.manufacturer == "GE"].index),
                      sorted(meta[meta.manufacturer == "SIEMENS"].index)))
    # E: pixel spacing, terzili entro la stessa cella per non confondere col protocollo
    for cell in sorted(meta.cell.unique()):
        sub = meta[meta.cell == cell].sort_values("ps")
        if len(sub) < 20:
            continue
        k = len(sub) // 3
        contrasts.append((f"E: pixel spacing, terzile basso vs alto — {cell}",
                          "campionamento spaziale",
                          sorted(sub.index[:k]), sorted(sub.index[-k:])))

    rows = []
    for i, (name, axis, a, b) in enumerate(contrasts):
        if len(a) < 5 or len(b) < 5:
            print(f"  salto {name}: gruppi troppo piccoli ({len(a)}, {len(b)})")
            continue
        r = contrast_row(name, axis, cache, by_full, a, b, args.p, args.B,
                         args.seed + 31 * i, anchor_excess)
        rows.append(r)
        print(f"  {name[:52]:52s} D={r['D_oss']:.4f} nullo={r['nullo_mediana']:.4f} "
              f"p={r['p_perm']:.3f} eta={r['eta']:+.3f}", flush=True)

    rows.append({
        "contrasto": "ANCORA: dose piena vs ridotta (stessi pazienti)",
        "asse": "dose", "n_A": len(dose_pids), "n_B": len(dose_pids),
        "D_oss": d_dose, "nullo_mediana": float(np.nanmedian(null_dose)),
        "nullo_p95": float(np.nanpercentile(null_dose, 95)),
        "p_perm": float((np.nansum(null_dose >= d_dose) + 1) / (np.sum(np.isfinite(null_dose)) + 1)),
        "eta": 1.0, "KL_sim": symmetric_kl(m_full_d, m_low_d),
        "cond_A": m_full_d.cond(), "cond_B": m_low_d.cond(),
        "patch_A": m_full_d.n_patches, "patch_B": m_low_d.n_patches,
        "feature_dominanti": "", "significativo_a": True, "rilevante_b": True,
    })

    df = pd.DataFrame(rows)
    df["decisione_ab"] = np.where(
        df.significativo_a & df.rilevante_b,
        "serve (a) e (b): verificare (c) operativo",
        np.where(df.significativo_a, "distinguibile ma sotto soglia di rilevanza",
                 "indistinguibile dal rumore di campionamento"))
    out = ROOT / args.out
    df.to_csv(out, index=False)
    print(f"\nscritto {out}  ({(time.time()-t0)/60:.1f} min)")
    print(df[["contrasto", "asse", "n_A", "n_B", "D_oss", "nullo_mediana",
              "p_perm", "eta", "decisione_ab"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
