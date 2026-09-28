#!/usr/bin/env python3
"""Ricerca degli iperparametri (P, C, p) con criterio lessicografico.

Il criterio e' dichiarato **prima** di eseguire (docs/01-proposta.md 4), e in
quest'ordine:

  1. SICUREZZA AL SOVRAFILTRAGGIO.  Numero di casi in cui un'immagine
     filtrata prende un punteggio migliore dell'originale a dose piena.
     Deve essere 0: le configurazioni che falliscono sono **eliminate**, non
     penalizzate.
  2. MONOTONICITA'.  Ordinamento corretto a ogni passo delle scale di rumore
     (bianco e tipo FBP) e di sfocatura.
  3. STABILITA'.  Divergenza mediana di bootstrap sui pazienti.
  4. PARSIMONIA E CONDIZIONAMENTO.  A pari prestazione, la configurazione con
     piu' patch per immagine e Sigma meglio condizionata.

Non e' il criterio di Computers 2025, che ottimizza la frazione di immagini
in cui la rete piu' profonda risulta il miglior denoiser: quello assume la
risposta e la usa come bersaglio.

CRITERIO RIVISTO, dichiarato come tale (docs/05 §8, docs/06).  Il criterio
sopra era cieco alla proprieta' piu' importante -- l'ordinamento della dose
ridotta reale -- e ha scelto una configurazione che la ordina correttamente
solo nel 38,5% degli addomi.  Dopo aver visto i dati di validazione, e con
l'accordo esplicito del committente, il criterio diventa:

  1. DOSE REALE.  Frazione di coppie (dose piena, dose ridotta reale, stessa
     slice) in cui la dose ridotta ha punteggio peggiore, presa al minimo fra
     torace e addome.  Soglia di superamento 0,95.
  2. SOVRAFILTRAGGIO.  Frazione di immagini filtrate che battono l'originale:
     minima.
  3. RUMORE RELATIVO.  Frazione di immagini con rumore aggiunto >= 20% del
     rumore nativo che peggiorano: massima.
  4. STABILITA', poi patch per immagine.
Se nessuna configurazione supera la soglia del punto 1, si ordina per il
punto 1 e poi per il punto 2.  La scelta del criterio originale viene
calcolata e riportata accanto, non sostituita di nascosto.

Tutto sul solo split interno di FIT.  TEST non viene toccato.

    .venv/bin/python scripts/search_hparams.py \
        --moments experiments/bank_val_inner_moments.npz
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
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.cache import FeatureCache  # noqa: E402
from riqe.dosetest import dose_ordering, dose_pairs, image_moments, summarize  # noqa: E402
from riqe.model import RCOND, mahalanobis_mixed, model_divergence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P_GRID = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 0.90)
DOSE_PASS = 0.95
REL_NOISE_MIN = 0.20


def load_moments(paths):
    """Uno o piu' file di momenti, concatenati; l'indice resta la riga."""
    NUs, SGs, metas = [], [], []
    for path in paths:
        z = np.load(path, allow_pickle=False)
        meta = pd.DataFrame(json.loads(str(z["meta"])))
        meta["slice_path"] = json.loads(str(z["slice_path"]))
        meta["patient_id"] = json.loads(str(z["patient_id"]))
        meta["cell"] = json.loads(str(z["cell"]))
        NUs.append(z["nu"])
        SGs.append(z["sigma"])
        metas.append(meta)
    meta = pd.concat(metas, ignore_index=True)
    return np.concatenate(NUs), np.concatenate(SGs), meta


def score_rows(model, NU, SG, idx) -> np.ndarray:
    """Punteggio di un insieme di righe di momenti contro un modello."""
    out = np.full(len(idx), np.nan)
    for k, i in enumerate(idx):
        if np.isnan(NU[i, 0]):
            continue
        out[k] = mahalanobis_mixed(model.nu, model.sigma, NU[i], SG[i], RCOND)
    return out


def bootstrap_divergence(cache, paths_by_patient, p, B, seed) -> np.ndarray:
    """D(modello completo, modello su un bootstrap dei pazienti)."""
    rng = np.random.default_rng(seed)
    pids = sorted(paths_by_patient)
    allp = [q for pid in pids for q in paths_by_patient[pid]]
    full = cache.fit_fast(allp, p, n_patients=len(pids))
    ds = []
    for _ in range(B):
        take = rng.choice(pids, size=len(pids), replace=True)
        paths = [q for pid in take for q in paths_by_patient[pid]]
        try:
            ds.append(model_divergence(full, cache.fit_fast(paths, p, n_patients=len(pids))))
        except ValueError:
            ds.append(np.nan)
    return np.asarray(ds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moments", required=True, nargs="+")
    ap.add_argument("--bootstrap", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default="experiments/hparam_search.csv")
    args = ap.parse_args()

    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit_inner = set(split["fit_inner"])

    kept = corpus[(corpus["kind"] == "full") & corpus["keep"]]
    fit_slices = kept[kept.patient_id.isin(fit_inner)]
    by_patient = defaultdict(list)
    for r in fit_slices.itertuples():
        by_patient[r.patient_id].append(r.path)
    print(f"fitting interno: {len(by_patient)} pazienti, {len(fit_slices)} slice")

    NU, SG, meta = load_moments(args.moments)
    configs = sorted({(int(r.P), float(r.C)) for r in meta.itertuples()})
    print(f"banco: {len(meta)} righe, {meta.slice_path.nunique()} slice, "
          f"{len(configs)} configurazioni (P, C)")

    # rumore nativo per slice, per esprimere il rumore aggiunto in termini relativi
    slices_tab = pd.read_parquet(ROOT / "corpus" / "slices.parquet").set_index("path")
    sigma_nat = slices_tab["body_sigma_med"]

    # coppie di dose reale sullo split di validazione interno
    pairs = dose_pairs(corpus, split["val_inner"])
    print(f"coppie di dose reale (validazione): {len(pairs)} "
          f"(torace {int((pairs.region=='torace').sum())}, addome {int((pairs.region=='addome').sum())})")

    # indici delle righe utili, per configurazione
    full_rows = meta[meta["source"] == "full"]
    results = []
    t0 = time.time()

    for P, C in configs:
        cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")
        sub = full_rows[(full_rows.P == P) & (full_rows.C == C)]
        orig = {r.slice_path: r.Index for r in sub[sub.kind == "original"].itertuples()}
        pair_moments = image_moments(cache, list(pairs.path_full) + list(pairs.path_low))
        noise = sub[sub.kind.isin(["noise_white", "noise_fbp", "noise_rel"])].copy()
        sn = noise["slice_path"].map(sigma_nat).to_numpy(dtype=float)
        noise["rel"] = np.sqrt(sn ** 2 + noise["sigma_hu"].to_numpy(dtype=float) ** 2) / sn - 1.0
        noise = noise[(noise["rel"] >= REL_NOISE_MIN) & noise["slice_path"].isin(orig)]

        for p in P_GRID:
            try:
                model = cache.fit_fast([q for v in by_patient.values() for q in v], p,
                                       n_patients=len(by_patient))
            except ValueError as e:
                results.append({"P": P, "C": C, "p": p, "error": str(e)})
                continue

            # --- criterio 1: sicurezza al sovrafiltraggio -----------------
            den = sub[sub.kind == "denoise"]
            s_orig = score_rows(model, NU, SG, [orig[sp] for sp in den.slice_path])
            s_den = score_rows(model, NU, SG, list(den.index))
            ok = np.isfinite(s_orig) & np.isfinite(s_den)
            fail_mask = ok & (s_den < s_orig)
            n_fail = int(fail_mask.sum())
            frac_fail = float(fail_mask.sum() / max(ok.sum(), 1))

            # sfocatura: e' anch'essa un sovrafiltraggio del pristine
            bl = sub[sub.kind == "blur"]
            sb_o = score_rows(model, NU, SG, [orig[sp] for sp in bl.slice_path])
            sb = score_rows(model, NU, SG, list(bl.index))
            okb = np.isfinite(sb_o) & np.isfinite(sb)
            n_fail_blur = int((okb & (sb < sb_o)).sum())

            # --- criterio 2: monotonicita' --------------------------------
            mono = {}
            for kind, param in (("noise_white", "sigma_hu"), ("noise_fbp", "sigma_hu"),
                                ("blur", "sigma_px")):
                g = sub[sub.kind == kind]
                perfect = total = 0
                for sp, gg in g.groupby("slice_path"):
                    gg = gg.sort_values(param)
                    sc = score_rows(model, NU, SG, list(gg.index))
                    s0 = score_rows(model, NU, SG, [orig[sp]])[0]
                    seq = np.concatenate([[s0], sc])
                    if np.isfinite(seq).all():
                        total += 1
                        if np.all(np.diff(seq) > 0):
                            perfect += 1
                mono[kind] = perfect / total if total else np.nan
            mono_mean = float(np.nanmean(list(mono.values())))
            mono_min = float(np.nanmin(list(mono.values())))

            # --- dose reale ------------------------------------------------
            dsum = summarize(dose_ordering(model, pairs, pair_moments))

            # --- rumore relativo -------------------------------------------
            s_n = score_rows(model, NU, SG, list(noise.index))
            # l'originale deve esistere: un indice di ripiego come -1 leggerebbe
            # l'ultima riga dell'array, cioe' il punteggio di un'altra immagine
            s_n0 = score_rows(model, NU, SG, [orig[sp] for sp in noise.slice_path])
            okn = np.isfinite(s_n) & np.isfinite(s_n0)
            rel_detect = float((s_n[okn] > s_n0[okn]).mean()) if okn.any() else np.nan

            # --- criterio 3: stabilita' -----------------------------------
            d_boot = bootstrap_divergence(cache, by_patient, p, args.bootstrap, args.seed)
            d_med = float(np.nanmedian(d_boot))
            d_p95 = float(np.nanpercentile(d_boot, 95))

            # --- criterio 4: patch e condizionamento ----------------------
            counts = cache.patch_counts([q for v in by_patient.values() for q in v], p)
            n_score_patches = np.asarray([
                r.n_patches for r in sub[sub.kind == "original"].itertuples()
            ])

            results.append({
                "P": P, "C": C, "p": p,
                "dose_corretto_min": dsum["corretto_min"],
                "dose_corretto_torace": dsum["corretto_torace"],
                "dose_corretto_addome": dsum["corretto_addome"],
                "dose_n_torace": dsum["n_torace"],
                "dose_n_addome": dsum["n_addome"],
                "rumore_rel_rilevato": rel_detect,
                "n_overfilter_fail": n_fail,
                "frac_overfilter_fail": frac_fail,
                "n_blur_fail": n_fail_blur,
                "mono_mean": mono_mean,
                "mono_min": mono_min,
                **{f"mono_{k}": v for k, v in mono.items()},
                "d_boot_median": d_med,
                "d_boot_p95": d_p95,
                "fit_patches_per_slice": float(counts.mean()),
                "fit_patches_total": int(counts.sum()),
                "score_patches_median": float(np.median(n_score_patches)),
                "cond_sigma": model.cond(),
                "score_orig_median": float(np.nanmedian(
                    score_rows(model, NU, SG, list(sub[sub.kind == "original"].index)))),
            })
            r = results[-1]
            print(f"  P={P:3d} C={C:<5g} p={p:<5.2f} | dose t/a={dsum['corretto_torace']:.2f}/"
                  f"{dsum['corretto_addome']:.2f} | rum.rel={rel_detect:.2f} | sovrafiltr.={n_fail:4d} "
                  f"({100*frac_fail:5.1f}%) blur_fail={n_fail_blur:3d} | mono={mono_mean:.3f} "
                  f"(min {mono_min:.3f}) | D_boot={d_med:.4f} | patch/slice={r['fit_patches_per_slice']:6.1f}"
                  f" | cond={model.cond():.1e}", flush=True)

    df = pd.DataFrame(results)
    out = ROOT / args.out
    df.to_csv(out, index=False)
    print(f"\nscritto {out}  ({(time.time()-t0)/60:.1f} min)")

    cols = ["P", "C", "p", "dose_corretto_min", "dose_corretto_torace", "dose_corretto_addome",
            "rumore_rel_rilevato", "frac_overfilter_fail", "mono_min", "d_boot_median",
            "fit_patches_per_slice", "score_patches_median", "cond_sigma"]

    # --- 1. criterio ORIGINALE, dichiarato prima di eseguire ----------------
    print("\n=== criterio originale (dichiarato prima di eseguire) ===")
    safe = df[df["n_overfilter_fail"] == 0]
    if len(safe) == 0:
        print("NESSUNA configurazione e' sicura al sovrafiltraggio (criterio 1 = 0).")
        ranked_o = df.sort_values(
            ["frac_overfilter_fail", "mono_min", "mono_mean", "d_boot_median"],
            ascending=[True, False, False, True])
    else:
        ranked_o = safe.sort_values(
            ["mono_min", "mono_mean", "d_boot_median", "fit_patches_per_slice"],
            ascending=[False, False, True, False])
    best_o = ranked_o.iloc[0]
    print(f"scelta originale: P={int(best_o.P)} C={best_o.C:g} p={best_o.p:g}  "
          f"(dose t/a {best_o.dose_corretto_torace:.3f}/{best_o.dose_corretto_addome:.3f}, "
          f"sovrafiltr. {100*best_o.frac_overfilter_fail:.1f}%)")

    # --- 2. criterio RIVISTO -------------------------------------------------
    print("\n=== criterio rivisto (dose reale prima) ===")
    passing = df[df["dose_corretto_min"] >= DOSE_PASS]
    if len(passing):
        print(f"configurazioni con dose reale ordinata >= {DOSE_PASS:.0%} in entrambe le regioni: "
              f"{len(passing)}/{len(df)}")
        ranked = passing.sort_values(
            ["frac_overfilter_fail", "rumore_rel_rilevato", "d_boot_median", "fit_patches_per_slice"],
            ascending=[True, False, True, False])
    else:
        print(f"NESSUNA configurazione ordina la dose reale >= {DOSE_PASS:.0%} in entrambe le "
              f"regioni: ordinamento per dose, poi sovrafiltraggio.")
        ranked = df.sort_values(
            ["dose_corretto_min", "frac_overfilter_fail", "rumore_rel_rilevato", "d_boot_median"],
            ascending=[False, True, False, True])
    print(ranked[cols].head(15).to_string(index=False))
    best = ranked.iloc[0]

    (ROOT / "experiments" / "hparam_choice.json").write_text(json.dumps({
        "P": int(best.P), "C": float(best.C), "p": float(best.p),
        "criterion": ("revised: real-dose ordering (min over chest/abdomen, pass >= 0.95), "
                      "then overfiltering failure fraction, then relative-noise detection, "
                      "then bootstrap stability, then patches per slice"),
        "declared_before_run": False,
        "revision_reason": ("the pre-declared criterion ignored real-dose ordering and selected "
                            "a configuration ordering reduced-dose abdomen correctly in 38.5% "
                            "of validation pairs; revised after seeing validation data, "
                            "with the commissioner's explicit agreement (docs/06)"),
        "any_config_passes_dose": bool(len(passing) > 0),
        "any_safe_config_overfiltering": bool(len(safe) > 0),
        "metrics": {k: (float(best[k]) if k in best else None) for k in cols},
        "original_criterion_choice": {
            "P": int(best_o.P), "C": float(best_o.C), "p": float(best_o.p),
            "criterion": "lexicographic: overfilter safety, then step monotonicity, "
                         "then bootstrap stability, then patches per slice",
            "declared_before_run": True,
            "metrics": {k: (float(best_o[k]) if k in best_o else None) for k in cols},
        },
    }, indent=1))
    print(f"\nscelta (criterio rivisto): P={int(best.P)} C={best.C:g} p={best.p:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
