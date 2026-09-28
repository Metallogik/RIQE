#!/usr/bin/env python3
"""Batteria di validazione, esperimenti 1-4, sul solo split TEST.

  1. MONOTONICITA'.  Il punteggio peggiora a ogni passo al crescere di rumore
     (bianco e tipo FBP) e di sfocatura.
  2. SOVRAFILTRAGGIO, forma 1 e forma 3.
     Forma 1: filtrare un'immagine a dose piena non puo' migliorarla; ogni
     caso in cui il punteggio migliora e' un fallimento, contato e riportato.
     Forma 3: scala fine di rumore a partire da zero, per cercare un minimo
     del punteggio a rumore **non nullo** -- il meccanismo con cui una
     variante di NIQE ha classificato un'ecografia rumorosa meglio
     dell'originale (MTAP 2024).
     La forma 2, che confronta l'ottimo della metrica con la fedelta' del
     segnale di lesioni inserite, sta in scripts/exp_lesions.py.
  3. DISCRIMINAZIONE.  Cinque denoiser a intensita' appaiata sullo stesso
     ingresso a dose ridotta reale: ordinamento, concordanza fra immagini,
     accordo con i riferimenti pieni.
  4. STABILITA'.  Bootstrap e leave-one-patient-out sul corpus di fitting,
     curva di apprendimento, intervalli di confidenza sui punteggi.

Gli esiti sono riportati **comunque vadano**: le definizioni di fallimento
sono dichiarate in docs/01-proposta.md e non si rinegoziano dopo aver visto i
numeri.

    .venv/bin/python scripts/run_battery.py --moments experiments/bank_test_moments.npz
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

from riqe import degrade as dg  # noqa: E402
from riqe.cache import FeatureCache  # noqa: E402
from riqe.evaluate import (  # noqa: E402
    attach_scores, holm, kendall_w, load_moments, sign_test, spearman, step_monotone,
)
from riqe.model import model_divergence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"


# ---------------------------------------------------------------------------
# 0. Dose reale  (aggiunto dopo la validazione: docs/05 §8, docs/06)
# ---------------------------------------------------------------------------

def exp0_real_dose(P, C, p, model, d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """La dose ridotta reale deve peggiorare il punteggio: e' il caso d'uso."""
    from riqe.dosetest import dose_ordering, dose_pairs, image_moments, summarize

    print("\n" + "=" * 78)
    print("ESPERIMENTO 0 - DOSE RIDOTTA REALE CONTRO DOSE PIENA, STESSA SLICE")
    print("=" * 78)
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")
    pairs = dose_pairs(corpus, split["test"])
    res = dose_ordering(model, pairs, image_moments(cache, list(pairs.path_full) + list(pairs.path_low)))
    sm = summarize(res)
    print(f"  coppie TEST: {sm['n_coppie']} punteggiabili "
          f"(torace {sm['n_torace']}, addome {sm['n_addome']}; non punteggiabili {sm['n_non_punteggiabili']})")
    print(f"  dose ridotta con punteggio peggiore (corretto):  torace {100*sm['corretto_torace']:.1f}%   "
          f"addome {100*sm['corretto_addome']:.1f}%   tutte {100*sm['corretto_tutte']:.1f}%")
    ok = res.dropna(subset=["score_full", "score_low"])
    for reg in ("torace", "addome"):
        g = ok[ok.region == reg]
        if len(g):
            print(f"    {reg}: delta mediano {np.median(g.score_low - g.score_full):+.4f}")
    res.to_csv(OUT / "exp0_real_dose.csv", index=False)
    return res, sm


# ---------------------------------------------------------------------------
# 1b. Rumore relativo al rumore nativo  (aggiunto dopo la validazione)
# ---------------------------------------------------------------------------

def exp1b_relative_noise(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 78)
    print("ESPERIMENTO 1b - RUMORE AGGIUNTO RELATIVO AL RUMORE NATIVO")
    print("=" * 78)
    g = d[(d.kind == "noise_rel") & (d.source == "full")].copy()
    if g.empty:
        print("  nessuna voce noise_rel nel banco")
        return pd.DataFrame(), {}
    orig = {r.slice_path: r.score for r in d[(d.kind == "original") & (d.source == "full")].itertuples()}
    g["base"] = g.slice_path.map(orig)
    g = g.dropna(subset=["score", "base"])
    g["peggiora"] = g.score > g.base
    g["regione"] = np.where(g.cell.str.contains("CHEST"), "torace", "addome")
    t = g.pivot_table(index="regione", columns="rel_increase", values="peggiora", aggfunc="mean")
    print("  frazione in cui il punteggio peggiora, per aumento relativo del rumore:")
    print((100 * t).round(1).to_string())
    perfect = total = 0
    for sp, gg in g.groupby("slice_path"):
        seq = np.concatenate([[orig[sp]], gg.sort_values("rel_increase").score.to_numpy()])
        if np.isfinite(seq).all():
            total += 1
            perfect += int(step_monotone(seq))
    print(f"  monotone a ogni passo: {perfect}/{total} ({100*perfect/max(total,1):.1f}%)")
    g.to_csv(OUT / "exp1b_relative_noise.csv", index=False)
    return g, {"peggiora_per_livello": {f"{k}": {f"{c:g}": float(v) for c, v in row.items()}
                                        for k, row in t.iterrows()},
               "frazione_monotone": perfect / max(total, 1)}


# ---------------------------------------------------------------------------
# 1. Monotonicita'
# ---------------------------------------------------------------------------

def exp1_monotonicity(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 78)
    print("ESPERIMENTO 1 - MONOTONICITA'")
    print("=" * 78)
    rows, detail = [], []
    orig = {r.slice_path: r.score for r in d[(d.kind == "original") & (d.source == "full")].itertuples()}
    for kind, param, unit in (("noise_white", "sigma_hu", "HU"),
                              ("noise_fbp", "sigma_hu", "HU"),
                              ("blur", "sigma_px", "px")):
        g = d[(d.kind == kind) & (d.source == "full")]
        if g.empty:
            continue
        # Sul banco TEST la scala fine di rumore (forma 3) ripete alcuni
        # livelli della scala standard con realizzazioni diverse: qui si usa
        # solo la scala standard, mediando le realizzazioni dello stesso
        # livello, altrimenti una slice avrebbe due punteggi per passo.
        if param == "sigma_hu":
            g = g[g[param].isin(dg.NOISE_SIGMAS_HU)]
        g = g.groupby(["slice_path", param], as_index=False).agg(
            score=("score", "mean"), cell=("cell", "first"))
        levels = sorted(g[param].unique())
        perfect = total = 0
        rhos = []
        for sp, gg in g.groupby("slice_path"):
            gg = gg.sort_values(param)
            seq = np.concatenate([[orig.get(sp, np.nan)], gg["score"].to_numpy()])
            xs = np.concatenate([[0.0], gg[param].to_numpy()])
            if not np.isfinite(seq).all():
                continue
            total += 1
            ok = step_monotone(seq, increasing=True)
            perfect += int(ok)
            rhos.append(spearman(xs, seq))
            detail.append({"kind": kind, "slice_path": sp, "monotona": ok,
                           "spearman": rhos[-1], **{f"s{j}": v for j, v in enumerate(seq)}})
        # test dei segni fra livelli consecutivi
        pvals = {}
        prev_lab, prev = "originale", np.array([orig.get(sp, np.nan) for sp in sorted(orig)])
        prev_idx = sorted(orig)
        for lv in levels:
            cur = g[g[param] == lv].set_index("slice_path")["score"]
            cur = np.array([cur.get(sp, np.nan) for sp in prev_idx])
            k, n, p = sign_test(cur, prev)
            # la chiave include la scala: "originale -> 5HU" compare sia in
            # noise_white sia in noise_fbp, e una chiave sul solo passo
            # farebbe sovrascrivere la correzione della prima dalla seconda
            pvals[(kind, f"{prev_lab} -> {lv:g}{unit}")] = p
            rows.append({"scala": kind, "passo": f"{prev_lab} -> {lv:g}{unit}",
                         "n_peggiorate": k, "n_validi": n,
                         "frazione_peggiorate": k / n if n else np.nan, "p_segni": p})
            prev, prev_lab = cur, f"{lv:g}{unit}"
        adj = holm(pvals)
        for r in rows:
            key = (r["scala"], r["passo"])
            if key in adj:
                r["p_holm"] = adj[key]
        print(f"\n  {kind}: {perfect}/{total} immagini monotone a ogni passo "
              f"({100*perfect/max(total,1):.1f}%), Spearman mediano {np.nanmedian(rhos):+.3f}")
        for r in [x for x in rows if x["scala"] == kind]:
            print(f"    {r['passo']:28s} peggiora in {r['n_peggiorate']:3d}/{r['n_validi']:3d} "
                  f"({100*r['frazione_peggiorate']:5.1f}%)  p_Holm={r.get('p_holm', np.nan):.3g}")
    df = pd.DataFrame(rows)
    det = pd.DataFrame(detail)
    summary = {
        "frazione_monotone": det.groupby("kind")["monotona"].mean().to_dict() if len(det) else {},
        "spearman_mediano": det.groupby("kind")["spearman"].median().to_dict() if len(det) else {},
        "tutti_i_passi_peggiorano": bool((df["frazione_peggiorate"] > 0.99).all()) if len(df) else False,
    }
    det.to_csv(OUT / "exp1_monotonicity_detail.csv", index=False)
    return df, summary


# ---------------------------------------------------------------------------
# 2. Sovrafiltraggio, forme 1 e 3
# ---------------------------------------------------------------------------

def exp2_overfiltering(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 78)
    print("ESPERIMENTO 2 - SOVRAFILTRAGGIO (forme 1 e 3)")
    print("=" * 78)
    orig = {r.slice_path: r.score for r in d[(d.kind == "original") & (d.source == "full")].itertuples()}

    # --- forma 1 -----------------------------------------------------------
    den = d[(d.kind == "denoise") & (d.source == "full")].copy()
    den["score_orig"] = den["slice_path"].map(orig)
    den["delta"] = den["score"] - den["score_orig"]
    den["fallimento"] = den["delta"] < 0
    ok = den["delta"].notna()
    print(f"\n  FORMA 1 - filtrare un'immagine a dose piena")
    print(f"  definizione di fallimento, dichiarata prima: punteggio(filtrata) < punteggio(originale)")
    print(f"  casi valutati: {int(ok.sum())}")
    print(f"  FALLIMENTI: {int(den.loc[ok,'fallimento'].sum())} "
          f"({100*den.loc[ok,'fallimento'].mean():.2f}%)")
    tab = den[ok].pivot_table(index="denoiser", columns="target_residual_hu",
                              values="fallimento", aggfunc="mean")
    print("\n  frazione di fallimenti per denoiser x intensita' (residuo HU):")
    print((100 * tab).round(1).to_string())
    tab2 = den[ok].pivot_table(index="denoiser", columns="target_residual_hu",
                               values="delta", aggfunc="median")
    print("\n  variazione mediana del punteggio (positiva = peggiora, come deve):")
    print(tab2.round(3).to_string())

    # --- forma 3 -----------------------------------------------------------
    fine = d[(d.kind == "noise_fbp") & (d.source == "full")]
    print(f"\n  FORMA 3 - esiste un livello di rumore che MIGLIORA il punteggio?")
    rows3 = []
    if not fine.empty:
        for sp, gg in fine.groupby("slice_path"):
            gg = gg.sort_values("sigma_hu")
            sig = np.concatenate([[0.0], gg["sigma_hu"].to_numpy()])
            sc = np.concatenate([[orig.get(sp, np.nan)], gg["score"].to_numpy()])
            if not np.isfinite(sc).all():
                continue
            j = int(np.argmin(sc))
            rows3.append({"slice_path": sp, "sigma_ottimo_hu": float(sig[j]),
                          "score_min": float(sc[j]), "score_a_zero": float(sc[0]),
                          "guadagno": float(sc[0] - sc[j]),
                          "minimo_non_a_zero": bool(j > 0)})
    r3 = pd.DataFrame(rows3)
    if len(r3):
        frac = r3["minimo_non_a_zero"].mean()
        print(f"  immagini il cui punteggio migliora aggiungendo rumore: "
              f"{int(r3['minimo_non_a_zero'].sum())}/{len(r3)} ({100*frac:.1f}%)")
        if frac > 0:
            q = r3[r3.minimo_non_a_zero]
            print(f"  sigma preferita (mediana): {q.sigma_ottimo_hu.median():.1f} HU; "
                  f"guadagno mediano {q.guadagno.median():.4f}")
            print("  => il modello ha un livello di rumore preferito diverso da zero.")
            print("     E' il meccanismo con cui NIQE-K ha classificato un'ecografia")
            print("     rumorosa meglio dell'originale. Va dichiarato come proprieta'.")
        else:
            print("  => nessuna: il punteggio e' minimo a rumore nullo, come deve essere.")
        r3.to_csv(OUT / "exp2_form3_noise_optimum.csv", index=False)

    den.to_csv(OUT / "exp2_form1_overfiltering.csv", index=False)
    summary = {
        "forma1_n_valutati": int(ok.sum()),
        "forma1_n_fallimenti": int(den.loc[ok, "fallimento"].sum()),
        "forma1_frazione_fallimenti": float(den.loc[ok, "fallimento"].mean()),
        "forma1_per_denoiser": (100 * tab).round(2).to_dict(),
        "forma3_frazione_minimo_non_a_zero": float(r3["minimo_non_a_zero"].mean()) if len(r3) else None,
        "forma3_sigma_preferita_mediana_hu": (
            float(r3.loc[r3.minimo_non_a_zero, "sigma_ottimo_hu"].median()) if len(r3) and r3["minimo_non_a_zero"].any() else None),
    }
    return den, summary


# ---------------------------------------------------------------------------
# 3. Discriminazione
# ---------------------------------------------------------------------------

def exp3_discrimination(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 78)
    print("ESPERIMENTO 3 - DISCRIMINAZIONE FRA DENOISER")
    print("=" * 78)
    low = d[(d.kind == "denoise") & (d.source == "low")]
    if low.empty:
        print("  nessuna immagine a dose ridotta nel banco: esperimento non eseguibile")
        return pd.DataFrame(), {"eseguibile": False}
    rows, W = [], {}
    for lvl, g in low.groupby("target_residual_hu"):
        piv = g.pivot_table(index="slice_path", columns="denoiser", values="score")
        piv = piv.dropna()
        if len(piv) < 3:
            continue
        ranks = piv.rank(axis=1)
        w = kendall_w(ranks.to_numpy())
        W[float(lvl)] = w
        mean_rank = ranks.mean().sort_values()
        rows.append({"residuo_hu": float(lvl), "n_immagini": len(piv), "kendall_W": w,
                     "ordine": " < ".join(mean_rank.index),
                     **{f"rank_{k}": v for k, v in mean_rank.items()},
                     **{f"score_{k}": v for k, v in piv.mean().items()}})
        print(f"\n  residuo {lvl:g} HU  ({len(piv)} immagini)  W di Kendall = {w:.3f}")
        print("    ordine (migliore -> peggiore secondo RIQE): " + " < ".join(mean_rank.index))
        print("    punteggio medio: " + "  ".join(f"{k}={v:.3f}" for k, v in piv.mean().items()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp3_discrimination.csv", index=False)
    print(f"\n  concordanza fra immagini (W di Kendall): "
          f"mediana {np.nanmedian(list(W.values())):.3f} su {len(W)} livelli")
    print("  W alto = il modello ordina i denoiser in modo stabile; non dice che")
    print("  l'ordine sia quello giusto. L'accordo con la fedelta' del segnale e'")
    print("  in scripts/exp_lesions.py.")
    return df, {"eseguibile": True, "kendall_W_per_livello": W,
                "kendall_W_mediano": float(np.nanmedian(list(W.values()))) if W else None}


# ---------------------------------------------------------------------------
# 4. Stabilita'
# ---------------------------------------------------------------------------

def exp4_stability(P, C, p, B, seed, d_test, NU, SG) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 78)
    print("ESPERIMENTO 4 - STABILITA'")
    print("=" * 78)
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit_pids = sorted(set(split["fit"]))
    full = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(fit_pids)]
    by_patient = defaultdict(list)
    for r in full.itertuples():
        by_patient[r.patient_id].append(r.path)
    cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")

    allp = [q for pid in fit_pids for q in by_patient[pid]]
    full_model = cache.fit_fast(allp, p, n_patients=len(fit_pids))
    print(f"  modello completo: {len(fit_pids)} pazienti, {full_model.n_patches} patch, "
          f"cond={full_model.cond():.3e}")

    rng = np.random.default_rng(seed)
    d_boot, boot_models = [], []
    for _ in range(B):
        take = rng.choice(fit_pids, size=len(fit_pids), replace=True)
        m = cache.fit_fast([q for pid in take for q in by_patient[pid]], p, n_patients=len(fit_pids))
        d_boot.append(model_divergence(full_model, m))
        boot_models.append(m)
    d_boot = np.asarray(d_boot)
    print(f"  bootstrap sui pazienti (B={B}): D mediana {np.median(d_boot):.4f}, "
          f"p95 {np.percentile(d_boot,95):.4f}, max {d_boot.max():.4f}")

    d_lopo = []
    for pid in fit_pids:
        m = cache.fit_fast([q for q2 in fit_pids if q2 != pid for q in by_patient[q2]], p,
                           n_patients=len(fit_pids) - 1)
        d_lopo.append({"patient_id": pid, "D": model_divergence(full_model, m)})
    lopo = pd.DataFrame(d_lopo).sort_values("D", ascending=False)
    print(f"  leave-one-patient-out: D mediana {lopo.D.median():.5f}, "
          f"massima {lopo.D.max():.5f} ({lopo.patient_id.iloc[0]})")

    cv = np.sqrt(np.diag(full_model.sigma)) / np.abs(full_model.nu)
    print(f"  coefficiente di variazione delle 36 componenti di nu: "
          f"mediano {np.median(cv):.3f}, massimo {cv.max():.3f}")

    curve = []
    for n in (10, 20, 40, 80, 120, len(fit_pids)):
        if n > len(fit_pids):
            continue
        ds = []
        for rep in range(10):
            take = np.random.default_rng(seed + rep).choice(fit_pids, size=n, replace=False)
            m = cache.fit_fast([q for pid in take for q in by_patient[pid]], p, n_patients=n)
            ds.append(model_divergence(full_model, m))
        curve.append({"n_pazienti": n, "D_mediana": float(np.median(ds)),
                      "D_p95": float(np.percentile(ds, 95))})
        print(f"    curva di apprendimento: {n:3d} pazienti -> D = {np.median(ds):.4f}")

    # intervalli di confidenza sui punteggi di un banco fisso
    orig_rows = d_test[(d_test.kind == "original") & (d_test.source == "full")]["row"].to_numpy()
    from riqe.evaluate import score_all
    S = np.stack([score_all(m, NU, SG, orig_rows) for m in boot_models[: min(B, 60)]])
    base = score_all(full_model, NU, SG, orig_rows)
    lo, hi = np.nanpercentile(S, [2.5, 97.5], axis=0)
    width = np.nanmedian(hi - lo)
    rel = np.nanmedian((hi - lo) / base)
    rho = [spearman(base, S[i]) for i in range(S.shape[0])]
    print(f"  punteggi su {len(orig_rows)} immagini TEST: ampiezza mediana IC 95% = "
          f"{width:.4f} ({100*rel:.1f}% del punteggio)")
    print(f"  Spearman fra ordinamenti prodotti da modelli bootstrap diversi: "
          f"mediano {np.nanmedian(rho):.4f}, minimo {np.nanmin(rho):.4f}")

    pd.DataFrame(curve).to_csv(OUT / "exp4_learning_curve.csv", index=False)
    lopo.to_csv(OUT / "exp4_lopo.csv", index=False)
    summary = {
        "bootstrap_B": B,
        "D_boot_mediana": float(np.median(d_boot)),
        "D_boot_p95": float(np.percentile(d_boot, 95)),
        "D_lopo_mediana": float(lopo.D.median()),
        "D_lopo_max": float(lopo.D.max()),
        "paziente_piu_influente": lopo.patient_id.iloc[0],
        "cv_nu_mediano": float(np.median(cv)),
        "cv_nu_max": float(cv.max()),
        "curva_apprendimento": curve,
        "ic95_ampiezza_mediana": float(width),
        "ic95_relativa_mediana": float(rel),
        "spearman_bootstrap_mediano": float(np.nanmedian(rho)),
        "spearman_bootstrap_min": float(np.nanmin(rho)),
        "cond_sigma": full_model.cond(),
    }
    return lopo, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moments", default="experiments/bank_test_moments.npz")
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--skip", default="", help="esperimenti da saltare, es. '4'")
    args = ap.parse_args()

    ch_file = ROOT / "experiments" / "hparam_choice.json"
    ch = json.loads(ch_file.read_text()) if ch_file.exists() else {}
    P = args.P if args.P is not None else ch.get("P")
    C = args.C if args.C is not None else ch.get("C")
    p = args.p if args.p is not None else ch.get("p")
    if P is None:
        ap.error("servono --P --C --p, oppure experiments/hparam_choice.json")

    split = json.load(open(ROOT / "corpus" / "split.json"))
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    fit_pids = set(split["fit"])
    fit = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(fit_pids)]
    cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")
    model = cache.fit(list(fit["path"]), p, n_patients=fit.patient_id.nunique())

    meta, NU, SG = load_moments(args.moments)
    meta = meta[(meta.P == P) & (meta.C == C)].copy()
    d = attach_scores(meta, model, NU, SG)
    test_pids = set(split["test"])
    intruders = set(d.patient_id) - test_pids
    print(f"modello: P={P} C={C:g} p={p:g}, {model.n_patches} patch da "
          f"{model.n_patients} pazienti FIT")
    print(f"banco: {len(d)} immagini, {d.slice_path.nunique()} slice, "
          f"{d.patient_id.nunique()} pazienti"
          + (f"  ATTENZIONE: {len(intruders)} pazienti fuori da TEST" if intruders else " (tutti TEST)"))
    print(f"punteggi non definibili: {int(d.score.isna().sum())}/{len(d)}")

    t0 = time.time()
    summaries = {}
    skip = set(args.skip.split(","))
    if "0" not in skip:
        _, summaries["esp0_dose_reale"] = exp0_real_dose(P, C, p, model, d)
    if "1" not in skip:
        _, summaries["esp1_monotonicita"] = exp1_monotonicity(d)
        _, summaries["esp1b_rumore_relativo"] = exp1b_relative_noise(d)
    if "2" not in skip:
        _, summaries["esp2_sovrafiltraggio"] = exp2_overfiltering(d)
    if "3" not in skip:
        _, summaries["esp3_discriminazione"] = exp3_discrimination(d)
    if "4" not in skip:
        _, summaries["esp4_stabilita"] = exp4_stability(P, C, p, args.bootstrap, args.seed, d, NU, SG)

    summaries["_config"] = {"P": P, "C": C, "p": p, "moments": args.moments,
                            "n_test_patients": int(d.patient_id.nunique())}
    (OUT / "battery_summary.json").write_text(json.dumps(summaries, indent=1, default=float))
    d.to_csv(OUT / "battery_scores.csv", index=False)
    print(f"\nscritti {OUT}/battery_summary.json e battery_scores.csv "
          f"({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
