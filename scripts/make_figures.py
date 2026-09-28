#!/usr/bin/env python3
"""Figure dell'articolo, dai risultati degli esperimenti.

Regole seguite, dichiarate in riqe/figstyle.py:
  * mai due assi y.  La figura principale confronta punteggio e ritenzione
    del segnale, che hanno scale diverse: due pannelli impilati che
    condividono la x, non un secondo asse.
  * identita' di serie mai affidata al solo colore: colore + tratteggio +
    marcatore, piu' legenda ed etichette dirette quando le serie sono poche.
  * griglia e assi recessivi; nessun numero su ogni punto.

    .venv/bin/python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe import figstyle as fs  # noqa: E402

fs.apply(matplotlib)
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
FIG = ROOT / "paper" / "figures"


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  scritta {name}")


def fig_main_overfiltering() -> None:
    """Figura principale: il punteggio contro la fedelta' del segnale."""
    f = EXP / "exp2_form2_lesions.csv"
    if not f.exists():
        print("  salto figura principale: manca exp2_form2_lesions.csv")
        return
    d = pd.read_csv(f)
    base = d[d.denoiser == "nessuno"]
    den = d[d.denoiser != "nessuno"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.15]})

    # pannello alto: punteggio per denoiser
    for i, (name, g) in enumerate(den.groupby("denoiser")):
        m = g.groupby("target_residual_hu")["riqe"].median()
        ax1.plot(m.index, m.values, label=name, **fs.style_for(i))
    b = base["riqe"].median()
    ax1.axhline(b, color=fs.INK_MUTED, lw=1.2, ls=":", zorder=1)
    ax1.annotate("non filtrata", xy=(den.target_residual_hu.min(), b),
                 xytext=(2, 4), textcoords="offset points",
                 fontsize=8, color=fs.INK_2, va="bottom")
    ax1.set_ylabel("punteggio RIQE\n(piu' basso = piu' vicino al modello)")
    ax1.set_title("Il punteggio premia un filtraggio che il segnale non sopravvive")
    ax1.legend(ncol=3, loc="upper left", fontsize=7.5)
    fs.despine(ax1)

    # pannello basso: ritenzione del segnale per dimensione di lesione
    for i, (dmm, g) in enumerate(den.groupby("diametro_mm")):
        m = g.groupby("target_residual_hu")["ritenzione_matched"].median()
        c = den[den.diametro_mm == dmm]["contrasto_hu"].iloc[0]
        ax2.plot(m.index, m.values, label=f"{dmm:.0f} mm, +{c:.0f} HU", **fs.style_for(i))
        ax2.annotate(f"{dmm:.0f} mm", xy=(m.index[-1], m.values[-1]),
                     xytext=(4, 0), textcoords="offset points",
                     fontsize=8, color=fs.SERIES[i % len(fs.SERIES)], va="center")
    ax2.axhline(0.5, color=fs.INK_MUTED, lw=1.0, ls="--", zorder=1)
    ax2.annotate("meta' del segnale perduta", xy=(den.target_residual_hu.min(), 0.5),
                 xytext=(2, -10), textcoords="offset points", fontsize=8, color=fs.INK_2)
    ax2.set_ylabel("ritenzione del segnale\n(filtro adattato)")
    ax2.set_xlabel("intensita' di filtraggio (deviazione standard del residuo, HU)")
    ax2.set_ylim(0, 1.05)
    ax2.set_xscale("log")
    ax2.set_xticks(sorted(den.target_residual_hu.unique()))
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.legend(loc="lower left", fontsize=7.5)
    fs.despine(ax2)

    fig.tight_layout()
    save(fig, "fig1_sovrafiltraggio")


def fig_monotonicity() -> None:
    f = EXP / "exp1_monotonicity_detail.csv"
    if not f.exists():
        print("  salto monotonicita': manca exp1_monotonicity_detail.csv")
        return
    d = pd.read_csv(f)
    kinds = [k for k in ("noise_white", "noise_fbp", "blur") if k in set(d["kind"])]
    titles = {"noise_white": "rumore bianco", "noise_fbp": "rumore tipo FBP",
              "blur": "sfocatura gaussiana"}
    cols = [c for c in d.columns if c.startswith("s") and c[1:].isdigit()]
    cols = sorted(cols, key=lambda c: int(c[1:]))
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.0 * len(kinds), 3.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, kinds):
        g = d[d["kind"] == k]
        M = g[cols].to_numpy(dtype=float)
        x = np.arange(M.shape[1])
        for row in M[: min(len(M), 120)]:
            ax.plot(x, row, color=fs.SERIES[0], alpha=0.10, lw=0.8, zorder=1)
        med = np.nanmedian(M, axis=0)
        ax.plot(x, med, color=fs.SERIES[0], lw=2.4, marker="o", zorder=3)
        frac = g["monotona"].mean()
        ax.set_title(f"{titles[k]}\n{100*frac:.0f}% monotone a ogni passo", fontsize=9)
        ax.set_xlabel("livello di degradazione")
        ax.set_xticks(x)
        fs.despine(ax)
    axes[0].set_ylabel("punteggio RIQE")
    fig.tight_layout()
    save(fig, "fig2_monotonicita")


def fig_noise_optimum() -> None:
    f = EXP / "exp2_form3_noise_optimum.csv"
    if not f.exists():
        print("  salto ottimo di rumore: manca exp2_form3_noise_optimum.csv")
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    frac = d["minimo_non_a_zero"].mean()
    vals = d.loc[d["minimo_non_a_zero"], "sigma_ottimo_hu"]
    if len(vals):
        bins = np.histogram_bin_edges(vals, bins="auto")
        ax.hist(vals, bins=bins, color=fs.SERIES[1], edgecolor=fs.SURFACE, linewidth=1.2)
        ax.set_xlabel("deviazione standard del rumore che minimizza il punteggio (HU)")
        ax.set_ylabel("immagini")
        ax.set_title(f"Il modello ha un livello di rumore preferito diverso da zero\n"
                     f"{100*frac:.0f}% delle immagini migliora aggiungendo rumore", fontsize=9)
    else:
        ax.text(0.5, 0.5, "nessuna immagine migliora aggiungendo rumore",
                ha="center", va="center", transform=ax.transAxes, color=fs.INK_2)
        ax.set_title("Il punteggio e' minimo a rumore nullo, come deve essere", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fs.despine(ax)
    fig.tight_layout()
    save(fig, "fig3_ottimo_di_rumore")


def fig_stratification() -> None:
    f = EXP / "stratification.csv"
    if not f.exists():
        print("  salto stratificazione: manca stratification.csv")
        return
    d = pd.read_csv(f).sort_values("eta")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(d) + 1.9))
    # banda del nullo di permutazione: da mediana a p95
    for i, r in enumerate(d.itertuples()):
        ax.plot([r.nullo_mediana, r.nullo_p95], [i, i], color=fs.GRID, lw=6,
                solid_capstyle="butt", zorder=1)
    ax.scatter(d["nullo_mediana"], y, s=22, color=fs.INK_MUTED, marker="|",
               zorder=2, label="nullo di permutazione (mediana - p95)")
    anchor = d["asse"] == "dose"
    ax.scatter(d.loc[~anchor, "D_oss"], y[~anchor.to_numpy()], s=46,
               color=fs.SERIES[0], marker="o", zorder=3, label="divergenza osservata")
    ax.scatter(d.loc[anchor, "D_oss"], y[anchor.to_numpy()], s=64,
               color=fs.SERIES[1], marker="D", zorder=3,
               label="ancora: dose piena vs ridotta")
    ax.set_yticks(y)
    ax.set_yticklabels([c[:58] for c in d["contrasto"]], fontsize=7.5)
    ax.set_xlabel("divergenza fra modelli D (stesse unita' del punteggio)")
    ax.set_title("Quanto divergono i sotto-modelli, fra rumore di campionamento\n"
                 "e una differenza fisica nota", fontsize=9)
    for i, r in enumerate(d.itertuples()):
        if np.isfinite(r.eta):
            ax.annotate(f"eta={r.eta:+.2f}", xy=(r.D_oss, i), xytext=(7, 0),
                        textcoords="offset points", fontsize=7, color=fs.INK_2, va="center")
    ax.legend(loc="lower right", fontsize=7.5)
    fs.despine(ax)
    fig.tight_layout()
    save(fig, "fig4_stratificazione")


def fig_photo_baseline() -> None:
    f = EXP / "exp5_scores_three_models.csv"
    if not f.exists():
        print("  salto baseline fotografica: manca exp5_scores_three_models.csv")
        return
    d = pd.read_csv(f).dropna(subset=["riqe", "photo"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.6))
    ax1.scatter(d["riqe"], d["photo"], s=9, color=fs.SERIES[0], alpha=0.35,
                edgecolors="none")
    ax1.set_xlabel("punteggio RIQE (corpus TC)")
    ax1.set_ylabel("punteggio col modello fotografico")
    try:
        rho = json.loads((EXP / "exp5_summary.json").read_text())["spearman_riqe_vs_photo"]
        ax1.set_title(f"Stesse immagini, due modelli\nSpearman {rho:+.3f}", fontsize=9)
    except Exception:  # noqa: BLE001
        ax1.set_title("Stesse immagini, due modelli", fontsize=9)
    fs.despine(ax1)

    v = pd.read_csv(EXP / "exp5_verdicts.csv") if (EXP / "exp5_verdicts.csv").exists() else None
    if v is not None:
        metrics = [c for c in v.columns if c != "modello"]
        x = np.arange(len(metrics))
        w = 0.26
        for i, r in enumerate(v.itertuples()):
            vals = [getattr(r, m) for m in metrics]
            ax2.bar(x + (i - 1) * w, vals, width=w - 0.02,
                    color=fs.SERIES[i], edgecolor=fs.SURFACE, linewidth=1.2,
                    label=r.modello)
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace("monotone_", "mon. ").replace(
            "frazione_fallimenti_sovrafiltraggio", "fallimenti\nsovrafiltr.")
            for m in metrics], fontsize=7.5)
        ax2.set_ylabel("frazione")
        ax2.set_title("I verdetti cambiano fra i tre modelli?", fontsize=9)
        ax2.legend(fontsize=7)
        fs.despine(ax2)
    fig.tight_layout()
    save(fig, "fig5_baseline_fotografica")


def fig_stability() -> None:
    f = EXP / "exp4_learning_curve.csv"
    if not f.exists():
        print("  salto stabilita': manca exp4_learning_curve.csv")
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(d["n_pazienti"], d["D_mediana"], **fs.style_for(0))
    ax.fill_between(d["n_pazienti"], d["D_mediana"], d["D_p95"],
                    color=fs.SERIES[0], alpha=0.14, linewidth=0)
    ax.set_xlabel("pazienti nel corpus di fitting")
    ax.set_ylabel("D dal modello completo")
    ax.set_title("Quanti pazienti bastano", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(d["n_pazienti"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fs.despine(ax)
    fig.tight_layout()
    save(fig, "fig6_stabilita")


def main() -> int:
    print("figure:")
    for fn in (fig_main_overfiltering, fig_monotonicity, fig_noise_optimum,
               fig_stratification, fig_photo_baseline, fig_stability):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  ERRORE in {fn.__name__}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
