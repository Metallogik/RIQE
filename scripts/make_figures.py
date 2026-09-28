#!/usr/bin/env python3
"""Figures for the paper, from the experiment results.

Written to paper/figures/. Rules followed, declared in riqe/figstyle.py:
  * never two y axes. The main figure compares score and signal retention,
    which have different scales: two stacked panels sharing x, not a second
    axis.
  * series identity never carried by colour alone: colour + dash +
    marker, plus legend and direct labels when there are few series.
  * recessive grid and axes; no number on every point.

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
    print(f"  wrote {name}")


NICE = {"gaussian": "Gaussian", "wavelet": "wavelet", "tv": "TV", "nlm": "NLM",
        "bilateral": "bilateral"}


def fig_main_overfiltering() -> None:
    """Main figure, per denoiser: how often RIQE prefers the filtered image,
    and how much of the lesion signal is left.

    Per denoiser and not pooled: the mean over denoisers hides the result,
    because linear smoothing is always penalised while edge-preserving
    non-linear filters are rewarded.
    """
    f = EXP / "exp2_form2_lesions.csv"
    if not f.exists():
        print("  skipping main figure: exp2_form2_lesions.csv missing")
        return
    d = pd.read_csv(f)
    dmm = 4.0
    base = d[d.denoiser == "none"][["slice_path", "diameter_mm", "riqe"]].rename(
        columns={"riqe": "base"})
    den = d[d.denoiser != "none"].merge(base, on=["slice_path", "diameter_mm"])
    den = den[den.diameter_mm == dmm]
    den["preferred"] = den.riqe < den.base

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    order = ["gaussian", "wavelet", "tv", "nlm", "bilateral"]
    for i, name in enumerate(order):
        g = den[den.denoiser == name]
        if g.empty:
            continue
        a = g.groupby("target_residual_hu")["preferred"].mean() * 100
        b = g.groupby("target_residual_hu")["retention_matched"].median()
        st = fs.style_for(i)
        lab = NICE.get(name, name)
        ax1.plot(a.index, a.values, label=lab, **st)
        ax2.plot(b.index, b.values, label=lab, **st)
    ax1.set_ylabel("RIQE prefers the filtered\nimage over the unfiltered (%)")
    ax1.set_ylim(-4, 104)
    ax1.set_title("Edge-preserving denoisers are rewarded while the lesion fades",
                  fontsize=9.5)
    fs.despine(ax1)
    ax2.axhline(0.5, color=fs.INK_MUTED, lw=1.0, ls="--", zorder=1)
    ax2.annotate("half of the lesion signal lost", xy=(2, 0.5), xytext=(2, -11),
                 textcoords="offset points", fontsize=7.5, color=fs.INK_2)
    ax2.set_ylabel(f"signal retained, {dmm:.0f} mm +10 HU lesion\n(matched filter)")
    ax2.set_xlabel("filtering strength (residual standard deviation, HU)")
    ax2.set_ylim(0, 1.05)
    ax2.set_xscale("log")
    ax2.set_xticks(sorted(den.target_residual_hu.unique()))
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax2.legend(loc="lower left", fontsize=7.5, ncol=2)
    fs.despine(ax2)
    fig.tight_layout()
    save(fig, "fig_lesions")


def fig_tradeoff() -> None:
    """Search trade-off: real-dose ordering against overfiltering, for every
    (P, C, p) combination."""
    f = EXP / "hparam_search.csv"
    if not f.exists():
        print("  skipping trade-off: hparam_search.csv missing")
        return
    d = pd.read_csv(f).dropna(subset=["dose_correct_min", "frac_overfilter_fail"])
    ch = json.loads((EXP / "hparam_choice.json").read_text())
    o = ch["original_criterion_choice"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(100 * d.frac_overfilter_fail, 100 * d.dose_correct_min, s=16,
               color=fs.SERIES[0], alpha=0.55, edgecolors="none",
               label="one (P, C, p) configuration")
    for sel, lab, k in ((ch, "selected (revised criterion)", 1),
                        (o, "selected (original criterion)", 2)):
        r = d[(d.P == sel["P"]) & (np.isclose(d.C, sel["C"])) & (np.isclose(d.p, sel["p"]))]
        if len(r):
            ax.scatter(100 * r.frac_overfilter_fail, 100 * r.dose_correct_min, s=70,
                       color=fs.SERIES[k], marker=fs.MARKERS[k], edgecolors=fs.SURFACE,
                       linewidths=1.5, zorder=4, label=lab)
    ax.axhline(95, color=fs.INK_MUTED, lw=1.0, ls="--", zorder=1)
    ax.annotate("95% threshold", xy=(ax.get_xlim()[0], 95), xytext=(4, 3),
                textcoords="offset points", fontsize=7.5, color=fs.INK_2)
    ax.set_xlabel("filtered full-dose images scoring better than the original (%)")
    ax.set_ylabel("real reduced-dose images ranked worse,\nworst of chest/abdomen (%)")
    ax.set_title("No single-model configuration is good at both", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7.5)
    fs.despine(ax)
    fig.tight_layout()
    save(fig, "fig_tradeoff")


def fig_monotonicity() -> None:
    f = EXP / "exp1_monotonicity_detail.csv"
    if not f.exists():
        print("  skipping monotonicity: exp1_monotonicity_detail.csv missing")
        return
    d = pd.read_csv(f)
    kinds = [k for k in ("noise_white", "noise_fbp", "blur") if k in set(d["kind"])]
    titles = {"noise_white": "white noise", "noise_fbp": "FBP-like noise",
              "blur": "Gaussian blur"}
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
        frac = g["monotone"].mean()
        ax.set_title(f"{titles[k]}\n{100*frac:.0f}% monotone at every step", fontsize=9)
        ax.set_xlabel("degradation level")
        ax.set_xticks(x)
        fs.despine(ax)
    axes[0].set_ylabel("RIQE score")
    fig.tight_layout()
    save(fig, "fig_monotonicity")


def fig_noise_optimum() -> None:
    f = EXP / "exp2_form3_noise_optimum.csv"
    if not f.exists():
        print("  skipping noise optimum: exp2_form3_noise_optimum.csv missing")
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    frac = d["min_not_at_zero"].mean()
    vals = d.loc[d["min_not_at_zero"], "sigma_opt_hu"]
    if len(vals):
        bins = np.histogram_bin_edges(vals, bins="auto")
        ax.hist(vals, bins=bins, color=fs.SERIES[1], edgecolor=fs.SURFACE, linewidth=1.2)
        ax.set_xlabel("standard deviation of the noise that minimises the score (HU)")
        ax.set_ylabel("images")
        ax.set_title(f"The model has a preferred non-zero noise level\n"
                     f"{100*frac:.0f}% of images improve when noise is added", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no image improves when noise is added",
                ha="center", va="center", transform=ax.transAxes, color=fs.INK_2)
        ax.set_title("The score is minimal at zero noise, as it should be", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fs.despine(ax)
    fig.tight_layout()
    save(fig, "fig_noise_optimum")


STRAT_LABELS = [
    ("A:", "Abdomen 5 mm: GE STANDARD vs Siemens B30f"),
    ("B:", "Chest: GE STANDARD 1.25 mm vs Siemens B50f 1.5 mm"),
    ("C:", "GE: chest 1.25 mm vs abdomen 5 mm"),
    ("D:", "Siemens: chest B50f 1.5 mm vs abdomen B30f 5 mm"),
    ("G:", "All chest vs all abdomen (confounded)"),
    ("H:", "Vendor: all GE vs all Siemens"),
]


def _strat_label(c: str) -> str:
    for k, v in STRAT_LABELS:
        if c.startswith(k):
            return v
    if c.startswith("E:"):
        cell = c.split("—")[-1].strip().split("|")
        v = {"GE": "GE", "SIEMENS": "Siemens"}.get(cell[0], cell[0])
        return f"Pixel spacing tertiles: {v} {cell[1].lower()}"
    return c


def fig_stratification() -> None:
    f = EXP / "stratification.csv"
    if not f.exists():
        print("  skipping stratification: stratification.csv missing")
        return
    d = pd.read_csv(f)
    anchor = d[d["axis"] == "dose"].iloc[0]
    d = d[d["axis"] != "dose"].sort_values("D_obs")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.2, 0.40 * len(d) + 1.6))
    for i, r in enumerate(d.itertuples()):
        ax.plot([r.null_median, r.null_p95], [i, i], color=fs.GRID, lw=7,
                solid_capstyle="butt", zorder=1)
    ax.axvline(anchor.D_obs, color=fs.SERIES[1], lw=1.6, ls="--", zorder=2)
    ax.annotate("real dose reduction\n(same patients)", xy=(anchor.D_obs, len(d) - 0.6),
                xytext=(6, 0), textcoords="offset points", fontsize=7.5,
                color=fs.INK_2, va="top")
    ax.scatter(d["D_obs"], y, s=40, color=fs.SERIES[0], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([_strat_label(c) for c in d["contrast"]], fontsize=7.5)
    ax.set_xlabel("divergence between the two sub-models, D (score units)")
    ax.set_xlim(0, max(d["D_obs"].max(), anchor.D_obs) * 1.08)
    ax.set_title("Divergence between sub-models, against sampling noise (grey)\n"
                 "and a real dose reduction (dashed)", fontsize=9.5)
    fs.despine(ax)
    fig.tight_layout()
    save(fig, "fig_stratification")


def fig_photo_baseline() -> None:
    f = EXP / "exp5_scores_three_models.csv"
    if not f.exists():
        print("  skipping photographic baseline: exp5_scores_three_models.csv missing")
        return
    d = pd.read_csv(f).dropna(subset=["riqe", "photo"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.6))
    ax1.scatter(d["riqe"], d["photo"], s=9, color=fs.SERIES[0], alpha=0.35,
                edgecolors="none")
    ax1.set_xlabel("RIQE score (CT corpus)")
    ax1.set_ylabel("score under the photographic model")
    try:
        rho = json.loads((EXP / "exp5_summary.json").read_text())["spearman_riqe_vs_photo"]
        ax1.set_title(f"Same images, two models\nSpearman {rho:+.3f}", fontsize=9)
    except Exception:  # noqa: BLE001
        ax1.set_title("Same images, two models", fontsize=9)
    fs.despine(ax1)

    v = pd.read_csv(EXP / "exp5_verdicts.csv") if (EXP / "exp5_verdicts.csv").exists() else None
    if v is not None:
        metrics = [c for c in v.columns if c != "model"]
        x = np.arange(len(metrics))
        w = 0.26
        for i, r in enumerate(v.itertuples()):
            vals = [getattr(r, m) for m in metrics]
            ax2.bar(x + (i - 1) * w, vals, width=w - 0.02,
                    color=fs.SERIES[i], edgecolor=fs.SURFACE, linewidth=1.2,
                    label=r.model)
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace("monotone_", "mon. ").replace(
            "overfilter_failure_fraction", "overfilter.\nfailures")
            for m in metrics], fontsize=7.5)
        ax2.set_ylabel("fraction")
        ax2.set_title("Do the verdicts change between the three models?", fontsize=9)
        ax2.legend(fontsize=7)
        fs.despine(ax2)
    fig.tight_layout()
    save(fig, "fig_photo_baseline")


def fig_stability() -> None:
    f = EXP / "exp4_learning_curve.csv"
    if not f.exists():
        print("  skipping stability: exp4_learning_curve.csv missing")
        return
    d = pd.read_csv(f)
    d = d[d["n_patients"] < d["n_patients"].max()]  # the last point is 0 by construction
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(d["n_patients"], d["D_median"], **fs.style_for(0))
    ax.fill_between(d["n_patients"], d["D_median"], d["D_p95"],
                    color=fs.SERIES[0], alpha=0.14, linewidth=0)
    try:
        st = pd.read_csv(EXP / "stratification.csv")
        a = float(st[st["axis"] == "dose"]["D_obs"].iloc[0])
        ax.axhline(a, color=fs.SERIES[1], lw=1.4, ls="--")
        ax.annotate("real dose reduction", xy=(d["n_patients"].min(), a), xytext=(2, 4),
                    textcoords="offset points", fontsize=7.5, color=fs.INK_2)
    except Exception:  # noqa: BLE001
        pass
    ax.set_xlabel("patients in the fitting corpus")
    ax.set_ylabel("D from the full model")
    ax.set_title("How many patients are enough", fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xticks(d["n_patients"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0, None)
    fs.despine(ax)
    fig.tight_layout()
    save(fig, "fig_stability")


def main() -> int:
    print("figures:")
    for fn in (fig_tradeoff, fig_main_overfiltering, fig_monotonicity, fig_noise_optimum,
               fig_stratification, fig_photo_baseline, fig_stability):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
