#!/usr/bin/env python3
"""Figures for the paper, from the experiment results.

Written to paper/figures/ as PDF and PNG. The style is that of the QICOM
manuscript (riqe/figstyle.py): monochrome, Latin Modern, "A  Title" panel
labels. Figure 4 (example images) re-reads one held-out DICOM slice and needs
the downloaded corpus; the others read experiments/ only.

    .venv/bin/python scripts/make_figures.py [--only fig_examples ...]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe import figstyle as fs  # noqa: E402

FAMILY = fs.apply(matplotlib)
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
FIG = ROOT / "paper" / "figures"

NICE = {"gaussian": "Gaussian", "wavelet": "Wavelet", "tv": "TV", "nlm": "NLM",
        "bilateral": "Bilateral"}
ORDER = ["gaussian", "wavelet", "tv", "nlm", "bilateral"]
REGIONS = (("chest", "Chest"), ("abdomen", "Abdomen"))
FULL_WIDTH = 7.2  # inches; included at \linewidth


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        kw = {"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {}
        fig.savefig(FIG / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"  wrote {name}")


def need(*paths: Path) -> bool:
    missing = [p for p in paths if not p.exists()]
    for p in missing:
        print(f"  skipping: {p.relative_to(ROOT)} missing")
    return not missing


def region_of(cell: pd.Series) -> np.ndarray:
    return np.where(cell.str.contains("CHEST"), "chest", "abdomen")


def log2_axis(ax, ticks) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


# ---------------------------------------------------------------------------
# Figure 1: study design
# ---------------------------------------------------------------------------

def fig_design() -> None:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.5))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, w, h, text, color=fs.LIGHT, dashed=False, size=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.009", linewidth=0.75,
                                    facecolor=color, edgecolor=fs.GRAY,
                                    linestyle="--" if dashed else "-"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

    def arrow(x, y1, y2):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=10,
                                     color=fs.GRAY, linewidth=0.75))

    box(0.015, 0.894, 0.97, 0.077,
        "Public data only  |  TCIA LDCT-and-Projection-data v7, CC BY 4.0  |  "
        "fixed HU mapping and masks", size=8.4)
    columns = [
        ("Fitting",
         "158 patients, full dose\n3,792 slices after S1–S6",
         "36 NSS features per patch\nsharpness selection, Gaussian fit",
         "Model: mean, covariance\nand every slice with its SHA-256",
         "MAIN FINDING\nThe refit from public data\nis bit-identical"),
        ("Hyperparameter selection",
         "Inner split of the fitting patients\n118 fit / 40 validation",
         "168 settings of P, C and p\nfour ranked criteria",
         "P = 24, C = 0.01, p = 0.5\ncriterion revised once, declared",
         "MAIN FINDING\nNo setting both ranks real dose\nand resists overfiltering"),
        ("Held-out evaluation",
         "40 test patients\nLDCTIQAC 2023, CC0 photographs",
         "Real dose, noise, blur, denoisers\ninserted lesions, sub-models",
         "Detection, overfiltering,\nlesion signal, stratification",
         "MAIN FINDING\nDetects degradation, but rewards\nedge-preserving filtering"),
    ]
    for k, (title, inputs, process, outputs, finding) in enumerate(columns):
        x, w = 0.015 + k * 0.332, 0.306
        c = x + w / 2
        ax.text(c, 0.839, title, ha="center", fontsize=9.5)
        box(x, 0.716, w, 0.083, inputs, color="white", size=7.6)
        arrow(c, 0.703, 0.657)
        box(x, 0.535, w, 0.115, process, size=7.9)
        arrow(c, 0.523, 0.477)
        box(x, 0.356, w, 0.115, outputs, color="white", size=7.9)
        box(x, 0.154, w, 0.151, finding, color="white", dashed=True, size=7.7)
    box(0.015, 0.026, 0.97, 0.079,
        "Shared traceability: slice UIDs and SHA-256, frozen patient split, code hash, "
        "verification script", size=8.4)
    save(fig, "fig_design")


# ---------------------------------------------------------------------------
# Figure 2: hyperparameter trade-off
# ---------------------------------------------------------------------------

def fig_tradeoff() -> None:
    f, c = EXP / "hparam_search.csv", EXP / "hparam_choice.json"
    if not need(f, c):
        return
    d = pd.read_csv(f).dropna(subset=["dose_correct_min", "frac_overfilter_fail"])
    ch = json.loads(c.read_text())
    o = ch["original_criterion_choice"]
    fig, ax = plt.subplots(figsize=(4.9, 3.3), layout="constrained")
    ax.scatter(100 * d.frac_overfilter_fail, 100 * d.dose_correct_min, s=14,
               facecolors="none", edgecolors=fs.MUTED, linewidths=0.7,
               label="One setting of P, C and p")
    for sel, lab, kw in (
        (ch, "Selected (revised criterion)", {"marker": "o", "facecolors": fs.INK}),
        (o, "Selected (original criterion)", {"marker": "s", "facecolors": "white"}),
    ):
        r = d[(d.P == sel["P"]) & np.isclose(d.C, sel["C"]) & np.isclose(d.p, sel["p"])]
        ax.scatter(100 * r.frac_overfilter_fail, 100 * r.dose_correct_min, s=42,
                   edgecolors=fs.INK, linewidths=1.0, zorder=4, label=lab, **kw)
    ax.axhline(95, color=fs.MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(ax.get_xlim()[0], 95.8, " 95% threshold", ha="left", va="bottom", fontsize=8)
    ax.set_xlabel("Filtered full-dose images scoring better (%)")
    ax.set_ylabel("Real reduced dose ranked worse,\nworse of chest and abdomen (%)")
    ax.set_ylim(-3, 103)
    ax.legend(loc="lower right")
    fs.clean(ax, "both")
    save(fig, "fig_tradeoff")


# ---------------------------------------------------------------------------
# Figure 3: score response to controlled degradation
# ---------------------------------------------------------------------------

def fig_response() -> None:
    f = EXP / "battery_scores.csv"
    if not need(f):
        return
    d = pd.read_csv(f)
    d = d[d.source == "full"].copy()
    orig = d[d.kind == "original"].set_index("slice_path")["score"]
    d["delta"] = d["score"] - d["slice_path"].map(orig)
    d["region"] = region_of(d["cell"])

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.55), layout="constrained")

    # A: noise relative to the native noise of each image
    ax = axes[0]
    g = d[d.kind == "noise_rel"]
    for i, (reg, lab) in enumerate(REGIONS):
        m = g[g.region == reg].groupby("rel_increase")["delta"].median()
        ax.plot(np.r_[0, 100 * m.index], np.r_[0, m.values], label=lab, **fs.style_for(i))
    fs.panel(ax, "A", "Added noise")
    ax.set_xlabel("Increase over native noise (%)")
    ax.set_ylabel("Change in RIQE score")
    ax.legend(loc="upper left")

    # B: Gaussian blur
    ax = axes[1]
    g = d[d.kind == "blur"]
    for i, (reg, lab) in enumerate(REGIONS):
        m = g[g.region == reg].groupby("sigma_px")["delta"].median()
        ax.plot(np.r_[0, m.index], np.r_[0, m.values], label=lab, **fs.style_for(i))
    fs.panel(ax, "B", "Gaussian blur")
    ax.set_xlabel(r"Kernel $\sigma$ (pixels)")

    # C: denoising of full-dose images
    ax = axes[2]
    g = d[d.kind == "denoise"]
    for i, name in enumerate(ORDER):
        m = g[g.denoiser == name].groupby("target_residual_hu")["delta"].median()
        ax.plot(m.index, m.values, label=NICE[name], **fs.style_for(i))
    fs.panel(ax, "C", "Denoising a full-dose image")
    ax.set_xlabel("Residual standard deviation (HU)")
    log2_axis(ax, [2, 4, 8, 16, 32, 64])
    ax.legend(loc="upper left", ncol=1, handlelength=2.6)

    for ax in axes:
        ax.axhline(0, color=fs.MUTED, lw=0.8, ls="--", zorder=1)
        fs.clean(ax)
    save(fig, "fig_response")


# ---------------------------------------------------------------------------
# Figure 4: example images
# ---------------------------------------------------------------------------

#: lesions shown in the example: (diameter mm, contrast HU), as in the experiment
EXAMPLE_LESIONS = ((10.0, 25.0), (6.0, 15.0), (4.0, 10.0))
#: a held-out Siemens abdominal slice of the lesion experiment, chosen for
#: legible anatomy; its RIQE preferences are checked against the majority
EXAMPLE_SLICE = "data/dicom/L064/full/00000133.dcm"
#: narrow soft-tissue window, so that the larger lesions are visible in print
WINDOW = (-65.0, 185.0)
#: the four conditions shown, all at the same residual
EXAMPLE_CONDITIONS = (("Noisy input", None), ("Gaussian", "gaussian"), ("NLM", "nlm"),
                      ("Bilateral", "bilateral"))
EXAMPLE_RESIDUAL = 16.0


def _example_layout(hu, body, ps, rng):
    """Three homogeneous sites close enough to share one crop, found with the
    same rule as the experiment (local mean 40-80 HU, low local variance)."""
    from riqe import degrade as dg

    r10 = EXAMPLE_LESIONS[0][0] / 2 / ps
    cand = np.asarray(dg.find_homogeneous_sites(hu, body, r10, 400, rng, min_sep_px=3.5 * r10),
                      dtype=float)
    best = None
    for i, c in enumerate(cand):
        dist = np.hypot(*(cand - c).T)
        near = [k for k in np.argsort(dist)[1:] if dist[k] < 64][:2]
        if len(near) == 2 and np.hypot(*(cand[near[0]] - cand[near[1]])) >= 3.5 * r10:
            spread = dist[near].max()
            if best is None or spread < best[0]:
                best = (spread, [i, *near])
    if best is None:
        return None
    sites = [tuple(int(v) for v in cand[j]) for j in best[1]]
    half, margin = 72, int(r10) + 12
    ys, xs = np.array(sites).T
    # among crops that keep every lesion inside, the one with the least air
    best_crop = None
    for cy in range(ys.max() + margin - half, ys.min() - margin + half + 1, 2):
        for cx in range(xs.max() + margin - half, xs.min() - margin + half + 1, 4):
            if not (half <= cy <= hu.shape[0] - half and half <= cx <= hu.shape[1] - half):
                continue
            air = float((hu[cy - half:cy + half, cx - half:cx + half] < -500).mean())
            if best_crop is None or air < best_crop[0]:
                best_crop = (air, cy, cx)
    _, cy, cx = best_crop
    return sites, (cy - half, cy + half, cx - half, cx + half)


def fig_examples() -> None:
    les = EXP / "exp2_form2_lesions.csv"
    if not need(les, EXP / "hparam_choice.json", ROOT / EXAMPLE_SLICE):
        return
    from riqe import degrade as dg
    from riqe.cache import FeatureCache
    from riqe.dicomio import read_hu
    from riqe.extract import MIN_PATCHES_FOR_SCORE, Spec, features_from_hu, masks_for
    from riqe.model import RCOND, mahalanobis_mixed

    # the example must behave like the majority of slices in the experiment
    d = pd.read_csv(les)
    x = d[(d.diameter_mm == 10.0) & (d.target_residual_hu.isna() | (d.target_residual_hu == EXAMPLE_RESIDUAL))]
    base = x[x.denoiser == "none"].set_index("slice_path")["riqe"]
    x = x[x.denoiser != "none"].assign(pref=lambda t: t.riqe < t.slice_path.map(base))
    shown = [n for _, n in EXAMPLE_CONDITIONS if n]
    majority = x[x.denoiser.isin(shown)].groupby("denoiser")["pref"].mean().round().astype(bool)
    mine = x[x.slice_path == EXAMPLE_SLICE].set_index("denoiser")["pref"]
    if not (mine.reindex(majority.index) == majority).all():
        print(f"  WARNING: {EXAMPLE_SLICE} does not follow the majority preference")
    noise = float(d[d.slice_path == EXAMPLE_SLICE]["noise_sigma_hu"].iloc[0])

    ch = json.loads((EXP / "hparam_choice.json").read_text())
    P, C, p = ch["P"], ch["C"], ch["p"]
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    fit = corpus[(corpus["kind"] == "full") & corpus["keep"] & corpus.patient_id.isin(set(split["fit"]))]
    model = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}").fit(
        list(fit["path"]), p, n_patients=fit.patient_id.nunique())
    spec = Spec(P=P, C=C, p=None)

    hu, pad, meta = read_hu(str(ROOT / EXAMPLE_SLICE))
    fov, body = masks_for(hu, pad)
    layout = _example_layout(hu, body, meta["pixel_spacing"], np.random.default_rng(20260917))
    if layout is None:
        print("  skipping fig_examples: no compact group of lesion sites")
        return
    sites, (y0, y1, x0, x1) = layout
    field = np.zeros_like(hu, dtype=np.float32)
    radii = []
    for (dmm, contrast), (cy, cx) in zip(EXAMPLE_LESIONS, sites):
        r = dmm / 2 / meta["pixel_spacing"]
        radii.append(r)
        field += (contrast * dg._disc(hu.shape, cy, cx, r)).astype(np.float32)
    noisy = dg.add_fbp_noise(hu, noise, np.random.default_rng(20260918))

    imgs = []
    for title, name in EXAMPLE_CONDITIONS:
        if name is None:
            img, label = noisy + field, title
        else:
            # strength calibrated without lesions, as in the experiment
            s, _, _ = dg.calibrate_strength(noisy, name, EXAMPLE_RESIDUAL, body)
            img, label = dg.apply_denoiser(noisy + field, name, s), f"{title}, {EXAMPLE_RESIDUAL:g} HU residual"
        pf = features_from_hu(img, pad, spec, fitting=False, masks=(fov, body))
        f = pf.feat[pf.valid]
        sc = (mahalanobis_mixed(model.nu, model.sigma, f.mean(axis=0), np.cov(f, rowvar=False), RCOND)
              if f.shape[0] >= MIN_PATCHES_FOR_SCORE else np.nan)
        imgs.append((label, img[y0:y1, x0:x1], sc))

    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 2.25), layout="constrained")
    for ax, (label, crop, sc) in zip(axes, imgs):
        ax.imshow(crop, cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1], interpolation="nearest")
        for (cy, cx), r in zip(sites, radii):
            ax.add_patch(Circle((cx - x0, cy - y0), r + 4, fill=False, color="white", lw=0.5))
        ax.set_title(label, fontsize=8.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"RIQE {sc:.2f}", fontsize=8.2)
        for sp in ax.spines.values():
            sp.set_visible(False)
    save(fig, "fig_examples")
    info = {"slice_path": EXAMPLE_SLICE, "noise_sigma_hu": noise, "window_hu": WINDOW,
            "sites": sites, "crop": [int(y0), int(y1), int(x0), int(x1)],
            "tissue_hu_at_sites": [float(np.median(hu[cy - 3:cy + 4, cx - 3:cx + 4])) for cy, cx in sites],
            "scores": {lab: float(sc) for lab, _, sc in imgs}}
    (FIG / "fig_examples.json").write_text(json.dumps(info, indent=1))
    print("    " + ", ".join(f"{k}: {v:.3f}" for k, v in info["scores"].items()))
    print(f"    tissue HU at sites: {info['tissue_hu_at_sites']}")


# ---------------------------------------------------------------------------
# Figure 5: preference against lesion signal
# ---------------------------------------------------------------------------

def fig_lesions() -> None:
    f = EXP / "exp2_form2_lesions.csv"
    if not need(f):
        return
    d = pd.read_csv(f)
    dmm = 4.0
    base = d[d.denoiser == "none"][["slice_path", "diameter_mm", "riqe"]].rename(columns={"riqe": "base"})
    den = d[d.denoiser != "none"].merge(base, on=["slice_path", "diameter_mm"])
    den = den[den.diameter_mm == dmm]
    den["preferred"] = den.riqe < den.base

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8), layout="constrained")
    for i, name in enumerate(ORDER):
        g = den[den.denoiser == name]
        a = g.groupby("target_residual_hu")["preferred"].mean() * 100
        b = g.groupby("target_residual_hu")["retention_matched"].median()
        ax1.plot(a.index, a.values, label=NICE[name], **fs.style_for(i))
        ax2.plot(b.index, b.values, label=NICE[name], **fs.style_for(i))
    fs.panel(ax1, "A", "Filtered image preferred by RIQE")
    ax1.set_ylabel("Images (%)")
    ax1.set_ylim(-4, 104)
    fs.panel(ax2, "B", f"Signal of a {dmm:.0f} mm, +10 HU lesion")
    ax2.set_ylabel("Fraction retained (matched filter)")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.5, color=fs.MUTED, lw=0.8, ls="--", zorder=1)
    ax2.legend(loc="lower left", handlelength=2.6)
    for ax in (ax1, ax2):
        ax.set_xlabel("Residual standard deviation (HU)")
        log2_axis(ax, [2, 4, 8, 16, 32, 64])
        fs.clean(ax)
    save(fig, "fig_lesions")


# ---------------------------------------------------------------------------
# Figure 6: agreement with radiologists (LDCTIQAC 2023)
# ---------------------------------------------------------------------------

def fig_ldctiqac() -> None:
    f, s = EXP / "exp6_ldctiqac_scores.csv", EXP / "exp6_ldctiqac_summary.json"
    if not need(f, s):
        return
    d = pd.read_csv(f)
    summ = json.loads(s.read_text())
    window = "W350/L40"
    d = d[(d.window == window)].dropna(subset=["score"])
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.7), layout="constrained")
    rng = np.random.default_rng(0)
    for ax, (letter, model, title) in zip(axes, (("A", "RIQE", "RIQE, fitted on CT"),
                                                 ("B", "photographic", "Same code fitted on photographs"))):
        g = d[d.model == model]
        x = g["radiologist"].to_numpy(float)
        ax.scatter(x + rng.uniform(-0.05, 0.05, len(x)), g["score"], s=5, color=fs.MUTED,
                   alpha=0.35, linewidths=0)
        med = g.groupby(g["radiologist"].round(1))["score"].median()
        ax.plot(med.index, med.values, "o-", color=fs.INK, markersize=3.2, lw=1.2)
        rho = summ["windows"][window][model]["spearman"]
        fs.panel(ax, letter, f"{title}  ($\\rho = {rho:+.2f}$)")
        ax.set_xlabel("Mean radiologist score (0 worst, 4 best)")
        fs.clean(ax)
    axes[0].set_ylabel("Score")
    save(fig, "fig_ldctiqac")


FIGURES = {f.__name__: f for f in (fig_design, fig_tradeoff, fig_response, fig_examples,
                                    fig_lesions, fig_ldctiqac)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES))
    args = ap.parse_args()
    print(f"figures (font: {FAMILY}):")
    for name, fn in FIGURES.items():
        if args.only and name not in args.only:
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR in {name}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
