"""Figure style: monochrome, Latin Modern, the same as the QICOM manuscript.

Rules followed:

  * black, neutral greys and white only. Series identity is carried by grey
    level **and** marker **and** dash pattern, so figures survive
    black-and-white printing and colour-vision deficiency.
  * never two y axes in one figure: quantities of different scale go in
    separate panels sharing an axis.
  * panels are labelled "A  Title", left-aligned, in the body font.
  * light horizontal grid, left and bottom spines only, ticks outward.
  * text in figures uses Latin Modern Roman, the font of the manuscript
    body, with Computer Modern for mathematics. The font file lives with the
    paper sources (paper/assets/fonts, GUST Font License); without it the
    figures fall back to DejaVu Serif.
"""

from __future__ import annotations

from pathlib import Path

INK = "#000000"
#: near-black to light grey, in fixed order
SERIES = ("#202020", "#707070", "#454545", "#9a9a9a", "#000000")
MARKERS = ("o", "s", "^", "D", "v")
DASHES = ((), (4, 1.6), (1.2, 1.4), (6, 1.8, 1.2, 1.8), (2.5, 1.2))
GRAY = "#606060"
MUTED = "#808080"
LIGHT = "#f2f2f2"
SURFACE = "#ffffff"

FONT = Path(__file__).resolve().parents[1] / "paper" / "assets" / "fonts" / "lmroman10-regular.ttf"


def apply(mpl) -> str:
    """Apply the style to matplotlib; returns the font family used."""
    import logging

    from matplotlib import font_manager

    # fontTools warns about the 1970 timestamps in the Latin Modern tables
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    family = "DejaVu Serif"
    if FONT.exists():
        font_manager.fontManager.addfont(str(FONT))
        family = font_manager.FontProperties(fname=str(FONT)).get_name()
    mpl.rcParams.update({
        "font.family": family,
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRAY,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.axisbelow": True,
        "grid.alpha": 0.22,
        "grid.color": GRAY,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.8,
        # TrueType outlines keep the Latin Modern letterforms and embed cleanly
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })
    return family


def style_for(i: int) -> dict:
    """Grey level, marker and dash pattern of series i, in fixed order."""
    j = i % len(SERIES)
    out = {"color": SERIES[j], "marker": MARKERS[j]}
    if DASHES[j]:
        out["dashes"] = DASHES[j]
    return out


def panel(ax, letter: str, title: str) -> None:
    """Panel label in the QICOM form: "A  Title", left-aligned."""
    ax.set_title(f"{letter}  {title}", loc="left")


def clean(ax, axis: str = "y") -> None:
    """Light grid on one axis, short outward ticks."""
    ax.grid(axis=axis)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)
