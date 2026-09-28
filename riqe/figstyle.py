"""Figure style, with the rules it follows stated explicitly.

The palette is a validated reference instance: slots are assigned **in fixed
order, never cycled**. Figures are for print, so light mode only.

Rules followed:

  * never two y axes in one figure. When two quantities of different scale
    share an x axis -- the main figure, score against signal retention --
    **two stacked panels sharing x** are used, not a secondary axis.
  * series identity is never carried by colour alone: every series also has
    its own dash pattern and marker, plus a direct label when series are few.
    This serves readers with colour-vision deficiency and black-and-white
    printing.
  * recessive grid and axes, thin markers, no number on every point.
  * sequential = one hue light->dark; diverging = two hues with a neutral grey
    midpoint. Never a rainbow.

The categorical palette passes colour-vision-deficiency thresholds on the
list of **adjacent** pairs up to five slots; beyond three slots it is not
valid for all-pairs forms (scatter, bubbles), which fold into "other" or
facet instead.
"""

from __future__ import annotations

#: categorical palette, fixed order
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948")
#: secondary encoding, paired with the slots
DASHES = ((), (5, 2), (1, 1.6), (7, 2, 1.5, 2), (3, 1.5, 1, 1.5),
          (6, 3), (2, 2, 6, 2), (9, 2))
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

#: one-hue sequential ramp (blue), light -> dark
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
#: diverging, warm/cool poles with a neutral grey midpoint
DIVERGING = ("#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f19b9a", "#e34948", "#8f2322")

SURFACE = "#ffffff"  # white: figures are for print
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e3e2de"


def apply(mpl) -> None:
    """Apply the style to matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def style_for(i: int) -> dict:
    """Colour, dash pattern and marker of slot i, in fixed order."""
    j = i % len(SERIES)
    d = DASHES[j]
    out = {"color": SERIES[j], "marker": MARKERS[j]}
    if d:
        out["dashes"] = d
    return out


def despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
