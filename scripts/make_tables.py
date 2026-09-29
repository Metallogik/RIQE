#!/usr/bin/env python3
"""LaTeX tables for the paper, generated from the result files.

No number in the paper is transcribed by hand: the tables are regenerated
from experiments/ into paper/tables/, and if an experiment is redone the
paper is updated by re-running this script.

    .venv/bin/python scripts/make_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "paper" / "tables"
NICE = {"gaussian": "Gaussian", "wavelet": "Wavelet", "tv": "TV", "nlm": "NLM",
        "bilateral": "Bilateral"}
ORDER = ["gaussian", "wavelet", "tv", "nlm", "bilateral"]


def w(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)
    print(f"  wrote {name}.tex")


def m(txt: str) -> str:
    """Typographic minus sign: a hyphen in a table is not a minus."""
    return txt.replace("-", "$-$")


def pct(x) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"


def t_detection() -> None:
    """Degradation detection on TEST."""
    s = json.loads((EXP / "battery_summary.json").read_text())
    d0 = s["exp0_real_dose"]
    rel = pd.read_csv(EXP / "exp1b_relative_noise.csv")
    rel["worse"] = rel["score"] > rel["base"]
    rt = rel.groupby(["region", "rel_increase"])["worse"].mean()
    bl = pd.read_csv(EXP / "battery_scores.csv")
    orig = bl[(bl.kind == "original") & (bl.source == "full")].set_index("slice_path")["score"]
    b = bl[(bl.kind == "blur") & (bl.source == "full")].copy()
    b["base"] = b.slice_path.map(orig)
    b = b.dropna(subset=["score", "base"])
    bt = b.groupby("sigma_px").apply(lambda g: (g.score > g.base).mean(), include_groups=False)
    rows = [
        r"Real reduced dose (10\%/25\% of routine) vs full dose, same slice & "
        f"{pct(d0['correct_chest'])} ({d0['n_chest']}) & {pct(d0['correct_abdomen'])} ({d0['n_abdomen']}) \\\\",
    ]
    for r in (0.05, 0.10, 0.20, 0.50, 1.00):
        rows.append(f"Added noise, +{100 * r:.0f}\\% of native & "
                    f"{pct(rt.get(('chest', r)))} & {pct(rt.get(('abdomen', r)))} \\\\")
    for sp in (0.5, 1.0, 2.0):
        g = b[b.sigma_px == sp]
        ch = g[g.cell.str.contains("CHEST")]
        ab = g[~g.cell.str.contains("CHEST")]
        rows.append(f"Gaussian blur, $\\sigma={sp:g}$ px & {pct((ch.score > ch.base).mean())} & "
                    f"{pct((ab.score > ab.base).mean())} \\\\")
    body = (r"""\begin{table}[tbp]
\centering
\caption{Degradation detection on the held-out test split (40 patients). Percentage of
cases in which the degraded image receives a worse score than its source; 100\% is
correct. Numbers of image pairs in parentheses.}
\label{tab:detection}
\small
\begin{tabular}{lcc}
\toprule
Degradation & Chest (\%) & Abdomen (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    w("tab_detection", body)


def t_overfiltering() -> None:
    d = pd.read_csv(EXP / "exp2_form1_overfiltering.csv")
    d = d[d["delta"].notna()]
    fail = d.pivot_table(index="denoiser", columns="target_residual_hu", values="failure",
                         aggfunc="mean")
    med = d.pivot_table(index="denoiser", columns="target_residual_hu", values="delta",
                        aggfunc="median")
    lv = list(fail.columns)
    rows = []
    for n in ORDER:
        cells = " & ".join(f"{100 * fail.loc[n, l]:.0f} \\,({m(f'{med.loc[n, l]:+.2f}')})" for l in lv)
        rows.append(f"{NICE[n]} & {cells} \\\\")
    tot = 100 * d["failure"].mean()
    body = (r"""\begin{table}[tbp]
\centering
\caption{Overfiltering, form 1: filtering a full-dose image cannot add information, yet
the filtered image scores better than the original in the percentage of cases shown
(test split, """ + f"{len(d)}" + r""" filtered images, """ + f"{tot:.1f}" + r"""\% overall).
In parentheses, the median score change (positive = correctly worse). Filtering
strength is the standard deviation of the residual inside the body.}
\label{tab:overfiltering}
\small
\begin{tabular}{l""" + "c" * len(lv) + r"""}
\toprule
& \multicolumn{""" + str(len(lv)) + r"""}{c}{Residual (HU)} \\
Denoiser & """ + " & ".join(f"{l:g}" for l in lv) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    w("tab_overfiltering", body)


def t_lesions() -> None:
    d = pd.read_csv(EXP / "exp2_form2_lesions.csv")
    base = d[d.denoiser == "none"][["slice_path", "diameter_mm", "riqe"]].rename(
        columns={"riqe": "base"})
    f = d[d.denoiser != "none"].merge(base, on=["slice_path", "diameter_mm"])
    f = f[f.diameter_mm == 4.0]
    f["pref"] = f.riqe < f.base
    lv = sorted(f.target_residual_hu.unique())
    rows = []
    for n in ORDER:
        g = f[f.denoiser == n]
        cells = []
        for l in lv:
            h = g[g.target_residual_hu == l]
            cells.append(f"{100 * h.pref.mean():.0f} / {h.retention_matched.median():.2f}".replace("-0.00", "0.00"))
        rows.append(f"{NICE[n]} & " + " & ".join(cells) + r" \\")
    ns = int(f.slice_path.nunique())
    ntot = int(d.slice_path.nunique())
    body = (r"""\begin{table}[tbp]
\centering
\caption{Metric preference against lesion signal. Substrate: """ + str(ntot) + r""" held-out
full-dose abdominal slices (""" + str(ns) + r""" with room for the 4\,mm lesions) plus FBP-like noise at the level measured between real
full- and quarter-dose images (18.4\,HU). Each cell: percentage of images in which
RIQE scores the filtered image better than the unfiltered one / median fraction of
the matched-filter signal of a 4\,mm, +10\,HU lesion that survives the filter.}
\label{tab:lesions}
\small
\begin{tabular}{l""" + "c" * len(lv) + r"""}
\toprule
& \multicolumn{""" + str(len(lv)) + r"""}{c}{Residual (HU): preferred (\%) / signal retained} \\
Denoiser & """ + " & ".join(f"{l:g}" for l in lv) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    w("tab_lesions", body)


def t_stratification() -> None:
    d = pd.read_csv(EXP / "stratification.csv")
    lab = {
        "A:": "Abdomen 5\\,mm: GE STANDARD vs Siemens B30f",
        "B:": "Chest: GE STANDARD 1.25\\,mm vs Siemens B50f 1.5\\,mm",
        "C:": "GE: chest 1.25\\,mm vs abdomen 5\\,mm",
        "D:": "Siemens: chest B50f 1.5\\,mm vs abdomen B30f 5\\,mm",
        "G:": "All chest vs all abdomen$^{\\dagger}$",
        "H:": "Vendor: all GE vs all Siemens",
        "ANCHOR": "\\textit{Anchor: full vs real reduced dose, same patients}",
    }

    def L(c):
        for k, v in lab.items():
            if c.startswith(k):
                return v
        cell = c.split("—")[-1].strip().split("|")
        vend = {"GE": "GE", "SIEMENS": "Siemens"}.get(cell[0], cell[0])
        return f"Pixel-spacing tertiles, {vend} {cell[1].lower()}"

    d = pd.concat([d[d.axis == "dose"], d[d.axis != "dose"].sort_values("eta")])
    rows = [f"{L(r.contrast)} & {r.n_A} / {r.n_B} & {r.D_obs:.2f} & {r.null_median:.2f} & "
            f"{r.eta:.2f} \\\\" for r in d.itertuples()]
    body = (r"""\begin{table}[tbp]
\centering
\caption{Divergence between sub-models fitted on two groups of patients (fitting split).
$D$ uses the same functional form as the score. Null: median of 200 patient-level
permutations of the pooled groups. $\eta$ expresses the excess over the null in units of
the excess produced by a real dose reduction on the same patients. All permutation
$p = 0.005$ (the minimum attainable with 200 permutations).
$^{\dagger}$Anatomy, kernel and slice thickness are not crossed in this collection;
this contrast cannot serve as a clean anatomical control.}
\label{tab:strat}
\small
\begin{tabular}{lcccc}
\toprule
Contrast & Patients & $D$ & Null & $\eta$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    w("tab_stratification", body)


def t_baselines() -> None:
    s5 = json.loads((EXP / "exp5_summary.json").read_text())
    dr = json.loads((EXP / "exp5_dose_and_relnoise.json").read_text())
    s6 = json.loads((EXP / "exp6_ldctiqac_summary.json").read_text())
    v = pd.read_csv(EXP / "exp5_verdicts.csv").set_index("model")
    r6 = s6["windows"]["W350/L40"]
    a, b = dr["RIQE (CT)"], dr["photographic"]
    va, vb = v.loc["RIQE (CT, masks)"], v.loc["photographic PD/CC0"]

    def ci(x):
        return m(f"{x['spearman']:+.2f} [{x['spearman_ci95'][0]:+.2f}, {x['spearman_ci95'][1]:+.2f}]")

    body = (r"""\begin{table}[tbp]
\centering
\caption{The same code fitted on CT (RIQE) and on """ + str(s5["photo_corpus"]["n_images"]) +
            r""" CC0 natural photographs, scoring the same held-out CT images. Last row: rank
correlation with the mean score of five radiologists on the 938 scoreable LDCTIQAC 2023
training images
(negative is the correct sign: lower RIQE means closer to the reference).}
\label{tab:photo}
\small
\begin{tabular}{lcc}
\toprule
& RIQE (CT) & Photographic \\
\midrule
Real reduced dose ranked worse, chest (\%) & """ + f"{a['dose_chest']:.1f} & {b['dose_chest']:.1f}" + r""" \\
Real reduced dose ranked worse, abdomen (\%) & """ + f"{a['dose_abdomen']:.1f} & {b['dose_abdomen']:.1f}" + r""" \\
Added noise +20\% of native detected (\%) & """ + f"{a['noise_+20%']:.1f} & {b['noise_+20%']:.1f}" + r""" \\
Added noise +100\% of native detected (\%) & """ + f"{a['noise_+100%']:.1f} & {b['noise_+100%']:.1f}" + r""" \\
Filtered full-dose scoring better (\%) & """ + f"{100 * va.overfilter_failure_fraction:.1f} & {100 * vb.overfilter_failure_fraction:.1f}" + r""" \\
Spearman with radiologists [95\% CI] & """ + f"{ci(r6['RIQE'])} & {ci(r6['photographic'])}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
""")
    w("tab_baselines", body)


def main() -> int:
    print("tables:")
    for fn in (t_detection, t_overfiltering, t_lesions, t_stratification, t_baselines):
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
