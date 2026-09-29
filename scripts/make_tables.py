#!/usr/bin/env python3
"""LaTeX tables for the paper, generated from the result files.

No number in the paper is transcribed by hand: the tables are regenerated
from experiments/ into paper/tables/, and if an experiment is redone the
paper is updated by re-running this script. Confidence intervals come from
experiments/uncertainty.json (patient-level bootstrap) and, for LDCTIQAC,
from exp6_ldctiqac_summary.json (bootstrap over inferred source slices).

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


def ci(e: dict | None, scale: float = 100.0, fmt: str = ".1f") -> str:
    """Estimate [low, high] from an uncertainty entry."""
    if not e:
        return "--"
    return (f"{scale * e['estimate']:{fmt}} {{\\scriptsize [{scale * e['ci_low']:{fmt}}, "
            f"{scale * e['ci_high']:{fmt}}]}}")


def U() -> dict:
    return json.loads((EXP / "uncertainty.json").read_text())


def table(name, caption, label, colspec, header, rows, sep: str | None = None) -> None:
    w(name, "\\begin{table}[tbp]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label
      + "}\n\\small\n" + (f"\\setlength{{\\tabcolsep}}{{{sep}}}\n" if sep else "")
      + "\\begin{tabular}{" + colspec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def t_detection() -> None:
    u = U()
    d = u["dose"]
    rows = [f"Reduced dose (10\\%/25\\% of routine), same slice & {ci(d['chest'])} & {ci(d['abdomen'])} \\\\"]
    for r in (0.05, 0.10, 0.20, 0.50, 1.00):
        rows.append(f"Added noise, +{100 * r:.0f}\\% of native & {ci(u['relative_noise'].get(f'chest|{r:g}'))} & "
                    f"{ci(u['relative_noise'].get(f'abdomen|{r:g}'))} \\\\")
    for sp in (0.5, 1.0, 2.0):
        rows.append(f"Gaussian blur, $\\sigma={sp:g}$ px & {ci(u['blur'].get(f'chest|{sp:g}'))} & "
                    f"{ci(u['blur'].get(f'abdomen|{sp:g}'))} \\\\")
    nd = d["chest"]
    table("tab_detection",
          "Paired ranking of degraded images on the held-out test split. Percentage of pairs "
          "in which the degraded image scores worse than its source (100\\% is correct), with "
          "95\\% confidence intervals from a bootstrap over patients. Reduced dose: "
          f"{d['chest']['n']} chest and {d['abdomen']['n']} abdominal pairs from "
          f"{nd['n_patients']} patients each, reconstructions simulated "
          "by the data providers through projection-domain noise insertion. Added noise and blur: "
          "240 full-dose slices from 40 patients (120 chest, 120 abdomen).",
          "tab:detection", "lcc", "Degradation & Chest (\\%) & Abdomen (\\%)", rows)


def t_filtered() -> None:
    u = U()
    d = pd.read_csv(EXP / "exp2_form1_overfiltering.csv")
    pref = d.pivot_table(index="denoiser", columns="target_residual_hu", values="preferred", aggfunc="mean")
    med = d.pivot_table(index="denoiser", columns="target_residual_hu", values="delta", aggfunc="median")
    lv = list(pref.columns)
    rows = [f"{NICE[n]} & " + " & ".join(
        f"{100 * pref.loc[n, l]:.0f} \\,({m(f'{med.loc[n, l]:+.2f}')})" for l in lv) + " \\\\" for n in ORDER]
    fp = u["filtered_preference"]
    table("tab_filtered",
          "Preference for filtered full-dose images: percentage of cases in which the filtered "
          "image scores better than its unfiltered source (test split, "
          f"{fp['all']['n']} filtered images from {fp['all']['n_patients']} patients), and in "
          "parentheses the median score change (positive = worse). A full-dose image still "
          "contains noise, so a preference is not by itself an error; whether the preferred "
          "filtering removes signal is tested in Table~\\ref{tab:lesions}. Overall rate "
          f"{ci(fp['all'])}\\%; counting only changes larger than 0.01, 0.05 and 0.1 score units: "
          f"{pct(fp['eps0.01']['estimate'])}, {pct(fp['eps0.05']['estimate'])} and "
          f"{pct(fp['eps0.1']['estimate'])}\\%. Strength is the standard deviation of the residual "
          "inside the body.",
          "tab:filtered", "l" + "c" * len(lv),
          "& \\multicolumn{" + str(len(lv)) + "}{c}{Residual (HU)} \\\\\nDenoiser & "
          + " & ".join(f"{l:g}" for l in lv), rows)


def t_lesions() -> None:
    d = pd.read_csv(EXP / "exp2_form2_lesions.csv")
    base = d[d.denoiser == "none"][["slice_path", "diameter_mm", "riqe"]].rename(columns={"riqe": "base"})
    f = d[d.denoiser != "none"].merge(base, on=["slice_path", "diameter_mm"])
    f = f[f.diameter_mm == 4.0]
    f["pref"] = f.riqe < f.base
    lv = sorted(f.target_residual_hu.unique())
    rows = []
    for n in ORDER:
        g = f[f.denoiser == n]
        cells = [f"{100 * g[g.target_residual_hu == l].pref.mean():.0f} / "
                 f"{g[g.target_residual_hu == l].retention_matched.median():.2f}".replace("-0.00", "0.00")
                 for l in lv]
        rows.append(f"{NICE[n]} & " + " & ".join(cells) + r" \\")
    noise = float(d.noise_sigma_hu.median())
    table("tab_lesions",
          "Metric preference against lesion signal. Substrate: " + str(d.slice_path.nunique())
          + " held-out full-dose abdominal slices (" + str(f.slice_path.nunique())
          + " with room for the 4\\,mm lesions), one per patient, plus FBP-like noise at the level "
          f"measured between full-dose and simulated quarter-dose abdominal images ({noise:.1f}\\,HU); "
          f"{json.loads((EXP / 'exp2_form2_summary.json').read_text())['realizations']} noise "
          "realisations per condition. Each cell: percentage of images in which RIQE scores the "
          "filtered image better than the unfiltered one / median fraction of the matched-filter "
          "signal of a 4\\,mm, +10\\,HU lesion that survives the filter.",
          "tab:lesions", "l" + "c" * len(lv),
          "& \\multicolumn{" + str(len(lv)) + "}{c}{Residual (HU): preferred (\\%) / signal retained} \\\\\n"
          "Denoiser & " + " & ".join(f"{l:g}" for l in lv), rows)


def t_stratification() -> None:
    d = pd.read_csv(EXP / "stratification.csv")
    lab = {
        "A:": "Abdomen 5\\,mm: GE STANDARD vs Siemens B30f",
        "B:": "Chest: GE STANDARD 1.25\\,mm vs Siemens B50f 1.5\\,mm",
        "C:": "GE: chest 1.25\\,mm vs abdomen 5\\,mm",
        "D:": "Siemens: chest B50f 1.5\\,mm vs abdomen B30f 5\\,mm",
        "G:": "All chest vs all abdomen$^{\\dagger}$",
        "H:": "Vendor: all GE vs all Siemens",
        "ANCHOR": "\\textit{Anchor: full vs reduced dose, same slices}",
    }

    def L(c):
        for k, v in lab.items():
            if c.startswith(k):
                return v
        cell = c.split("—")[-1].strip().split("|")
        vend = {"GE": "GE", "SIEMENS": "Siemens"}.get(cell[0], cell[0])
        return f"Pixel-spacing tertiles, {vend} {cell[1].lower()}"

    d = pd.concat([d[d.axis == "dose"], d[d.axis != "dose"].sort_values("eta")])
    rows = [f"{L(r.contrast)} & {r.n_A} / {r.n_B} & {r.D_obs:.2f} & {r.null_median:.2f} & {r.eta:.2f} \\\\"
            for r in d.itertuples()]
    table("tab_stratification",
          "Divergence between sub-models fitted on two groups of patients (fitting split). "
          "$D$ uses the same functional form as the score. Null: median of 200 patient-level "
          "permutations of the pooled groups. $\\eta$ expresses the excess over the null in units "
          "of the excess between models fitted on the full-dose slices and on the reduced-dose "
          "reconstructions of the same slices. All permutation $p = 0.005$ (the minimum attainable "
          "with 200 permutations). $^{\\dagger}$Anatomy, kernel and slice thickness are not crossed "
          "in this collection; this contrast cannot serve as a clean anatomical control.",
          "tab:strat", "lcccc", "Contrast & Patients & $D$ & Null & $\\eta$", rows)


def t_baselines() -> None:
    s5 = json.loads((EXP / "exp5_summary.json").read_text())
    dr = json.loads((EXP / "exp5_dose_and_relnoise.json").read_text())
    s6 = json.loads((EXP / "exp6_ldctiqac_summary.json").read_text())["models"]
    v = pd.read_csv(EXP / "exp5_verdicts.csv").set_index("model")
    cols = [("RIQE (CT)", "RIQE (CT, masks)", "riqe"),
            ("photographic", "photographic PD/CC0", "photographic"),
            ("NIQE parameters", "NIQE parameters, PD/CC0", "niqe_params")]

    def row(label, vals):
        return f"{label} & " + " & ".join(vals) + " \\\\"

    def rho(k):
        e = s6.get(k)
        if not e:
            return "--"
        lo, hi = e["spearman_ci95_by_source_slice"]
        return m(f"{e['spearman']:+.2f} {{\\scriptsize [{lo:+.2f}, {hi:+.2f}]}}")

    def within(k):
        e = s6.get(k)
        if not e:
            return "--"
        lo, hi = e["within_slice_ci95_by_source_slice"]
        return m(f"{e['within_slice_spearman_median']:+.2f} {{\\scriptsize [{lo:+.2f}, {hi:+.2f}]}}")

    rows = [
        row("Reduced dose ranked worse, chest (\\%)", [f"{dr[a]['dose_chest']:.1f}" for a, _, _ in cols]),
        row("Reduced dose ranked worse, abdomen (\\%)", [f"{dr[a]['dose_abdomen']:.1f}" for a, _, _ in cols]),
        row("Noise +20\\% of native detected (\\%)", [f"{dr[a]['noise_+20%']:.1f}" for a, _, _ in cols]),
        row("Noise +100\\% of native detected (\\%)", [f"{dr[a]['noise_+100%']:.1f}" for a, _, _ in cols]),
        row("Filtered full-dose preferred (\\%)", [f"{100 * v.loc[b].overfilter_failure_fraction:.1f}" for _, b, _ in cols]),
        row("LDCTIQAC: Spearman, pooled", [rho(k) for _, _, k in cols]),
        row("LDCTIQAC: Spearman within slice", [within(k) for _, _, k in cols]),
    ]
    nq = s5["niqe_parameters_model"]
    table("tab_baselines",
          "RIQE against two models fitted with the same code on " + str(s5["photo_corpus"]["n_images"])
          + " CC0 natural photographs: one with the RIQE settings, one with the parameters "
          f"published for NIQE ($P={nq['P']}$, $C={nq['C']:g}$, $p={nq['p']:g}$, no masks). The "
          "reference NIQE model fitted on the LIVE photographs is not used. Rows 1--5: held-out test "
          "split. Rows 6--7: the LDCTIQAC~2023 training images, decoded as $\\mathrm{HU} = 1400x - "
          "1000$; 95\\% intervals resample the inferred source slices; negative is the expected sign "
          "(lower score = closer to the reference). Within-slice: rank correlation among the degraded "
          "versions of one source slice, median over slices.",
          "tab:photo", "lccc", "& RIQE & Photographs, & Photographs, \\\\\n"
          "& (CT) & RIQE settings & NIQE parameters", rows, sep="3.5pt")


def t_flow() -> None:
    """Denominators of every experiment: available, selected, analysed."""
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    u = U()
    kept = corpus[(corpus.kind == "full") & corpus.keep]
    fit = kept[kept.patient_id.isin(split["fit"])]
    test = kept[kept.patient_id.isin(split["test"])]
    d0 = pd.read_csv(EXP / "exp0_real_dose.csv")
    f1 = pd.read_csv(EXP / "exp2_form1_overfiltering.csv")
    bs = pd.read_csv(EXP / "battery_scores.csv")
    low = bs[bs.source == "low"]
    les = pd.read_csv(EXP / "exp2_form2_lesions.csv")
    s6 = json.loads((EXP / "exp6_ldctiqac_summary.json").read_text())
    st = pd.read_csv(EXP / "stratification.csv")
    anchor = st[st.axis == "dose"].iloc[0]
    rows = [
        f"Model fitting & {fit.patient_id.nunique()} fitting patients, 24 slices each & -- & -- & "
        f"{fit.patient_id.nunique()} / {len(fit):,} slices \\\\",
        "Hyperparameter search & inner split 118 / 40 patients & 4 evenly spaced slices per validation patient; "
        "all 24 for dose pairs & -- & 40 / 160 slices, 480 dose pairs \\\\",
        f"Reduced dose (test) & {d0.patient_id.nunique()} Siemens patients with reduced-dose series & "
        f"all kept slices & {int(d0.correct.isna().sum())} unscoreable & "
        f"{d0.dropna(subset=['correct']).patient_id.nunique()} / {int(d0.correct.notna().sum())} pairs \\\\",
        f"Noise, blur, filtering (test) & {test.patient_id.nunique()} patients, {len(test)} kept slices & "
        "6 evenly spaced slices per patient & 0 unscoreable & "
        f"{f1.patient_id.nunique()} / 240 slices, {len(f1):,} filtered \\\\",
        f"Denoisers on reduced dose & 20 Siemens test patients & slices of the test bank with a reduced-dose "
        f"counterpart & {int(low.score.isna().sum())} unscoreable & "
        f"{low.patient_id.nunique()} / {low.slice_path.nunique()} slices \\\\",
        f"Inserted lesions & 20 abdominal test patients & central slice; sites by rule & "
        f"{les.slice_path.nunique() - les[les.diameter_mm == 4.0].slice_path.nunique()} without 4\\,mm sites & "
        f"{les.slice_path.nunique()} / {les.slice_path.nunique()} slices \\\\",
        f"Stratification & {fit.patient_id.nunique()} fitting patients & per contrast & -- & "
        f"anchor {int(anchor.n_A)} patients \\\\",
        f"LDCTIQAC 2023 & {s6['n_images']} images, {s6['n_source_slice_groups']} source-slice groups & all & "
        f"{s6['n_images'] - s6['models']['riqe']['n_scored']} unscoreable & "
        f"{s6['models']['riqe']['n_scored']} images \\\\",
    ]
    table("tab_flow",
          "Denominators. Patients are the unit of the split and of every confidence interval. "
          "Unscoreable: fewer than 72 valid patches in the image domain.",
          "tab:flow", "".join(f">{{\\raggedright\\arraybackslash}}p{{{w}cm}}" for w in (2.6, 3.3, 3.6, 1.9, 2.6)),
          "Experiment & Available & Selection & Excluded & Analysed (patients / units)", rows)


def main() -> int:
    print("tables:")
    for fn in (t_detection, t_filtered, t_lesions, t_stratification, t_baselines, t_flow):
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
