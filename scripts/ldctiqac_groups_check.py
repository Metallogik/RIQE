#!/usr/bin/env python3
"""Sensitivity of the LDCTIQAC analysis to the inferred groupings.

LDCTIQAC does not distribute patient or source-slice identifiers. The
groupings used for its intervals are inferred from image similarity
(scripts/exp6_ldctiqac.py): source slices by a correlation threshold,
anatomy groups by clustering the source slices into as many groups as there
are patients. The right number of clusters does not guarantee the right
membership, so these are exploratory groupings, not patients. This script
reports how the results move when the threshold and the number of anatomy
groups change, and writes a montage (one image per source-slice group, one
row per anatomy group) for visual inspection.

Reads experiments/exp6_ldctiqac_scores.csv; writes
experiments/ldctiqac_group_sensitivity.csv and
experiments/ldctiqac_groups_montage.png.

    .venv/bin/python scripts/ldctiqac_groups_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp6_ldctiqac import DATA, GROUP_CORR, N_ANATOMY_GROUPS, OUT, cluster_boot, median_boot  # noqa: E402

THRESHOLDS = (0.98, 0.99, 0.995, 0.998, 0.999)
N_ANATOMY = (5, 7, 9)
B = 1000


def similarity(names) -> np.ndarray:
    X = []
    for n in names:
        x = np.asarray(Image.open(DATA / "image" / n), dtype=np.float32)
        X.append(gaussian_filter(x, 4)[::8, ::8].ravel())
    X = np.asarray(X)
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    return X @ X.T / X.shape[1]


def groups(R, thr, n_anat):
    _, sg = connected_components(R > thr, directed=False)
    k = sg.max() + 1
    Rg = np.array([[R[np.ix_(sg == a, sg == b)].mean() for b in range(k)] for a in range(k)])
    Dg = np.clip(1.0 - Rg, 0.0, None)
    np.fill_diagonal(Dg, 0.0)
    anat = fcluster(linkage(squareform(Dg, checks=False), "average"), n_anat, criterion="maxclust")
    return sg, anat[sg] - 1


def rho(col):
    return lambda f: spearmanr(f[col], f["radiologist"])[0]


def within(f, col, grp):
    return pd.Series({k: spearmanr(g[col], g["radiologist"])[0] for k, g in f.groupby(grp)
                      if len(g) >= 5 and g["radiologist"].nunique() > 1}).dropna()


def main() -> int:
    d = pd.read_csv(OUT / "exp6_ldctiqac_scores.csv")
    names = list(d["image"])
    R = similarity(names)
    rows = []
    for thr in THRESHOLDS:
        for na in N_ANATOMY:
            sg, ag = groups(R, thr, na)
            f = d.assign(sg=sg, ag=ag)
            sizes = np.bincount(sg)
            w_r, w_p = within(f, "riqe", "sg"), within(f, "photographic", "sg")
            diff = (w_r - w_p).dropna().to_numpy()
            rows.append({
                "threshold": thr, "n_anatomy_groups": na, "n_source_slice_groups": int(len(sizes)),
                "group_size_min": int(sizes.min()), "group_size_median": float(np.median(sizes)),
                "group_size_max": int(sizes.max()),
                "anatomy_group_sizes": "/".join(str(int(x)) for x in sorted(np.bincount(ag), reverse=True)),
                "riqe_rho": rho("riqe")(f),
                "riqe_ci_by_slice": cluster_boot(f, rho("riqe"), "sg", B),
                "riqe_ci_by_anatomy": cluster_boot(f, rho("riqe"), "ag", B, seed=1),
                "riqe_within_median": float(w_r.median()),
                "riqe_within_ci": median_boot(w_r.to_numpy(), B),
                "within_diff_riqe_minus_photo": float(np.median(diff)),
                "within_diff_ci": median_boot(diff, B, seed=3),
            })
            r = rows[-1]
            print(f"thr {thr:.3f} anat {na}: {r['n_source_slice_groups']:3d} slice groups "
                  f"(sizes {r['group_size_min']}-{r['group_size_max']}, median {r['group_size_median']:g}); "
                  f"anatomy {r['anatomy_group_sizes']}; rho {r['riqe_rho']:+.3f} "
                  f"slice CI [{r['riqe_ci_by_slice'][0]:+.2f}, {r['riqe_ci_by_slice'][1]:+.2f}] "
                  f"anat CI [{r['riqe_ci_by_anatomy'][0]:+.2f}, {r['riqe_ci_by_anatomy'][1]:+.2f}]; "
                  f"within {r['riqe_within_median']:+.2f} [{r['riqe_within_ci'][0]:+.2f}, "
                  f"{r['riqe_within_ci'][1]:+.2f}]; within diff {r['within_diff_riqe_minus_photo']:+.2f} "
                  f"[{r['within_diff_ci'][0]:+.2f}, {r['within_diff_ci'][1]:+.2f}]")
    pd.DataFrame(rows).to_csv(OUT / "ldctiqac_group_sensitivity.csv", index=False)

    # montage of the grouping in use: first image of each source-slice group,
    # one row per anatomy group, for visual inspection
    sg, ag = groups(R, GROUP_CORR, N_ANATOMY_GROUPS)
    reps = {}
    for i, (s, a) in enumerate(zip(sg, ag)):
        reps.setdefault(a, {}).setdefault(s, i)
    t = 96
    ncol = max(len(v) for v in reps.values())
    sheet = Image.new("L", (ncol * t, len(reps) * t), 255)
    for r, a in enumerate(sorted(reps)):
        for c, (s, i) in enumerate(sorted(reps[a].items())):
            x = np.asarray(Image.open(DATA / "image" / names[i]), dtype=np.float32)
            im = Image.fromarray((255 * np.clip(x, 0, 1)).astype(np.uint8)).resize((t, t))
            sheet.paste(im, (c * t, r * t))
    sheet.save(OUT / "ldctiqac_groups_montage.png")
    print(f"wrote {OUT / 'ldctiqac_group_sensitivity.csv'} and ldctiqac_groups_montage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
