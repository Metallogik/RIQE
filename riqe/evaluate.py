"""Utilities shared by the validation battery.

Loading bank moments, scoring against a model, and the ranking statistics
used repeatedly. Kept in one place so that the experiments cannot diverge on
details such as the handling of undefined scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import RCOND, MVGModel, mahalanobis_mixed


def load_moments(path: str | Path):
    """Returns (meta, NU, SIGMA). `meta` has one row per test image."""
    z = np.load(path, allow_pickle=False)
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    meta["slice_path"] = json.loads(str(z["slice_path"]))
    meta["patient_id"] = json.loads(str(z["patient_id"]))
    meta["cell"] = json.loads(str(z["cell"]))
    meta["pixel_spacing"] = json.loads(str(z["pixel_spacing"]))
    meta["row"] = np.arange(len(meta))
    return meta, z["nu"], z["sigma"]


def score_all(model: MVGModel, NU: np.ndarray, SG: np.ndarray, rows) -> np.ndarray:
    """Scores of a set of rows. NaN where undefined."""
    rows = np.asarray(rows, dtype=int)
    out = np.full(len(rows), np.nan)
    for k, i in enumerate(rows):
        if np.isfinite(NU[i, 0]):
            out[k] = mahalanobis_mixed(model.nu, model.sigma, NU[i], SG[i], RCOND)
    return out


def attach_scores(meta: pd.DataFrame, model: MVGModel, NU, SG, col: str = "score") -> pd.DataFrame:
    d = meta.copy()
    d[col] = score_all(model, NU, SG, d["row"].to_numpy())
    return d


def step_monotone(seq: np.ndarray, increasing: bool = True) -> bool:
    """Monotonicity **at every step**, not merely overall."""
    s = np.asarray(seq, dtype=float)
    if not np.isfinite(s).all() or len(s) < 2:
        return False
    d = np.diff(s)
    return bool(np.all(d > 0) if increasing else np.all(d < 0))


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_w(rank_matrix: np.ndarray) -> float:
    """Kendall's W: agreement among several judges (here: images) on a set of
    objects (here: denoisers). 1 = perfect agreement."""
    R = np.asarray(rank_matrix, dtype=float)
    R = R[np.isfinite(R).all(axis=1)]
    m, n = R.shape
    if m < 2 or n < 2:
        return float("nan")
    Rj = R.sum(axis=0)
    S = ((Rj - Rj.mean()) ** 2).sum()
    return float(12.0 * S / (m ** 2 * (n ** 3 - n)))


def sign_test(a, b) -> tuple[int, int, float]:
    """Paired sign test. Returns (n_positive, n_valid, two-sided p)."""
    from scipy.stats import binomtest

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & (a != b)
    n = int(ok.sum())
    if n == 0:
        return 0, 0, float("nan")
    k = int((a[ok] > b[ok]).sum())
    return k, n, float(binomtest(k, n, 0.5).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni correction for multiple comparisons."""
    items = sorted(((k, v) for k, v in pvals.items() if np.isfinite(v)), key=lambda t: t[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, v) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * v))
        out[k] = adj
        prev = adj
    for k, v in pvals.items():
        out.setdefault(k, float("nan"))
    return out


def score_all_refcov(model: MVGModel, NU: np.ndarray, rows) -> np.ndarray:
    """Ablation of the score: the image covariance is left out, so the
    distance is sqrt((nu_ref - nu_img)^T Sigma_ref^+ (nu_ref - nu_img)).

    If a preference survives this ablation it comes from the shift of the
    mean features towards the reference, not from a change in the spread of
    the image's own patch features."""
    rows = np.asarray(rows, dtype=int)
    pinv = np.linalg.pinv(np.asarray(model.sigma, dtype=np.float64), rcond=RCOND)
    out = np.full(len(rows), np.nan)
    for k, i in enumerate(rows):
        if np.isfinite(NU[i, 0]):
            dv = np.asarray(model.nu, dtype=np.float64) - NU[i].astype(np.float64)
            out[k] = float(np.sqrt(max(dv @ pinv @ dv, 0.0)))
    return out


def cluster_bootstrap(frame: pd.DataFrame, stat, cluster: str = "patient_id",
                      B: int = 2000, seed: int = 20260917, level: float = 95.0) -> dict:
    """Percentile confidence interval of `stat(frame)`, resampling whole
    clusters (patients) with replacement, so that slices, pairs and
    transformations of one patient stay together."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in frame.groupby(cluster)]
    est = float(stat(frame))
    vals = np.empty(B)
    for b in range(B):
        take = rng.integers(0, len(groups), len(groups))
        vals[b] = stat(pd.concat([groups[i] for i in take], ignore_index=True))
    a = (100.0 - level) / 2.0
    lo, hi = np.nanpercentile(vals, [a, 100.0 - a])
    return {"estimate": est, "ci_low": float(lo), "ci_high": float(hi),
            "n_items": int(len(frame)), "n_clusters": int(len(groups)), "B": B}
