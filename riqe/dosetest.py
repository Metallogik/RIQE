"""Real-dose test: the score of the reduced-dose image must be worse.

For 100 Siemens patients the collection provides the reconstruction of the
**same** acquisition at a real reduced dose (10% of routine for chest, 25% for
abdomen), on the same z grid. It is the only test of the battery with native
rather than synthetic noise texture, and it is the use case the metric exists
for: low-dose CT.

Computed from the feature cache without re-reading DICOM files: the cache
holds, for every kept slice of both doses, the patches inside the domain,
which is exactly what scoring needs (sharpness selection is not applied at
scoring time).

Added to the battery after inspecting validation data: without it, the
hyperparameter criterion was blind to the most important property.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cache import FeatureCache
from .extract import MIN_PATCHES_FOR_SCORE
from .model import RCOND, MVGModel, mahalanobis_mixed


def dose_pairs(corpus: pd.DataFrame, patient_ids) -> pd.DataFrame:
    """(full dose, reduced dose) pairs of kept slices at identical z."""
    k = corpus[corpus["keep"] & corpus.patient_id.isin(set(patient_ids))]
    rows = []
    for pid, g in k.groupby("patient_id"):
        lo = g[g["kind"] == "low"]
        fu = g[g["kind"] == "full"]
        if lo.empty or fu.empty:
            continue
        fz = fu["z"].to_numpy()
        for r in lo.itertuples():
            j = int(np.abs(fz - r.z).argmin())
            if abs(fz[j] - r.z) <= 0.01:
                rows.append({
                    "patient_id": pid,
                    "cell": fu["cell"].iloc[j],
                    "region": "chest" if "CHEST" in fu["cell"].iloc[j] else "abdomen",
                    "z": float(r.z),
                    "path_full": fu["path"].iloc[j],
                    "path_low": r.path,
                })
    return pd.DataFrame(rows)


def image_moments(cache: FeatureCache, paths) -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
    """Mean and covariance of the in-domain patches, per image."""
    out = {}
    for q in paths:
        f, _ = cache.slice_rows(q)
        if f.shape[0] < MIN_PATCHES_FOR_SCORE:
            out[q] = None
        else:
            f = f.astype(np.float64)
            out[q] = (f.mean(axis=0), np.cov(f, rowvar=False))
    return out


def dose_ordering(model: MVGModel, pairs: pd.DataFrame, moments: dict) -> pd.DataFrame:
    """Scores of the pairs; `correct` = the reduced-dose image is worse."""
    d = pairs.copy()
    sf, sl = [], []
    for r in d.itertuples():
        a, b = moments.get(r.path_full), moments.get(r.path_low)
        sf.append(np.nan if a is None else mahalanobis_mixed(model.nu, model.sigma, a[0], a[1], RCOND))
        sl.append(np.nan if b is None else mahalanobis_mixed(model.nu, model.sigma, b[0], b[1], RCOND))
    d["score_full"] = sf
    d["score_low"] = sl
    # float, not bool: an unscoreable pair is NaN, not "wrong"
    ok = d["score_full"].notna() & d["score_low"].notna()
    d["correct"] = np.where(ok, (d["score_low"] > d["score_full"]).astype(float), np.nan)
    return d


def summarize(d: pd.DataFrame) -> dict:
    ok = d.dropna(subset=["score_full", "score_low"])
    out = {"n_pairs": int(len(ok)), "n_unscoreable": int(len(d) - len(ok))}
    for reg in ("chest", "abdomen"):
        s = ok[ok["region"] == reg]
        out[f"n_{reg}"] = int(len(s))
        out[f"correct_{reg}"] = float((s["score_low"] > s["score_full"]).mean()) if len(s) else np.nan
    vals = [out[f"correct_{r}"] for r in ("chest", "abdomen") if np.isfinite(out[f"correct_{r}"])]
    out["correct_min"] = float(min(vals)) if vals else np.nan
    out["correct_all"] = float((ok["score_low"] > ok["score_full"]).mean()) if len(ok) else np.nan
    return out
