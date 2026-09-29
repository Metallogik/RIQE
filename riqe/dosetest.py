"""Reduced-dose test: the score of the reduced-dose image must be worse.

For the 100 Siemens patients the collection provides a second reconstruction
at reduced dose (10% of routine for chest, 25% for abdomen). The reduced-dose
data were **simulated** by the data providers, by inserting noise into the
projection data of the same full-dose scan (Moen et al., Med Phys 2021), and
reconstructed on the same z grid. The noise texture is therefore that of the
reconstruction chain rather than of an image-domain model, but it is not an
independent acquisition.

Pairing and domain. Every kept full-dose slice is paired with the
reduced-dose slice of the same patient at identical table position; the
reduced-dose series is not subsampled on its own. Both images of a pair are
scored on the **full-dose masks**: the body mask of a 10%-dose chest image
fragments under noise, and a domain that changes with dose would confound
the effect of dose with that of the domain. The reduced-dose features live in
a separate cache built with those masks (scripts/cache_features.py
--paired-low).

Added to the battery after inspecting validation data: without it, the
hyperparameter criterion was blind to the most important property.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cache import FeatureCache
from .extract import MIN_PATCHES_FOR_SCORE
from .model import RCOND, MVGModel, mahalanobis_mixed

_ROOT = Path(__file__).resolve().parents[1]


def low_cache_dir(P: int, C: float) -> Path:
    """Cache of the reduced-dose slices paired to kept full-dose slices,
    extracted on the full-dose masks."""
    return _ROOT / "data" / "features_lowpaired" / f"P{P}_C{C:g}"


def dose_pairs(corpus: pd.DataFrame, patient_ids=None) -> pd.DataFrame:
    """Every kept full-dose slice with the reduced-dose slice of the same
    patient at identical z (tolerance 0.01 mm)."""
    c = corpus if patient_ids is None else corpus[corpus.patient_id.isin(set(patient_ids))]
    full = c[(c["kind"] == "full") & c["keep"]]
    low = c[c["kind"] == "low"]
    low_by_patient = {pid: g for pid, g in low.groupby("patient_id")}
    rows = []
    for pid, fu in full.groupby("patient_id"):
        lo = low_by_patient.get(pid)
        if lo is None or lo.empty:
            continue
        lz = lo["z"].to_numpy()
        for r in fu.itertuples():
            j = int(np.abs(lz - r.z).argmin())
            if abs(lz[j] - r.z) <= 0.01:
                rows.append({
                    "patient_id": pid,
                    "cell": r.cell,
                    "region": "chest" if "CHEST" in r.cell else "abdomen",
                    "z": float(r.z),
                    "path_full": r.path,
                    "path_low": lo["path"].iloc[j],
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


def pair_moments(full_cache: FeatureCache, low_cache: FeatureCache, pairs: pd.DataFrame) -> dict:
    """Moments of both members of every pair, each from its own cache."""
    m = image_moments(full_cache, list(pairs["path_full"]))
    m.update(image_moments(low_cache, list(pairs["path_low"])))
    return m


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
        out[f"n_patients_{reg}"] = int(s["patient_id"].nunique()) if len(s) else 0
        out[f"correct_{reg}"] = float((s["score_low"] > s["score_full"]).mean()) if len(s) else np.nan
    vals = [out[f"correct_{r}"] for r in ("chest", "abdomen") if np.isfinite(out[f"correct_{r}"])]
    out["correct_min"] = float(min(vals)) if vals else np.nan
    out["correct_all"] = float((ok["score_low"] > ok["score_full"]).mean()) if len(ok) else np.nan
    return out
