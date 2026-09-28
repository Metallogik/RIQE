"""Inclusion criteria of the pristine corpus, executable and auditable.

Design: a single pass computes for every slice a vector of objective
statistics (`slice_stats`); criteria S1-S6 are then **pure operations on the
table** of statistics (`apply_criteria`). Criteria can therefore be re-run,
re-thresholded and audited without re-reading the DICOM files, and the count
of excluded slices per criterion ends up in the model artefact.

No criterion is applied "by eye": every rule is a boolean function with a
declared threshold. Motion is not detected directly: S1 mitigates it, and the
limitation is stated in the paper. The alternative "fit a preliminary model
and exclude high-scoring slices" was rejected as circular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hu import BODY_HU, domain_mask
from .nss import local_stats

# ---- declared thresholds ---------------------------------------------------

#: S1: fraction of slices discarded at each end of a series
S1_TRIM = 0.10
#: S2: admissible body area as a fraction of the field-of-view area
S2_BODY_FRAC = (0.15, 0.85)
#: S3: width in pixels of the ring inside the field-of-view edge, and the
#: admissible fraction of body pixels on it
S3_RING_PX = 3
S3_MAX_ON_RING = 0.02
#: S4: metal HU threshold and admissible fraction of body pixels above it.
#: Calibrated by visual inspection of stratified samples: 1500 HU is dense
#: cortical bone, not metal -- 43% of slices exceed it and the median of the
#: maximum in-body HU is 1471. At 2000 HU with fraction 1e-4, 3.1% of slices
#: are excluded and all patients but one keep at least 58 slices. The rule is
#: deliberately conservative: the corpus is ten times larger than needed, so
#: losing some artefact-free dense-bone slices is preferable to contaminating
#: the pristine set.
S4_METAL_HU = 2000.0
S4_MAX_FRAC = 1e-4
#: S5: DISABLED as an exclusion criterion. The statistic is still computed
#: and recorded, but excludes nothing.
#:
#: It was intended to detect metal streaks and motion: median local sigma in
#: air, robust z per protocol cell. Calibration falsified it twice. In its
#: first form it fired on a single large patient with clean images, because
#: with little air around the body the statistic was computed on a few pixels
#: at the field-of-view edge. Restricted to air far from the body and to at
#: least 5000 pixels, the extreme cases were again a patient with
#: body_frac_fov = 0.94 -- habitus, not artefacts, and already excluded by S2.
#:
#: Decisive counter-evidence: on slices with bilateral hip prostheses, the
#: clearest metal case in the corpus, the robust z of in-body sigma is -1.0 to
#: -1.6, i.e. it moves in the **wrong** direction (the pelvis is a large
#: homogeneous soft-tissue region).
#:
#: Keeping it enabled would pass off a habitus measure as an artefact
#: detector. It stays as a recorded diagnostic, and the limitation -- motion
#: and streaks not detected directly -- is stated in the paper.
S5_ENABLED = False
S5_MAX_Z = 4.0
S5_MIN_AIR_PX = 5000
#: S6: slices kept per patient
S6_PER_PATIENT = 24


def slice_stats(hu: np.ndarray, padding_value: float | None) -> dict:
    """Objective statistics of one slice, independent of the thresholds."""
    dom, fov, body = domain_mask(hu, padding_value)
    n_fov = int(fov.sum())
    n_body = int(body.sum())
    out = {
        "n_fov": n_fov,
        "n_body": n_body,
        "body_frac_fov": n_body / n_fov if n_fov else 0.0,
        "metal_frac": 0.0,
        "frac_gt_1500": 0.0,
        "frac_gt_2000": 0.0,
        "frac_gt_2500": 0.0,
        "frac_gt_3000": 0.0,
        "hu_max_body": np.nan,
        "hu_p9999_body": np.nan,
        "ring_frac": 0.0,
        "n_air": 0,
        "air_sigma_med": np.nan,
        "body_sigma_med": np.nan,
        "hu_clip_hi_frac": 0.0,
    }
    if n_body == 0:
        return out

    hu_body = hu[body]
    out["metal_frac"] = float((hu_body > S4_METAL_HU).mean())
    out["hu_clip_hi_frac"] = float((hu_body > 1000.0).mean())
    # several thresholds: 1500 HU confuses dense cortical bone with metal,
    # which lies much higher. The threshold is chosen from the data.
    for t in (1500, 2000, 2500, 3000):
        out[f"frac_gt_{t}"] = float((hu_body > t).mean())
    out["hu_max_body"] = float(hu_body.max())
    out["hu_p9999_body"] = float(np.percentile(hu_body, 99.99))

    # S3: body touching the edge of the reconstructed field of view
    from scipy.ndimage import binary_erosion

    inner = binary_erosion(fov, np.ones((2 * S3_RING_PX + 1, 2 * S3_RING_PX + 1), dtype=bool))
    ring = fov & ~inner
    out["ring_frac"] = float((body & ring).sum() / n_body)

    # S5: noise in the air inside the field of view and outside the body
    _, sigma = local_stats(hu)
    # only air well outside the body: near the skin the reconstruction is
    # noisy for reasons unrelated to artefacts
    from scipy.ndimage import binary_dilation

    near_body = binary_dilation(body, np.ones((15, 15), dtype=bool))
    air = fov & ~near_body & (hu < -700.0)
    out["n_air"] = int(air.sum())
    if out["n_air"] >= S5_MIN_AIR_PX:
        out["air_sigma_med"] = float(np.median(sigma[air]))
    out["body_sigma_med"] = float(np.median(sigma[body]))
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    """z based on median and MAD, robust to the outliers it must find."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


def apply_criteria(
    df: pd.DataFrame,
    cell_col: str = "cell",
    per_patient: int = S6_PER_PATIENT,
) -> pd.DataFrame:
    """Apply S1-S6. Returns the dataframe with one boolean column per
    criterion, the final `keep` and the `exclude_reason` of the first failed
    criterion.

    Expects the columns patient_id, series_uid, sop_uid, z, cell and the
    statistics produced by `slice_stats`.
    """
    d = df.copy()

    # S1 - series ends, on the anatomical ordering by z
    d = d.sort_values(["series_uid", "z"]).reset_index(drop=True)
    rank = d.groupby("series_uid").cumcount()
    n = d.groupby("series_uid")["z"].transform("size")
    frac = rank / np.maximum(n - 1, 1)
    d["S1_interior"] = (frac >= S1_TRIM) & (frac <= 1.0 - S1_TRIM)

    # S2 - sufficient anatomy
    lo, hi = S2_BODY_FRAC
    d["S2_anatomy"] = d["body_frac_fov"].between(lo, hi)

    # S3 - truncation
    d["S3_untruncated"] = d["ring_frac"] <= S3_MAX_ON_RING

    # S4 - metal. Uses the explicit column at the declared threshold, not
    # `metal_frac`: the latter is computed during the DICOM pass and would
    # stay tied to the value of S4_METAL_HU in force at that time, making the
    # criterion silently inconsistent if the threshold changes.
    metal_col = f"frac_gt_{int(S4_METAL_HU)}"
    if metal_col not in d.columns:
        raise KeyError(
            f"the slice table has no {metal_col}: re-run build_slice_table.py "
            f"after changing S4_METAL_HU"
        )
    d["S4_no_metal"] = d[metal_col] <= S4_MAX_FRAC

    # S5 - streak / motion artefacts, robust threshold per protocol cell
    z = np.full(len(d), np.nan)
    for _, idx in d.groupby(cell_col).groups.items():
        pos = d.index.get_indexer(idx)
        z[pos] = _robust_z(d.loc[idx, "air_sigma_med"].to_numpy())
    d["air_sigma_z"] = z
    if S5_ENABLED:
        d["S5_no_streak"] = ~(d["air_sigma_z"] > S5_MAX_Z)
        d.loc[d["air_sigma_med"].isna(), "S5_no_streak"] = False
    else:
        # disabled: the statistic stays recorded but excludes nothing
        d["S5_no_streak"] = True

    surviving = (
        d["S1_interior"] & d["S2_anatomy"] & d["S3_untruncated"] & d["S4_no_metal"] & d["S5_no_streak"]
    )

    # S6 - evenly spaced sampling among surviving slices, per patient
    d["S6_sampled"] = False
    for pid, g in d[surviving].groupby("patient_id"):
        g = g.sort_values("z")
        k = min(per_patient, len(g))
        if k == 0:
            continue
        pick = np.linspace(0, len(g) - 1, k).round().astype(int)
        d.loc[g.index[np.unique(pick)], "S6_sampled"] = True

    d["keep"] = surviving & d["S6_sampled"]

    order = ["S1_interior", "S2_anatomy", "S3_untruncated", "S4_no_metal", "S5_no_streak", "S6_sampled"]
    reason = pd.Series("", index=d.index, dtype=object)
    for c in order:
        reason = reason.where(reason.ne("") | d[c], c)
    d["exclude_reason"] = reason
    return d


def exclusion_report(d: pd.DataFrame) -> pd.DataFrame:
    """Count of excluded slices per criterion, in cascade (first failure)."""
    order = ["S1_interior", "S2_anatomy", "S3_untruncated", "S4_no_metal", "S5_no_streak", "S6_sampled"]
    rows = []
    remaining = len(d)
    alive = pd.Series(True, index=d.index)
    for c in order:
        failed = int((alive & ~d[c]).sum())
        alive &= d[c]
        rows.append({"criterion": c, "excluded": failed, "remaining": int(alive.sum())})
        remaining = int(alive.sum())
    rows.append({"criterion": "TOTAL kept", "excluded": len(d) - remaining, "remaining": remaining})
    return pd.DataFrame(rows)
