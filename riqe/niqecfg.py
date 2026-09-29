"""Scoring with the parameters published for NIQE, for the baseline only.

Mittal, Soundararajan and Bovik (2013) use 96x96 patches, C = 1 and a
sharpness threshold of 0.75 at fitting time, on 8-bit photographs, without
any mask; the test image's own patch covariance enters the distance however
few its patches are, through the pseudo-inverse. This module applies those
parameters with the code of this package. It is **not** the reference NIQE:
the reference model fitted on the LIVE photographs is deliberately not used,
and neither is any third-party code. The baseline is refitted on CC0
photographs (scripts/exp5_photo_baseline.py).

Two choices that the paper leaves open are declared here:
  * constant patches (air, padding) have zero MSCN variance and undefined
    distribution parameters; they are left out of the test image's
    statistics rather than entering them as degenerate values;
  * an image needs at least two defined patches to have a covariance.
"""

from __future__ import annotations

import numpy as np

from .model import RCOND, MVGModel, mahalanobis_mixed
from .nss import patch_features

P, C, P_SELECT = 96, 1.0, 0.75
MIN_PATCHES = 2
#: indices of the GGD variance at the two scales (18 features per scale)
_VAR_IDX = (1, 19)


def defined(feat: np.ndarray) -> np.ndarray:
    """Patches whose MSCN distribution is defined (non-zero variance)."""
    return np.all(feat[:, _VAR_IDX] > 0, axis=1) & np.isfinite(feat).all(axis=1)


def image_moments(lum: np.ndarray):
    """(mean, covariance, n) of the defined patches of a luminance image."""
    pf = patch_features(lum, P=P, C=C, fov=None, body=None, p=None)
    f = pf.feat[defined(pf.feat)].astype(np.float64)
    if f.shape[0] < MIN_PATCHES:
        return None, None, int(f.shape[0])
    return f.mean(axis=0), np.cov(f, rowvar=False), int(f.shape[0])


def fitting_features(lum: np.ndarray) -> np.ndarray:
    """Patches selected for fitting: sharpness above P_SELECT of the maximum."""
    pf = patch_features(lum, P=P, C=C, fov=None, body=None, p=P_SELECT)
    f = pf.selected
    return f[defined(f)] if f.shape[0] else f


def score(model: MVGModel, mu, cov) -> float:
    if mu is None:
        return float("nan")
    return mahalanobis_mixed(model.nu, model.sigma, mu, cov, RCOND)
