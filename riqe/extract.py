"""Full pipeline: HU -> luminance -> masks -> 36 features per patch.

A single entry point, so that fitting and scoring cannot diverge by accident:
the only admissible difference between the two is the parameter `p`, which
must be None at scoring time as the NIQE paper prescribes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hu import HU_HI, HU_LO, domain_mask, to_luminance
from .nss import GAUSS_SIGMA, N_FEATURES, patch_features


@dataclass(frozen=True)
class Spec:
    """Complete model specification: everything that changes the numbers."""

    hu_lo: float = HU_LO
    hu_hi: float = HU_HI
    code_max: float = 255.0
    C: float = 0.25
    P: int = 32
    p: float = 0.20
    gauss_sigma: float = GAUSS_SIGMA
    body_min: float = 0.90
    use_masks: bool = True

    def as_dict(self) -> dict:
        return {
            "hu_window": [self.hu_lo, self.hu_hi],
            "code_max": self.code_max,
            "mscn_C": self.C,
            "patch_size": self.P,
            "sharpness_fraction_p": self.p,
            "gauss_sigma": self.gauss_sigma,
            "patch_body_min_fraction": self.body_min,
            "fov_required_fraction": 1.0,
            "use_masks": self.use_masks,
            "quantize": False,
        }

    def key(self) -> tuple:
        """Key of the components that change per-patch features (the
        threshold p acts only on selection, downstream)."""
        return (self.hu_lo, self.hu_hi, self.code_max, self.C, self.P,
                self.gauss_sigma, self.body_min, self.use_masks)


def luminance(hu: np.ndarray, spec: Spec) -> np.ndarray:
    lum = to_luminance(hu, spec.hu_lo, spec.hu_hi)
    if spec.code_max != 255.0:
        lum = lum * np.float32(spec.code_max / 255.0)
    return lum


def features_from_hu(
    hu: np.ndarray,
    padding_value: float | None,
    spec: Spec,
    fitting: bool,
    masks: tuple[np.ndarray, np.ndarray] | None = None,
):
    """Features of one slice.

    Parameters
    ----------
    fitting : True for fitting the corpus (applies sharpness selection),
              False for scoring (does not).
    masks   : precomputed (fov, body). Pass them when scoring degraded
              versions of the same slice: the mask must remain that of the
              original, otherwise a degradation that moves the body threshold
              also changes the domain, confounding the effect of the
              degradation with that of the domain.

    Returns a PatchFeatures; `.selected` are the patches to use.
    """
    if spec.use_masks:
        if masks is None:
            _, fov, body = domain_mask(hu, padding_value)
        else:
            fov, body = masks
    else:
        fov = body = None

    return patch_features(
        luminance(hu, spec),
        P=spec.P,
        C=spec.C,
        fov=fov,
        body=body,
        body_min=spec.body_min,
        p=spec.p if fitting else None,
    )


def masks_for(hu: np.ndarray, padding_value: float | None):
    _, fov, body = domain_mask(hu, padding_value)
    return fov, body


#: minimum number of patches to estimate a non-degenerate 36x36 covariance.
#: Below this the score is undefined and must be reported as NaN.
MIN_PATCHES_FOR_SCORE = 2 * N_FEATURES
