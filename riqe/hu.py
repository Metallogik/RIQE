"""Canonical HU -> luminance mapping and domain masks.

The mapping is an inseparable part of the model specification: different
windows produce mutually incompatible models.

Frozen specification (RIQE v1):

    L = 255 * (clip(HU, HU_LO, HU_HI) - HU_LO) / (HU_HI - HU_LO)

with HU_LO = -1000 and HU_HI = +1000, in float32 and without rounding. The
0-255 scale is kept because the MSCN stabilising constant is an absolute
additive constant that presumes that scale; 8-bit quantisation is not part of
the algorithm and is not applied.

The masks are not optional. GE images in this collection carry 55,772 pixels
per slice (21.3% of the area) at the padding value -3024 HU outside the
reconstructed field of view; Siemens images carry none. Without a
field-of-view mask, the between-vendor difference one would measure is a
DICOM padding convention, not acquisition physics.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening, label

#: canonical HU window of the model
HU_LO = -1000.0
HU_HI = 1000.0

#: body-mask threshold, in HU
BODY_HU = -300.0
#: side of the structuring element of the morphological opening
BODY_OPEN = 5


def to_luminance(hu: np.ndarray, lo: float = HU_LO, hi: float = HU_HI) -> np.ndarray:
    """Map HU to 0-255 float32, clipping to the canonical window."""
    x = np.clip(np.asarray(hu, dtype=np.float32), lo, hi)
    return (255.0 * (x - lo) / (hi - lo)).astype(np.float32)


def fov_mask(hu: np.ndarray, padding_value: float | None) -> np.ndarray:
    """Pixels inside the reconstructed field of view.

    The padding value declared in PixelPaddingValue is excluded. When the tag
    is absent, pixels below -1024 HU are excluded: they are not anatomy, since
    the physical floor of the scale is -1000 and -1024 is the minimum
    representable with RescaleIntercept = -1024.
    """
    hu = np.asarray(hu)
    if padding_value is not None:
        m = hu > float(padding_value)
    else:
        m = np.ones(hu.shape, dtype=bool)
    return m & (hu >= -1024.0)


def body_mask(hu: np.ndarray, fov: np.ndarray | None = None) -> np.ndarray:
    """Body mask: threshold, opening, largest connected component, holes filled."""
    hu = np.asarray(hu)
    m = hu > BODY_HU
    if fov is not None:
        m &= fov
    m = binary_opening(m, np.ones((BODY_OPEN, BODY_OPEN), dtype=bool))
    lab, n = label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == sizes.argmax()
    elif n == 0:
        return np.zeros(hu.shape, dtype=bool)
    return binary_fill_holes(m)


def domain_mask(hu: np.ndarray, padding_value: float | None):
    """Admissible patch domain: field of view and body. Returns (domain, fov, body)."""
    fov = fov_mask(hu, padding_value)
    body = body_mask(hu, fov)
    return fov & body, fov, body
