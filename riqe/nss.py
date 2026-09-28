"""NIQE-style NSS features, implemented from the original paper.

Source of the algorithm: A. Mittal, R. Soundararajan, A. C. Bovik, "Making a
'Completely Blind' Image Quality Analyzer", IEEE Signal Processing Letters
20(3):209-212, 2013. GGD parameters by moment matching after K. Sharifi,
A. Leon-Garcia, IEEE TCSVT 5(1):52-56, 1995 (ref. [14] of the NIQE paper);
AGGD parameters after N.-E. Lasmar, Y. Stitou, Y. Berthoumieu, ICIP 2009
(ref. [15]).

No code from the LIVE MATLAB package was consulted or translated.

Choices the paper leaves implicit, made explicit here and recorded in the
model artefact:

  * standard deviation of the Gaussian window: 1.0 on a 7x7 window. The paper
    says "sampled out to 3 standard deviations (K = L = 3)", which with a
    half-width of 3 literally means sigma = 1.0.
  * stabilising constant C in (I - mu)/(sigma + C): fixed to 1 in the paper
    for 0-255 images. Here it is a declared parameter, because in CT the local
    sigma in code units is much smaller than in photographs and C = 1 would
    make the divisive normalisation partial and protocol dependent.
  * low-pass filter of the second scale: the same Gaussian kernel, followed
    by 2:1 decimation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from scipy.ndimage import correlate1d
from scipy.special import gammaln

# --------------------------------------------------------------------------
# Gaussian kernel of the paper: 7x7, sigma = 1.0, unit volume
# --------------------------------------------------------------------------

GAUSS_SIGMA = 1.0
GAUSS_HALFWIDTH = 3  # K = L = 3


def gaussian_weights(sigma: float = GAUSS_SIGMA, halfwidth: int = GAUSS_HALFWIDTH):
    """Separable 1-D kernel equivalent to the circularly symmetric 2-D kernel.

    A circularly symmetric 2-D Gaussian is separable, so the 2-D kernel
    normalised to unit volume is the outer product of the 1-D kernel
    normalised to unit sum.
    """
    x = np.arange(-halfwidth, halfwidth + 1, dtype=np.float64)
    w = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return w / w.sum()


_W1D = gaussian_weights()


def _sepconv(img: np.ndarray, w: np.ndarray = _W1D) -> np.ndarray:
    """Separable convolution with reflective boundary extension."""
    out = correlate1d(img, w, axis=0, mode="reflect")
    return correlate1d(out, w, axis=1, mode="reflect")


def local_stats(img: np.ndarray):
    """Weighted local mean and standard deviation, equations (2) and (3)."""
    img = np.asarray(img, dtype=np.float64)
    mu = _sepconv(img)
    mu_sq = _sepconv(img * img)
    var = mu_sq - mu * mu
    np.clip(var, 0.0, None, out=var)
    return mu, np.sqrt(var)


def mscn(img: np.ndarray, C: float = 1.0):
    """MSCN coefficients, equation (1). Returns (mscn, sigma field)."""
    mu, sigma = local_stats(img)
    return (np.asarray(img, dtype=np.float64) - mu) / (sigma + C), sigma


# --------------------------------------------------------------------------
# Inversion of the moment ratios for GGD and AGGD
#
# Both estimates require inverting a monotone function of alpha made of gamma
# functions. We tabulate it on a fine grid and interpolate: the inversion is
# exact within the grid step, and the grid is recorded in the artefact.
# --------------------------------------------------------------------------

_ALPHA_MIN, _ALPHA_MAX, _ALPHA_N = 0.05, 20.0, 40001
_ALPHA_GRID = np.linspace(_ALPHA_MIN, _ALPHA_MAX, _ALPHA_N)


def _lgam(x):
    return gammaln(x)


# GGD:  rho(alpha) = Gamma(1/a) Gamma(3/a) / Gamma(2/a)^2,  decreasing in a
_GGD_RHO = np.exp(
    _lgam(1.0 / _ALPHA_GRID) + _lgam(3.0 / _ALPHA_GRID) - 2.0 * _lgam(2.0 / _ALPHA_GRID)
)
# np.interp needs increasing abscissae: rho decreases, so we reverse
_GGD_RHO_ASC = _GGD_RHO[::-1]
_GGD_ALPHA_ASC = _ALPHA_GRID[::-1]

# AGGD:  rho(alpha) = Gamma(2/a)^2 / (Gamma(1/a) Gamma(3/a)),  increasing in a
_AGGD_RHO = 1.0 / _GGD_RHO


def _invert_ggd(rho: np.ndarray) -> np.ndarray:
    return np.interp(rho, _GGD_RHO_ASC, _GGD_ALPHA_ASC)


def _invert_aggd(rho: np.ndarray) -> np.ndarray:
    return np.interp(rho, _AGGD_RHO, _ALPHA_GRID)


def _beta_scale(alpha: np.ndarray) -> np.ndarray:
    """sqrt(Gamma(1/a) / Gamma(3/a)), the factor from sigma to beta."""
    return np.exp(0.5 * (_lgam(1.0 / alpha) - _lgam(3.0 / alpha)))


# --------------------------------------------------------------------------
# Block reductions: all per-patch moments in vectorised form
# --------------------------------------------------------------------------

def _blocks(a: np.ndarray, P: int) -> np.ndarray:
    """(n_patch, P*P) view of the non-overlapping PxP blocks."""
    h, w = a.shape
    nh, nw = h // P, w // P
    return (
        a[: nh * P, : nw * P]
        .reshape(nh, P, nw, P)
        .transpose(0, 2, 1, 3)
        .reshape(nh * nw, P * P)
    )


def _ggd_features_blocks(blk: np.ndarray):
    """(alpha, beta) per block from a zero-mean GGD.

    Sharifi/Leon-Garcia moment matching: rho = E[x^2] / E[|x|]^2 determines
    alpha, then beta = sqrt(E[x^2]) * sqrt(Gamma(1/a)/Gamma(3/a)).
    """
    m2 = np.mean(blk * blk, axis=1)
    m1 = np.mean(np.abs(blk), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(m1 > 0, m2 / (m1 * m1), np.inf)
    alpha = _invert_ggd(rho)
    beta = np.sqrt(m2) * _beta_scale(alpha)
    return alpha, beta


def _aggd_features_blocks(blk: np.ndarray):
    """(gamma, beta_l, beta_r, eta) per block from a zero-mode AGGD.

    Lasmar/Stitou/Berthoumieu moment matching:
        sigma_l^2 = E[x^2 | x < 0],  sigma_r^2 = E[x^2 | x >= 0]
        gamma_hat = sigma_l / sigma_r
        r_hat     = E[|x|]^2 / E[x^2]
        R_hat     = r_hat (gamma_hat^3 + 1)(gamma_hat + 1) / (gamma_hat^2 + 1)^2
    and R_hat = Gamma(2/g)^2 / (Gamma(1/g) Gamma(3/g)) is inverted for g.
    Then beta_l = sigma_l sqrt(Gamma(1/g)/Gamma(3/g)), likewise beta_r, and
        eta = (beta_r - beta_l) Gamma(2/g) / Gamma(1/g)      (equation (8))
    """
    neg = blk < 0
    sq = blk * blk
    n_neg = neg.sum(axis=1)
    n_pos = blk.shape[1] - n_neg
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_l = np.sqrt(np.where(n_neg > 0, np.sum(sq * neg, axis=1) / np.maximum(n_neg, 1), 0.0))
        sigma_r = np.sqrt(np.where(n_pos > 0, np.sum(sq * ~neg, axis=1) / np.maximum(n_pos, 1), 0.0))
        gamma_hat = np.where(sigma_r > 0, sigma_l / sigma_r, np.inf)
        m2 = np.mean(sq, axis=1)
        m1 = np.mean(np.abs(blk), axis=1)
        r_hat = np.where(m2 > 0, (m1 * m1) / m2, 0.0)
        g3 = gamma_hat ** 3
        g2 = gamma_hat ** 2
        R_hat = r_hat * (g3 + 1.0) * (gamma_hat + 1.0) / ((g2 + 1.0) ** 2)
    R_hat = np.clip(np.nan_to_num(R_hat, nan=0.0, posinf=0.0), 0.0, _AGGD_RHO[-1])
    gam = _invert_aggd(R_hat)
    scale = _beta_scale(gam)
    beta_l = sigma_l * scale
    beta_r = sigma_r * scale
    eta = (beta_r - beta_l) * np.exp(_lgam(2.0 / gam) - _lgam(1.0 / gam))
    return gam, beta_l, beta_r, eta


# --------------------------------------------------------------------------
# Per-patch features: 18 per scale
# --------------------------------------------------------------------------

#: products of adjacent pairs, as in the paper:
#: horizontal, vertical, main diagonal, secondary diagonal
_SHIFTS = ((0, 1), (1, 0), (1, 1), (1, -1))

FEATURE_NAMES_PER_SCALE = ["ggd_alpha", "ggd_beta"] + [
    f"aggd_{name}_{orient}"
    for orient in ("h", "v", "d1", "d2")
    for name in ("gamma", "eta", "beta_l", "beta_r")
]
assert len(FEATURE_NAMES_PER_SCALE) == 18

FEATURE_NAMES = [f"s1_{n}" for n in FEATURE_NAMES_PER_SCALE] + [
    f"s2_{n}" for n in FEATURE_NAMES_PER_SCALE
]
N_FEATURES = 36


def _shifted_product(m: np.ndarray, di: int, dj: int) -> np.ndarray:
    """m(i,j) * m(i+di, j+dj), with reflective boundary extension."""
    h, w = m.shape
    mp = np.pad(m, 1, mode="reflect")
    return m * mp[1 + di : 1 + di + h, 1 + dj : 1 + dj + w]


def scale_features(
    img: np.ndarray,
    P: int,
    C: float,
    fov: np.ndarray | None = None,
    body: np.ndarray | None = None,
    body_min: float = 0.90,
):
    """Features at one scale.

    Returns (feat, delta, valid):
      feat  (n_patch, 18)  the 18 features per patch
      delta (n_patch,)     patch sharpness, sum of sigma (equation (4))
      valid (n_patch,)     domain admissibility (see below)

    The domain has two asymmetric requirements:

      * field of view at 100%: a single out-of-field padding pixel is fatal,
        because it is an artificial constant (-3024 HU on GE) that zeroes the
        local variance and corrupts the statistics.
      * body at a declared fraction `body_min`: a few air pixels at the skin
        boundary are real anatomy, not an artefact, and excluding them would
        remove every surface patch.
    """
    m, sigma = mscn(img, C=C)
    blk_m = _blocks(m, P)
    alpha, beta = _ggd_features_blocks(blk_m)
    cols = [alpha, beta]
    for di, dj in _SHIFTS:
        prod = _shifted_product(m, di, dj)
        g, bl, br, eta = _aggd_features_blocks(_blocks(prod, P))
        cols += [g, eta, bl, br]
    feat = np.stack(cols, axis=1)
    delta = _blocks(sigma, P).sum(axis=1)

    valid = np.ones(feat.shape[0], dtype=bool)
    if fov is not None:
        valid &= _blocks(fov.astype(np.float64), P).mean(axis=1) >= 1.0
    if body is not None:
        valid &= _blocks(body.astype(np.float64), P).mean(axis=1) >= body_min
    return feat, delta, valid


def downscale(img: np.ndarray) -> np.ndarray:
    """Low-pass filter (same kernel) and 2:1 decimation."""
    return _sepconv(np.asarray(img, dtype=np.float64))[::2, ::2]


def downscale_mask(mask: np.ndarray) -> np.ndarray:
    """Mask decimation by erosion: a second-scale pixel is valid only if all
    four corresponding original pixels are."""
    m = mask.astype(bool)
    h, w = m.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    q = m[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2)
    return q.all(axis=(1, 3))


def patch_features(
    img: np.ndarray,
    P: int = 32,
    C: float = 1.0,
    fov: np.ndarray | None = None,
    body: np.ndarray | None = None,
    body_min: float = 0.90,
    p: float | None = None,
):
    """36 features per patch, two scales.

    Parameters
    ----------
    img  : luminance map on the 0-255 scale (float, not quantised)
    P    : patch side at the first scale; P/2 at the second
    C    : MSCN stabilising constant
    fov  : reconstructed field of view, required at 100%
    body : body mask, required at fraction `body_min`
    p    : if given, applies sharpness selection delta > p * max(delta) among
           valid patches. At scoring time it must be None, as the paper
           prescribes.

    Returns a PatchFeatures.
    """
    if P % 2:
        raise ValueError("P must be even to obtain P/2 at the second scale")
    f1, delta, v1 = scale_features(img, P, C, fov, body, body_min)
    img2 = downscale(img)
    fov2 = None if fov is None else downscale_mask(fov)
    body2 = None if body is None else downscale_mask(body)
    f2, _, v2 = scale_features(img2, P // 2, C, fov2, body2, body_min)

    # the two grids have the same number of patches by construction, but odd
    # sizes can lose a row or column: truncate to the common grid
    nh1, nw1 = img.shape[0] // P, img.shape[1] // P
    nh2, nw2 = img2.shape[0] // (P // 2), img2.shape[1] // (P // 2)
    nh, nw = min(nh1, nh2), min(nw1, nw2)
    sel = (np.arange(nh)[:, None] * nw1 + np.arange(nw)[None, :]).ravel()
    sel2 = (np.arange(nh)[:, None] * nw2 + np.arange(nw)[None, :]).ravel()

    feat = np.concatenate([f1[sel], f2[sel2]], axis=1)
    valid = v1[sel] & v2[sel2]
    delta = delta[sel]

    finite = np.isfinite(feat).all(axis=1)
    valid &= finite
    keep = valid.copy()
    if p is not None:
        if valid.any():
            thr = p * delta[valid].max()
            keep &= delta > thr
        else:
            keep[:] = False
    return PatchFeatures(feat=feat, delta=delta, valid=valid, keep=keep)


@dataclass(frozen=True)
class PatchFeatures:
    """Per-patch features of one slice, with two admissibility levels.

    feat  (n_patch, 36)
    delta (n_patch,)      patch sharpness, equation (4)
    valid (n_patch,)      inside the domain (field of view, body) and finite
    keep  (n_patch,)      valid, plus sharpness selection when requested
    """

    feat: np.ndarray
    delta: np.ndarray
    valid: np.ndarray
    keep: np.ndarray

    @property
    def selected(self) -> np.ndarray:
        return self.feat[self.keep]

    @property
    def in_domain(self) -> np.ndarray:
        return self.feat[self.valid]
