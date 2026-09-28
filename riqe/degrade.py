"""Degradations and filters on declared ladders, plus the low-contrast lesion
detectability surrogate.

Everything operates in Hounsfield units, **before** the intensity mapping:
the only physically meaningful order, because noise and filtering act on the
reconstructed image, not on its 8-bit representation.

Denoisers come from scikit-image (BSD-3). No BM3D: the available
implementations are GPL or of unclear licence, and the licensing constraint
of the project is not negotiable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter  # noqa: F401
from skimage.restoration import (
    denoise_bilateral,
    denoise_nl_means,
    denoise_tv_chambolle,
    denoise_wavelet,
)

# ---------------------------------------------------------------------------
# Degradation ladders
# ---------------------------------------------------------------------------

NOISE_SIGMAS_HU = (5.0, 10.0, 20.0, 40.0, 80.0)
BLUR_SIGMAS_PX = (0.5, 1.0, 1.5, 2.0, 3.0)

#: fine ladder starting at zero, for form 3 of the overfiltering test: used to
#: look for a minimum of the score at non-zero noise
NOISE_FINE_HU = (0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 65.0, 90.0)


def add_white_noise(hu: np.ndarray, sigma_hu: float, rng: np.random.Generator) -> np.ndarray:
    if sigma_hu <= 0:
        return hu.astype(np.float32, copy=True)
    return (hu + rng.normal(0.0, sigma_hu, hu.shape)).astype(np.float32)


def _ramp_nps_filter(shape: tuple[int, int], cutoff: float = 0.55) -> np.ndarray:
    """Radial filter mimicking filtered back-projection.

    FBP noise is not white: the |f| ramp of the reconstruction filter shifts
    its power towards mid frequencies, with a roll-off at the kernel cut-off.
    The spectrum magnitude is approximated by |f| times a Hann window that
    vanishes at `cutoff` (in units of the Nyquist frequency).
    """
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2) / 0.5  # normalised to Nyquist = 1
    w = np.where(r <= cutoff, 0.5 * (1.0 + np.cos(np.pi * r / cutoff)), 0.0)
    return r * w


_NPS_CACHE: dict[tuple[int, int, float], np.ndarray] = {}


def add_fbp_noise(
    hu: np.ndarray, sigma_hu: float, rng: np.random.Generator, cutoff: float = 0.55
) -> np.ndarray:
    """Correlated FBP-like noise, rescaled to an overall sigma_hu."""
    if sigma_hu <= 0:
        return hu.astype(np.float32, copy=True)
    key = (hu.shape[0], hu.shape[1], cutoff)
    if key not in _NPS_CACHE:
        _NPS_CACHE[key] = _ramp_nps_filter(hu.shape, cutoff)
    h = _NPS_CACHE[key]
    w = rng.normal(0.0, 1.0, hu.shape)
    n = np.real(np.fft.ifft2(np.fft.fft2(w) * h))
    s = n.std()
    if s > 0:
        n *= sigma_hu / s
    return (hu + n).astype(np.float32)


def blur(hu: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return hu.astype(np.float32, copy=True)
    return gaussian_filter(hu.astype(np.float32), sigma_px, mode="nearest")


# ---------------------------------------------------------------------------
# Denoisers, at increasing strength
# ---------------------------------------------------------------------------

def _f_gaussian(hu, s):
    return gaussian_filter(hu.astype(np.float32), s, mode="nearest")


def _f_tv(hu, w):
    return denoise_tv_chambolle(hu.astype(np.float64), weight=w).astype(np.float32)


def _f_bilateral(hu, sc):
    return denoise_bilateral(
        hu.astype(np.float64), sigma_color=sc, sigma_spatial=2.0, channel_axis=None
    ).astype(np.float32)


def _f_nlm(hu, h):
    return denoise_nl_means(
        hu.astype(np.float64), h=h, sigma=h / 1.5, patch_size=5, patch_distance=6,
        fast_mode=True, channel_axis=None,
    ).astype(np.float32)


def _f_wavelet(hu, s):
    return denoise_wavelet(
        hu.astype(np.float64), sigma=s, mode="soft", method="BayesShrink",
        wavelet="db4", rescale_sigma=True, channel_axis=None,
    ).astype(np.float32)


#: name -> (function, bounds of the strength search interval)
#: The native parameter of each filter is not comparable with the others:
#: `bilateral` at sigma_color 10 alters the image ten times more than `tv` at
#: weight 2. Strength is therefore calibrated per image to hit a declared
#: residual standard deviation (see `calibrate_strength`), and ladders are
#: indexed by that.
DENOISERS: dict[str, tuple] = {
    "gaussian": (_f_gaussian, (0.05, 8.0)),
    "tv": (_f_tv, (0.05, 2000.0)),
    "bilateral": (_f_bilateral, (0.5, 4000.0)),
    "nlm": (_f_nlm, (0.2, 2000.0)),
    "wavelet": (_f_wavelet, (0.2, 2000.0)),
}

#: common filtering-strength scale, as the standard deviation of the residual
#: inside the body (HU). It is the x axis of the main figures: physically
#: interpretable and independent of the filter.
RESIDUAL_LEVELS_HU = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def apply_denoiser(hu: np.ndarray, name: str, strength) -> np.ndarray:
    fn, _ = DENOISERS[name]
    return fn(hu, strength)


def residual_std(original: np.ndarray, filtered: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Standard deviation of the residual: the common unit to match filters."""
    d = np.asarray(filtered, dtype=np.float64) - np.asarray(original, dtype=np.float64)
    if mask is not None:
        d = d[mask]
    return float(d.std())


def calibrate_strength(
    hu: np.ndarray,
    name: str,
    target_residual_hu: float,
    mask: np.ndarray | None = None,
    tol: float = 0.03,
    max_iter: int = 24,
):
    """Filter strength producing the requested residual, by bisection on a
    logarithmic scale.

    Returns (strength, residual_obtained, n_iterations). If the target lies
    outside the range the filter can reach on this image, returns the nearest
    bound and its residual: the caller sees it from the residual and reports
    it, instead of pretending to have hit the target.
    """
    fn, (lo, hi) = DENOISERS[name]
    r_lo = residual_std(hu, fn(hu, lo), mask)
    r_hi = residual_std(hu, fn(hu, hi), mask)
    if target_residual_hu <= r_lo:
        return lo, r_lo, 0
    if target_residual_hu >= r_hi:
        return hi, r_hi, 0
    a, b = lo, hi
    for it in range(1, max_iter + 1):
        m = float(np.sqrt(a * b))  # geometric bisection
        r = residual_std(hu, fn(hu, m), mask)
        if abs(r - target_residual_hu) <= tol * target_residual_hu:
            return m, r, it
        if r < target_residual_hu:
            a = m
        else:
            b = m
    m = float(np.sqrt(a * b))
    return m, residual_std(hu, fn(hu, m), mask), max_iter


def denoise_at_level(
    hu: np.ndarray, name: str, target_residual_hu: float, mask: np.ndarray | None = None
):
    """Apply the filter at the calibrated strength. Returns (image, info)."""
    s, r, it = calibrate_strength(hu, name, target_residual_hu, mask)
    return apply_denoiser(hu, name, s), {
        "denoiser": name,
        "target_residual_hu": target_residual_hu,
        "strength": s,
        "residual_hu": r,
        "bisect_iters": it,
        "target_reached": abs(r - target_residual_hu) <= 0.05 * target_residual_hu,
    }


# ---------------------------------------------------------------------------
# Detectability surrogate: inserted lesions and NPW matched filter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Lesion:
    cy: int
    cx: int
    radius_px: float
    contrast_hu: float
    diameter_mm: float


def _disc(shape, cy, cx, radius_px, edge_px: float = 0.7):
    """Disc with an edge softened over ~1 px, so as not to introduce an
    artificial step that would be easier to detect than a real lesion."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return np.clip((radius_px + edge_px - r) / (2 * edge_px), 0.0, 1.0)


def find_homogeneous_sites(
    hu: np.ndarray,
    body: np.ndarray,
    radius_px: float,
    n_sites: int,
    rng: np.random.Generator,
    hu_range: tuple[float, float] = (40.0, 80.0),
    sigma_quantile: float = 0.5,
    min_sep_px: float | None = None,
):
    """Sites of homogeneous parenchyma where lesions are inserted.

    Criteria: local mean within `hu_range` (soft tissue / liver parenchyma),
    local standard deviation below the image median, and distance from the
    body boundary of at least one diameter.
    """
    from scipy.ndimage import binary_erosion

    k = int(np.ceil(radius_px * 2)) | 1
    mu = uniform_filter(hu.astype(np.float64), k, mode="nearest")
    mu2 = uniform_filter(hu.astype(np.float64) ** 2, k, mode="nearest")
    sd = np.sqrt(np.clip(mu2 - mu ** 2, 0, None))

    interior = binary_erosion(body, np.ones((k + 4, k + 4), dtype=bool))
    ok = interior & (mu >= hu_range[0]) & (mu <= hu_range[1])
    if not ok.any():
        return []
    thr = np.quantile(sd[ok], sigma_quantile)
    ok &= sd <= thr
    ys, xs = np.nonzero(ok)
    if len(ys) == 0:
        return []

    sep = min_sep_px if min_sep_px is not None else 3.0 * radius_px
    order = rng.permutation(len(ys))
    picked: list[tuple[int, int]] = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if all((y - py) ** 2 + (x - px) ** 2 >= sep ** 2 for py, px in picked):
            picked.append((y, x))
            if len(picked) >= n_sites:
                break
    return picked


def insert_lesions(
    hu: np.ndarray,
    sites: list[tuple[int, int]],
    radius_px: float,
    contrast_hu: float,
    diameter_mm: float,
) -> tuple[np.ndarray, list[Lesion]]:
    out = hu.astype(np.float32, copy=True)
    lesions = []
    for cy, cx in sites:
        out += (contrast_hu * _disc(hu.shape, cy, cx, radius_px)).astype(np.float32)
        lesions.append(Lesion(cy, cx, radius_px, contrast_hu, diameter_mm))
    return out, lesions


def _template(radius_px: float, half: int) -> np.ndarray:
    """Matched-filter template: the known lesion profile, zero mean,
    normalised. It does not prewhiten (NPW) and applies no eye filter: the
    latter would need arbitrary assumptions about viewing distance and
    display, which cannot be justified in a study without human readers.
    """
    n = 2 * half + 1
    t = _disc((n, n), half, half, radius_px)
    t = t - t.mean()
    nrm = np.sqrt((t ** 2).sum())
    return t / nrm if nrm > 0 else t


def npw_response(img: np.ndarray, sites, radius_px: float) -> np.ndarray:
    """Response of the NPW matched filter at the given sites."""
    half = int(np.ceil(radius_px * 2.5))
    t = _template(radius_px, half)
    out = []
    for cy, cx in sites:
        y0, y1 = cy - half, cy + half + 1
        x0, x1 = cx - half, cx + half + 1
        if y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1]:
            out.append(np.nan)
            continue
        out.append(float((img[y0:y1, x0:x1].astype(np.float64) * t).sum()))
    return np.asarray(out)


def dprime_task(
    hu_clean: np.ndarray,
    body: np.ndarray,
    sites,
    radius_px: float,
    contrast_hu: float,
    diameter_mm: float,
    noise_sigma_hu: float,
    n_realizations: int,
    rng: np.random.Generator,
    filt=None,
    noise: str = "fbp",
):
    """NPW matched-filter d', with the variance taken **over noise
    realisations at the same site**.

    Why this and not a comparison across different sites: the template
    response at different anatomical sites varies mostly because of
    anatomical variability, orders of magnitude larger than the signal of a
    low-contrast lesion. A d' built on that variance measures how
    heterogeneous the liver is, not how detectable the lesion is. Comparing
    instead the same site with and without the lesion over independent noise
    realisations cancels the anatomical term exactly and leaves the noise,
    which is what the filter modifies.

    Procedure, for each of `n_realizations` realisations:
      1. generate a noise realisation at `noise_sigma_hu`;
      2. build the image with and without lesions, same noise;
      3. apply the same filter `filt` to both (None = no filter);
      4. record the template response at the sites.

    Then, per site:
      Delta = mean(lambda_present - lambda_absent)   (signal left after the
              filter, capturing loss of contrast)
      sigma = standard deviation of lambda_absent over realisations
      d' of the site = Delta / sigma
    and the overall d' is the mean over sites.

    Returns (d_prime, details).
    """
    if len(sites) == 0:
        return float("nan"), {}
    add = add_fbp_noise if noise == "fbp" else add_white_noise
    sig = np.zeros((n_realizations, len(sites)))
    abs_ = np.zeros((n_realizations, len(sites)))
    lesion_field = np.zeros_like(hu_clean, dtype=np.float32)
    for cy, cx in sites:
        lesion_field += (contrast_hu * _disc(hu_clean.shape, cy, cx, radius_px)).astype(np.float32)
    for k in range(n_realizations):
        n = add(np.zeros_like(hu_clean), noise_sigma_hu, rng)
        base = hu_clean + n
        with_les = base + lesion_field
        if filt is not None:
            base_f = filt(base)
            with_f = filt(with_les)
        else:
            base_f, with_f = base, with_les
        abs_[k] = npw_response(base_f, sites, radius_px)
        sig[k] = npw_response(with_f, sites, radius_px)
    delta = np.nanmean(sig - abs_, axis=0)
    sd = np.nanstd(abs_, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dp = np.where(sd > 0, delta / sd, np.nan)

    # Signal retention: how much of the lesion amplitude survives the filter.
    # Needed because the d' of a matched filter with known template is barely
    # sensitive to smoothing (the optimal filter reduces signal and noise in
    # the same band), whereas loss of amplitude is what erases a lesion to the
    # eye. The two measures are reported together: when they diverge, that is
    # information, not noise.
    half = int(np.ceil(radius_px * 2.5))
    t = _template(radius_px, half)
    ref_delta = float(np.nansum(
        t * _disc((2 * half + 1, 2 * half + 1), half, half, radius_px) * contrast_hu
    ))
    peak_ret = []
    for k in range(min(n_realizations, 4)):
        n = add(np.zeros_like(hu_clean), noise_sigma_hu, rng)
        base = hu_clean + n
        with_les = base + lesion_field
        bf = filt(base) if filt is not None else base
        wf = filt(with_les) if filt is not None else with_les
        diff = wf.astype(np.float64) - bf.astype(np.float64)
        for cy, cx in sites:
            y0, y1 = max(cy - half, 0), min(cy + half + 1, diff.shape[0])
            x0, x1 = max(cx - half, 0), min(cx + half + 1, diff.shape[1])
            peak_ret.append(float(diff[y0:y1, x0:x1].max()) / contrast_hu)

    return float(np.nanmean(dp)), {
        "diameter_mm": diameter_mm,
        "contrast_hu": contrast_hu,
        "radius_px": radius_px,
        "n_sites": len(sites),
        "n_realizations": n_realizations,
        "noise_sigma_hu": noise_sigma_hu,
        "noise_model": noise,
        "dprime_per_site": dp.tolist(),
        "signal_amplitude_mean": float(np.nanmean(delta)),
        "noise_sd_mean": float(np.nanmean(sd)),
        "matched_signal_retention": float(np.nanmean(delta) / ref_delta) if ref_delta else float("nan"),
        "peak_retention": float(np.nanmean(peak_ret)) if peak_ret else float("nan"),
    }
