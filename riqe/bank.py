"""Bank of test images, defined deterministically.

A bank entry describes a transformation of a source slice in a fully
reproducible way: kind, parameters, random seed, calibrated filter strength.
The image is not stored: it is **regenerated** by `render`, bit for bit,
whenever needed.

Rationale: the hyperparameter search scores the same set of images under many
(P, C) settings, and the validation battery scores it again. Storing the
images would cost tens of GB; storing their recipes costs kilobytes, and
makes reproduction by third parties a checkable fact rather than a promise.

Calibrating filter strengths is the expensive part (bisection with many
filter applications) and **does not depend on (P, C)**: it is done once,
recorded in the bank, and never repeated.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from . import degrade as dg


@dataclass(frozen=True)
class BankEntry:
    """Recipe of one test image."""

    kind: str                     # original | noise_white | noise_fbp | noise_rel | blur | denoise
    source: str = "full"          # full | low  (which slice of the patient)
    sigma_hu: float | None = None
    sigma_px: float | None = None
    seed: int | None = None
    denoiser: str | None = None
    target_residual_hu: float | None = None
    strength: float | None = None
    residual_hu: float | None = None
    target_reached: bool | None = None
    rel_increase: float | None = None

    def label(self) -> str:
        if self.kind == "original":
            return f"{self.source}:original"
        if self.kind in ("noise_white", "noise_fbp"):
            return f"{self.source}:{self.kind}:{self.sigma_hu:g}HU"
        if self.kind == "noise_rel":
            return f"{self.source}:noise_rel:+{100 * self.rel_increase:g}%"
        if self.kind == "blur":
            return f"{self.source}:blur:{self.sigma_px:g}px"
        return f"{self.source}:{self.denoiser}:res{self.target_residual_hu:g}HU"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def render(entry: BankEntry, hu: np.ndarray) -> np.ndarray:
    """Regenerate the image of an entry from the source slice in HU."""
    if entry.kind == "original":
        return hu.astype(np.float32, copy=True)
    if entry.kind == "noise_white":
        return dg.add_white_noise(hu, entry.sigma_hu, np.random.default_rng(entry.seed))
    if entry.kind in ("noise_fbp", "noise_rel"):
        return dg.add_fbp_noise(hu, entry.sigma_hu, np.random.default_rng(entry.seed))
    if entry.kind == "blur":
        return dg.blur(hu, entry.sigma_px)
    if entry.kind == "denoise":
        if entry.strength is None:
            raise ValueError("denoise entry without a calibrated strength")
        return dg.apply_denoiser(hu, entry.denoiser, entry.strength)
    raise ValueError(f"unknown entry kind: {entry.kind}")


_K_FBP: float | None = None


def _fbp_local_ratio() -> float:
    """Ratio between the median local sigma and the global sigma of the
    synthetic FBP-like noise, estimated once on fixed-seed realisations."""
    global _K_FBP
    if _K_FBP is None:
        from .nss import local_stats

        ks = []
        for seed in range(3):
            n = dg.add_fbp_noise(np.zeros((512, 512), np.float32), 1.0, np.random.default_rng(seed))
            _, s = local_stats(n)
            ks.append(float(np.median(s)))
        _K_FBP = float(np.mean(ks))
    return _K_FBP


def build_bank(
    hu: np.ndarray,
    body: np.ndarray,
    source: str,
    seed: int,
    noise_sigmas=dg.NOISE_SIGMAS_HU,
    blur_sigmas=dg.BLUR_SIGMAS_PX,
    residual_levels=dg.RESIDUAL_LEVELS_HU,
    denoisers=tuple(dg.DENOISERS),
    noise_fine=(),
    noise_rel=(),
) -> list[BankEntry]:
    """Build the bank of one slice, calibrating the filter strengths.

    This is the expensive function: call it once per slice and serialise the
    result.
    """
    entries: list[BankEntry] = [BankEntry(kind="original", source=source)]
    for i, s in enumerate(noise_sigmas):
        entries.append(BankEntry(kind="noise_white", source=source, sigma_hu=float(s), seed=seed + i))
        entries.append(BankEntry(kind="noise_fbp", source=source, sigma_hu=float(s), seed=seed + 1000 + i))
    for s in blur_sigmas:
        entries.append(BankEntry(kind="blur", source=source, sigma_px=float(s)))
    for i, s in enumerate(noise_fine):
        if s > 0:
            entries.append(
                BankEntry(kind="noise_fbp", source=source, sigma_hu=float(s), seed=seed + 2000 + i)
            )
    if noise_rel:
        # Noise ladder RELATIVE to the native noise of the slice. A ladder in
        # absolute HU confounds protocols: 5 HU on a chest slice with 60 HU of
        # native noise is less than a 1% increase, physically undetectable.
        # Here the increase r of the total noise is the same for every slice:
        # sigma_added = sigma_native * sqrt((1 + r)^2 - 1).
        from .nss import local_stats

        _, sig = local_stats(hu)
        sigma_nat = float(np.median(sig[body])) if body.any() else float("nan")
        # sigma_nat is measured by the local estimator (7x7 window), which
        # sees only part of a correlated noise: for the synthetic FBP-like
        # noise, local sigma = 0.637 x global sigma (measured, stable to
        # +-0.002). The added noise must be expressed in the same units as the
        # measurement, otherwise a requested +100% becomes +55%.
        k = _fbp_local_ratio()
        for i, r in enumerate(noise_rel):
            s_add = sigma_nat * float(np.sqrt((1.0 + r) ** 2 - 1.0)) / k
            entries.append(BankEntry(kind="noise_rel", source=source, sigma_hu=s_add,
                                     seed=seed + 3000 + i, rel_increase=float(r)))
    for name in denoisers:
        for lvl in residual_levels:
            st, res, _ = dg.calibrate_strength(hu, name, float(lvl), body)
            entries.append(
                BankEntry(
                    kind="denoise",
                    source=source,
                    denoiser=name,
                    target_residual_hu=float(lvl),
                    strength=float(st),
                    residual_hu=float(res),
                    target_reached=bool(abs(res - lvl) <= 0.05 * lvl),
                )
            )
    return entries


def from_dict(d: dict) -> BankEntry:
    return BankEntry(**{k: v for k, v in d.items() if k in BankEntry.__dataclass_fields__})
