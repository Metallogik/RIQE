"""MVG model, score, and divergence between models.

Score, equation (10) of the NIQE paper:

    D = sqrt( (nu1 - nu2)^T ((Sigma1 + Sigma2)/2)^-1 (nu1 - nu2) )

The same functional form is used as the divergence between two models, so
that a divergence reads in the same units as the scores the model produces.

The inverse is always an SVD pseudo-inverse with a declared relative
tolerance: with small P or few patches the averaged covariance can be
ill-conditioned, and a hidden implicit ridge would make the numbers
irreproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

#: relative singular-value tolerance of the pseudo-inverse
RCOND = 1e-10


@dataclass
class MVGModel:
    """36-dimensional multivariate Gaussian model."""

    nu: np.ndarray            # (36,)
    sigma: np.ndarray         # (36, 36)
    n_patches: int
    n_images: int
    n_patients: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.nu.shape[0])

    def cond(self) -> float:
        return float(np.linalg.cond(self.sigma))

    def to_npz(self, path) -> None:
        np.savez_compressed(
            path,
            nu=self.nu,
            sigma=self.sigma,
            n_patches=self.n_patches,
            n_images=self.n_images,
            n_patients=self.n_patients,
            meta=json.dumps(self.meta, default=str),
        )

    @classmethod
    def from_npz(cls, path) -> "MVGModel":
        z = np.load(path, allow_pickle=False)
        return cls(
            nu=z["nu"],
            sigma=z["sigma"],
            n_patches=int(z["n_patches"]),
            n_images=int(z["n_images"]),
            n_patients=int(z["n_patients"]),
            meta=json.loads(str(z["meta"])),
        )


def fit_mvg(
    features: np.ndarray,
    n_images: int = 0,
    n_patients: int = 0,
    meta: dict[str, Any] | None = None,
) -> MVGModel:
    """Fit mean and covariance on a set of patches (n_patch, 36)."""
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError("features must be (n_patch, n_feature)")
    if f.shape[0] <= f.shape[1]:
        raise ValueError(
            f"more patches than features are needed: {f.shape[0]} patches, {f.shape[1]} features"
        )
    return MVGModel(
        nu=f.mean(axis=0),
        sigma=np.cov(f, rowvar=False),
        n_patches=int(f.shape[0]),
        n_images=int(n_images),
        n_patients=int(n_patients),
        meta=dict(meta or {}),
    )


def mahalanobis_mixed(
    nu1: np.ndarray, sigma1: np.ndarray, nu2: np.ndarray, sigma2: np.ndarray, rcond: float = RCOND
) -> float:
    """Equation (10): distance with averaged covariance."""
    d = np.asarray(nu1, dtype=np.float64) - np.asarray(nu2, dtype=np.float64)
    s = (np.asarray(sigma1, dtype=np.float64) + np.asarray(sigma2, dtype=np.float64)) / 2.0
    val = float(d @ np.linalg.pinv(s, rcond=rcond) @ d)
    return float(np.sqrt(max(val, 0.0)))


def score(model: MVGModel, features: np.ndarray, rcond: float = RCOND) -> float:
    """RIQE score of an image from its patches (without selection)."""
    f = np.asarray(features, dtype=np.float64)
    if f.shape[0] <= 1:
        return float("nan")
    return mahalanobis_mixed(model.nu, model.sigma, f.mean(axis=0), np.cov(f, rowvar=False), rcond)


def model_divergence(a: MVGModel, b: MVGModel, rcond: float = RCOND) -> float:
    """Primary divergence between two models: the same D as the score."""
    return mahalanobis_mixed(a.nu, a.sigma, b.nu, b.sigma, rcond)


def symmetric_kl(a: MVGModel, b: MVGModel, rcond: float = RCOND) -> float:
    """Symmetrised (Jeffreys) KL divergence between the two Gaussians."""
    d = a.dim
    sa, sb = np.asarray(a.sigma), np.asarray(b.sigma)
    ia, ib = np.linalg.pinv(sa, rcond=rcond), np.linalg.pinv(sb, rcond=rcond)
    dm = a.nu - b.nu
    kl_ab = 0.5 * (np.trace(ib @ sa) + dm @ ib @ dm - d + _logdet(sb) - _logdet(sa))
    kl_ba = 0.5 * (np.trace(ia @ sb) + dm @ ia @ dm - d + _logdet(sa) - _logdet(sb))
    return float(kl_ab + kl_ba)


def _logdet(s: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(np.asarray(s, dtype=np.float64))
    return float(ld) if sign > 0 else float("nan")


def feature_shift(a: MVGModel, b: MVGModel) -> np.ndarray:
    """Standardised per-feature shift, (nu_a - nu_b) / averaged sd.

    Tells *which* of the 36 components move between two models, which the
    aggregate D hides.
    """
    sd = np.sqrt((np.diag(a.sigma) + np.diag(b.sigma)) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (a.nu - b.nu) / sd, np.nan)
