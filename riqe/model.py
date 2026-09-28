"""Modello MVG, punteggio e divergenza fra modelli.

Punteggio, formula (10) dell'articolo NIQE:

    D = sqrt( (nu1 - nu2)^T ((Sigma1 + Sigma2)/2)^-1 (nu1 - nu2) )

La stessa forma funzionale serve da metrica di divergenza fra due modelli
(cfr. docs/01-proposta.md 5.2): la divergenza si legge cosi' nelle stesse
unita' dei punteggi che il modello produce.

L'inversa e' sempre una pseudo-inversa per SVD con soglia relativa
dichiarata: con P piccolo o con poche patch la covarianza mediata puo'
essere mal condizionata, e un ridge implicito nascosto renderebbe i numeri
non riproducibili.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

#: soglia relativa dei valori singolari per la pseudo-inversa
RCOND = 1e-10


@dataclass
class MVGModel:
    """Modello gaussiano multivariato a 36 dimensioni."""

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
    """Fit di media e covarianza su un insieme di patch (n_patch, 36)."""
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError("features deve essere (n_patch, n_feature)")
    if f.shape[0] <= f.shape[1]:
        raise ValueError(
            f"servono piu' patch che feature: {f.shape[0]} patch, {f.shape[1]} feature"
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
    """Formula (10): distanza con covarianza mediata."""
    d = np.asarray(nu1, dtype=np.float64) - np.asarray(nu2, dtype=np.float64)
    s = (np.asarray(sigma1, dtype=np.float64) + np.asarray(sigma2, dtype=np.float64)) / 2.0
    val = float(d @ np.linalg.pinv(s, rcond=rcond) @ d)
    return float(np.sqrt(max(val, 0.0)))


def score(model: MVGModel, features: np.ndarray, rcond: float = RCOND) -> float:
    """Punteggio RIQE di un'immagine dalle sue patch (senza selezione)."""
    f = np.asarray(features, dtype=np.float64)
    if f.shape[0] <= 1:
        return float("nan")
    return mahalanobis_mixed(model.nu, model.sigma, f.mean(axis=0), np.cov(f, rowvar=False), rcond)


def model_divergence(a: MVGModel, b: MVGModel, rcond: float = RCOND) -> float:
    """Divergenza primaria fra due modelli: la stessa D del punteggio."""
    return mahalanobis_mixed(a.nu, a.sigma, b.nu, b.sigma, rcond)


def symmetric_kl(a: MVGModel, b: MVGModel, rcond: float = RCOND) -> float:
    """Divergenza KL simmetrizzata (Jeffreys) fra le due gaussiane."""
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
    """Scostamento standardizzato per feature, (nu_a - nu_b) / sd mediata.

    Dice *quali* delle 36 componenti si muovono fra due modelli, che la D
    aggregata nasconde.
    """
    sd = np.sqrt((np.diag(a.sigma) + np.diag(b.sigma)) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (a.nu - b.nu) / sd, np.nan)
