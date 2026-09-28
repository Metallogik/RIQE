"""Banco di immagini di prova, definito in modo deterministico.

Una voce del banco descrive una trasformazione di una slice sorgente in modo
completamente riproducibile: tipo, parametri, seme del generatore, forza
calibrata del filtro.  L'immagine non viene salvata: viene **rigenerata** da
`render`, identica bit per bit, ogni volta che serve.

Motivo: la ricerca degli iperparametri deve valutare lo stesso insieme di
immagini sotto 12 combinazioni (P, C) diverse, e la batteria di validazione
deve valutarlo ancora.  Salvare le immagini costerebbe decine di GB;
salvarne la ricetta costa kilobyte, e rende la riproduzione da parte di terzi
un fatto verificabile invece di una promessa.

La calibrazione della forza dei filtri e' la parte costosa (bisezione con
molte applicazioni del filtro) e **non dipende da (P, C)**: si fa una volta,
si registra nel banco, e non si rifa'.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from . import degrade as dg


@dataclass(frozen=True)
class BankEntry:
    """Ricetta di una singola immagine di prova."""

    kind: str                     # original | noise_white | noise_fbp | blur | denoise
    source: str = "full"          # full | low  (quale slice del paziente)
    sigma_hu: float | None = None
    sigma_px: float | None = None
    seed: int | None = None
    denoiser: str | None = None
    target_residual_hu: float | None = None
    strength: float | None = None
    residual_hu: float | None = None
    target_reached: bool | None = None

    def label(self) -> str:
        if self.kind == "original":
            return f"{self.source}:original"
        if self.kind in ("noise_white", "noise_fbp"):
            return f"{self.source}:{self.kind}:{self.sigma_hu:g}HU"
        if self.kind == "blur":
            return f"{self.source}:blur:{self.sigma_px:g}px"
        return f"{self.source}:{self.denoiser}:res{self.target_residual_hu:g}HU"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def render(entry: BankEntry, hu: np.ndarray) -> np.ndarray:
    """Rigenera l'immagine della voce dalla slice sorgente in HU."""
    if entry.kind == "original":
        return hu.astype(np.float32, copy=True)
    if entry.kind == "noise_white":
        return dg.add_white_noise(hu, entry.sigma_hu, np.random.default_rng(entry.seed))
    if entry.kind == "noise_fbp":
        return dg.add_fbp_noise(hu, entry.sigma_hu, np.random.default_rng(entry.seed))
    if entry.kind == "blur":
        return dg.blur(hu, entry.sigma_px)
    if entry.kind == "denoise":
        if entry.strength is None:
            raise ValueError("voce denoise senza forza calibrata")
        return dg.apply_denoiser(hu, entry.denoiser, entry.strength)
    raise ValueError(f"tipo di voce sconosciuto: {entry.kind}")


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
) -> list[BankEntry]:
    """Costruisce il banco per una slice, calibrando le forze dei filtri.

    E' la funzione costosa: va chiamata una volta per slice, e il risultato
    va serializzato.
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
