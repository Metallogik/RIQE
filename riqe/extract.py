"""Pipeline completa: HU -> luminanza -> maschere -> 36 feature per patch.

Un unico punto di ingresso, cosi' che fitting e valutazione non possano
divergere per sbaglio: l'unica differenza ammessa fra i due e' il parametro
`p`, che in valutazione deve essere None come prescrive l'articolo NIQE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hu import HU_HI, HU_LO, domain_mask, to_luminance
from .nss import GAUSS_SIGMA, N_FEATURES, patch_features


@dataclass(frozen=True)
class Spec:
    """Specifica completa del modello: tutto cio' che cambia i numeri."""

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
        """Chiave delle sole componenti che cambiano le feature per patch
        (la soglia p agisce solo sulla selezione, a valle)."""
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
    """Feature di una slice.

    Parametri
    ---------
    fitting : True per il fitting del corpus (applica la selezione per
              nitidezza), False per il punteggio (non la applica).
    masks   : (fov, body) precalcolate.  Vanno passate quando si valutano
              versioni degradate della stessa slice: la maschera deve
              restare quella dell'originale, altrimenti una degradazione che
              sposta la soglia del corpo cambia anche il dominio e si
              confonde l'effetto della degradazione con quello del dominio.

    Ritorna un PatchFeatures; `.selected` sono le patch da usare.
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


#: numero minimo di patch per stimare una covarianza 36x36 non degenere.
#: Con meno di questo il punteggio non e' definibile e va riportato NaN.
MIN_PATCHES_FOR_SCORE = 2 * N_FEATURES
