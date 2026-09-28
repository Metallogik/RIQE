"""Mappatura canonica HU -> luminanza e maschere di dominio.

Questa mappatura e' parte inseparabile della specifica del modello: finestre
diverse producono modelli fra loro incompatibili.  Cfr. docs/01-proposta.md 2.

Specifica congelata (RIQE v1):

    L = 255 * (clip(HU, HU_LO, HU_HI) - HU_LO) / (HU_HI - HU_LO)

con HU_LO = -1000, HU_HI = +1000, in float32 e senza arrotondamento.  La
scala 0-255 va conservata perche' la costante di stabilizzazione dell'MSCN e'
additiva assoluta e presuppone quella scala; la quantizzazione a 8 bit no,
non fa parte dell'algoritmo.

Le maschere non sono un accessorio.  Le immagini GE di questa collezione
portano 55772 pixel per slice (21,3% dell'area) al valore di riempimento
-3024 HU fuori dal campo ricostruito, le Siemens no: senza maschera FOV la
differenza fra costruttori che si misurerebbe e' una convenzione DICOM, non
la fisica dell'acquisizione.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening, label

#: finestra HU canonica del modello
HU_LO = -1000.0
HU_HI = 1000.0

#: soglia della maschera corpo, in HU
BODY_HU = -300.0
#: lato dell'elemento strutturante per l'apertura morfologica
BODY_OPEN = 5


def to_luminance(hu: np.ndarray, lo: float = HU_LO, hi: float = HU_HI) -> np.ndarray:
    """Mappa HU su 0-255 float32, con clip alla finestra canonica."""
    x = np.clip(np.asarray(hu, dtype=np.float32), lo, hi)
    return (255.0 * (x - lo) / (hi - lo)).astype(np.float32)


def fov_mask(hu: np.ndarray, padding_value: float | None) -> np.ndarray:
    """Pixel dentro il campo ricostruito.

    Il valore di riempimento dichiarato in PixelPaddingValue e' escluso.  In
    sua assenza si escludono i pixel sotto -1024 HU, che non sono anatomia:
    il pavimento fisico della scala e' -1000 e -1024 e' il minimo
    rappresentabile con RescaleIntercept = -1024.
    """
    hu = np.asarray(hu)
    if padding_value is not None:
        m = hu > float(padding_value)
    else:
        m = np.ones(hu.shape, dtype=bool)
    return m & (hu >= -1024.0)


def body_mask(hu: np.ndarray, fov: np.ndarray | None = None) -> np.ndarray:
    """Maschera del corpo: soglia, apertura, riempimento, componente maggiore."""
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
    """Dominio ammesso per le patch: FOV e corpo.  Ritorna (dominio, fov, corpo)."""
    fov = fov_mask(hu, padding_value)
    body = body_mask(hu, fov)
    return fov & body, fov, body
