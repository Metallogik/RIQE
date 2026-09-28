"""Feature NSS in stile NIQE, implementate dall'articolo primario.

Fonte dell'algoritmo: A. Mittal, R. Soundararajan, A. C. Bovik, "Making a
'Completely Blind' Image Quality Analyzer", IEEE Signal Processing Letters
20(3):209-212, 2013.  Stima dei parametri GGD per momento secondo
K. Sharifi, A. Leon-Garcia, IEEE TCSVT 5(1):52-56, 1995 (rif. [14]
dell'articolo NIQE); stima AGGD secondo N.-E. Lasmar, Y. Stitou,
Y. Berthoumieu, ICIP 2009 (rif. [15]).

Nessun codice del pacchetto MATLAB di LIVE e' stato consultato o tradotto.

Scelte che l'articolo lascia implicite, qui esplicite e registrate
nell'artefatto modello (cfr. docs/02-spec-niqe.md):

  * sigma del kernel gaussiano: 1.0 su finestra 7x7.  L'articolo dice
    "sampled out to 3 standard deviations (K = L = 3)", che con semiampiezza
    3 implica letteralmente sigma = 1.0.
  * costante di stabilizzazione C in (I - mu)/(sigma + C): nell'articolo e'
    fissata a 1 su immagini 0-255.  Qui e' un parametro dichiarato, perche'
    su TC la sigma locale in unita' di codice e' molto piu' piccola che in
    fotografia e C = 1 renderebbe la normalizzazione divisiva parziale e
    dipendente dal protocollo (cfr. docs/01-proposta.md 2.4).
  * filtro passa-basso della seconda scala: lo stesso kernel gaussiano,
    seguito da decimazione 2:1.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from scipy.ndimage import correlate1d
from scipy.special import gammaln

# --------------------------------------------------------------------------
# Kernel gaussiano dell'articolo: 7x7, sigma = 1.0, volume unitario
# --------------------------------------------------------------------------

GAUSS_SIGMA = 1.0
GAUSS_HALFWIDTH = 3  # K = L = 3


def gaussian_weights(sigma: float = GAUSS_SIGMA, halfwidth: int = GAUSS_HALFWIDTH):
    """Kernel 1-D separabile equivalente al kernel 2-D a simmetria circolare.

    Un gaussiano 2-D a simmetria circolare e' separabile, quindi il kernel
    2-D normalizzato a volume unitario si ottiene dal prodotto esterno del
    kernel 1-D normalizzato a somma unitaria.
    """
    x = np.arange(-halfwidth, halfwidth + 1, dtype=np.float64)
    w = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return w / w.sum()


_W1D = gaussian_weights()


def _sepconv(img: np.ndarray, w: np.ndarray = _W1D) -> np.ndarray:
    """Convoluzione separabile con estensione a riflessione dei bordi."""
    out = correlate1d(img, w, axis=0, mode="reflect")
    return correlate1d(out, w, axis=1, mode="reflect")


def local_stats(img: np.ndarray):
    """Media e deviazione standard locali pesate, formule (2) e (3)."""
    img = np.asarray(img, dtype=np.float64)
    mu = _sepconv(img)
    mu_sq = _sepconv(img * img)
    var = mu_sq - mu * mu
    np.clip(var, 0.0, None, out=var)
    return mu, np.sqrt(var)


def mscn(img: np.ndarray, C: float = 1.0):
    """Coefficienti MSCN, formula (1).  Ritorna (mscn, campo sigma)."""
    mu, sigma = local_stats(img)
    return (np.asarray(img, dtype=np.float64) - mu) / (sigma + C), sigma


# --------------------------------------------------------------------------
# Inversione dei rapporti di momenti per GGD e AGGD
#
# Entrambe le stime richiedono di invertire una funzione monotona di alpha
# fatta di funzioni gamma.  Tabuliamo su una griglia fitta e interpoliamo:
# l'inversione e' esatta entro il passo della griglia, e il passo e'
# registrato nell'artefatto.
# --------------------------------------------------------------------------

_ALPHA_MIN, _ALPHA_MAX, _ALPHA_N = 0.05, 20.0, 40001
_ALPHA_GRID = np.linspace(_ALPHA_MIN, _ALPHA_MAX, _ALPHA_N)


def _lgam(x):
    return gammaln(x)


# GGD:  rho(alpha) = Gamma(1/a) Gamma(3/a) / Gamma(2/a)^2,  decrescente in a
_GGD_RHO = np.exp(
    _lgam(1.0 / _ALPHA_GRID) + _lgam(3.0 / _ALPHA_GRID) - 2.0 * _lgam(2.0 / _ALPHA_GRID)
)
# per np.interp servono ascisse crescenti: rho decresce, quindi invertiamo
_GGD_RHO_ASC = _GGD_RHO[::-1]
_GGD_ALPHA_ASC = _ALPHA_GRID[::-1]

# AGGD:  rho(alpha) = Gamma(2/a)^2 / (Gamma(1/a) Gamma(3/a)),  crescente in a
_AGGD_RHO = 1.0 / _GGD_RHO


def _invert_ggd(rho: np.ndarray) -> np.ndarray:
    return np.interp(rho, _GGD_RHO_ASC, _GGD_ALPHA_ASC)


def _invert_aggd(rho: np.ndarray) -> np.ndarray:
    return np.interp(rho, _AGGD_RHO, _ALPHA_GRID)


def _beta_scale(alpha: np.ndarray) -> np.ndarray:
    """sqrt(Gamma(1/a) / Gamma(3/a)), fattore che porta da sigma a beta."""
    return np.exp(0.5 * (_lgam(1.0 / alpha) - _lgam(3.0 / alpha)))


# --------------------------------------------------------------------------
# Riduzioni a blocchi: tutti i momenti per patch in forma vettorizzata
# --------------------------------------------------------------------------

def _blocks(a: np.ndarray, P: int) -> np.ndarray:
    """Vista (n_patch, P*P) dei blocchi non sovrapposti PxP."""
    h, w = a.shape
    nh, nw = h // P, w // P
    return (
        a[: nh * P, : nw * P]
        .reshape(nh, P, nw, P)
        .transpose(0, 2, 1, 3)
        .reshape(nh * nw, P * P)
    )


def _ggd_features_blocks(blk: np.ndarray):
    """(alpha, beta) per blocco da una GGD a media nulla.

    Sharpe/Leon-Garcia per momenti: rho = E[x^2] / E[|x|]^2 determina alpha,
    poi beta = sqrt(E[x^2]) * sqrt(Gamma(1/a)/Gamma(3/a)).
    """
    m2 = np.mean(blk * blk, axis=1)
    m1 = np.mean(np.abs(blk), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(m1 > 0, m2 / (m1 * m1), np.inf)
    alpha = _invert_ggd(rho)
    beta = np.sqrt(m2) * _beta_scale(alpha)
    return alpha, beta


def _aggd_features_blocks(blk: np.ndarray):
    """(gamma, beta_l, beta_r, eta) per blocco da una AGGD a moda nulla.

    Lasmar/Stitou/Berthoumieu per momenti:
        sigma_l^2 = E[x^2 | x < 0],  sigma_r^2 = E[x^2 | x >= 0]
        gamma_hat = sigma_l / sigma_r
        r_hat     = E[|x|]^2 / E[x^2]
        R_hat     = r_hat (gamma_hat^3 + 1)(gamma_hat + 1) / (gamma_hat^2 + 1)^2
    e R_hat = Gamma(2/g)^2 / (Gamma(1/g) Gamma(3/g)) da invertire in g.
    Poi beta_l = sigma_l sqrt(Gamma(1/g)/Gamma(3/g)), idem per beta_r, e
        eta = (beta_r - beta_l) Gamma(2/g) / Gamma(1/g)      (formula (8))
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
# Feature per patch: 18 per scala
# --------------------------------------------------------------------------

#: prodotti di coppie adiacenti, come nell'articolo:
#: orizzontale, verticale, diagonale principale, diagonale secondaria
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
    """m(i,j) * m(i+di, j+dj), con estensione a riflessione dei bordi."""
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
    """Feature a una scala.

    Ritorna (feat, delta, valid):
      feat  (n_patch, 18)  le 18 feature per patch
      delta (n_patch,)     nitidezza di patch, somma di sigma (formula (4))
      valid (n_patch,)     ammissibilita' del dominio (vedi sotto)

    Il dominio ha due requisiti asimmetrici, per la ragione spiegata in
    docs/01-proposta.md 2.3:

      * FOV al 100%: un solo pixel di riempimento fuori campo e' fatale,
        perche' e' una costante artificiale (-3024 HU sulle GE) che azzera la
        varianza locale e falsa la statistica.
      * corpo a una frazione dichiarata `body_min`: qualche pixel d'aria al
        confine cutaneo e' anatomia reale, non un artefatto, e escluderlo
        eliminerebbe tutte le patch di superficie.
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
    """Filtro passa-basso (stesso kernel) e decimazione 2:1."""
    return _sepconv(np.asarray(img, dtype=np.float64))[::2, ::2]


def downscale_mask(mask: np.ndarray) -> np.ndarray:
    """Decimazione della maschera per erosione: la patch di scala 2 e' valida
    solo se tutti i quattro pixel originali corrispondenti lo sono."""
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
    """36 feature per patch, due scale.

    Parametri
    ---------
    img  : mappa di luminanza sulla scala 0-255 (float, non quantizzata)
    P    : lato della patch alla prima scala; alla seconda e' P/2
    C    : costante di stabilizzazione MSCN
    fov  : campo ricostruito, requisito al 100%
    body : maschera del corpo, requisito alla frazione `body_min`
    p    : se dato, applica la selezione per nitidezza delta > p * max(delta)
           fra le patch valide.  In valutazione va lasciato None, come
           prescrive l'articolo.

    Ritorna (feat, keep) con feat (n_patch, 36) e keep (n_patch,) booleano.
    """
    if P % 2:
        raise ValueError("P deve essere pari per ottenere P/2 alla seconda scala")
    f1, delta, v1 = scale_features(img, P, C, fov, body, body_min)
    img2 = downscale(img)
    fov2 = None if fov is None else downscale_mask(fov)
    body2 = None if body is None else downscale_mask(body)
    f2, _, v2 = scale_features(img2, P // 2, C, fov2, body2, body_min)

    # le due griglie hanno lo stesso numero di patch per costruzione, ma le
    # dimensioni dispari possono far perdere una riga o colonna: tronchiamo
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
    """Feature per patch di una slice, con i due livelli di ammissibilita'.

    feat  (n_patch, 36)
    delta (n_patch,)      nitidezza di patch, formula (4)
    valid (n_patch,)      dentro il dominio (FOV, corpo) e finita
    keep  (n_patch,)      valid, piu' la selezione per nitidezza se richiesta
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
