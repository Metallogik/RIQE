"""Degradazioni e filtri, con scale dichiarate, piu' il surrogato di
rilevabilita' di lesioni a basso contrasto.

Tutto opera in unita' Hounsfield, **prima** della mappatura di intensita':
e' l'unico ordine fisicamente sensato, perche' rumore e filtraggio agiscono
sull'immagine ricostruita, non sulla sua rappresentazione a 8 bit.

I denoiser vengono da scikit-image (BSD-3).  Niente BM3D: le
implementazioni disponibili sono GPL o di licenza incerta, e il vincolo di
licenza del progetto e' non negoziabile.
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
# Scale di degradazione
# ---------------------------------------------------------------------------

NOISE_SIGMAS_HU = (5.0, 10.0, 20.0, 40.0, 80.0)
BLUR_SIGMAS_PX = (0.5, 1.0, 1.5, 2.0, 3.0)

#: scala fine a partire da zero, per la forma 3 del test del sovrafiltraggio:
#: serve a cercare un eventuale minimo del punteggio a rumore non nullo
NOISE_FINE_HU = (0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 65.0, 90.0)


def add_white_noise(hu: np.ndarray, sigma_hu: float, rng: np.random.Generator) -> np.ndarray:
    if sigma_hu <= 0:
        return hu.astype(np.float32, copy=True)
    return (hu + rng.normal(0.0, sigma_hu, hu.shape)).astype(np.float32)


def _ramp_nps_filter(shape: tuple[int, int], cutoff: float = 0.55) -> np.ndarray:
    """Filtro radiale tipo retroproiezione filtrata.

    Il rumore della FBP non e' bianco: la rampa |f| del filtro di
    ricostruzione ne sposta la potenza verso le frequenze medie, con una
    caduta alla frequenza di taglio del kernel.  Approssimiamo il modulo
    dello spettro con |f| moltiplicato per una finestra di Hann che si
    annulla a `cutoff` (in frequenza di Nyquist).
    """
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2) / 0.5  # normalizzata a Nyquist = 1
    w = np.where(r <= cutoff, 0.5 * (1.0 + np.cos(np.pi * r / cutoff)), 0.0)
    return r * w


_NPS_CACHE: dict[tuple[int, int, float], np.ndarray] = {}


def add_fbp_noise(
    hu: np.ndarray, sigma_hu: float, rng: np.random.Generator, cutoff: float = 0.55
) -> np.ndarray:
    """Rumore correlato con spettro tipo FBP, riscalato a sigma_hu complessiva."""
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
# Denoiser, con scale di intensita' crescente
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


#: nome -> (funzione, estremi dell'intervallo di ricerca della forza)
#: Il parametro nativo di ciascun filtro non e' confrontabile con quello
#: degli altri: `bilateral` a sigma_color 10 tocca l'immagine dieci volte
#: piu' di `tv` a weight 2.  La forza si calibra quindi per immagine, per
#: colpire una deviazione standard del residuo dichiarata (vedi
#: `calibrate_strength`), e le scale sono indicizzate da quella.
DENOISERS: dict[str, tuple] = {
    "gaussian": (_f_gaussian, (0.05, 8.0)),
    "tv": (_f_tv, (0.05, 2000.0)),
    "bilateral": (_f_bilateral, (0.5, 4000.0)),
    "nlm": (_f_nlm, (0.2, 2000.0)),
    "wavelet": (_f_wavelet, (0.2, 2000.0)),
}

#: scala comune di intensita' di filtraggio, in deviazione standard del
#: residuo dentro il corpo (HU).  E' l'asse x delle figure principali:
#: fisicamente interpretabile e indipendente dal filtro.
RESIDUAL_LEVELS_HU = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def apply_denoiser(hu: np.ndarray, name: str, strength) -> np.ndarray:
    fn, _ = DENOISERS[name]
    return fn(hu, strength)


def residual_std(original: np.ndarray, filtered: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Deviazione standard del residuo, l'unita' comune per appaiare filtri."""
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
    """Forza del filtro che produce il residuo richiesto, per bisezione in
    scala logaritmica.

    Ritorna (forza, residuo_ottenuto, n_iterazioni).  Se il bersaglio e'
    fuori dall'intervallo raggiungibile dal filtro su questa immagine,
    ritorna l'estremo piu' vicino e il residuo corrispondente: il chiamante
    lo vede dal residuo e lo riporta, invece di far finta di averlo colpito.
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
        m = float(np.sqrt(a * b))  # bisezione geometrica
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
    """Applica il filtro alla forza calibrata.  Ritorna (immagine, info)."""
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
# Surrogato di rilevabilita': lesioni inserite e filtro adattato NPW
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Lesion:
    cy: int
    cx: int
    radius_px: float
    contrast_hu: float
    diameter_mm: float


def _disc(shape, cy, cx, radius_px, edge_px: float = 0.7):
    """Disco con bordo sfumato su ~1 px, per non introdurre un gradino
    artificiale che sarebbe piu' facile da rilevare di una lesione vera."""
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
    """Siti di parenchima omogeneo dove inserire lesioni o misurare controlli.

    Criteri: media locale in `hu_range` (tessuto molle / parenchima epatico),
    deviazione standard locale sotto la mediana dell'immagine, e distanza dal
    bordo del corpo di almeno un diametro.
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
    """Template del filtro adattato: il profilo noto della lesione, a media
    nulla, normalizzato.  Non pre-imbianca (NPW) e non applica filtro
    oculare: quest'ultimo richiederebbe ipotesi arbitrarie su distanza di
    visione e caratteristiche del display, indifendibili in un lavoro che
    non fa validazione con lettori umani.
    """
    n = 2 * half + 1
    t = _disc((n, n), half, half, radius_px)
    t = t - t.mean()
    nrm = np.sqrt((t ** 2).sum())
    return t / nrm if nrm > 0 else t


def npw_response(img: np.ndarray, sites, radius_px: float) -> np.ndarray:
    """Risposta del filtro adattato NPW nei siti indicati."""
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
    """d\' del filtro adattato NPW, con la varianza presa **sulle
    realizzazioni di rumore allo stesso sito**.

    Perche' cosi' e non confrontando siti diversi: la risposta del template
    su siti anatomici diversi varia soprattutto per la variabilita'
    dell'anatomia, che e' di ordini di grandezza maggiore del segnale di una
    lesione a basso contrasto.  Un d\' costruito su quella varianza misura
    quanto e' eterogeneo il fegato, non quanto e' rilevabile la lesione.
    Confrontando invece lo stesso sito fra presenza e assenza di lesione su
    realizzazioni di rumore indipendenti, il termine anatomico si cancella
    esattamente e resta solo il rumore, che e' cio' che il filtro modifica.

    Procedura, per ciascuna delle `n_realizations` realizzazioni:
      1. genera una realizzazione di rumore a `noise_sigma_hu`;
      2. costruisce l'immagine con e senza lesioni, stesso rumore;
      3. applica lo stesso filtro `filt` a entrambe (None = nessun filtro);
      4. registra la risposta del template nei siti.

    Poi, per ogni sito:
      Delta = media(lambda_presenza - lambda_assenza)   (segnale residuo
              dopo il filtro, che cattura la perdita di contrasto)
      sigma = deviazione standard di lambda_assenza sulle realizzazioni
      d' del sito = Delta / sigma
    e d\' complessivo = media sui siti.

    Ritorna (d_prime, dettagli).
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

    # Ritenzione del segnale: quanta ampiezza della lesione sopravvive al
    # filtro.  Serve perche' il d' del filtro adattato con template noto e'
    # poco sensibile al lisciamento (il filtro ottimale riduce segnale e
    # rumore nella stessa banda), mentre la perdita di ampiezza e' cio' che
    # cancella una lesione all'occhio.  Le due misure vanno riportate
    # insieme: se divergono, e' informazione, non rumore.
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
