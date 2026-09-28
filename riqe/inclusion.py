"""Criteri di inclusione del corpus pristine, eseguibili e verificabili.

Disegno: una sola passata calcola per ogni slice un vettore di statistiche
oggettive (`slice_stats`); i criteri S1-S6 sono poi **operazioni pure sulla
tabella** delle statistiche (`apply_criteria`).  Cosi' i criteri si possono
rieseguire, cambiare di soglia e verificare senza rileggere i DICOM, e il
conteggio degli esclusi per criterio finisce nell'artefatto modello.

Nessun criterio "a occhio": ogni regola e' una funzione booleana con soglia
dichiarata.  Sul moto sono esplicito: non lo rileviamo direttamente, S1 e S5
lo mitigano, e la limitazione va dichiarata nell'articolo.  L'alternativa
"fitta un modello preliminare ed escludi le slice con punteggio alto" e'
stata scartata perche' circolare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hu import BODY_HU, domain_mask
from .nss import local_stats

# ---- soglie dichiarate -----------------------------------------------------

#: S1: frazione di slice scartata a ciascun estremo della serie
S1_TRIM = 0.10
#: S2: frazione ammessa di area del corpo rispetto all'area del FOV
S2_BODY_FRAC = (0.15, 0.85)
#: S3: spessore in pixel dell'anello interno al bordo FOV, e frazione ammessa
S3_RING_PX = 3
S3_MAX_ON_RING = 0.02
#: S4: soglia HU del metallo e frazione ammessa di pixel del corpo oltre.
#: Calibrata per ispezione visiva su campioni stratificati (docs/04):
#: 1500 HU e' osso corticale denso, non metallo -- il 43% delle slice supera
#: quel valore e la mediana del massimo HU nel corpo e' 1471.  A 2000 HU con
#: frazione 1e-4 si escludono il 3,1% delle slice, restano tutti i 199
#: pazienti con almeno 58 slice ciascuno, e la regola e' deliberatamente
#: conservativa: il corpus e' dieci volte piu' grande del necessario, quindi
#: si preferisce perdere qualche slice di osso denso senza artefatti che
#: contaminare il pristine.
S4_METAL_HU = 2000.0
S4_MAX_FRAC = 1e-4
#: S5: DISATTIVATO come criterio di esclusione.  La statistica resta
#: calcolata e registrata, ma non esclude nulla.
#:
#: Era pensata come rilevatore di strisce da metallo e di moto: sigma locale
#: mediana nell'aria, z robusto per cella.  La calibrazione l'ha falsificata,
#: due volte.  Nella prima forma scattava su un solo paziente corpulento, con
#: immagini pulite, perche' con poca aria attorno al corpo la statistica si
#: calcolava su pochi pixel al bordo del FOV.  Ristretta all'aria lontana dal
#: corpo e con un minimo di 5000 pixel, il caso limite resta lo stesso: i sei
#: valori piu' alti (z ~ 7,5-7,9) sono un paziente con body_frac_fov = 0,94,
#: cioe' corporatura, non artefatti -- e sono gia' esclusi da S2.
#:
#: Prova decisiva in senso contrario: sulle slice con protesi d'anca
#: bilaterali, il caso di metallo piu' netto del corpus, lo z della sigma nel
#: corpo vale da -1,0 a -1,6, cioe' va nella direzione **sbagliata** (il
#: bacino e' una grande regione omogenea di tessuti molli).
#:
#: Tenerlo attivo significherebbe spacciare per rilevatore di artefatti una
#: misura di corporatura.  Resta quindi registrato come diagnostica, e il
#: limite -- moto e strisce non rilevati direttamente -- e' dichiarato
#: nell'articolo.  Cfr. docs/04-calibrazione-criteri.md.
S5_ENABLED = False
S5_MAX_Z = 4.0
S5_MIN_AIR_PX = 5000
#: S6: slice trattenute per paziente
S6_PER_PATIENT = 24


def slice_stats(hu: np.ndarray, padding_value: float | None) -> dict:
    """Statistiche oggettive di una slice, indipendenti dalle soglie."""
    dom, fov, body = domain_mask(hu, padding_value)
    n_fov = int(fov.sum())
    n_body = int(body.sum())
    out = {
        "n_fov": n_fov,
        "n_body": n_body,
        "body_frac_fov": n_body / n_fov if n_fov else 0.0,
        "metal_frac": 0.0,
        "frac_gt_1500": 0.0,
        "frac_gt_2000": 0.0,
        "frac_gt_2500": 0.0,
        "frac_gt_3000": 0.0,
        "hu_max_body": np.nan,
        "hu_p9999_body": np.nan,
        "ring_frac": 0.0,
        "n_air": 0,
        "air_sigma_med": np.nan,
        "body_sigma_med": np.nan,
        "hu_clip_hi_frac": 0.0,
    }
    if n_body == 0:
        return out

    hu_body = hu[body]
    out["metal_frac"] = float((hu_body > S4_METAL_HU).mean())
    out["hu_clip_hi_frac"] = float((hu_body > 1000.0).mean())
    # soglie multiple: 1500 HU confonde l'osso corticale denso col metallo,
    # che sta molto piu' in alto.  La soglia si scegliera' dai dati.
    for t in (1500, 2000, 2500, 3000):
        out[f"frac_gt_{t}"] = float((hu_body > t).mean())
    out["hu_max_body"] = float(hu_body.max())
    out["hu_p9999_body"] = float(np.percentile(hu_body, 99.99))

    # S3: corpo che tocca il bordo del campo ricostruito
    from scipy.ndimage import binary_erosion

    inner = binary_erosion(fov, np.ones((2 * S3_RING_PX + 1, 2 * S3_RING_PX + 1), dtype=bool))
    ring = fov & ~inner
    out["ring_frac"] = float((body & ring).sum() / n_body)

    # S5: rumore nell'aria dentro il FOV e fuori dal corpo
    _, sigma = local_stats(hu)
    # solo aria ben fuori dal corpo: vicino alla pelle la ricostruzione e'
    # rumorosa per ragioni che non c'entrano con gli artefatti
    from scipy.ndimage import binary_dilation

    near_body = binary_dilation(body, np.ones((15, 15), dtype=bool))
    air = fov & ~near_body & (hu < -700.0)
    out["n_air"] = int(air.sum())
    if out["n_air"] >= S5_MIN_AIR_PX:
        out["air_sigma_med"] = float(np.median(sigma[air]))
    out["body_sigma_med"] = float(np.median(sigma[body]))
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    """z basato su mediana e MAD, robusto agli outlier che deve trovare."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


def apply_criteria(
    df: pd.DataFrame,
    cell_col: str = "cell",
    per_patient: int = S6_PER_PATIENT,
) -> pd.DataFrame:
    """Applica S1-S6.  Ritorna il dataframe con una colonna booleana per
    criterio, `keep` finale e `exclude_reason` del primo criterio fallito.

    Attende le colonne: patient_id, series_uid, sop_uid, z, cell, e le
    statistiche prodotte da `slice_stats`.
    """
    d = df.copy()

    # S1 - estremi di serie, sull'ordinamento anatomico per z
    d = d.sort_values(["series_uid", "z"]).reset_index(drop=True)
    rank = d.groupby("series_uid").cumcount()
    n = d.groupby("series_uid")["z"].transform("size")
    frac = rank / np.maximum(n - 1, 1)
    d["S1_interior"] = (frac >= S1_TRIM) & (frac <= 1.0 - S1_TRIM)

    # S2 - anatomia sufficiente
    lo, hi = S2_BODY_FRAC
    d["S2_anatomy"] = d["body_frac_fov"].between(lo, hi)

    # S3 - troncamento
    d["S3_untruncated"] = d["ring_frac"] <= S3_MAX_ON_RING

    # S4 - metallo.  Si usa la colonna esplicita alla soglia dichiarata, non
    # `metal_frac`: quest'ultima e' calcolata al momento della passata sui
    # DICOM e resterebbe legata al valore di S4_METAL_HU vigente allora,
    # rendendo il criterio silenziosamente incoerente se la soglia cambia.
    metal_col = f"frac_gt_{int(S4_METAL_HU)}"
    if metal_col not in d.columns:
        raise KeyError(
            f"la tabella delle slice non ha {metal_col}: rieseguire "
            f"build_slice_table.py dopo aver cambiato S4_METAL_HU"
        )
    d["S4_no_metal"] = d[metal_col] <= S4_MAX_FRAC

    # S5 - artefatti a striscia / moto, soglia robusta per cella
    z = np.full(len(d), np.nan)
    for _, idx in d.groupby(cell_col).groups.items():
        pos = d.index.get_indexer(idx)
        z[pos] = _robust_z(d.loc[idx, "air_sigma_med"].to_numpy())
    d["air_sigma_z"] = z
    if S5_ENABLED:
        d["S5_no_streak"] = ~(d["air_sigma_z"] > S5_MAX_Z)
        d.loc[d["air_sigma_med"].isna(), "S5_no_streak"] = False
    else:
        # disattivato: la statistica resta registrata ma non esclude
        d["S5_no_streak"] = True

    surviving = (
        d["S1_interior"] & d["S2_anatomy"] & d["S3_untruncated"] & d["S4_no_metal"] & d["S5_no_streak"]
    )

    # S6 - campionamento equispaziato fra le sopravvissute, per paziente
    d["S6_sampled"] = False
    for pid, g in d[surviving].groupby("patient_id"):
        g = g.sort_values("z")
        k = min(per_patient, len(g))
        if k == 0:
            continue
        pick = np.linspace(0, len(g) - 1, k).round().astype(int)
        d.loc[g.index[np.unique(pick)], "S6_sampled"] = True

    d["keep"] = surviving & d["S6_sampled"]

    order = ["S1_interior", "S2_anatomy", "S3_untruncated", "S4_no_metal", "S5_no_streak", "S6_sampled"]
    reason = pd.Series("", index=d.index, dtype=object)
    for c in order:
        reason = reason.where(reason.ne("") | d[c], c)
    d["exclude_reason"] = reason
    return d


def exclusion_report(d: pd.DataFrame) -> pd.DataFrame:
    """Conteggio degli esclusi per criterio, in cascata (primo che fallisce)."""
    order = ["S1_interior", "S2_anatomy", "S3_untruncated", "S4_no_metal", "S5_no_streak", "S6_sampled"]
    rows = []
    remaining = len(d)
    alive = pd.Series(True, index=d.index)
    for c in order:
        failed = int((alive & ~d[c]).sum())
        alive &= d[c]
        rows.append({"criterio": c, "esclusi": failed, "residui": int(alive.sum())})
        remaining = int(alive.sum())
    rows.append({"criterio": "TOTALE trattenute", "esclusi": len(d) - remaining, "residui": remaining})
    return pd.DataFrame(rows)
