"""Test della dose reale: il punteggio della dose ridotta deve essere peggiore.

La collezione fornisce, per 100 pazienti Siemens, la ricostruzione della
**stessa** acquisizione a dose ridotta reale (10% della dose di routine sul
torace, 25% sull'addome), sulla stessa griglia di z.  E' il solo test della
batteria con rumore di tessitura nativa invece che sintetica, ed e' il caso
d'uso per cui la metrica esiste: la TC a bassa dose.

Si calcola dalla cache delle feature, senza rileggere i DICOM: la cache
contiene, per ogni slice trattenuta di entrambe le dosi, le patch nel
dominio, che sono esattamente cio' che serve al punteggio (in valutazione la
selezione per nitidezza non si applica).

Aggiunto alla batteria dopo aver visto i dati di validazione (docs/05 §8):
senza, il criterio di scelta degli iperparametri era cieco alla proprieta'
piu' importante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cache import FeatureCache
from .extract import MIN_PATCHES_FOR_SCORE
from .model import RCOND, MVGModel, mahalanobis_mixed


def dose_pairs(corpus: pd.DataFrame, patient_ids) -> pd.DataFrame:
    """Coppie (dose piena, dose ridotta) fra slice trattenute, a z identica."""
    k = corpus[corpus["keep"] & corpus.patient_id.isin(set(patient_ids))]
    rows = []
    for pid, g in k.groupby("patient_id"):
        lo = g[g["kind"] == "low"]
        fu = g[g["kind"] == "full"]
        if lo.empty or fu.empty:
            continue
        fz = fu["z"].to_numpy()
        for r in lo.itertuples():
            j = int(np.abs(fz - r.z).argmin())
            if abs(fz[j] - r.z) <= 0.01:
                rows.append({
                    "patient_id": pid,
                    "cell": fu["cell"].iloc[j],
                    "region": "torace" if "CHEST" in fu["cell"].iloc[j] else "addome",
                    "z": float(r.z),
                    "path_full": fu["path"].iloc[j],
                    "path_low": r.path,
                })
    return pd.DataFrame(rows)


def image_moments(cache: FeatureCache, paths) -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
    """Media e covarianza delle patch nel dominio, per immagine."""
    out = {}
    for q in paths:
        f, _ = cache.slice_rows(q)
        if f.shape[0] < MIN_PATCHES_FOR_SCORE:
            out[q] = None
        else:
            f = f.astype(np.float64)
            out[q] = (f.mean(axis=0), np.cov(f, rowvar=False))
    return out


def dose_ordering(model: MVGModel, pairs: pd.DataFrame, moments: dict) -> pd.DataFrame:
    """Punteggi delle coppie; `corretto` = la dose ridotta e' peggiore."""
    d = pairs.copy()
    sf, sl = [], []
    for r in d.itertuples():
        a, b = moments.get(r.path_full), moments.get(r.path_low)
        sf.append(np.nan if a is None else mahalanobis_mixed(model.nu, model.sigma, a[0], a[1], RCOND))
        sl.append(np.nan if b is None else mahalanobis_mixed(model.nu, model.sigma, b[0], b[1], RCOND))
    d["score_full"] = sf
    d["score_low"] = sl
    # float, non bool: una coppia non punteggiabile e' NaN, non "sbagliata"
    ok = d["score_full"].notna() & d["score_low"].notna()
    d["corretto"] = np.where(ok, (d["score_low"] > d["score_full"]).astype(float), np.nan)
    return d


def summarize(d: pd.DataFrame) -> dict:
    ok = d.dropna(subset=["score_full", "score_low"])
    out = {"n_coppie": int(len(ok)), "n_non_punteggiabili": int(len(d) - len(ok))}
    for reg in ("torace", "addome"):
        s = ok[ok["region"] == reg]
        out[f"n_{reg}"] = int(len(s))
        out[f"corretto_{reg}"] = float((s["score_low"] > s["score_full"]).mean()) if len(s) else np.nan
    vals = [out[f"corretto_{r}"] for r in ("torace", "addome") if np.isfinite(out[f"corretto_{r}"])]
    out["corretto_min"] = float(min(vals)) if vals else np.nan
    out["corretto_tutte"] = float((ok["score_low"] > ok["score_full"]).mean()) if len(ok) else np.nan
    return out
