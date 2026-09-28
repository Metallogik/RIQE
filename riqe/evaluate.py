"""Utilita' condivise dalla batteria di validazione.

Caricamento dei momenti del banco, punteggio contro un modello, e le
statistiche di ordinamento usate piu' volte.  Sta qui per evitare che i sei
esperimenti divergano su dettagli come il trattamento dei punteggi non
definiti.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import RCOND, MVGModel, mahalanobis_mixed


def load_moments(path: str | Path):
    """Ritorna (meta, NU, SIGMA).  `meta` ha una riga per immagine di prova."""
    z = np.load(path, allow_pickle=False)
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    meta["slice_path"] = json.loads(str(z["slice_path"]))
    meta["patient_id"] = json.loads(str(z["patient_id"]))
    meta["cell"] = json.loads(str(z["cell"]))
    meta["pixel_spacing"] = json.loads(str(z["pixel_spacing"]))
    meta["row"] = np.arange(len(meta))
    return meta, z["nu"], z["sigma"]


def score_all(model: MVGModel, NU: np.ndarray, SG: np.ndarray, rows) -> np.ndarray:
    """Punteggi di un insieme di righe.  NaN dove non definibile."""
    rows = np.asarray(rows, dtype=int)
    out = np.full(len(rows), np.nan)
    for k, i in enumerate(rows):
        if np.isfinite(NU[i, 0]):
            out[k] = mahalanobis_mixed(model.nu, model.sigma, NU[i], SG[i], RCOND)
    return out


def attach_scores(meta: pd.DataFrame, model: MVGModel, NU, SG, col: str = "score") -> pd.DataFrame:
    d = meta.copy()
    d[col] = score_all(model, NU, SG, d["row"].to_numpy())
    return d


def step_monotone(seq: np.ndarray, increasing: bool = True) -> bool:
    """Monotonicita' **a ogni passo**, non solo globale."""
    s = np.asarray(seq, dtype=float)
    if not np.isfinite(s).all() or len(s) < 2:
        return False
    d = np.diff(s)
    return bool(np.all(d > 0) if increasing else np.all(d < 0))


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_w(rank_matrix: np.ndarray) -> float:
    """W di Kendall: concordanza fra piu' giudici (qui: piu' immagini) su un
    insieme di oggetti (qui: i denoiser).  1 = accordo perfetto."""
    R = np.asarray(rank_matrix, dtype=float)
    R = R[np.isfinite(R).all(axis=1)]
    m, n = R.shape
    if m < 2 or n < 2:
        return float("nan")
    Rj = R.sum(axis=0)
    S = ((Rj - Rj.mean()) ** 2).sum()
    return float(12.0 * S / (m ** 2 * (n ** 3 - n)))


def sign_test(a, b) -> tuple[int, int, float]:
    """Test dei segni appaiato.  Ritorna (n_positivi, n_validi, p bilaterale)."""
    from scipy.stats import binomtest

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & (a != b)
    n = int(ok.sum())
    if n == 0:
        return 0, 0, float("nan")
    k = int((a[ok] > b[ok]).sum())
    return k, n, float(binomtest(k, n, 0.5).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Correzione di Holm-Bonferroni per confronti multipli."""
    items = sorted(((k, v) for k, v in pvals.items() if np.isfinite(v)), key=lambda t: t[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, v) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * v))
        out[k] = adj
        prev = adj
    for k, v in pvals.items():
        out.setdefault(k, float("nan"))
    return out
