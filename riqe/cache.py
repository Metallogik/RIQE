"""Accesso alla cache delle feature, con fitting a soglia `p` variabile.

La cache contiene solo le patch nel dominio, con la loro nitidezza `delta`.
La selezione per nitidezza (soglia relativa al massimo **per immagine**) si
applica qui, a valle: cambiare `p` non richiede di riestrarre nulla.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import MVGModel, fit_mvg


#: radice del progetto, per normalizzare i percorsi
_ROOT = Path(__file__).resolve().parents[1]


def canonical(path: str | Path) -> str:
    """Percorso in forma canonica, relativa alla radice del progetto.

    La cache e' stata scritta con percorsi assoluti, le tabelle del corpus
    usano percorsi relativi.  Senza normalizzazione la ricerca fallisce in
    silenzio e il fit si ritrova con meno slice del previsto, o con nessuna.
    """
    q = Path(path)
    if q.is_absolute():
        try:
            q = q.relative_to(_ROOT)
        except ValueError:
            pass
    return q.as_posix()


class FeatureCache:
    def __init__(self, directory: str | Path):
        d = Path(directory)
        self.dir = d
        self.features: np.ndarray = np.load(d / "features.npy", mmap_mode="r")
        self.delta: np.ndarray = np.load(d / "delta.npy", mmap_mode="r")
        self.index: list[dict] = json.loads((d / "index.json").read_text())
        self.info: dict = json.loads((d / "info.json").read_text())
        self.by_path: dict[str, dict] = {canonical(e["path"]): e for e in self.index}
        self._stats: dict = {}

    @property
    def P(self) -> int:
        return int(self.info["P"])

    @property
    def C(self) -> float:
        return float(self.info["C"])

    def slice_rows(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        e = self.by_path[canonical(path)]
        s, n = e["start"], e["n"]
        return np.asarray(self.features[s : s + n]), np.asarray(self.delta[s : s + n])

    def select(self, path: str, p: float | None) -> np.ndarray:
        """Patch selezionate di una slice alla soglia `p`."""
        f, d = self.slice_rows(path)
        if p is None or f.shape[0] == 0:
            return f
        return f[d > p * d.max()]

    # -- statistiche sufficienti ------------------------------------------
    #
    # Media e covarianza di un gruppo di slice dipendono solo da tre
    # quantita' per slice: il numero di patch, la somma dei vettori e la
    # somma dei prodotti esterni.  Precalcolarle rende ogni rifit una
    # aggregazione di 36 + 1296 numeri per slice invece di una rilettura e
    # concatenazione di centinaia di migliaia di patch.  Serve perche' la
    # ricerca degli iperparametri rifitta 96 volte per 60 bootstrap, e la
    # stratificazione 200 volte per contrasto: senza, sono ore.

    def stats(self, p: float | None) -> dict[str, tuple[int, np.ndarray, np.ndarray]]:
        """(n, somma, somma dei prodotti esterni) per slice, alla soglia `p`."""
        key = ("all" if p is None else round(float(p), 6))
        if key not in self._stats:
            out = {}
            for e in self.index:
                path = canonical(e["path"])
                f = self.select(path, p).astype(np.float64)
                if f.shape[0] == 0:
                    continue
                out[path] = (int(f.shape[0]), f.sum(axis=0), f.T @ f)
            self._stats[key] = out
        return self._stats[key]

    def fit_fast(self, paths, p: float | None, n_patients: int = 0,
                 meta: dict | None = None) -> MVGModel:
        """Come `fit`, ma dalle statistiche sufficienti.

        Risultato identico a `fit` entro l'errore di arrotondamento: la
        covarianza e' calcolata come (S2 - N mu mu^T) / (N - 1), la stessa
        stima non distorta di np.cov con ddof=1.
        """
        st = self.stats(p)
        n = 0
        s1 = np.zeros(self.n_features)
        s2 = np.zeros((self.n_features, self.n_features))
        n_img = 0
        for q in (canonical(x) for x in paths):
            v = st.get(q)
            if v is None:
                continue
            n += v[0]
            s1 += v[1]
            s2 += v[2]
            n_img += 1
        if n <= self.n_features:
            raise ValueError(
                f"solo {n} patch per {self.n_features} feature: soglia p troppo alta?")
        mu = s1 / n
        cov = (s2 - n * np.outer(mu, mu)) / (n - 1)
        m = dict(meta or {})
        m.update({"P": self.P, "C": self.C, "p": p,
                  "patches_per_slice": float(n / max(n_img, 1))})
        return MVGModel(nu=mu, sigma=cov, n_patches=n, n_images=n_img,
                        n_patients=n_patients, meta=m)

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    def fit(self, paths, p: float | None, n_patients: int = 0, meta: dict | None = None) -> MVGModel:
        """Fit del modello sulle slice indicate, alla soglia `p`."""
        paths = [canonical(q) for q in paths]
        known = [q for q in paths if q in self.by_path]
        if len(known) < len(paths):
            raise KeyError(
                f"{len(paths) - len(known)} delle {len(paths)} slice richieste non "
                f"sono nella cache {self.dir.name}. Rieseguire cache_features.py, "
                f"oppure i percorsi non corrispondono (esempio mancante: "
                f"{next(q for q in paths if q not in self.by_path)})"
            )
        parts = [self.select(pp, p) for pp in known]
        parts = [x for x in parts if x.shape[0] > 0]
        if not parts:
            raise ValueError("nessuna patch selezionata: soglia p troppo alta?")
        X = np.concatenate(parts).astype(np.float64)
        m = dict(meta or {})
        m.update({"P": self.P, "C": self.C, "p": p,
                  "patches_per_slice": float(X.shape[0] / len(parts))})
        return fit_mvg(X, n_images=len(parts), n_patients=n_patients, meta=m)

    def patch_counts(self, paths, p: float | None) -> np.ndarray:
        out = []
        for pp in (canonical(q) for q in paths):
            if pp not in self.by_path:
                continue
            f, d = self.slice_rows(pp)
            out.append(int((d > p * d.max()).sum()) if (p is not None and f.shape[0]) else f.shape[0])
        return np.asarray(out)
