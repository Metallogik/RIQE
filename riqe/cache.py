"""Access to the feature cache, with fitting at a variable threshold `p`.

The cache holds only the patches inside the domain, with their sharpness
`delta`. Sharpness selection (threshold relative to the maximum **per
image**) is applied here, downstream: changing `p` needs no re-extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import MVGModel, fit_mvg


#: project root, to normalise paths
_ROOT = Path(__file__).resolve().parents[1]


def canonical(path: str | Path) -> str:
    """Canonical path, relative to the project root.

    Caches may have been written with absolute paths while the corpus tables
    use relative ones. Without normalisation the lookup fails silently and a
    fit ends up with fewer slices than intended, or none.
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
        """Selected patches of one slice at threshold `p`."""
        f, d = self.slice_rows(path)
        if p is None or f.shape[0] == 0:
            return f
        return f[d > p * d.max()]

    # -- sufficient statistics ------------------------------------------------
    #
    # Mean and covariance of a group of slices depend only on three per-slice
    # quantities: number of patches, sum of vectors, sum of outer products.
    # Precomputing them turns every refit into an aggregation of 36 + 1296
    # numbers per slice instead of re-reading and concatenating hundreds of
    # thousands of patches. Needed because the hyperparameter search refits
    # 168 settings times 60 bootstraps, and stratification 200 times per
    # contrast: without it, hours.

    def stats(self, p: float | None) -> dict[str, tuple[int, np.ndarray, np.ndarray]]:
        """(n, sum, sum of outer products) per slice, at threshold `p`."""
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
        """Like `fit`, but from sufficient statistics.

        Identical to `fit` within rounding error: the covariance is computed
        as (S2 - N mu mu^T) / (N - 1), the same unbiased estimate as np.cov
        with ddof=1.
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
                f"only {n} patches for {self.n_features} features: threshold p too high?")
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
        """Fit the model on the given slices, at threshold `p`."""
        paths = [canonical(q) for q in paths]
        known = [q for q in paths if q in self.by_path]
        if len(known) < len(paths):
            raise KeyError(
                f"{len(paths) - len(known)} of the {len(paths)} requested slices are "
                f"not in cache {self.dir.name}. Re-run cache_features.py, or the "
                f"paths do not match (missing example: "
                f"{next(q for q in paths if q not in self.by_path)})"
            )
        parts = [self.select(pp, p) for pp in known]
        parts = [x for x in parts if x.shape[0] > 0]
        if not parts:
            raise ValueError("no patch selected: threshold p too high?")
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
