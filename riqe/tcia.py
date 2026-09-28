"""Client per l'API REST pubblica NBIA/TCIA, con cache su disco e hash.

L'API v1 non richiede autenticazione per le collezioni pubbliche.  Il punto
importante e' `getSingleImage`: permette di scaricare una singola slice DICOM
invece dell'intera serie, quindi non serve scaricare il TB di dati di
proiezione della collezione.

Ogni file scaricato e' registrato con il suo SHA-256, che finisce
nell'artefatto modello come identita' del corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "LDCT-and-Projection-data"

USER_AGENT = "riqe/0.1 (research; no-reference CT IQA model fitting)"


def _get(url: str, timeout: int = 180, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"richiesta fallita dopo {retries} tentativi: {url}") from last


def get_series(collection: str = COLLECTION) -> list[dict]:
    return json.loads(_get(f"{BASE}/getSeries?Collection={collection}"))


def get_sop_uids(series_uid: str) -> list[str]:
    data = json.loads(_get(f"{BASE}/getSOPInstanceUIDs?SeriesInstanceUID={series_uid}"))
    return [d["SOPInstanceUID"] for d in data]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_slice(series_uid: str, sop_uid: str, cache_dir: Path) -> tuple[Path, str]:
    """Scarica (o riusa dalla cache) una slice.  Ritorna (percorso, sha256)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{sop_uid}.dcm"
    if path.exists():
        return path, sha256(path.read_bytes())
    raw = _get(f"{BASE}/getSingleImage?SeriesInstanceUID={series_uid}&SOPInstanceUID={sop_uid}")
    if not raw.startswith(b"DICM", 128) and b"DICM" not in raw[:200]:
        raise RuntimeError(f"risposta non DICOM per {sop_uid}: {raw[:120]!r}")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    return path, sha256(raw)
