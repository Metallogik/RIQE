"""Client for the public NBIA/TCIA REST API, with on-disk cache and hashes.

The v1 API needs no authentication for public collections. The important
endpoint is `getSingleImage`: it downloads a single DICOM slice instead of a
whole series, so the terabyte of projection data in the collection never has
to be downloaded.

Every downloaded file is recorded with its SHA-256, which ends up in the model
artefact as the identity of the corpus.
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
    raise RuntimeError(f"request failed after {retries} attempts: {url}") from last


def get_series(collection: str = COLLECTION) -> list[dict]:
    return json.loads(_get(f"{BASE}/getSeries?Collection={collection}"))


def get_sop_uids(series_uid: str) -> list[str]:
    data = json.loads(_get(f"{BASE}/getSOPInstanceUIDs?SeriesInstanceUID={series_uid}"))
    return [d["SOPInstanceUID"] for d in data]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_slice(series_uid: str, sop_uid: str, cache_dir: Path) -> tuple[Path, str]:
    """Download (or reuse from cache) one slice. Returns (path, sha256)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{sop_uid}.dcm"
    if path.exists():
        return path, sha256(path.read_bytes())
    raw = _get(f"{BASE}/getSingleImage?SeriesInstanceUID={series_uid}&SOPInstanceUID={sop_uid}")
    if not raw.startswith(b"DICM", 128) and b"DICM" not in raw[:200]:
        raise RuntimeError(f"non-DICOM response for {sop_uid}: {raw[:120]!r}")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    return path, sha256(raw)
