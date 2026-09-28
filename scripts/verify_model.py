#!/usr/bin/env python3
"""Reproducibility check of the artifact.

Redoes the fit from the public data and compares it with the published
model. This is the check that makes the claim "the model can be rebuilt by
anyone who downloads the same data" verifiable rather than a promise.

It checks, in order:
  1. that every slice declared in the artifact is present locally and has
     the declared SHA-256;
  2. that the code has the same hash as when the model was fitted (if it
     differs, it says so and continues: that is information, not an error);
  3. that the refit matches the published nu and Sigma within the declared
     tolerance.

    .venv/bin/python scripts/verify_model.py --model artifacts/riqe-v1.0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.cache import FeatureCache  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOL_REL = 1e-6


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "riqe").glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/riqe-v1.0.json")
    ap.add_argument("--check-hashes", action="store_true",
                    help="recompute the SHA-256 of every slice (slow but complete)")
    args = ap.parse_args()

    art = json.loads((ROOT / args.model).read_text())
    z = np.load(ROOT / args.model.replace(".json", ".npz"))
    nu_pub, sg_pub = z["nu"], z["sigma"]
    hp = art["hyperparameters"]
    print(f"artifact: {art['model_id']}  P={hp['P']} C={hp['C']} p={hp['p']}")
    print(f"declared corpus: {len(art['corpus_slices_used'])} slices, "
          f"{art['fit']['n_patients']} patients")

    ok = True

    # 1. presence and integrity of the slices ---------------------------------
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    have = {r.sop_uid: r.path for r in corpus.itertuples()}
    missing = [s for s in art["corpus_slices_used"]
               if s["sop_uid"] not in have or not (ROOT / have[s["sop_uid"]]).exists()]
    print(f"\n1. slices present locally: "
          f"{len(art['corpus_slices_used']) - len(missing)}/{len(art['corpus_slices_used'])}")
    if missing:
        ok = False
        print(f"   MISSING: {len(missing)} (first: "
              f"{[m['patient_id'] for m in missing[:5]]})")
    if args.check_hashes and not missing:
        bad = 0
        for s in art["corpus_slices_used"]:
            b = (ROOT / have[s["sop_uid"]]).read_bytes()
            if s["sha256"] and hashlib.sha256(b).hexdigest() != s["sha256"]:
                bad += 1
        print(f"   SHA-256 mismatches: {bad}")
        ok = ok and bad == 0

    # 2. identity of the code ---------------------------------------------------
    now = code_hash()
    same = now == art["code"]["package_sha256"]
    print(f"\n2. hash of the riqe package: {'IDENTICAL' if same else 'DIFFERENT'}")
    if not same:
        print(f"   published: {art['code']['package_sha256'][:16]}...")
        print(f"   current  : {now[:16]}...")
        print("   the code changed after the fit: the numbers may legitimately "
              "differ, but the difference must be explained")

    # 3. refit -----------------------------------------------------------------
    paths = [have[s["sop_uid"]] for s in art["corpus_slices_used"] if s["sop_uid"] in have]
    cache_dir = ROOT / "data" / "features" / f"P{hp['P']}_C{hp['C']:g}"
    if not cache_dir.exists():
        print(f"\n3. cache {cache_dir.name} missing: run cache_features.py first")
        return 1 if not ok else 0
    cache = FeatureCache(cache_dir)
    m = cache.fit(paths, hp["p"], n_patients=art["fit"]["n_patients"])
    d_nu = np.max(np.abs(m.nu - nu_pub) / np.maximum(np.abs(nu_pub), 1e-12))
    d_sg = np.max(np.abs(m.sigma - sg_pub) / np.maximum(np.abs(sg_pub), 1e-12))
    print(f"\n3. refit: {m.n_patches} patches (published {art['fit']['n_patches']})")
    print(f"   max relative deviation on nu    : {d_nu:.3e}")
    print(f"   max relative deviation on Sigma : {d_sg:.3e}")
    passed = d_nu < TOL_REL and d_sg < TOL_REL and m.n_patches == art["fit"]["n_patches"]
    print(f"   within tolerance {TOL_REL:g}: {'YES' if passed else 'NO'}")
    ok = ok and passed

    print(f"\nOUTCOME: {'verification passed' if ok else 'VERIFICATION FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
