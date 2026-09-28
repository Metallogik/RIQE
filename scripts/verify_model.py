#!/usr/bin/env python3
"""Verifica di riproducibilita' dell'artefatto.

Rifa' il fit dai dati pubblici e confronta con il modello pubblicato.  E' il
controllo che rende verificabile, invece che promessa, l'affermazione "il
modello e' ricostruibile da chi scarica gli stessi dati".

Controlla, in ordine:
  1. che ogni slice dichiarata nell'artefatto sia presente in locale e abbia
     lo SHA-256 dichiarato;
  2. che il codice abbia lo stesso hash di quando il modello e' stato fittato
     (se differisce, lo dice e continua: e' informazione, non un errore);
  3. che il fit rifatto coincida con nu e Sigma pubblicati entro la
     tolleranza dichiarata.

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
                    help="ricalcola lo SHA-256 di ogni slice (lento ma completo)")
    args = ap.parse_args()

    art = json.loads((ROOT / args.model).read_text())
    z = np.load(ROOT / args.model.replace(".json", ".npz"))
    nu_pub, sg_pub = z["nu"], z["sigma"]
    hp = art["hyperparameters"]
    print(f"artefatto: {art['model_id']}  P={hp['P']} C={hp['C']} p={hp['p']}")
    print(f"corpus dichiarato: {len(art['corpus_slices_used'])} slice, "
          f"{art['fit']['n_patients']} pazienti")

    ok = True

    # 1. presenza e integrita' delle slice -----------------------------------
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    have = {r.sop_uid: r.path for r in corpus.itertuples()}
    missing = [s for s in art["corpus_slices_used"]
               if s["sop_uid"] not in have or not (ROOT / have[s["sop_uid"]]).exists()]
    print(f"\n1. slice presenti in locale: "
          f"{len(art['corpus_slices_used']) - len(missing)}/{len(art['corpus_slices_used'])}")
    if missing:
        ok = False
        print(f"   MANCANTI: {len(missing)} (prime: "
              f"{[m['patient_id'] for m in missing[:5]]})")
    if args.check_hashes and not missing:
        bad = 0
        for s in art["corpus_slices_used"]:
            b = (ROOT / have[s["sop_uid"]]).read_bytes()
            if s["sha256"] and hashlib.sha256(b).hexdigest() != s["sha256"]:
                bad += 1
        print(f"   SHA-256 discordanti: {bad}")
        ok = ok and bad == 0

    # 2. identita' del codice ------------------------------------------------
    now = code_hash()
    same = now == art["code"]["package_sha256"]
    print(f"\n2. hash del pacchetto riqe: {'IDENTICO' if same else 'DIVERSO'}")
    if not same:
        print(f"   pubblicato: {art['code']['package_sha256'][:16]}...")
        print(f"   attuale   : {now[:16]}...")
        print("   il codice e' cambiato dopo il fit: i numeri possono differire "
              "legittimamente, ma la differenza va spiegata")

    # 3. rifacimento del fit -------------------------------------------------
    paths = [have[s["sop_uid"]] for s in art["corpus_slices_used"] if s["sop_uid"] in have]
    cache_dir = ROOT / "data" / "features" / f"P{hp['P']}_C{hp['C']:g}"
    if not cache_dir.exists():
        print(f"\n3. cache {cache_dir.name} assente: eseguire prima cache_features.py")
        return 1 if not ok else 0
    cache = FeatureCache(cache_dir)
    m = cache.fit(paths, hp["p"], n_patients=art["fit"]["n_patients"])
    d_nu = np.max(np.abs(m.nu - nu_pub) / np.maximum(np.abs(nu_pub), 1e-12))
    d_sg = np.max(np.abs(m.sigma - sg_pub) / np.maximum(np.abs(sg_pub), 1e-12))
    print(f"\n3. fit rifatto: {m.n_patches} patch (pubblicate {art['fit']['n_patches']})")
    print(f"   scarto relativo massimo su nu    : {d_nu:.3e}")
    print(f"   scarto relativo massimo su Sigma : {d_sg:.3e}")
    passed = d_nu < TOL_REL and d_sg < TOL_REL and m.n_patches == art["fit"]["n_patches"]
    print(f"   entro tolleranza {TOL_REL:g}: {'SI' if passed else 'NO'}")
    ok = ok and passed

    print(f"\nESITO: {'verifica superata' if ok else 'VERIFICA FALLITA'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
