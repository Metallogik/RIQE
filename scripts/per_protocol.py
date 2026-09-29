#!/usr/bin/env python3
"""One model per protocol cell against the single model, on the inner
validation split.

With the released setting (P, C, p), fits one model per protocol cell on the
inner fitting patients of that cell and compares it with the single model
fitted on all inner fitting patients, on two endpoints:

  * reduced dose ranked worse than full dose, same slice (Siemens cells);
  * preference for filtered full-dose images.

Every validation image is scored by the model of its own cell. Validation
only: TEST is not touched, and the released model remains the single one.
This answers whether the cost of a single model, measured in
stratification.py, is what drives the two endpoints.

    .venv/bin/python scripts/per_protocol.py --moments experiments/bank_val_inner_moments_lowC.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riqe.cache import FeatureCache  # noqa: E402
from riqe.dosetest import dose_ordering, dose_pairs, low_cache_dir, pair_moments  # noqa: E402
from riqe.evaluate import attach_scores, load_moments  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moments", default="experiments/bank_val_inner_moments_lowC.npz",
                    help="validation moments that contain the released (P, C)")
    args = ap.parse_args()

    ch = json.loads((OUT / "hparam_choice.json").read_text())
    P, C, p = int(ch["P"]), float(ch["C"]), float(ch["p"])
    corpus = pd.read_parquet(ROOT / "corpus" / "corpus.parquet")
    split = json.load(open(ROOT / "corpus" / "split.json"))
    kept = corpus[(corpus["kind"] == "full") & corpus["keep"]]
    fit = kept[kept.patient_id.isin(set(split["fit_inner"]))]
    cache = FeatureCache(ROOT / "data" / "features" / f"P{P}_C{C:g}")

    meta, NU, SG = load_moments(args.moments)
    meta = meta[(meta.P == P) & (meta.C == C) & (meta.source == "full")].copy()
    pairs = dose_pairs(corpus, split["val_inner"])
    moments_pairs = pair_moments(cache, FeatureCache(low_cache_dir(P, C)), pairs)

    single = cache.fit_fast(list(fit["path"]), p, n_patients=fit.patient_id.nunique())
    rows = []
    for cell, fc in fit.groupby("cell"):
        own = cache.fit_fast(list(fc["path"]), p, n_patients=fc.patient_id.nunique())
        sub = meta[meta.cell == cell]
        pc = pairs[pairs.cell == cell]
        for name, model in (("single", single), ("per_cell", own)):
            d = attach_scores(sub, model, NU, SG)
            orig = d[d.kind == "original"].set_index("slice_path")["score"]
            den = d[d.kind == "denoise"].copy()
            den["delta"] = den["score"] - den["slice_path"].map(orig)
            den = den.dropna(subset=["delta"])
            r = {"cell": cell, "model": name, "n_fit_patients": int(fc.patient_id.nunique()),
                 "n_val_patients": int(sub.patient_id.nunique()),
                 "filtered_n": int(len(den)),
                 "filtered_preferred": float((den["delta"] < 0).mean()) if len(den) else np.nan}
            if len(pc):
                res = dose_ordering(model, pc, moments_pairs).dropna(subset=["correct"])
                r.update({"dose_n": int(len(res)), "dose_correct": float(res["correct"].mean())})
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_protocol.csv", index=False)
    show = df.copy()
    for c in ("filtered_preferred", "dose_correct"):
        show[c] = (100 * show[c]).round(1)
    print(f"setting P={P} C={C:g} p={p:g}, inner validation split")
    print(show.to_string(index=False))
    print(f"wrote {OUT / 'per_protocol.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
