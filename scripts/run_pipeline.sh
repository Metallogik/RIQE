#!/usr/bin/env bash
# Runs in sequence everything that follows corpus construction.
#
# Prerequisites (see README.md): download_corpus.py, build_slice_table.py,
# make_corpus.py, cache_features.py and cache_features.py --paired-low have
# been run, and, for experiment 6, the LDCTIQAC 2023 training set is unpacked
# in data/ldctiqac/.
#
# Each step writes its own log to logs/, and the pipeline stops at the first
# error: a step that fails silently would leave the following steps working
# on stale results, which is the worst way to be wrong.
#
#   nohup bash scripts/run_pipeline.sh > logs/pipeline.log 2>&1 &
#
# Resumable: steps already completed (output file present) are skipped.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs experiments artifacts

W=${WORKERS:-30}

step() {                       # step <name> <output-file> <command...>
  local name="$1" out="$2"; shift 2
  if [ -e "$out" ]; then
    echo "[$(date +%H:%M:%S)] skipping $name (already present: $out)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] --- $name ---"
  "$@" 2>&1 | tee "logs/$name.log"
  echo "[$(date +%H:%M:%S)] done $name"
}

step bank_val experiments/bank_val_inner.json \
  $PY scripts/prepare_bank.py --split val_inner --per-patient 4 --workers "$W"

step moments_val experiments/bank_val_inner_moments.npz \
  $PY scripts/score_bank.py --bank experiments/bank_val_inner.json --workers "$W"

# C grid extended downwards (0.05 / 0.025 / 0.01): the mechanism of the
# preferred noise level goes through C
step moments_val_lowC experiments/bank_val_inner_moments_lowC.npz \
  $PY scripts/score_bank.py --bank experiments/bank_val_inner.json --workers "$W" \
     --P 16,24,32 --C 0.05,0.025,0.01 --out experiments/bank_val_inner_moments_lowC.npz

step hparam_search experiments/hparam_choice.json \
  $PY scripts/search_hparams.py --bootstrap 60 --moments \
     experiments/bank_val_inner_moments.npz experiments/bank_val_inner_moments_lowC.npz

step per_protocol experiments/per_protocol.csv \
  $PY scripts/per_protocol.py --moments experiments/bank_val_inner_moments_lowC.npz

step fit_model artifacts/riqe-v1.0.json \
  $PY scripts/fit_model.py

step stratification experiments/stratification.csv \
  $PY scripts/stratification.py --B 200

step bank_test experiments/bank_test.json \
  $PY scripts/prepare_bank.py --split test --per-patient 6 --fine-noise --rel-noise --workers "$W"

# on TEST only the chosen setting and the original-criterion setting are needed
PLIST=$($PY -c "import json;c=json.load(open('experiments/hparam_choice.json'));o=c['original_criterion_choice'];print(','.join(sorted({str(c['P']),str(o['P'])})))")
CLIST=$($PY -c "import json;c=json.load(open('experiments/hparam_choice.json'));o=c['original_criterion_choice'];print(','.join(sorted({repr(float(c['C'])),repr(float(o['C']))})))")
echo "[$(date +%H:%M:%S)] settings for TEST: P=$PLIST C=$CLIST"

step moments_test experiments/bank_test_moments.npz \
  $PY scripts/score_bank.py --bank experiments/bank_test.json --workers "$W" --P "$PLIST" --C "$CLIST"

step moments_test_niqecfg experiments/bank_test_moments_niqecfg.npz \
  $PY scripts/score_bank.py --bank experiments/bank_test.json --workers "$W" --niqe-config

step battery experiments/battery_summary.json \
  $PY scripts/run_battery.py --moments experiments/bank_test_moments.npz --bootstrap 200

step lesions experiments/exp2_form2_summary.json \
  $PY scripts/exp_lesions.py --n-slices 24 --realizations 64 --workers "$W"

step photo_baseline experiments/exp5_summary.json \
  $PY scripts/exp5_photo_baseline.py --n-photos 125 --moments experiments/bank_test_moments.npz \
     --workers 16

step photo_sanity experiments/photo_sanity.json \
  $PY scripts/photo_sanity.py --workers 16

if [ -e data/ldctiqac/LDCTIQAG2023_train/train.json ]; then
  step ldctiqac experiments/exp6_ldctiqac_summary.json \
    $PY scripts/exp6_ldctiqac.py --workers "$W"
  step ldctiqac_groups experiments/ldctiqac_group_sensitivity.csv \
    $PY scripts/ldctiqac_groups_check.py
else
  echo "[$(date +%H:%M:%S)] skipping ldctiqac: data/ldctiqac/LDCTIQAG2023_train not found"
fi

step uncertainty experiments/uncertainty.json \
  $PY scripts/uncertainty.py

step verify artifacts/verify_ok.txt \
  bash -c "$PY scripts/verify_model.py | tee artifacts/verify_ok.txt"

echo "[$(date +%H:%M:%S)] PIPELINE COMPLETE"
