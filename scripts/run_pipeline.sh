#!/usr/bin/env bash
# Esegue in sequenza tutto cio' che resta dopo la costruzione del corpus.
#
# Ogni passo scrive il proprio log in logs/, e la pipeline si ferma al primo
# errore: un passo che fallisce in silenzio lascerebbe i passi successivi a
# lavorare su risultati vecchi, che e' il modo peggiore di sbagliare.
#
#   nohup bash scripts/run_pipeline.sh > logs/pipeline.log 2>&1 &
#
# Riprendibile: i passi gia' completati (file di uscita presente) si saltano.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs experiments artifacts

W=${WORKERS:-30}

step() {                       # step <nome> <file-di-uscita> <comando...>
  local name="$1" out="$2"; shift 2
  if [ -e "$out" ]; then
    echo "[$(date +%H:%M:%S)] salto $name (gia' presente: $out)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] --- $name ---"
  "$@" 2>&1 | tee "logs/$name.log"
  echo "[$(date +%H:%M:%S)] fine $name"
}

# attende il banco di validazione se un'altra esecuzione lo sta ancora creando
while pgrep -f "prepare_bank.py --split val_inner" > /dev/null; do
  echo "[$(date +%H:%M:%S)] attendo il banco val_inner ancora in costruzione..."
  sleep 30
done

step banco_val experiments/bank_val_inner.json \
  $PY scripts/prepare_bank.py --split val_inner --per-patient 4 --workers "$W"

step momenti_val experiments/bank_val_inner_moments.npz \
  $PY scripts/score_bank.py --bank experiments/bank_val_inner.json --workers "$W"

step ricerca_iperparametri experiments/hparam_choice.json \
  $PY scripts/search_hparams.py --moments experiments/bank_val_inner_moments.npz --bootstrap 60

step fit_modello artifacts/riqe-v1.0.json \
  $PY scripts/fit_model.py

step stratificazione experiments/stratification.csv \
  $PY scripts/stratification.py --B 200

step banco_test experiments/bank_test.json \
  $PY scripts/prepare_bank.py --split test --per-patient 6 --fine-noise --workers "$W"

step momenti_test experiments/bank_test_moments.npz \
  $PY scripts/score_bank.py --bank experiments/bank_test.json --workers "$W"

step batteria experiments/battery_summary.json \
  $PY scripts/run_battery.py --moments experiments/bank_test_moments.npz --bootstrap 200

step lesioni experiments/exp2_form2_summary.json \
  $PY scripts/exp_lesions.py --n-slices 24 --realizations 16 --workers "$W"

step baseline_fotografica experiments/exp5_summary.json \
  $PY scripts/exp5_photo_baseline.py --n-photos 125 --moments experiments/bank_test_moments.npz \
     --workers 16

step verifica artifacts/verify_ok.txt \
  bash -c "$PY scripts/verify_model.py | tee artifacts/verify_ok.txt"

echo "[$(date +%H:%M:%S)] PIPELINE COMPLETA"
