#!/usr/bin/env bash
# WP6: CML-regularised ConcatMatched-AU. Resumable, one MPS job at a time.
# Skip rule matches scripts/mtap_overnight.sh train_one: a metrics status of
# "complete" is left alone; a "failed" status is kept and not rerun.
# Usage: caffeinate -dims bash scripts/mtap_cml_grid.sh
#        bash scripts/mtap_cml_grid.sh smoke   # one RAVDESS run (seed 0, fold 1)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-python3}"
LOG_DIR=mtap_outputs/logs
STATUS=$LOG_DIR/cml_status.md
mkdir -p "$LOG_DIR"
SEEDS=(0 1 7)
FOLDS=(1 2 3 4 5)

log() { echo "- $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }

metrics_status() {  # $1 = metrics json path -> complete | failed | missing
  [ -f "$1" ] || { echo missing; return; }
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('status','complete'))" "$1" 2>/dev/null || echo failed
}

train_one() {  # $1 corpus, $2 config, $3 model, $4 system, $5 seed, $6 fold, $7 artifacts root
  local tag="$1_$4_seed$5" stem="$3_fold_$6_$1_$4_seed$5"
  local metrics="$7/checkpoints/${stem}.metrics.json" st
  st=$(metrics_status "$metrics")
  if [ "$st" = complete ]; then return 0; fi
  if [ "$st" = failed ]; then echo "$stem failed earlier; kept, not rerun" >> "$LOG_DIR/cml_failed_runs.txt"; return 0; fi
  local start=$(date +%s)
  if "$PY" -m iemocap_mer.cli --config "$2" train --model "$3" --fold "fold_$6" --seed "$5" --tag "$tag" --device mps >> "$LOG_DIR/cml_train_$1.log" 2>&1; then
    echo "$(date '+%H:%M:%S') ok $stem $(( $(date +%s) - start ))s" >> "$LOG_DIR/cml_train_$1_progress.txt"
  else
    echo "$(date '+%H:%M:%S') FAILED $stem $(( $(date +%s) - start ))s" >> "$LOG_DIR/cml_train_$1_progress.txt"
    echo "$stem" >> "$LOG_DIR/cml_failed_runs.txt"
  fi
}

run_corpus() {  # $1 corpus, $2 config, $3 artifacts root
  local c=$1
  log "$c: CML ConcatMatched-AU grid (seeds 0,1,7; folds 1-5)"
  for seed in "${SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
    train_one "$c" "$2" concat_matched concat_matched_au_cml "$seed" "$fold" "$3"
  done; done
  log "$c: CML grid pass finished"
}

if [ "${1:-}" = smoke ]; then
  log "smoke: ravdess concat_matched_au_cml seed 0 fold 1"
  train_one ravdess configs/ravdess_w2v_video_au_cml.yaml concat_matched concat_matched_au_cml 0 1 artifacts_ravdess
  exit 0
fi

log "CML grid start (pid $$)"
run_corpus ravdess configs/ravdess_w2v_video_au_cml.yaml artifacts_ravdess
run_corpus cremad configs/cremad_w2v_video_au_cml.yaml artifacts_cremad
run_corpus iemocap configs/iemocap_w2v_video_au_cml.yaml artifacts
log "CML grid finished"
