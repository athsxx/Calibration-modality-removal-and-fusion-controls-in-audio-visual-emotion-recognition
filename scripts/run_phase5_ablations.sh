#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python3}
CFG=configs/essence_v2.yaml
LOG_DIR=artifacts/results/phase5_ablations
mkdir -p "$LOG_DIR"
MAIN_LOG=artifacts/results/phase5_ablations.log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }

train_ablation() {
  local fold=$1 tag=$2 disable_loss=$3
  local metrics="artifacts/checkpoints/essence_${fold}_${tag}.metrics.json"
  local run_log="${LOG_DIR}/essence_${fold}_${tag}.log"
  if [[ -f "$metrics" ]]; then
    log "skip essence ${fold} ${tag} (exists)"
    return 0
  fi
  log "start essence ${fold} ${tag} (disable ${disable_loss})"
  "$PY" -m iemocap_mer.cli --config "$CFG" train \
    --model essence --fold "$fold" --tag "$tag" \
    --disable-loss "$disable_loss" \
    > "$run_log" 2>&1
  local f1
  f1=$("$PY" -c "import json; print(f\"{json.load(open('$metrics'))['test']['macro_f1']:.3f}\")")
  log "done essence ${fold} ${tag} test_macro_f1=${f1}"
}

run_parallel_wave() {
  local tag=$1 disable_loss=$2
  local -a pids=()
  log "=== wave: ${tag} (5 folds in parallel) ==="
  for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
    train_ablation "$fold" "$tag" "$disable_loss" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    log "ERROR: one or more jobs failed in wave ${tag}"
    exit 1
  fi
}

log "=== Phase 5: essence v2 ablations (video_v2 features) ==="
run_parallel_wave "v2_no_audio_to_video" "audio_to_video_distillation"
run_parallel_wave "v2_no_video_to_audio" "video_to_audio_distillation"

log "summarize + research report"
"$PY" -m iemocap_mer.cli summarize 2>&1 | tee -a "$MAIN_LOG"
"$PY" -m iemocap_mer.cli research-report 2>&1 | tee -a "$MAIN_LOG"
log "=== Phase 5 complete ==="
