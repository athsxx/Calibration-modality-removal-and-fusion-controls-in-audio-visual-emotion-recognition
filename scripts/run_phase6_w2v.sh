#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python3}
CFG=configs/audio_w2v.yaml
LOG_DIR=artifacts/results/phase6_w2v
MAIN_LOG=artifacts/results/phase6_w2v_runner.log
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }

count_audio_w2v() {
  find artifacts/features/audio_w2v -name '*.pt' ! -name '._*' 2>/dev/null | wc -l | tr -d ' '
}

extract_session() {
  local session=$1
  log "start w2v extract session ${session} ($(count_audio_w2v)/5531 so far)"
  "$PY" -m iemocap_mer.cli --config "$CFG" extract-features \
    --modality audio --backend wav2vec2 --device mps --session "$session" \
    >> "${LOG_DIR}/extract_session${session}.log" 2>&1
  log "done w2v extract session ${session} ($(count_audio_w2v)/5531)"
}

train_fold() {
  local fold=$1
  local metrics="artifacts/checkpoints/audio_${fold}_w2v.metrics.json"
  if [[ -f "$metrics" ]]; then
    log "skip audio ${fold} w2v (exists)"
    return 0
  fi
  log "start audio ${fold} w2v"
  "$PY" -m iemocap_mer.cli --config "$CFG" train --model audio --fold "$fold" --tag w2v \
    > "${LOG_DIR}/audio_${fold}_w2v.log" 2>&1
  local f1
  f1=$("$PY" -c "import json; print(f\"{json.load(open('$metrics'))['test']['macro_f1']:.3f}\")")
  log "done audio ${fold} w2v test_macro_f1=${f1}"
}

log "=== Phase 6: Wav2Vec2 audio features + baseline ==="
log "current cache: $(count_audio_w2v)/5531"

# Sequential extraction — parallel MPS jobs keep dying mid-run
if [[ "$(count_audio_w2v)" -lt 5531 ]]; then
  log "extract wav2vec2 sequentially (skip existing)"
  for session in 1 2 3 4 5; do
    extract_session "$session"
  done
else
  log "skip extraction ($(count_audio_w2v) present)"
fi

log "verify audio_w2v cache"
"$PY" -m iemocap_mer.cli --config "$CFG" verify-features --modality audio 2>&1 | tee -a "$MAIN_LOG"
if [[ "$(count_audio_w2v)" -lt 5531 ]]; then
  log "ERROR: expected 5531 features, got $(count_audio_w2v)"
  exit 1
fi

log "train audio baseline (5 folds in parallel)"
pids=()
for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
  train_fold "$fold" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  log "ERROR: one or more training jobs failed"
  exit 1
fi

log "summarize + research report"
"$PY" -m iemocap_mer.cli summarize 2>&1 | tee -a "$MAIN_LOG"
"$PY" -m iemocap_mer.cli research-report 2>&1 | tee -a "$MAIN_LOG"
log "=== Phase 6 complete ==="
