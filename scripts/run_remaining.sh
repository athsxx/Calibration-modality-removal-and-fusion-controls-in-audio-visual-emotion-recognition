#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python3}
LOG=artifacts/results/experiment_run.log
CKPT=artifacts/checkpoints

log() {
  echo "$*" | tee -a "$LOG"
}

train_if_missing() {
  local model=$1 fold=$2 tag=${3:-}
  local suffix=""
  [[ -n "$tag" ]] && suffix="_${tag}"
  local metrics="${CKPT}/${model}_${fold}${suffix}.metrics.json"
  if [[ -f "$metrics" ]]; then
    log "skip ${model} ${fold}${suffix}"
    return 0
  fi
  log "=== ${model} ${fold}${suffix} $(date) ==="
  local args=(train --model "$model" --fold "$fold")
  [[ -n "$tag" ]] && args+=(--tag "$tag")
  shift 3 2>/dev/null || shift 2
  args+=("$@")
  "$PY" -m iemocap_mer.cli "${args[@]}"
}

for model in concat cross_attention essence; do
  for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
    train_if_missing "$model" "$fold"
  done
done

for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
  ckpt="${CKPT}/essence_${fold}.pt"
  if [[ -f "$ckpt" ]] && [[ ! -f "artifacts/results/essence_${fold}_robustness.json" ]]; then
    log "=== analyze essence ${fold} $(date) ==="
    "$PY" -m iemocap_mer.cli analyze --model essence --fold "$fold" || true
  fi
  train_if_missing essence "$fold" no_audio_to_video --disable-loss audio_to_video_distillation
  train_if_missing essence "$fold" no_video_to_audio --disable-loss video_to_audio_distillation
done

"$PY" -m iemocap_mer.cli summarize
log "=== ALL DONE $(date) ==="
