#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python3}
CKPT=artifacts/checkpoints

train_if_missing() {
  local model=$1
  local fold=$2
  local tag=${3:-}
  shift 2
  [[ -n "$tag" ]] && shift 1
  local suffix=""
  [[ -n "$tag" ]] && suffix="_${tag}"
  local metrics="${CKPT}/${model}_${fold}${suffix}.metrics.json"
  if [[ -f "$metrics" ]]; then
    echo "skip ${model} ${fold}${suffix} (metrics exist)"
    return 0
  fi
  local args=(train --model "$model" --fold "$fold")
  [[ -n "$tag" ]] && args+=(--tag "$tag")
  shift 3 2>/dev/null || shift 2
  args+=("$@")
  "$PY" -m iemocap_mer.cli "${args[@]}"
}

for model in audio video concat cross_attention essence; do
  for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
    train_if_missing "$model" "$fold"
  done
done

for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
  if [[ ! -f "artifacts/results/essence_${fold}_robustness.json" ]]; then
    "$PY" -m iemocap_mer.cli analyze --model essence --fold "$fold" || true
  else
    echo "skip analyze essence ${fold}"
  fi
  train_if_missing essence "$fold" no_audio_to_video --disable-loss audio_to_video_distillation
  train_if_missing essence "$fold" no_video_to_audio --disable-loss video_to_audio_distillation
done

"$PY" -m iemocap_mer.cli summarize

