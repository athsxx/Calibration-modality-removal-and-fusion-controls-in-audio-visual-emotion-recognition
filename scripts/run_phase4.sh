#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python3}
LOG=artifacts/results/phase4_multimodal_v2.log
mkdir -p artifacts/results

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

train_if_missing() {
  local cfg=$1 model=$2 fold=$3 tag=$4
  local metrics="artifacts/checkpoints/${model}_${fold}_${tag}.metrics.json"
  if [[ -f "$metrics" ]]; then
    log "skip ${model} ${fold} ${tag} (exists)"
    return 0
  fi
  log "train ${model} ${fold} ${tag}"
  "$PY" -m iemocap_mer.cli --config "$cfg" train --model "$model" --fold "$fold" --tag "$tag" \
    2>&1 | tee -a "$LOG"
}

log "=== Phase 4: multimodal models with video_v2 features ==="

for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
  train_if_missing configs/video_v2.yaml concat "$fold" v2
done

for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
  train_if_missing configs/essence_v2.yaml essence "$fold" v2
done

for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
  if [[ ! -f "artifacts/results/essence_${fold}_v2_robustness.json" ]]; then
    log "analyze essence ${fold} v2"
    "$PY" -m iemocap_mer.cli --config configs/essence_v2.yaml analyze --model essence --fold "$fold" --tag v2 \
      2>&1 | tee -a "$LOG"
  fi
done

log "summarize + research report"
"$PY" -m iemocap_mer.cli summarize 2>&1 | tee -a "$LOG"
"$PY" -m iemocap_mer.cli research-report 2>&1 | tee -a "$LOG"
log "=== Phase 4 complete ==="
