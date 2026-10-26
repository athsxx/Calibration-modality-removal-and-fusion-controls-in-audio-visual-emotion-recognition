#!/usr/bin/env bash
# Train and analyze Wav2Vec2 + AU video models (5 LOSO folds).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PYTHON:-python3}"
CONFIG="configs/w2v_video_au.yaml"
TAG="w2v_au"
LOG="artifacts/reports/w2v_au_training.log"
mkdir -p artifacts/reports

echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
  echo "=== video-only AU $fold ===" | tee -a "$LOG"
  $PY -m iemocap_mer.cli --config configs/video_au.yaml train --model video --fold "$fold" --tag au \
    2>&1 | tee -a "$LOG"

  echo "=== concat_matched W2V+AU $fold ===" | tee -a "$LOG"
  $PY -m iemocap_mer.cli --config "$CONFIG" train --model concat_matched --fold "$fold" --tag "$TAG" \
    2>&1 | tee -a "$LOG"

  echo "=== essence W2V+AU $fold ===" | tee -a "$LOG"
  $PY -m iemocap_mer.cli --config "$CONFIG" train --model essence --fold "$fold" --tag "$TAG" \
    2>&1 | tee -a "$LOG"
done

for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
  echo "=== analyze video AU $fold ===" | tee -a "$LOG"
  $PY -m iemocap_mer.cli --config configs/video_au.yaml analyze --model video --fold "$fold" --tag au \
    2>&1 | tee -a "$LOG"

  echo "=== analyze concat_matched $fold ===" | tee -a "$LOG"
  $PY -m iemocap_mer.cli --config "$CONFIG" analyze --model concat_matched --fold "$fold" --tag "$TAG" \
    2>&1 | tee -a "$LOG"

  echo "=== analyze essence $fold ===" | tee -a "$LOG"
  $PY -m iemocap_mer.cli --config "$CONFIG" analyze --model essence --fold "$fold" --tag "$TAG" \
    2>&1 | tee -a "$LOG"
done

$PY -m iemocap_mer.cli summarize 2>&1 | tee -a "$LOG"
echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
