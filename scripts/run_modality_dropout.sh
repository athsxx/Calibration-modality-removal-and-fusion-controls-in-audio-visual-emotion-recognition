#!/usr/bin/env bash
# Train ConcatMatched with train-time modality dropout (5-fold LOSO).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export PYTHONPATH=src
CONFIG=configs/w2v_video_v2_moddropout.yaml
TAG=w2v_v2_moddropout
LOG_DIR=/tmp/moddropout_train
mkdir -p "$LOG_DIR"
START=$(date +%s)
echo "=== Modality-dropout ConcatMatched training started $(date -Iseconds) ==="
for FOLD in fold_1 fold_2 fold_3 fold_4 fold_5; do
  echo "--- Training $FOLD ---"
  $PY -m iemocap_mer.cli --config "$CONFIG" train \
    --model concat_matched \
    --fold "$FOLD" \
    --tag "$TAG" \
    --modality-dropout-video 0.2 \
    --modality-dropout-audio 0.1 \
    2>&1 | tee "$LOG_DIR/${FOLD}.log"
done
END=$(date +%s)
echo "=== Training finished in $((END - START))s ($(date -Iseconds)) ==="

echo "=== Running analyze on all folds ==="
for FOLD in fold_1 fold_2 fold_3 fold_4 fold_5; do
  echo "--- Analyze $FOLD ---"
  $PY -m iemocap_mer.cli --config "$CONFIG" analyze \
    --model concat_matched \
    --fold "$FOLD" \
    --tag "$TAG" \
    2>&1 | tee "$LOG_DIR/${FOLD}_analyze.log"
done
echo "=== Done ==="
