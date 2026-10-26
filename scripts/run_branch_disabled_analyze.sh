#!/usr/bin/env bash
# Re-run analyze for existing checkpoints to add branch-disabled robustness conditions.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export PYTHONPATH=src
CONFIG=configs/w2v_video_v2.yaml
LOG_DIR=/tmp/branch_disabled_analyze
mkdir -p "$LOG_DIR"
START=$(date +%s)
echo "=== Branch-disabled re-analyze started $(date -Iseconds) ==="
for FOLD in fold_1 fold_2 fold_3 fold_4 fold_5; do
  echo "--- ConcatMatched $FOLD ---"
  $PY -m iemocap_mer.cli --config "$CONFIG" analyze \
    --model concat_matched --fold "$FOLD" --tag w2v_v2 \
    2>&1 | tee "$LOG_DIR/concat_matched_${FOLD}.log"
  echo "--- Essence S2-gated $FOLD ---"
  $PY -m iemocap_mer.cli --config "$CONFIG" analyze \
    --model essence --fold "$FOLD" --tag w2v_v2_stage2fix \
    2>&1 | tee "$LOG_DIR/essence_s2_${FOLD}.log"
done
END=$(date +%s)
echo "=== Re-analyze finished in $((END - START))s ==="
