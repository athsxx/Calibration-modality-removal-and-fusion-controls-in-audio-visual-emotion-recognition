#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python3}
CFG=configs/video_v2.yaml

echo "=== Phase 3: Session 3 video_v2 extraction (worst v1 face-track session) ==="
"$PY" -m iemocap_mer.cli --config "$CFG" extract-features \
  --modality video --pretrained --device mps --session 3 --overwrite \
  --encoder-batch-size 128

"$PY" -m iemocap_mer.cli --config "$CFG" verify-features --modality video
"$PY" -m iemocap_mer.cli --config "$CFG" video-audit

echo "=== Phase 3: video baseline fold_3 with video_v2 features ==="
"$PY" -m iemocap_mer.cli --config "$CFG" train --model video --fold fold_3 --tag v2

echo "=== Phase 3 session-3 smoke complete ==="
