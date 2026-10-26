#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python3}
CFG=configs/essence_v2.yaml

for fold in fold_1 fold_4; do
  echo "=== essence v2 $fold $(date) ==="
  "$PY" -m iemocap_mer.cli --config "$CFG" train --model essence --fold "$fold" --tag v2
done

"$PY" -m iemocap_mer.cli research-report
