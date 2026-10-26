#!/usr/bin/env bash
# Resume AU feature extraction (skips existing .pt files unless --overwrite).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PYTHON:-python3}"
LOG="artifacts/reports/video_au_extraction.log"
PROGRESS="artifacts/reports/video_au_progress.log"
EXPECTED=5531
mkdir -p artifacts/reports

count_pt() {
  find artifacts/features/video_au -name '*.pt' ! -name '._*' 2>/dev/null | wc -l | tr -d ' '
}

log_progress() {
  local n pct
  n="$(count_pt)"
  pct="$(awk "BEGIN {printf \"%.1f\", 100*$n/$EXPECTED}")"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] extracted=$n/$EXPECTED (${pct}%)" | tee -a "$PROGRESS"
}

echo "RESUME_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG" "$PROGRESS"
log_progress

$PY -m iemocap_mer.cli --config configs/video_au.yaml extract-features --modality video --device cpu \
  >> "$LOG" 2>&1 &
MAIN_PID=$!

# Background progress watcher (logs every ~500 new files or every 10 min)
(
  last="$(count_pt)"
  last_time=$(date +%s)
  while kill -0 "$MAIN_PID" 2>/dev/null; do
    sleep 60
    n="$(count_pt)"
    now=$(date +%s)
    if (( n - last >= 500 )) || (( now - last_time >= 600 )); then
      log_progress
      last=$n
      last_time=$now
    fi
  done
) &
WATCH_PID=$!

wait "$MAIN_PID"
EXIT=$?
kill "$WATCH_PID" 2>/dev/null || true

log_progress
echo "RESUME_END $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$EXIT" | tee -a "$LOG" "$PROGRESS"

if [[ "$EXIT" -ne 0 ]]; then
  exit "$EXIT"
fi

echo "VERIFY $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG" "$PROGRESS"
$PY -m iemocap_mer.cli --config configs/video_au.yaml verify-features --modality video \
  | tee -a "$LOG"

echo "TRAIN_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG" "$PROGRESS"
bash scripts/run_w2v_au.sh
TRAIN_EXIT=$?
echo "PIPELINE_END $(date -u +%Y-%m-%dT%H:%M:%SZ) train_exit=$TRAIN_EXIT" | tee -a "$LOG" "$PROGRESS"
exit "$TRAIN_EXIT"
