#!/usr/bin/env bash
# Multi-seed training grid (Priority A/B = Wav2Vec2+v2; C = Wav2Vec2+AU).
# Sequential on MPS. Never overwrites seed-42 canonical tags
# (w2v / w2v_v2 / w2v_v2_stage2fix / w2v_au).
set -uo pipefail
# Avoid set -e: arithmetic/false statuses in status writer must not abort the grid.

cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PYTHON:-python3}"
CKPT_DIR=artifacts/checkpoints
SEEDS=(0 1 7)
FOLDS=(1 2 3 4 5)
# PRIORITY=A|B|C|ALL — default A (A = audio+concat_matched+essence v2; B = concat v2; C = concat_matched+essence AU)
PRIORITY="${PRIORITY:-A}"
if [[ "$PRIORITY" == "C" ]]; then
  MASTER_LOG="${MASTER_LOG:-/tmp/multiseed_au_master.log}"
  STATUS_MD="${STATUS_MD:-paper/multiseed_au_status.md}"
else
  MASTER_LOG="${MASTER_LOG:-/tmp/multiseed_master.log}"
  STATUS_MD="${STATUS_MD:-paper/multiseed_grid_status.md}"
fi

mkdir -p "$CKPT_DIR" "$(dirname "$STATUS_MD")"
# Append-only master log (preserve prior sessions)
touch "$MASTER_LOG"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$MASTER_LOG"
}

metrics_path() {
  local model=$1 fold=$2 tag=$3
  echo "${CKPT_DIR}/${model}_fold_${fold}_${tag}.metrics.json"
}

job_log_path() {
  local model=$1 fold=$2 seed=$3
  echo "/tmp/multiseed_${model}_fold${fold}_seed${seed}.log"
}

# Counters (updated as we go)
PLANNED=0
SKIPPED=0
RAN=0
FAILED=0
CONSEC_CRASH=0
FALLBACK=0
CURRENT_JOB="(idle)"
STARTED_AT=$(date +%s)

declare -a REMAINING_JOBS=()
declare -a FALLBACK_NOTES=()

estimate_minutes() {
  local model=$1
  case "$model" in
    audio) echo 4 ;;
    concat) echo 7 ;;
    concat_matched) echo 8 ;;
    essence) echo 10 ;;
    *) echo 8 ;;
  esac
}

write_status() {
  local now elapsed remaining_n eta_min eta_lo eta_hi
  now=$(date +%s)
  elapsed=$((now - STARTED_AT))
  remaining_n=${#REMAINING_JOBS[@]}
  eta_min=0
  if ((remaining_n)); then
    for j in "${REMAINING_JOBS[@]}"; do
      local m
      m=$(echo "$j" | cut -d'|' -f4)
      eta_min=$((eta_min + ${m:-8}))
    done
  fi
  eta_lo=$eta_min
  eta_hi=$((eta_min + remaining_n * 2))

  {
    cat << STATUS
# Multi-seed grid status

Updated: $(date '+%Y-%m-%d %H:%M:%S')

## Summary

| Metric | Value |
|--------|-------|
| Priority scope | \`${PRIORITY}\` (A = audio + concat_matched + essence v2; B = concat v2; C = concat_matched + essence AU) |
| Seeds | {0, 1, 7} × folds {1..5} |
| Total jobs planned | ${PLANNED} |
| Skipped (metrics exist) | ${SKIPPED} |
| Completed this run | ${RAN} |
| Failed | ${FAILED} |
| Remaining | ${remaining_n} |
| Currently running | \`${CURRENT_JOB}\` |
| Elapsed | $((elapsed / 60))m $((elapsed % 60))s |
| ETA (remaining) | ~${eta_lo}–${eta_hi} min (~5–10 min/job for concat_matched/essence; less for audio) |

## Paths

- Master log: \`${MASTER_LOG}\`
- Per-job logs: \`/tmp/multiseed_<model>_fold<K>_seed<S>.log\`
- Checkpoints: \`${CKPT_DIR}/\`
- Orchestrator: \`scripts/run_multiseed_grid.sh\`

## Monitor

\`\`\`bash
tail -f ${MASTER_LOG}
ps aux | grep 'iemocap_mer.cli' | grep -v grep
ls -lt ${CKPT_DIR}/*seed*.metrics.json 2>/dev/null | head
\`\`\`

## Constraints

- Does **not** overwrite seed-42 canonical tags (\`w2v\`, \`w2v_v2\`, \`w2v_v2_stage2fix\`, \`w2v_au\`).
- Multi-seed tags: \`w2v_seed{S}\`, \`w2v_v2_seed{S}\`, \`w2v_v2_stage2fix_seed{S}\`, \`w2v_au_seed{S}\` (Priority C).
- If essence collapses with NaNs, training falls back to Stage-1 and sets \`used_stage1_fallback: true\` — run is kept.

## Stage-1 fallback notes

STATUS
    if ((${#FALLBACK_NOTES[@]})); then
      for note in "${FALLBACK_NOTES[@]}"; do
        echo "- ${note}"
      done
    else
      echo "_None recorded yet._"
    fi
    echo ""
    echo "## Remaining queue (up to 20)"
    echo ""
    if ((remaining_n == 0)); then
      echo "_None — grid complete for this priority scope._"
    else
      local shown=0 model fold seed mins
      for j in "${REMAINING_JOBS[@]}"; do
        IFS='|' read -r model fold seed mins _ <<< "$j"
        echo "- \`${model}\` fold_${fold} seed=${seed} (~${mins} min)"
        shown=$((shown + 1))
        if ((shown >= 20)); then break; fi
      done
      if ((remaining_n > 20)); then
        echo "- … and $((remaining_n - 20)) more"
      fi
    fi
  } > "$STATUS_MD"
}

enqueue_or_skip() {
  local model=$1 config=$2 tag_prefix=$3 seed=$4 fold=$5
  local tag="${tag_prefix}${seed}"
  local metrics
  metrics=$(metrics_path "$model" "$fold" "$tag")
  PLANNED=$((PLANNED + 1))
  if [[ -f "$metrics" ]]; then
    SKIPPED=$((SKIPPED + 1))
    log "SKIP ${model} fold_${fold} seed=${seed} tag=${tag} (metrics exist: $(basename "$metrics"))"
    # Record existing fallback if present
    if [[ "$model" == "essence" ]]; then
      local fb
      fb=$("$PY" -c "import json; d=json.load(open('$metrics')); print('1' if d.get('used_stage1_fallback') else '0')" 2>/dev/null || echo 0)
      if [[ "$fb" == "1" ]]; then
        FALLBACK=$((FALLBACK + 1))
        FALLBACK_NOTES+=("pre-existing: essence fold_${fold} seed=${seed} used_stage1_fallback=true")
      fi
    fi
    return 0
  fi
  local mins
  mins=$(estimate_minutes "$model")
  REMAINING_JOBS+=("${model}|${fold}|${seed}|${mins}|${config}|${tag}")
}

run_one_job() {
  local model=$1 fold=$2 seed=$3 mins=$4 config=$5 tag=$6
  local metrics jlog attempt max_attempts rc
  metrics=$(metrics_path "$model" "$fold" "$tag")
  jlog=$(job_log_path "$model" "$fold" "$seed")
  CURRENT_JOB="${model} fold_${fold} seed=${seed} tag=${tag}"
  write_status

  max_attempts=2
  rc=1
  for attempt in $(seq 1 "$max_attempts"); do
    log "START ${CURRENT_JOB} attempt=${attempt}/${max_attempts} config=${config} log=${jlog}"
    : > "$jlog"
    "$PY" -u -m iemocap_mer.cli --config "$config" train \
      --model "$model" --fold "fold_${fold}" --tag "$tag" --seed "$seed" \
      > "$jlog" 2>&1
    rc=$?
    if [[ $rc -eq 0 && -f "$metrics" ]]; then
      break
    fi
    # Segfault / abort: cool down MPS before retry or abort cascade
    if [[ $rc -eq 139 || $rc -eq 134 || $rc -eq 137 ]]; then
      CONSEC_CRASH=$((CONSEC_CRASH + 1))
      log "CRASH ${CURRENT_JOB} rc=${rc} consec_crash=${CONSEC_CRASH} (see ${jlog})"
      if ((CONSEC_CRASH >= 3)); then
        log "ABORT: ${CONSEC_CRASH} consecutive hard crashes (MPS likely wedged). Re-run later; metrics-absent jobs will resume."
        write_status
        exit 2
      fi
      if ((attempt < max_attempts)); then
        log "cooldown 15s before retry"
        sleep 15
      fi
    else
      CONSEC_CRASH=0
      break
    fi
  done

  if [[ $rc -ne 0 || ! -f "$metrics" ]]; then
    FAILED=$((FAILED + 1))
    log "FAIL ${CURRENT_JOB} rc=${rc} (see ${jlog})"
    tail -20 "$jlog" 2>/dev/null | tee -a "$MASTER_LOG" || true
  else
    CONSEC_CRASH=0
    RAN=$((RAN + 1))
    local f1 fb_note=""
    f1=$("$PY" -c "import json; print(round(json.load(open('$metrics'))['test']['macro_f1'], 4))" 2>/dev/null || echo "?")
    if [[ "$model" == "essence" ]]; then
      local fb s2
      fb=$("$PY" -c "import json; d=json.load(open('$metrics')); print(bool(d.get('used_stage1_fallback')))" 2>/dev/null || echo "?")
      s2=$("$PY" -c "import json; d=json.load(open('$metrics')); print(bool(d.get('stage2_reached')))" 2>/dev/null || echo "?")
      fb_note=" stage2_reached=${s2} used_stage1_fallback=${fb}"
      if [[ "$fb" == "True" ]]; then
        FALLBACK=$((FALLBACK + 1))
        FALLBACK_NOTES+=("essence fold_${fold} seed=${seed}: used_stage1_fallback=true (kept)")
      fi
    fi
    log "DONE ${CURRENT_JOB} macro_f1=${f1}${fb_note}"

    "$PY" -u -m iemocap_mer.cli --config "$config" analyze \
      --model "$model" --fold "fold_${fold}" --tag "$tag" \
      >> "$jlog" 2>&1
    local arc=$?
    if [[ $arc -ne 0 ]]; then
      log "WARN analyze failed for ${CURRENT_JOB} rc=${arc} (train kept)"
    else
      log "ANALYZE OK ${CURRENT_JOB}"
    fi
  fi

  if ((${#REMAINING_JOBS[@]})); then
    REMAINING_JOBS=("${REMAINING_JOBS[@]:1}")
  fi
  CURRENT_JOB="(idle between jobs)"
  write_status
  # brief pause between jobs to reduce MPS flakiness
  sleep 2
}

build_queue_priority_a() {
  log "=== Building Priority A queue ==="
  local seed fold
  for seed in "${SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      enqueue_or_skip audio configs/audio_w2v.yaml w2v_seed "$seed" "$fold"
    done
  done
  for seed in "${SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      enqueue_or_skip concat_matched configs/w2v_video_v2.yaml w2v_v2_seed "$seed" "$fold"
    done
  done
  for seed in "${SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      enqueue_or_skip essence configs/w2v_video_v2.yaml w2v_v2_stage2fix_seed "$seed" "$fold"
    done
  done
}

build_queue_priority_b() {
  log "=== Building Priority B queue (plain concat) ==="
  local seed fold
  for seed in "${SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      enqueue_or_skip concat configs/w2v_video_v2.yaml w2v_v2_seed "$seed" "$fold"
    done
  done
}

build_queue_priority_c() {
  log "=== Building Priority C queue (Wav2Vec2 + AU concat_matched + essence) ==="
  local seed fold
  for seed in "${SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      enqueue_or_skip concat_matched configs/w2v_video_au.yaml w2v_au_seed "$seed" "$fold"
    done
  done
  for seed in "${SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      enqueue_or_skip essence configs/w2v_video_au.yaml w2v_au_seed "$seed" "$fold"
    done
  done
}

# --- main ---
log "=== Multi-seed grid start (PRIORITY=${PRIORITY}) pid=$$ ==="
log "PY=${PY} cwd=$(pwd)"

case "$PRIORITY" in
  A) build_queue_priority_a ;;
  B) build_queue_priority_b ;;
  C) build_queue_priority_c ;;
  ALL)
    build_queue_priority_a
    build_queue_priority_b
    build_queue_priority_c
    ;;
  *)
    log "ERROR: PRIORITY must be A, B, C, or ALL (got ${PRIORITY})"
    exit 1
    ;;
esac

log "Planned=${PLANNED} Skipped=${SKIPPED} Queued=${#REMAINING_JOBS[@]}"
write_status

while ((${#REMAINING_JOBS[@]})); do
  IFS='|' read -r model fold seed mins config tag <<< "${REMAINING_JOBS[0]}"
  run_one_job "$model" "$fold" "$seed" "$mins" "$config" "$tag"
done

CURRENT_JOB="(complete)"
write_status
log "=== Multi-seed grid finished: ran=${RAN} skipped=${SKIPPED} failed=${FAILED} fallback=${FALLBACK} ==="
exit $([[ "$FAILED" -eq 0 ]] && echo 0 || echo 1)
