#!/usr/bin/env bash
# Overnight orchestrator for the MTAP revision (EXECUTION_PLAN.md, Section 3).
# Resumable: completed runs (metrics status "complete") are skipped; failed runs are logged and kept.
# Usage: caffeinate -dims bash scripts/mtap_overnight.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-python3}"
LOG_DIR=mtap_outputs/logs
STATUS=$LOG_DIR/overnight_status.md
mkdir -p "$LOG_DIR"
SEEDS=(0 1 7)
FOLDS=(1 2 3 4 5)

log() { echo "- $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }

metrics_status() {  # $1 = metrics json path -> complete | failed | missing
  [ -f "$1" ] || { echo missing; return; }
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('status','complete'))" "$1" 2>/dev/null || echo failed
}

verify_cache() {  # $1 config, $2 modality, $3 corpus, $4 name
  local out=$LOG_DIR/verify_$3_$4.json
  "$PY" -m iemocap_mer.cli --config "$1" verify-features --modality "$2" > "$out" 2>>"$LOG_DIR/$3_features.log"
  "$PY" - "$out" <<'PY'
import json, sys
text = open(sys.argv[1]).read()
report = json.loads(text[text.index("{"):])
bad = int(report.get("missing_count", 0)) + int(report.get("invalid_count", 0))
print(f"expected={report.get('expected')} valid={report.get('valid_count', report.get('valid'))} missing={report.get('missing_count')} invalid={report.get('invalid_count')}")
sys.exit(1 if bad else 0)
PY
}

extract_corpus() {  # $1 corpus, $2 audio device, $3 video device
  local c=$1 flog=$LOG_DIR/$1_features.log
  log "$c: feature extraction start (audio $2, video $3)"
  "$PY" -m iemocap_mer.cli --config configs/${c}_audio_w2v.yaml extract-features --modality audio --backend wav2vec2 --device "$2" >> "$flog" 2>&1 || { log "$c: audio extraction FAILED"; return 1; }
  "$PY" -m iemocap_mer.cli --config configs/${c}_video_v2.yaml extract-features --modality video --pretrained --device "$3" >> "$flog" 2>&1 || { log "$c: ResNet video extraction FAILED"; return 1; }
  "$PY" -m iemocap_mer.cli --config configs/${c}_video_au.yaml extract-features --modality video --device cpu >> "$flog" 2>&1 || { log "$c: AU extraction FAILED"; return 1; }
  local ok=0
  for spec in "audio_w2v audio audio_w2v" "video_v2 video video_v2" "video_au video video_au"; do
    set -- $spec
    if r=$(verify_cache configs/${c}_$1.yaml "$2" "$c" "$3"); then log "$c: $3 cache OK ($r)"; else log "$c: $3 cache FAILED ($r)"; ok=1; fi
  done
  return $ok
}

train_one() {  # $1 corpus, $2 config, $3 model, $4 system, $5 seed, $6 fold, $7 artifacts root
  local tag="$1_$4_seed$5" stem="$3_fold_$6_$1_$4_seed$5"
  local metrics="$7/checkpoints/${stem}.metrics.json" st
  st=$(metrics_status "$metrics")
  if [ "$st" = complete ]; then return 0; fi
  if [ "$st" = failed ]; then echo "$stem failed earlier; kept, not rerun" >> "$LOG_DIR/failed_runs.txt"; return 0; fi
  local start=$(date +%s)
  if "$PY" -m iemocap_mer.cli --config "$2" train --model "$3" --fold "fold_$6" --seed "$5" --tag "$tag" --device mps >> "$LOG_DIR/train_$1.log" 2>&1; then
    echo "$(date '+%H:%M:%S') ok $stem $(( $(date +%s) - start ))s" >> "$LOG_DIR/train_$1_progress.txt"
  else
    echo "$(date '+%H:%M:%S') FAILED $stem $(( $(date +%s) - start ))s" >> "$LOG_DIR/train_$1_progress.txt"
    echo "$stem" >> "$LOG_DIR/failed_runs.txt"
  fi
}

grid_corpus() {  # $1 corpus, $2 artifacts root
  local c=$1 root=$2
  log "$c: training grid start (7 systems x 3 seeds x 5 folds)"
  for seed in "${SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
    train_one "$c" configs/${c}_audio_w2v.yaml audio audio "$seed" "$fold" "$root"
    train_one "$c" configs/${c}_video_au.yaml video video_au "$seed" "$fold" "$root"
    train_one "$c" configs/${c}_video_v2.yaml video video_resnet "$seed" "$fold" "$root"
    train_one "$c" configs/${c}_w2v_video_v2.yaml concat_matched concat_matched_resnet "$seed" "$fold" "$root"
    train_one "$c" configs/${c}_w2v_video_v2.yaml essence essence_resnet "$seed" "$fold" "$root"
    train_one "$c" configs/${c}_w2v_video_au.yaml concat_matched concat_matched_au "$seed" "$fold" "$root"
    train_one "$c" configs/${c}_w2v_video_au.yaml essence essence_au "$seed" "$fold" "$root"
  done; done
  log "$c: training grid finished ($(grep -c ok "$LOG_DIR/train_${c}_progress.txt" 2>/dev/null || echo 0) ok lines total; failures in failed_runs.txt)"
}

cremad_cpu_lane() {
  if [ -f "$LOG_DIR/cremad_features.done" ]; then
    log "cremad: features already verified; skipping media and extraction"
    return 0
  fi
  log "cremad: waiting for media fetch"
  until grep -q '^exit=' "$LOG_DIR/cremad_fetch.log" 2>/dev/null; do sleep 60; done
  if ! grep -q '^exit=0' "$LOG_DIR/cremad_fetch.log"; then log "cremad: media fetch FAILED; see cremad_fetch.log"; return 1; fi
  log "cremad: media fetched; rebuilding manifest and splits"
  "$PY" -m iemocap_mer.cli --config configs/cremad_w2v_video_au.yaml build-corpus-manifest --corpus cremad --raw "${CREMAD_ROOT:-../CREMA-D}" >> "$LOG_DIR/cremad_features.log" 2>&1 || { log "cremad: manifest FAILED"; return 1; }
  "$PY" -m iemocap_mer.cli --config configs/cremad_w2v_video_au.yaml build-splits >> "$LOG_DIR/cremad_features.log" 2>&1 || { log "cremad: splits FAILED"; return 1; }
  extract_corpus cremad cpu cpu && touch "$LOG_DIR/cremad_features.done"
}

log "orchestrator start (pid $$)"

# Lane B (CPU): CREMA-D media -> manifest -> features, in parallel with RAVDESS work on MPS.
( cremad_cpu_lane ) &
CPU_LANE=$!

# Lane A (MPS).
if [ ! -f "$LOG_DIR/ravdess_features.done" ]; then
  extract_corpus ravdess mps mps && touch "$LOG_DIR/ravdess_features.done"
fi
if [ -f "$LOG_DIR/ravdess_features.done" ]; then
  grid_corpus ravdess artifacts_ravdess
else
  log "ravdess: features not verified; grid skipped"
fi

log "iemocap: video-only AU runs start"
for seed in "${SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
  train_one iemocap configs/video_au.yaml video video_au "$seed" "$fold" artifacts
done; done
log "iemocap: video-only AU runs finished"

wait "$CPU_LANE"
if [ -f "$LOG_DIR/cremad_features.done" ]; then
  grid_corpus cremad artifacts_cremad
else
  log "cremad: features not verified; grid skipped"
fi
log "orchestrator finished"
