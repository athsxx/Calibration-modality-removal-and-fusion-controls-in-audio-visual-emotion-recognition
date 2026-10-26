#!/usr/bin/env bash
# Faithful CML (Ma et al., Algorithm 1) on ConcatMatched-AU.
# Resumable, one MPS job at a time. Does not touch the CML-inspired penalty grid.
# A metrics status of "complete" is left alone. A "failed" status is kept.
# A checkpoint with no metrics file is a partial write and is deleted, then retrained.
# Usage: caffeinate -dims bash scripts/mtap_cml_faithful_grid.sh
#        bash scripts/mtap_cml_faithful_grid.sh smoke   # one RAVDESS run (seed 0, fold 1)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
LOG_DIR=mtap_outputs/logs
STATUS=$LOG_DIR/cmlf_status.md
STAGING="${STAGING:-../cmlf_export_staging}"
mkdir -p "$LOG_DIR"
SEEDS=(0 1 7)
FOLDS=(1 2 3 4 5)

log() { echo "- $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }

metrics_status() {  # $1 = metrics json path -> complete | failed | missing
  [ -f "$1" ] || { echo missing; return; }
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('status','complete'))" "$1" 2>/dev/null || echo failed
}

train_one() {  # $1 corpus, $2 config, $3 model, $4 system, $5 seed, $6 fold, $7 artifacts root
  local tag="$1_$4_seed$5" stem="$3_fold_$6_$1_$4_seed$5"
  local pt="$7/checkpoints/${stem}.pt" metrics="$7/checkpoints/${stem}.metrics.json" st
  if [ -f "$pt" ] && [ ! -f "$metrics" ]; then
    rm -f "$pt"
    echo "$(date '+%Y-%m-%d %H:%M:%S') removed partial $stem" >> "$LOG_DIR/cmlf_partial_removed.txt"
  fi
  st=$(metrics_status "$metrics")
  if [ "$st" = complete ]; then return 0; fi
  if [ "$st" = failed ]; then echo "$stem failed earlier; kept, not rerun" >> "$LOG_DIR/cmlf_failed_runs.txt"; return 0; fi
  local start=$(date +%s)
  if "$PY" -m iemocap_mer.cli --config "$2" train --model "$3" --fold "fold_$6" --seed "$5" --tag "$tag" --device mps >> "$LOG_DIR/cmlf_train_$1.log" 2>&1; then
    echo "$(date '+%H:%M:%S') ok $stem $(( $(date +%s) - start ))s" >> "$LOG_DIR/cmlf_train_$1_progress.txt"
  else
    echo "$(date '+%H:%M:%S') FAILED $stem $(( $(date +%s) - start ))s" >> "$LOG_DIR/cmlf_train_$1_progress.txt"
    echo "$stem" >> "$LOG_DIR/cmlf_failed_runs.txt"
  fi
}

run_corpus() {  # $1 corpus, $2 config, $3 artifacts root
  local c=$1
  log "$c: faithful CML ConcatMatched-AU (seeds 0,1,7; folds 1-5)"
  for seed in "${SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
    train_one "$c" "$2" concat_matched concat_matched_au_cmlfaithful "$seed" "$fold" "$3"
  done; done
  log "$c: faithful CML training pass finished"
}

manifest_complete() {  # $1 export dir -> yes or no
  [ -f "$1/export_manifest.json" ] || { echo no; return; }
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get('complete') else 1)" "$1/export_manifest.json" \
    && echo yes || echo no
}

export_corpus() {  # $1 corpus, $2 artifacts, $3 manifest, $4 splits
  local c=$1 final="mtap_outputs/exports_cmlf_$1" stage="$STAGING/$1" done
  done=$(manifest_complete "$final")
  if [ "$done" = yes ]; then log "$c: export already complete"; return 0; fi
  if [ -d "$final" ]; then log "$c: export destination exists and is not complete; left untouched"; return 1; fi
  mkdir -p "$STAGING"
  log "$c: exporting to $stage"
  if [ -d "$stage" ]; then
    if ! "$PY" scripts/mtap_export_corpus.py \
        --artifacts-root "$2" --manifest "$3" --splits "$4" --dataset "$c" \
        --run-spec "mtap_outputs/run_specs/${c}_cmlfaithful.csv" --out "$stage" \
        --device mps --resume >> "$LOG_DIR/cmlf_export_$c.log" 2>&1; then
      log "$c: export failed; staging kept for resume"
      return 1
    fi
  elif ! "$PY" scripts/mtap_export_corpus.py \
      --artifacts-root "$2" --manifest "$3" --splits "$4" --dataset "$c" \
      --run-spec "mtap_outputs/run_specs/${c}_cmlfaithful.csv" --out "$stage" \
      --device mps >> "$LOG_DIR/cmlf_export_$c.log" 2>&1; then
    log "$c: export failed; staging kept for resume"
    return 1
  fi
  done=$(manifest_complete "$stage")
  if [ "$done" != yes ]; then log "$c: export incomplete; not moved"; return 1; fi
  mv "$stage" "$final"
  log "$c: export moved to $final"
}

compare_and_analyse() {  # $1 corpus, $2 baseline export, $3 baseline model name
  local c=$1 view="mtap_outputs/exports_cmlf_compare_$1" analysis="mtap_outputs/analysis_cmlf_$1"
  local fresh="mtap_outputs/exports_cmlf_$1" penalty="mtap_outputs/exports_cml_$1"
  if [ ! -d "$view" ]; then
    log "$c: building combined view"
    "$PY" scripts/build_step9_combined_view.py \
      --include "$2" "$3" \
      --include "$penalty" concat_matched_au_cml \
      --include "$fresh" concat_matched_au_cmlfaithful \
      --out "$view" >> "$LOG_DIR/cmlf_compare_$c.log" 2>&1 || { log "$c: combined view failed"; return 1; }
  else
    log "$c: combined view already present"
  fi
  if [ -d "$analysis" ]; then log "$c: analysis already present"; return 0; fi
  log "$c: speaker-level analysis"
  "$PY" scripts/mtap_strong_analysis.py \
    --exports "$view" --out "$analysis" \
    --pairs-json "mtap_outputs/protocol/pairs_cmlfaithful_$c.json" \
    >> "$LOG_DIR/cmlf_analysis_$c.log" 2>&1 || { log "$c: analysis failed"; return 1; }
  log "$c: analysis finished"
}

if [ "${1:-}" = smoke ]; then
  log "smoke: ravdess concat_matched_au_cmlfaithful seed 0 fold 1"
  train_one ravdess configs/ravdess_w2v_video_au_cmlfaithful.yaml concat_matched concat_matched_au_cmlfaithful 0 1 artifacts_ravdess
  exit 0
fi

log "faithful CML grid start (pid $$)"
run_corpus iemocap configs/iemocap_w2v_video_au_cmlfaithful.yaml artifacts
run_corpus ravdess configs/ravdess_w2v_video_au_cmlfaithful.yaml artifacts_ravdess
run_corpus cremad configs/cremad_w2v_video_au_cmlfaithful.yaml artifacts_cremad

export_corpus iemocap artifacts manifests/iemocap_utterances.csv splits/emotion4_session_folds.json
export_corpus ravdess artifacts_ravdess manifests/utterances.csv splits/emotion4_session_folds.json
export_corpus cremad artifacts_cremad manifests/utterances.csv splits/emotion4_session_folds.json

compare_and_analyse iemocap mtap_outputs/exports concatmatched_au
compare_and_analyse ravdess mtap_outputs/exports_ravdess concat_matched_au
compare_and_analyse cremad mtap_outputs/exports_cremad concat_matched_au
log "faithful CML grid finished"
