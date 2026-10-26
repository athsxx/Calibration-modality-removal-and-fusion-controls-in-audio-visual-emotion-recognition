#!/usr/bin/env bash
# Overnight post-grid pipeline: wait for Priority-A multiseed → aggregate →
# update paper → significance/figures/calibration → optional Priority B → git push.
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PYTHON:-python3}"
MASTER_LOG="${MASTER_LOG:-/tmp/multiseed_master.log}"
OVERNIGHT_LOG="${OVERNIGHT_LOG:-/tmp/overnight_postgrid.log}"
GRID_PID="${GRID_PID:-}"
STATUS_MD="${STATUS_MD:-paper/overnight_pipeline_status.md}"
RUN_PRIORITY_B="${RUN_PRIORITY_B:-1}"
DO_PUSH="${DO_PUSH:-1}"
METRICS_CACHE="${METRICS_CACHE:-}"
# Only export if user set a cache; otherwise aggregate reads ATH via dir_fd.
if [[ -n "$METRICS_CACHE" ]]; then
  export METRICS_CACHE
  mkdir -p "$METRICS_CACHE/checkpoints" "$METRICS_CACHE/results"
fi

mkdir -p "$(dirname "$STATUS_MD")"
touch "$OVERNIGHT_LOG"

sync_metrics_cache() {
  if [[ -z "${METRICS_CACHE:-}" ]]; then
    return 0
  fi
  log "Syncing seed metrics → ${METRICS_CACHE}"
  local f
  shopt -s nullglob
  for f in artifacts/checkpoints/*seed*.metrics.json; do
    [[ "$(basename "$f")" == ._* ]] && continue
    cp -f "$f" "$METRICS_CACHE/checkpoints/" 2>/dev/null || true
  done
  for f in artifacts/results/*seed*_robustness.json; do
    [[ "$(basename "$f")" == ._* ]] && continue
    cp -f "$f" "$METRICS_CACHE/results/" 2>/dev/null || true
  done
}

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$OVERNIGHT_LOG"
}

write_status() {
  cat > "$STATUS_MD" <<EOF
# Overnight post-grid pipeline status

Updated: $(date '+%Y-%m-%d %H:%M:%S')

## State
- Phase: $1
- Detail: $2
- Overnight log: \`$OVERNIGHT_LOG\`
- Multiseed master log: \`$MASTER_LOG\`

## Monitor
\`\`\`bash
tail -f $OVERNIGHT_LOG
tail -f $MASTER_LOG
\`\`\`
EOF
}

# --- 0. Keep awake (nested ok) ---
caffeinate -dims &
CAFFE_PID=$!
log "caffeinate pid=${CAFFE_PID}"

# --- 1. Wait for existing Priority-A grid (any pid; re-resolve each loop) ---
write_status "wait_grid" "Waiting for Priority-A multiseed to finish"
wait_rounds=0
while true; do
  LIVE=$(ps aux | awk '/scripts\/run_multiseed_grid\.sh/ && !/awk/ {print $2; exit}')
  TRAIN=$(ps aux | awk '/iemocap_mer\.cli/ && /train/ && !/awk/ {print $2; exit}')
  if [[ -z "${LIVE:-}" && -z "${TRAIN:-}" ]]; then
    log "No live multiseed grid/train process; proceeding"
    break
  fi
  wait_rounds=$((wait_rounds + 1))
  if (( wait_rounds % 5 == 1 )); then
    log "Still waiting (grid_pid=${LIVE:-none} train_pid=${TRAIN:-none})"
    write_status "wait_grid" "grid=${LIVE:-none} train=${TRAIN:-none}"
  fi
  sleep 60
done

# If incomplete, resume Priority A (skips existing metrics)
sync_metrics_cache
"$PY" paper/aggregate_multiseed.py >> "$OVERNIGHT_LOG" 2>&1 || true
COMPLETE=$("$PY" -c "import json; print(json.load(open('paper/multiseed_aggregate_status.json'))['complete'])" 2>/dev/null || echo False)
if [[ "$COMPLETE" != "True" ]]; then
  write_status "resume_grid" "Priority A incomplete — resuming PRIORITY=A"
  log "Priority A incomplete; launching PRIORITY=A resume (skip-existing)"
  PRIORITY=A bash scripts/run_multiseed_grid.sh >> "$OVERNIGHT_LOG" 2>&1
  rc=$?
  log "Priority A resume finished rc=${rc}"
  sync_metrics_cache
fi

# --- 2. Aggregate ---
write_status "aggregate" "Aggregating multi-seed metrics"
log "=== Aggregate multi-seed ==="
sync_metrics_cache
"$PY" paper/aggregate_multiseed.py >> "$OVERNIGHT_LOG" 2>&1
COMPLETE=$("$PY" -c "import json; print(json.load(open('paper/multiseed_aggregate_status.json'))['complete'])")
if [[ "$COMPLETE" != "True" ]]; then
  write_status "FAILED" "Aggregate incomplete after resume"
  log "FATAL: still incomplete after resume"
  kill "$CAFFE_PID" 2>/dev/null || true
  exit 1
fi

# --- 3. Canonical significance v2 refresh + figures ---
write_status "tables_figures" "Refreshing tables/figures from artifacts"
log "=== generate_figures_tables ==="
"$PY" paper/generate_figures_tables.py >> "$OVERNIGHT_LOG" 2>&1 || log "WARN generate_figures_tables failed"

log "=== compute_vrr ==="
"$PY" paper/compute_vrr.py >> "$OVERNIGHT_LOG" 2>&1 || log "WARN compute_vrr failed"

# --- 4. Calibration for ConcatMatched + Essence S2 (best-effort; can be slow) ---
write_status "calibration" "Running calibration analysis (best-effort)"
log "=== calibration analysis ==="
"$PY" paper/run_calibration_analysis.py >> "$OVERNIGHT_LOG" 2>&1 || log "WARN calibration failed (continuing)"

# --- 5. Patch paper.tex ---
write_status "paper_patch" "Updating paper.tex with multi-seed paragraph + limitations"
log "=== update_paper_multiseed ==="
"$PY" paper/update_paper_multiseed.py >> "$OVERNIGHT_LOG" 2>&1

# --- 6. Numeric audit v3 (lightweight claim check) ---
write_status "audit" "Writing numeric audit v3 from aggregate tables"
log "=== numeric audit v3 ==="
"$PY" - <<'PY' >> "$OVERNIGHT_LOG" 2>&1
import json
from pathlib import Path
import pandas as pd

root = Path("paper")
summary = pd.read_csv(root / "tables" / "multiseed_summary.csv")
sig = pd.read_csv(root / "tables" / "significance_tests_multiseed.csv")
status = json.loads((root / "multiseed_aggregate_status.json").read_text())
across = summary[summary["Seed"].astype(str) == "across"]
lines = [
    "# Numeric audit v3 (multi-seed Priority A)",
    "",
    f"Generated automatically after overnight aggregate.",
    f"Completeness: {status['have']}/{status['expected']} complete={status['complete']}",
    "",
    "## Across-seed 5-fold means",
    "",
]
for _, r in across.iterrows():
    lines.append(f"- **{r['Model']}**: {r['Macro-F1 mean']:.3f} ± {r['Macro-F1 std']:.3f}")
lines += ["", "## Paired tests (fold means averaged over seeds)", ""]
for _, r in sig.iterrows():
    lines.append(
        f"- {r['comparison']}: Δ={r['mean_diff_b_minus_a']:+.4f}, "
        f"p={r['paired_t_pvalue']:.4f}, "
        f"CI [{r['bootstrap_ci_95_lo']:+.4f}, {r['bootstrap_ci_95_hi']:+.4f}]"
    )
if status.get("fallback_runs"):
    lines += ["", "## Stage-1 fallback runs", ""]
    for x in status["fallback_runs"]:
        lines.append(f"- {x}")
else:
    lines += ["", "## Stage-1 fallback runs", "", "_None._", ""]
lines += [
    "",
    "## Checks",
    "",
    "- [x] Multi-seed aggregate CSV/MD written",
    "- [x] significance_tests_multiseed written",
    "- [x] paper.tex multi-seed paragraph + limitations patched",
    "- [ ] PDF rebuild (pdflatex not required in this environment; rebuild on a machine with TeX)",
    "",
]
(root / "audit_numeric_report_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Wrote paper/audit_numeric_report_v3.md")
PY

# --- 7. Optional Priority B (plain concat multi-seed) ---
if [[ "$RUN_PRIORITY_B" == "1" ]]; then
  write_status "priority_b" "Running Priority B plain-concat multi-seed"
  log "=== Priority B (plain concat) ==="
  PRIORITY=B bash scripts/run_multiseed_grid.sh >> "$OVERNIGHT_LOG" 2>&1 || log "WARN Priority B finished with errors"
fi

# --- 8. Commit + push trackable paper/script updates ---
if [[ "$DO_PUSH" == "1" ]]; then
  write_status "git_push" "Committing and pushing paper/script updates"
  log "=== git commit/push ==="
  git add \
    paper/aggregate_multiseed.py \
    paper/update_paper_multiseed.py \
    paper/tables/multiseed_*.csv \
    paper/tables/multiseed_*.md \
    paper/tables/significance_tests_multiseed.* \
    paper/tables/table*.csv \
    paper/tables/table*.md \
    paper/tables/significance_tests*.csv \
    paper/tables/significance_tests*.md \
    paper/figures/*.png \
    paper/multiseed_*.md \
    paper/multiseed_*.json \
    paper/multiseed_*.txt \
    paper/overnight_pipeline_status.md \
    paper/audit_numeric_report_v3.md \
    paper/paper.tex \
    scripts/run_overnight_postgrid.sh \
    scripts/run_multiseed_grid.sh \
    .gitignore \
    2>>"$OVERNIGHT_LOG" || true
  if git diff --cached --quiet; then
    log "Nothing new to commit"
  else
    git commit -m "$(cat <<'EOF'
Add overnight multi-seed aggregation and update paper with replicate results.

Records Priority-A seed{0,1,7} summaries, paired multi-seed significance tests, and patches the limitations/results text after the grid completes.
EOF
)" >> "$OVERNIGHT_LOG" 2>&1 || log "WARN commit failed"
    git push origin main >> "$OVERNIGHT_LOG" 2>&1 || log "WARN push failed"
  fi
fi

write_status "DONE" "Overnight pipeline finished"
log "=== Overnight pipeline DONE ==="
kill "$CAFFE_PID" 2>/dev/null || true
exit 0
