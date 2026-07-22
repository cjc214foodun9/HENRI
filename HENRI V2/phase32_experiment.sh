#!/bin/bash
# HENRI Phase 3.2: Preference-blend goal wave experiment (SEQUENTIAL)
# Pre-registered criteria: see HANDOFF.md
# NOTE: ARMs must run sequentially — 5090 has 32 GiB, each run uses ~28 GiB

set -euo pipefail
cd "/workspace/HENRI/HENRI V2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="phase32_${TIMESTAMP}"
mkdir -p logs telemetry_logs

COMMON="BETA_PRAGMATIC=10.0 PROGRESS_VALENCE=0 CONSTRAINT_AXIOM=1 LAMBDA_CONSTRAINT_MAX=5.0"
PY="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 production_arc_run.py"

# ── ARM1: Baseline (identity fallback — preference store empty at start) ──
echo "=== ARM1: λ_goal=1.0, thresh=0.5 (baseline) ===" | tee "logs/${RUN_ID}_arm1.log"
env ${COMMON} CONSTRAINT_REJECT_THRESH=0.5 LAMBDA_GOAL=1.0 \
  ${PY} --envs 3 --steps 40 2>&1 | tee -a "logs/${RUN_ID}_arm1.log"
echo "ARM1 done."

# ── ARM2: HaPPY threshold (0.25) ──
echo "=== ARM2: λ_goal=1.0, thresh=0.25 (HaPPY tune) ===" | tee "logs/${RUN_ID}_arm2.log"
env ${COMMON} CONSTRAINT_REJECT_THRESH=0.25 LAMBDA_GOAL=1.0 \
  ${PY} --envs 3 --steps 40 2>&1 | tee -a "logs/${RUN_ID}_arm2.log"
echo "ARM2 done."

# ── ARM3: Aggressive goal + tight constraint ──
echo "=== ARM3: λ_goal=3.0, thresh=0.25 (aggressive) ===" | tee "logs/${RUN_ID}_arm3.log"
env ${COMMON} CONSTRAINT_REJECT_THRESH=0.25 LAMBDA_GOAL=3.0 \
  ${PY} --envs 3 --steps 40 2>&1 | tee -a "logs/${RUN_ID}_arm3.log"
echo "ARM3 done."

echo ""
echo "=== Phase 3.2 COMPLETE ==="
echo "Logs: logs/${RUN_ID}_arm*.log"
for arm in arm1 arm2 arm3; do
    LOG="logs/${RUN_ID}_${arm}.log"
    if [ -f "$LOG" ]; then
        echo "  ${arm}: $(grep -c 'FINAL SCORECARDS' $LOG || echo 0) scorecards"
    fi
done
