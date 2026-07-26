#!/bin/bash
# HENRI Phase 3.3: Chimera Phase-Lag Swarm Syncytium & Interoceptive Viability Experiment
# Pre-registered criteria: see HANDOFF.md
# NOTE: ARMs must run sequentially — 5090 has 32 GiB, each run uses ~28 GiB

set -euo pipefail
cd "/workspace/HENRI/HENRI V2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="phase33_${TIMESTAMP}"
mkdir -p logs telemetry_logs

COMMON="ZONE_C_ENV=prod BETA_PRAGMATIC=10.0 PROGRESS_VALENCE=0 CONSTRAINT_AXIOM=1 LAMBDA_CONSTRAINT_MAX=5.0 GRID_DIST_EPISTEMIC=1"
PY="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 production_arc_run.py"

# ── ARM1: Baseline (No Chimera, default constraint reject thresh = 0.25) ──
echo "=== ARM1: CHIMERA_MODE=0, thresh=0.25 (baseline) ===" | tee "logs/${RUN_ID}_arm1.log"
env ${COMMON} CHIMERA_MODE=0 CONSTRAINT_REJECT_THRESH=0.25 LAMBDA_GOAL=1.0 \
  ${PY} --envs 3 --steps 40 2>&1 | tee -a "logs/${RUN_ID}_arm1.log"
echo "ARM1 done."

# ── ARM2: Chimera Mode Enabled (alpha=1.4, explorer_fraction=0.25) ──
echo "=== ARM2: CHIMERA_MODE=1, alpha=1.4, fraction=0.25 ===" | tee "logs/${RUN_ID}_arm2.log"
env ${COMMON} CHIMERA_MODE=1 CHIMERA_ALPHA=1.4 CHIMERA_EXPLORER_FRACTION=0.25 CONSTRAINT_REJECT_THRESH=0.25 LAMBDA_GOAL=1.0 \
  ${PY} --envs 3 --steps 40 2>&1 | tee -a "logs/${RUN_ID}_arm2.log"
echo "ARM2 done."

# ── ARM3: Chimera Mode + Aggressive Explorer (alpha=1.8, fraction=0.35) ──
echo "=== ARM3: CHIMERA_MODE=1, alpha=1.8, fraction=0.35 ===" | tee "logs/${RUN_ID}_arm3.log"
env ${COMMON} CHIMERA_MODE=1 CHIMERA_ALPHA=1.8 CHIMERA_EXPLORER_FRACTION=0.35 CONSTRAINT_REJECT_THRESH=0.25 LAMBDA_GOAL=3.0 \
  ${PY} --envs 3 --steps 40 2>&1 | tee -a "logs/${RUN_ID}_arm3.log"
echo "ARM3 done."

echo ""
echo "=== Phase 3.3 COMPLETE ==="
echo "Logs: logs/${RUN_ID}_arm*.log"
for arm in arm1 arm2 arm3; do
    LOG="logs/${RUN_ID}_${arm}.log"
    if [ -f "$LOG" ]; then
        echo "  ${arm}: $(grep -c 'FINAL SCORECARDS' $LOG || echo 0) scorecards"
    fi
done
