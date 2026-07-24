#!/bin/bash
# HENRI Phase 3.4: Holographic HaPPY Tensor-Cut & Colored Langevin Noise Experiment
# Pre-registered criteria: see HANDOFF.md
# NOTE: ARMs must run sequentially — 5090 has 32 GiB, each run uses ~28 GiB

set -euo pipefail
cd "/workspace/HENRI/HENRI V2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="phase34_${TIMESTAMP}"
mkdir -p logs telemetry_logs

COMMON="BETA_PRAGMATIC=10.0 PROGRESS_VALENCE=0 CONSTRAINT_AXIOM=1 LAMBDA_CONSTRAINT_MAX=5.0 GRID_DIST_EPISTEMIC=1"
PY="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 production_arc_run.py"

# ── ARM1: Baseline (White Noise, No HaPPY Cut) ──
echo "=== ARM1: COLORED_LANGEVIN=0, HAPPY_TENSOR_CUT=0 (baseline) ===" | tee "logs/${RUN_ID}_arm1.log"
env ${COMMON} CHIMERA_MODE=1 COLORED_LANGEVIN=0 HAPPY_TENSOR_CUT=0 CONSTRAINT_REJECT_THRESH=0.25 LAMBDA_GOAL=1.0 \
  ${PY} --envs 3 --steps 40 2>&1 | tee -a "logs/${RUN_ID}_arm1.log"
echo "ARM1 done."

# ── ARM2: Colored Langevin Exploration ──
echo "=== ARM2: COLORED_LANGEVIN=1, HAPPY_TENSOR_CUT=0 ===" | tee "logs/${RUN_ID}_arm2.log"
env ${COMMON} CHIMERA_MODE=1 COLORED_LANGEVIN=1 HAPPY_TENSOR_CUT=0 CONSTRAINT_REJECT_THRESH=0.25 LAMBDA_GOAL=1.0 \
  ${PY} --envs 3 --steps 40 2>&1 | tee -a "logs/${RUN_ID}_arm2.log"
echo "ARM2 done."

# ── ARM3: Full Protocol (Colored Langevin + HaPPY Tensor Cut) ──
echo "=== ARM3: COLORED_LANGEVIN=1, HAPPY_TENSOR_CUT=1 ===" | tee "logs/${RUN_ID}_arm3.log"
env ${COMMON} CHIMERA_MODE=1 COLORED_LANGEVIN=1 HAPPY_TENSOR_CUT=1 CONSTRAINT_REJECT_THRESH=0.25 LAMBDA_GOAL=1.0 \
  ${PY} --envs 3 --steps 40 2>&1 | tee -a "logs/${RUN_ID}_arm3.log"
echo "ARM3 done."

echo ""
echo "=== Phase 3.4 COMPLETE ==="
echo "Logs: logs/${RUN_ID}_arm*.log"
for arm in arm1 arm2 arm3; do
    LOG="logs/${RUN_ID}_${arm}.log"
    if [ -f "$LOG" ]; then
        echo "  ${arm}: $(grep -c 'FINAL SCORECARDS' $LOG || echo 0) scorecards"
    fi
done
