#!/bin/bash
# Gate 1 payload-engagement factorial — 2x2 with HENRI_ARC_ACTION_PAYLOADS=1.
# Cells: A=action1+frozen, B=efe+frozen, C=action1+learning, D=efe+learning.
# Same envs, seed, budget, ordering as stages 3-5. Runs from the Gate-1
# worktree (henri_g1_wt @ 001fec8) so the payload code is live.
set -u
cd '/workspace/henri_g1_wt/HENRI V2'
declare -A POLICY FREEZE
POLICY[A]=action1; FREEZE[A]=1
POLICY[B]=efe;     FREEZE[B]=1
POLICY[C]=action1; FREEZE[C]=0
POLICY[D]=efe;     FREEZE[D]=0
for cell in A B C D; do
  for env in lf52 tn36 sc25; do
    mkdir -p "/workspace/henri_artifacts/arc_g1/$cell/$env"
    echo "=== G1 $cell $env start $(date -u +%FT%TZ) ==="
    env HENRI_TELEMETRY_DIR="/workspace/henri_artifacts/arc_g1/$cell/$env" \
        HENRI_OFFLINE_DIAG=1 \
        HENRI_SEED=20260809 \
        HENRI_SINGLE_ENV="$env" \
        HENRI_POLICY="${POLICY[$cell]}" \
        HENRI_FREEZE_LEARNING="${FREEZE[$cell]}" \
        HENRI_ARC_ACTION_PAYLOADS=1 \
        PYTHONPATH=. /venv/main/bin/python production_arc_run.py --envs 1 --steps 100 \
        >> "/tmp/arc_g1_${cell}_${env}.log" 2>&1
    echo "=== G1 $cell $env end rc=$? $(date -u +%FT%TZ) ==="
  done
done
touch /tmp/arc_g1_seq_done
