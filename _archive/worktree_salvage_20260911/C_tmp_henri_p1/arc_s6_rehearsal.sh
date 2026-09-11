#!/bin/bash
# Stage 6 diagnostic rehearsal (Gate 2 completion) — 33fed84, payloads ON.
# Cells A (action1+frozen) and D (efe+learning) x lf52/tn36/sc25, seed
# 20260809, 100-step budget, offline surrogate. Verifies the
# SCORE_ELIGIBILITY label appears in every episode (score_eligible=false,
# LOADED_COMPONENT_NOT_ON_ACTION_PATH). Non-score-eligible by design.
set -u
cd '/workspace/henri_g1_wt/HENRI V2'
declare -A POLICY FREEZE
POLICY[A]=action1; FREEZE[A]=1
POLICY[D]=efe;     FREEZE[D]=0
for cell in A D; do
  for env in lf52 tn36 sc25; do
    mkdir -p "/workspace/henri_artifacts/arc_s6_rehearsal/$cell/$env"
    echo "=== S6 $cell $env start $(date -u +%FT%TZ) ==="
    env HENRI_TELEMETRY_DIR="/workspace/henri_artifacts/arc_s6_rehearsal/$cell/$env" \
        HENRI_OFFLINE_DIAG=1 \
        HENRI_SEED=20260809 \
        HENRI_SINGLE_ENV="$env" \
        HENRI_POLICY="${POLICY[$cell]}" \
        HENRI_FREEZE_LEARNING="${FREEZE[$cell]}" \
        HENRI_ARC_ACTION_PAYLOADS=1 \
        PYTHONPATH=. /venv/main/bin/python production_arc_run.py --envs 1 --steps 100 \
        >> "/tmp/arc_s6_${cell}_${env}.log" 2>&1
    echo "=== S6 $cell $env end rc=$? $(date -u +%FT%TZ) ==="
  done
done
touch /tmp/arc_s6_rehearsal_done
