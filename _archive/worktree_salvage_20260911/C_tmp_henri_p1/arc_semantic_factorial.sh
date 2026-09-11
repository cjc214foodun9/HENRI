#!/bin/bash
# Semantic Gate-1 factorial: policy x learning x env, payloads ON (screen+oracle).
# Sequential (never concurrent d=65536 on 32 GB). Per-cell/env telemetry dirs.
# Uses the PROVEN stage-5 runner contract (HENRI_SINGLE_ENV / HENRI_POLICY /
# HENRI_FREEZE_LEARNING / HENRI_TELEMETRY_DIR / --envs 1 --steps 100).
set -u
cd "/workspace/henri_semantic_wt/HENRI V2"
export HENRI_ARC_ACTION_PAYLOADS=1
export HENRI_SEED=20260811
export HENRI_OFFLINE_DIAG=1
export PYTHONPATH=.
OUT=/workspace/semantic_telemetry
mkdir -p "$OUT"
for CELL in A B C D; do
  case "$CELL" in
    A) POLICY=action1; FREEZE=1 ;;
    B) POLICY=efe;     FREEZE=1 ;;
    C) POLICY=action1; FREEZE=0 ;;
    D) POLICY=efe;     FREEZE=0 ;;
  esac
  for ENV in lf52 tn36 sc25; do
    mkdir -p "$OUT/$CELL/$ENV"
    echo "=== $CELL $ENV start $(date -u +%FT%TZ) ==="
    env HENRI_TELEMETRY_DIR="$OUT/$CELL/$ENV" \
        HENRI_OFFLINE_DIAG=1 \
        HENRI_SEED=20260811 \
        HENRI_SINGLE_ENV="$ENV" \
        HENRI_POLICY="$POLICY" \
        HENRI_FREEZE_LEARNING="$FREEZE" \
        PYTHONPATH=. /venv/main/bin/python production_arc_run.py --envs 1 --steps 100 \
        >> "/tmp/semantic_factorial_${CELL}_${ENV}.log" 2>&1
    echo "=== $CELL $ENV end rc=$? $(date -u +%FT%TZ) ==="
  done
done
echo "SEMANTIC_FACTORIAL_DONE rc=0" >> /tmp/semantic_factorial_done.marker
