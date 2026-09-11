#!/bin/bash
# Stage 5 sequential driver — factorial cells B and C (diagnostic, not score-eligible).
# A=stage3 (action1+frozen) done; D=stage4 (efe+learning) done.
# B: efe + frozen | C: action1 + learning
# Same envs, seed, budget, ordering, per-env telemetry dirs.
set -u
cd '/workspace/henri_p1_run/HENRI V2'
for cell in B C; do
  if [ "$cell" = B ]; then POLICY=efe;   FREEZE=1; else POLICY=action1; FREEZE=0; fi
  for env in lf52 tn36 sc25; do
    mkdir -p "/workspace/henri_artifacts/arc_s5/$cell/$env"
    echo "=== S5 $cell $env start $(date -u +%FT%TZ) ==="
    env HENRI_TELEMETRY_DIR="/workspace/henri_artifacts/arc_s5/$cell/$env" \
        HENRI_OFFLINE_DIAG=1 \
        HENRI_SEED=20260809 \
        HENRI_SINGLE_ENV="$env" \
        HENRI_POLICY="$POLICY" \
        HENRI_FREEZE_LEARNING="$FREEZE" \
        PYTHONPATH=. /venv/main/bin/python production_arc_run.py --envs 1 --steps 100 \
        >> "/tmp/arc_s5_${cell}_${env}.log" 2>&1
    echo "=== S5 $cell $env end rc=$? $(date -u +%FT%TZ) ==="
  done
done
touch /tmp/arc_s5_seq_done
