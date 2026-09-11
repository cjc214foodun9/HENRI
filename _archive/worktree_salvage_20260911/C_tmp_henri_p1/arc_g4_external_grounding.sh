#!/bin/bash
# Gate 4: external Dnu grounding paired ablation (single-mechanism).
# Arms: A=EXTERNAL_OUTCOME_EFE=0 (control), B=EXTERNAL_OUTCOME_EFE=1 (treatment).
# Both: policy=efe, freeze=1, payloads=1, seed=20260811, steps=100, Zone C LIVE.
# Sequential (never concurrent d=65536 on 32 GB). Proven runner contract.
set -u
cd "/workspace/henri_semantic_wt/HENRI V2"
set -a; source /workspace/zonec_prod.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/workspace/g4_telemetry
mkdir -p "$OUT"
for ARM in A B; do
  case "$ARM" in
    A) EXT=0 ;;
    B) EXT=1 ;;
  esac
  for ENV in lf52 tn36 sc25; do
    mkdir -p "$OUT/$ARM/$ENV"
    echo "=== G4 $ARM $ENV start $(date -u +%FT%TZ) EXTERNAL_OUTCOME_EFE=$EXT policy=efe freeze=1 payloads=1 zonec=live ==="
    env EXTERNAL_OUTCOME_EFE="$EXT" \
        EXTERNAL_EIG_WEIGHT=0.25 \
        EXTERNAL_TASK_WEIGHT=1.0 \
        TASK_WEIGHTED_EIG=0 \
        ZONE_C_ENV=prod \
        HENRI_ARC_ACTION_PAYLOADS=1 \
        HENRI_SEED=20260811 \
        HENRI_SINGLE_ENV="$ENV" \
        HENRI_POLICY=efe \
        HENRI_FREEZE_LEARNING=1 \
        USE_ZONE_C_AXIOMS=1 \
        ZONE_C_AXIOM_ENV_FILE=/workspace/zonec_prod.env \
        HENRI_TELEMETRY_DIR="$OUT/$ARM/$ENV" \
        PYTHONPATH=. /venv/main/bin/python production_arc_run.py --envs 1 --steps 100 \
        >> "/tmp/g4_${ARM}_${ENV}.log" 2>&1
    echo "=== G4 $ARM $ENV end rc=$? $(date -u +%FT%TZ) ==="
  done
done
echo "G4_DONE rc=0" >> /tmp/g4_done.marker
