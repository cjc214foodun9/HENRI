#!/bin/bash
# Phase 3.0: Goal-conditioned EFE — test lambda_goal steering
# Does adding goal-distance to the EFE change action selection and scores?
#
# ARM design:
#   ARM1: λ_goal=0.0 (baseline — backward compatible, no goal conditioning)
#   ARM2: λ_goal=1.0 (goal distance weighted equally with surprise)
#   ARM3: λ_goal=3.0 (aggressive goal pulling)
#
# Pre-registered criteria:
#   PASS: goal_distance decreases over steps (planner approaches goal)
#   PASS: action distribution changes vs baseline
#   PASS: EFE bounded, fallback <5%
#   SCORE: any non-zero is a breakthrough
set -euo pipefail

REPO="/workspace/HENRI/HENRI V2"
cd "$REPO"
mkdir -p logs telemetry_logs

RUN_ID="phase30_$(date +%Y%m%d_%H%M%S)"
echo "=== Phase 3.0: Goal-Conditioned EFE ==="
echo "Run ID: $RUN_ID"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

export CONSTRAINT_AXIOM=1
export PROGRESS_VALENCE=1
export LAMBDA_CONSTRAINT_MAX=5.0
export CONSTRAINT_REJECT_THRESH=0.5
export BETA_PRAGMATIC=10.0  # Best from Phase 2.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- ARM1: λ_goal=0.0 (baseline) ----
echo ">>> ARM1: λ_goal=0.0 (baseline — no goal conditioning)"
export LAMBDA_GOAL=0.0
python3 production_arc_run.py --envs 3 --steps 30 \
    > "logs/${RUN_ID}_arm1_goal0.log" 2>&1
echo "  ARM1 done: $(date -u +%H:%M:%S)"

# ---- ARM2: λ_goal=1.0 ----
echo ">>> ARM2: λ_goal=1.0 (balanced goal steering)"
export LAMBDA_GOAL=1.0
python3 production_arc_run.py --envs 3 --steps 30 \
    > "logs/${RUN_ID}_arm2_goal1.log" 2>&1
echo "  ARM2 done: $(date -u +%H:%M:%S)"

# ---- ARM3: λ_goal=3.0 ----
echo ">>> ARM3: λ_goal=3.0 (aggressive goal pulling)"
export LAMBDA_GOAL=3.0
python3 production_arc_run.py --envs 3 --steps 30 \
    > "logs/${RUN_ID}_arm3_goal3.log" 2>&1
echo "  ARM3 done: $(date -u +%H:%M:%S)"

# ---- Summary ----
echo ""
echo "=== Phase 3.0 Complete ==="
echo "Run ID: $RUN_ID"
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

echo "--- Terminal States ---"
for arm in arm1_goal0 arm2_goal1 arm3_goal3; do
    echo "=== $arm ==="
    grep -E '\[end\]|BUDGET|terminal|scorecard' "logs/${RUN_ID}_${arm}.log" 2>/dev/null || echo "  (no terminal lines)"
done

echo ""
echo "--- cd82 Telemetry ---"
LATEST=$(ls -t telemetry_logs/production_run_*.jsonl 2>/dev/null | head -3)
for f in $LATEST; do
    echo "=== $(basename $f) ==="
    python3 -c "
import json
with open('$f') as fh:
    for line in fh:
        d = json.loads(line)
        if d.get('env','') == 'cd82-fb555c5d':
            fb = 'FALLBACK' if d.get('fallback_executed') else 'ok'
            gd = d.get('goal_distance', -1)
            print(f\"  step {d['step']:3d}: δ={d.get('sagnac_delta',0):.4f} r={d.get('kuramoto_r',0):.3f} EFE={d.get('efe_best',0):+.3f} penalty={d.get('constraint_penalty',0):.4f} λ={d.get('lambda_active',0):.4f} adm={d.get('admissible_count',0)} gdist={gd:.4f} pref={d.get('preference_store_size',0)} {fb}\")
" 2>/dev/null
done

echo ""
echo "--- Scorecards ---"
for f in telemetry_logs/production_run_*_scorecards.json; do
    [ -f "$f" ] && echo "$(basename $f):" && python3 -c "import json; d=json.load(open('$f')); [print(f'  {k}: score={v.get(\"score\",\"?\")}') for k,v in d.items() if isinstance(v,dict)]" 2>/dev/null
done
