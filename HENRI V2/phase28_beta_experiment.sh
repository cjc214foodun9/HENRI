#!/bin/bash
# Phase 2.8: β_pragmatic sweep — test preference-resonance steering strength
# Hypothesis: stronger β_pragmatic pulls planner toward historically favorable
# basins, reducing RESET rate and potentially improving action quality.
#
# ARM design:
#   ARM1: β=1.0  (baseline — matches Phase 2.7 ARM3)
#   ARM2: β=5.0  (moderate amplification)
#   ARM3: β=10.0 (aggressive amplification)
#
# Pre-registered criteria:
#   PASS: EFE bounded [-1.0,+2.0], fallback <5%, RESET <20%
#   FAIL: EFE > +5.0, fallback >20%, RESET >40%
#   SCORE: any non-zero is a breakthrough; 0.0 all envs = baseline expected
set -euo pipefail

REPO="/workspace/HENRI/HENRI V2"
cd "$REPO"
mkdir -p logs telemetry_logs

RUN_ID="phase28_$(date +%Y%m%d_%H%M%S)"
echo "=== Phase 2.8: β_pragmatic sweep ==="
echo "Run ID: $RUN_ID"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Common config
export CONSTRAINT_AXIOM=1
export PROGRESS_VALENCE=1
export LAMBDA_CONSTRAINT_MAX=5.0
export CONSTRAINT_REJECT_THRESH=0.5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- ARM1: β=1.0 (baseline) ----
echo ">>> ARM1: β_pragmatic=1.0 (baseline)"
export BETA_PRAGMATIC=1.0
python3 production_arc_run.py --envs 3 --steps 30 \
    > "logs/${RUN_ID}_arm1_beta1.log" 2>&1
echo "  ARM1 done: $(date -u +%H:%M:%S)"

# ---- ARM2: β=5.0 ----
echo ">>> ARM2: β_pragmatic=5.0"
export BETA_PRAGMATIC=5.0
python3 production_arc_run.py --envs 3 --steps 30 \
    > "logs/${RUN_ID}_arm2_beta5.log" 2>&1
echo "  ARM2 done: $(date -u +%H:%M:%S)"

# ---- ARM3: β=10.0 ----
echo ">>> ARM3: β_pragmatic=10.0"
export BETA_PRAGMATIC=10.0
python3 production_arc_run.py --envs 3 --steps 30 \
    > "logs/${RUN_ID}_arm3_beta10.log" 2>&1
echo "  ARM3 done: $(date -u +%H:%M:%S)"

# ---- Summary ----
echo ""
echo "=== Phase 2.8 Complete ==="
echo "Run ID: $RUN_ID"
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "--- Terminal States ---"
for arm in arm1_beta1 arm2_beta5 arm3_beta10; do
    echo "=== $arm ==="
    grep -E '\[end\]|BUDGET|terminal|scorecard' "logs/${RUN_ID}_${arm}.log" 2>/dev/null || echo "  (no terminal lines)"
done

echo ""
echo "--- cd82 Telemetry (all ARMs) ---"
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
            pref = d.get('preference_store_size', 0)
            print(f\"  step {d['step']:3d}: δ={d.get('sagnac_delta',0):.4f} r={d.get('kuramoto_r',0):.3f} EFE={d.get('efe_best',0):+.3f} penalty={d.get('constraint_penalty',0):.4f} λ={d.get('lambda_active',0):.4f} adm={d.get('admissible_count',0)} pref={pref} {fb}\")
" 2>/dev/null
done

echo ""
echo "--- Scorecards ---"
for f in telemetry_logs/production_run_*_scorecards.json; do
    [ -f "$f" ] && echo "$(basename $f):" && python3 -c "import json; print(json.dumps(json.load(open('$f')), indent=1)[:500])" 2>/dev/null
done
