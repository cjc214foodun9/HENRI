#!/usr/bin/env bash
# ACTION 5 COMPLETION: prove the veto now RUNS in the LIVE loop at reduced scale.
#
# THE FIX UNDER TEST (production_arc_run.py, lines ~744-780)
#     _num_channels = int(SCALE["num_blocks"])          <- was hardcoded 8192
#     ActionOutcomeGeneratorStore(num_channels=_num_channels, ...)
#     pretrain_action_generators(..., num_channels=_num_channels, ...)
#
#   At CPU scale SCALE["num_blocks"]=64, so the macro field is 64 channels ->
#   construct_macro_option -> [64,3,3] -> field_to_wave -> 512-wide, which MATCHES the
#   run's own references (num_blocks*8 = 512). Before the fix it was 8192 -> 65536-wide
#   -> the veto raised on every step and recorded UNAVAILABLE_SHAPE_MISMATCH 60/60.
#
# WHAT THIS RUN MUST SHOW (pre-registered)
#   * the telemetry payloads now carry delta_axiom + hard_vetoed (the veto RAN), and
#     NO payload carries gate_status=UNAVAILABLE_SHAPE_MISMATCH
#   * the bridge is NOT needed (default OFF), so this is the real path, not a crutch
#   * hard_vetoed is reported per step; note whether it takes both values in the live
#     loop. It may legitimately be one-sided here because the macro candidate and the
#     boundary axiom are different objects -- that is a MEASUREMENT, not a defect, and
#     bidirectionality is separately proven deterministically by contract test M5.
#
# Run from the WORKTREE with fix markers asserted first (an earlier gauntlet silently
# executed in the main checkout, which has no fixes).
set -uo pipefail

WT="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
EXPORTS="C:/Users/chan/HENRI_telemetry_exports"
cd "$WT" || { echo "FATAL cd"; exit 1; }
PY="$LOCALAPPDATA/Temp/arc312_env/Scripts/python.exe"

echo "=== tree + fix markers ==="
echo "  branch=$(git rev-parse --abbrev-ref HEAD) sha=$(git rev-parse --short HEAD)"
printf "  scale_bound_num_channels=%s  hardcoded_8192_left=%s  isinstance=%s  bridge=%s\n" \
  "$(grep -c '_num_channels = int(SCALE\["num_blocks"\])' production_arc_run.py)" \
  "$(grep -c 'num_channels=8192' production_arc_run.py)" \
  "$(grep -c 'isinstance(_veto_exc, SagnacGateUnavailable)' production_arc_run.py)" \
  "$(grep -c 'HENRI_SAGNAC_WIDTH_BRIDGE' sagnac_mcts_planner.py)"

echo "=== contract tests (macro resolution) ==="
python -m pytest tests/contract/test_macro_field_resolution.py \
  tests/contract/test_sagnac_width_contract.py -q -p no:cacheprovider 2>&1 | tail -6

export OPERATION_MODE=offline
export ARC_API_KEY=""
export ENVIRONMENTS_DIR="C:/Users/chan/Desktop/HENRI 7B SWARM/environment_files"
export HENRI_ARC_SAGNAC_VETO=1
unset HENRI_SAGNAC_WIDTH_BRIDGE          # default OFF: the real path, no crutch

echo "=== LIVE GAUNTLET with the resolution fix ==="
timeout 480 "$PY" production_arc_run.py --mode phase823_live_gauntlet --steps 70 \
    > "$LOCALAPPDATA/Temp/gauntlet_fixed.log" 2>&1
echo "  exit=$?"
TEL=$(ls -t "$EXPORTS"/production_run_*.jsonl 2>/dev/null | head -1)
echo "  telemetry=$TEL"
printf '%s' "$TEL" > "$LOCALAPPDATA/Temp/tel_fixed.txt"

echo "=== AUDIT: did the veto RUN? ==="
"$PY" experiments/verification/audit_gauntlet_gate.py \
   --off "$TEL" --on "$TEL" --out "experiments/verification/gauntlet_fixed_audit.json" \
   2>&1 | tail -14

echo "=== raw payload shapes (independent of the auditor) ==="
"$PY" - "$TEL" <<'PY'
import json, sys, collections
path = sys.argv[1]
shapes = collections.Counter()
deltas = []
def walk(o):
    if isinstance(o, dict):
        v = o.get("sagnac_veto")
        if isinstance(v, dict):
            shapes[tuple(sorted(v.keys()))] += 1
            d = v.get("delta_axiom")
            if isinstance(d, (int, float)):
                deltas.append(float(d))
        for x in o.values():
            walk(x)
    elif isinstance(o, list):
        for x in o:
            walk(x)
n = 0
with open(path, encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            walk(json.loads(line))
        except Exception:
            pass
print(f"  records={n} payloads={sum(shapes.values())}")
for k, c in shapes.most_common(6):
    print(f"  {c:>4}x keys={list(k)}")
if deltas:
    print(f"  delta_axiom: n={len(deltas)} min={min(deltas):.6f} max={max(deltas):.6f}")
    print(f"  vetoed(>0.35)={sum(1 for d in deltas if d > 0.35)}/{len(deltas)}")
PY
