#!/usr/bin/env bash
# ACTION 3: re-run the LIVE gauntlet FROM THE WORKTREE and audit the veto gate.
#
# WHY THIS RUN EXISTS
#   Every earlier gauntlet run in this session silently executed in the MAIN checkout
#   (carrier/e6-physical-verifier), which has NONE of the fixes. So no prior gauntlet
#   says anything about the current tree. This run:
#     (a) proves WHICH tree it used (branch + short SHA + fix markers),
#     (b) runs the veto flag ON,
#     (c) runs TWO ARMS: bridge OFF (default) and bridge ON (diagnostic),
#     (d) audits the written telemetry for gate_status / hard_vetoed / error.
#
# WHY TWO ARMS
#   At the reduced scale this gauntlet uses (num_blocks=64 -> state_wave 512) the
#   candidate is 65536-wide, so the veto CANNOT run: bridge OFF must record
#   gate_status=UNAVAILABLE_SHAPE_MISMATCH for every step and NO hard_vetoed key.
#   Bridge ON makes the comparison computable, so hard_vetoed can then take BOTH
#   values. That A/B is the evidence that the gate is now honest: it distinguishes
#   "did not run" from "passed" from "vetoed". A gate that only ever reads False is
#   still not proven live -- this is the test for that.
#
# The bridged verdict is PLUMBING ONLY: pooling a field wave to a grid wave's width
# does not make the two comparable, so no capability claim follows from arm ON.
set -uo pipefail

WT="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
EXPORTS="C:/Users/chan/HENRI_telemetry_exports"
cd "$WT" || { echo "FATAL cannot cd to worktree"; exit 1; }
PY="$LOCALAPPDATA/Temp/arc312_env/Scripts/python.exe"

export OPERATION_MODE=offline
export ARC_API_KEY=""
export ENVIRONMENTS_DIR="C:/Users/chan/Desktop/HENRI 7B SWARM/environment_files"
export HENRI_ARC_SAGNAC_VETO=1

echo "=== (a) WHICH TREE, AND ARE THE FIXES PRESENT? ==="
echo "  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "  sha   : $(git rev-parse --short HEAD)"
printf "  fix markers: class=%s isinstance=%s import=%s bridge=%s\n" \
  "$(grep -c 'class SagnacGateUnavailable' sagnac_mcts_planner.py)" \
  "$(grep -c 'isinstance(_veto_exc, SagnacGateUnavailable)' production_arc_run.py)" \
  "$(grep -c 'SagnacMCTSPlanner, SagnacGateUnavailable' production_arc_run.py)" \
  "$(grep -c 'HENRI_SAGNAC_WIDTH_BRIDGE' sagnac_mcts_planner.py)"

echo "=== checkpoint overlay (Action 5 prep) ==="
if [ -f models/henri_decoder_checkpoint.pt ]; then
  echo "  PRESENT $(du -h models/henri_decoder_checkpoint.pt | cut -f1)"
else
  echo "  ABSENT -> a d_model=65536 run would need checkpoint_policy=required overlay"
fi

run_arm () {
  local label="$1" bridge="$2" tel=""
  if [ "$bridge" = "1" ]; then export HENRI_SAGNAC_WIDTH_BRIDGE=1
  else unset HENRI_SAGNAC_WIDTH_BRIDGE; fi
  echo "=== ARM $label (HENRI_SAGNAC_WIDTH_BRIDGE=${bridge}) ==="
  timeout 420 "$PY" production_arc_run.py --mode phase823_live_gauntlet --steps 60 \
      > "$LOCALAPPDATA/Temp/gauntlet_$label.log" 2>&1
  echo "  exit=$?"
  tel=$(ls -t "$EXPORTS"/production_run_*.jsonl 2>/dev/null | head -1)
  echo "  telemetry=$tel"
  printf '%s' "$tel" > "$LOCALAPPDATA/Temp/tel_$label.txt"
  grep -c "GAME_OVER\|FINAL SCORECARDS" "$LOCALAPPDATA/Temp/gauntlet_$label.log" \
      | sed 's/^/  completion markers: /'
}

run_arm off 0
run_arm on 1

echo "=== AUDIT ==="
"$PY" experiments/verification/audit_gauntlet_gate.py \
  --off "$(cat "$LOCALAPPDATA/Temp/tel_off.txt")" \
  --on  "$(cat "$LOCALAPPDATA/Temp/tel_on.txt")" 2>&1 | tail -30
