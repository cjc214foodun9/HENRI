#!/usr/bin/env bash
# STEP 2 MEASUREMENT: reduced-scale A/B, COUPLED resolution fix, flag ON, bridge OFF.
#
# WHY THE FLAG MUST BE ON (confound caught before running)
#   `HENRI_MACRO_NUM_CHANNELS` is DEFAULT-OFF, so the 64-vs-8192 einsum failure
#   reproduces only with the flag set. A run without it measures the old default
#   path and says nothing about the fix.
#
# WHAT MUST BE TRUE (pre-registered)
#   veto payloads carry delta_axiom + hard_vetoed, ZERO
#   gate_status=UNAVAILABLE_SHAPE_MISMATCH, zero einsum errors, zero
#   `[opine] unavailable`, and no bridge consulted.
set -uo pipefail

WT="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
EXPORTS="C:/Users/chan/HENRI_telemetry_exports"
TMP="$LOCALAPPDATA/Temp"
cd "$WT" || { echo "FATAL cd"; exit 1; }
PY="$TMP/arc312_env/Scripts/python.exe"

echo "=== tree + COUPLED fix markers ==="
echo "  branch=$(git rev-parse --abbrev-ref HEAD) sha=$(git rev-parse --short HEAD)"
printf "  helper_def=%s  flag_reads=%s  pad_reads_helper=%s  bare_num_channels=%s  bridge_lines=%s\n" \
  "$(grep -c 'def _macro_num_blocks' production_arc_run.py)" \
  "$(grep -c 'HENRI_MACRO_NUM_CHANNELS' production_arc_run.py)" \
  "$(grep -c 'nb = _macro_num_blocks()' production_arc_run.py)" \
  "$(grep -c 'num_channels=8192' production_arc_run.py)" \
  "$(grep -c 'HENRI_SAGNAC_WIDTH_BRIDGE' sagnac_mcts_planner.py)"

echo "=== which try-block owns the '[opine] unavailable' print? ==="
grep -n '\[opine\] unavailable' production_arc_run.py | head -4

echo "=== contract tests (3 suites) ==="
python -m pytest tests/contract/test_macro_resolution_coupling.py \
  tests/contract/test_macro_field_resolution.py \
  tests/contract/test_sagnac_width_contract.py -q -p no:cacheprovider 2>&1 | tail -4

export OPERATION_MODE=offline
export ARC_API_KEY=""
export ENVIRONMENTS_DIR="C:/Users/chan/Desktop/HENRI 7B SWARM/environment_files"
export HENRI_ARC_SAGNAC_VETO=1
export HENRI_MACRO_NUM_CHANNELS=1       # THE FLAG: ON
unset HENRI_SAGNAC_WIDTH_BRIDGE          # default OFF: the real path, no crutch

echo "=== LIVE GAUNTLET: coupled fix, flag ON, bridge OFF ==="
timeout 480 "$PY" production_arc_run.py --mode phase823_live_gauntlet --steps 60 \
    > "$TMP/gauntlet_coupled.log" 2>&1
echo "  exit=$?"

TEL=$(ls -t "$EXPORTS"/production_run_*.jsonl 2>/dev/null | head -1)
echo "  telemetry=$TEL"
printf '%s' "$TEL" > "$TMP/tel_coupled.txt"

echo "=== AUDIT (independent of the runner) ==="
"$PY" experiments/verification/audit_coupled_veto.py "$TEL" "$TMP/gauntlet_coupled.log" \
    2>&1 | tail -20
echo "=== verdict line from log, if any ==="
grep -m3 -E 'opine|veto|GAME_OVER|Score|level' "$TMP/gauntlet_coupled.log" | cut -c1-150 || true
