#!/usr/bin/env bash
# ITEM 4: attempt the live ARC-AGI-3 gauntlet with the veto ENGAGED.
# The blocker (production adaptive-epsilon -> NEVER_FIRES) is fixed in ef07279 and
# verified SELECTIVE, so this run can now actually be affected by the flag.
set -uo pipefail
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2" || exit 1
PY="$LOCALAPPDATA/Temp/arc312_env/Scripts/python.exe"

export OPERATION_MODE=offline
export ENVIRONMENTS_DIR="/c/Users/chan/Desktop/HENRI 7B SWARM/environment_files"
export HENRI_ARC_SAGNAC_VETO=1
export ARC_API_KEY=""

echo "=== 1. argparse surface (what does the runner accept?) ==="
grep -nE "add_argument|args\.mode" production_arc_run.py | head -22

echo
echo "=== 2. --help ==="
timeout 150 "$PY" production_arc_run.py --help 2>&1 | tail -28

echo
echo "=== 3. attempt the gauntlet (bounded, flag ENGAGED) ==="
timeout 420 "$PY" production_arc_run.py --mode phase823_live_gauntlet 2>&1 \
  | grep -vE "SyntaxWarning|invalid escape|^\s+(Computes|W_task|Adapts|2\.|1\.)" \
  | tail -50
echo "GAUNTLET_EXIT=${PIPESTATUS[0]}"
