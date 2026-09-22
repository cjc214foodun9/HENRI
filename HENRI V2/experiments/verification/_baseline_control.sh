#!/usr/bin/env bash
# Are the ledger failures MINE, or PRE-EXISTING at the baseline?
#
# Instrument discipline: a failure count without a matched baseline is not a
# verdict. This runs the SAME failing test files against parent commit a039095 in
# a throwaway worktree, so the only variable is my commit.
#
# PATH DISCIPLINE: native git/python cannot open MSYS '/c/...' paths here. MSYS
# path conversion is disabled, so git -C "/c/Users/..." fails with
# "not a git repository". Use C:/... form for any NATIVE tool. (An earlier draft of
# this script hit exactly that and reported a false infrastructure error.)
set -uo pipefail

SRC="C:/Users/chan/Desktop/HENRI 7B SWARM"
TMP="$SRC/.worktrees/_baseline_ctrl"
FILES="tests/contract/test_phase8_batched_nav_probe.py tests/contract/test_phase817_in_context_alignment.py tests/contract/test_phase820_action_grounding.py tests/unit/test_henri_phase838_zonec_bridge_wiring.py tests/contract/test_sealed_egress_production_wiring.py"

echo "=== clear any stale control worktree ==="
git -C "$SRC" worktree remove --force "$TMP" 2>/dev/null || true
rm -rf "$TMP" 2>/dev/null || true

echo "=== create control worktree at a039095 (detached) ==="
git -C "$SRC" worktree add --detach "$TMP" a039095 2>&1 | tail -3
echo "--- control HEAD ---"
git -C "$TMP" log --oneline -1

cd "$TMP/HENRI V2" || { echo "FAIL: cannot cd to control HENRI V2"; exit 1; }

echo
echo "=== BASELINE a039095: the same failing files, same invocation ==="
python -m pytest $FILES -q -p no:cacheprovider 2>&1 | tail -28
echo "BASELINE_EXIT=${PIPESTATUS[0]}"
echo
echo "control worktree left for inspection at: $TMP"
