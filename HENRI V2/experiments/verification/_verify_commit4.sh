#!/usr/bin/env bash
# Verify commit 4's contents and push. Then the ledger runs separately.
set -uo pipefail
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone" || exit 1

echo "=== commit 4 stat ==="
git show --stat HEAD | head -30

echo
echo "=== is the PRODUCTION change actually in the commit? ==="
git show --name-only HEAD | grep -E "production_arc_run|henri_grid_observable" || echo "  MISSING"

echo
echo "=== did the pre-commit gates run and pass? ==="
git show --format="%H %s" --no-patch HEAD | head -2

echo
echo "=== reflog / gate evidence: any BLOCKED commit attempt? ==="
git reflog -8 | head -8

echo
echo "=== confirm epsilon_hard is in the COMMITTED blob (not just the worktree) ==="
git show HEAD:"HENRI V2/production_arc_run.py" | grep -n "epsilon_hard=sagnac_planner.tau_veto" | head -3

echo
echo "=== push ==="
git push origin milestone1-readout-decoupling 2>&1 | tail -6
echo "PUSH_EXIT=$?"
git ls-remote --heads origin milestone1-readout-decoupling | head -2
