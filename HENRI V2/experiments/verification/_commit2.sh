#!/usr/bin/env bash
# Commit 2: the Sagnac scale fix (the load-bearing production defect) + ingress.
set -uo pipefail
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone" || exit 1

echo "=== branch ==="
git rev-parse --abbrev-ref HEAD

echo "=== remove the commit helper that got swept in by add -A ==="
git rm -q --cached "HENRI V2/experiments/verification/_do_commit.sh" 2>/dev/null || true
rm -f "HENRI V2/experiments/verification/_do_commit.sh"

echo "=== compile + IMPORT check (py_compile does NOT resolve names) ==="
cd "HENRI V2" || exit 1
for f in sagnac_mcts_planner.py henri_role_filler_ingress.py \
         tests/contract/test_sagnac_scale_consistency.py \
         experiments/verification/ingress_role_filler_probe.py \
         experiments/verification/sagnac_scale_defect.py \
         experiments/verification/live_planner_defect_probe.py; do
  python -m py_compile "$f" 2>/dev/null && echo "  compile OK   $f" || { echo "  FAIL $f"; exit 1; }
done
python -c "
import sys; sys.path.insert(0,'.')
import sagnac_mcts_planner as m
import henri_role_filler_ingress as r
assert hasattr(m,'os'), 'os unbound in planner'
assert hasattr(m.SagnacMCTSPlanner,'_norm_consistent_similarity')
print('  IMPORT+NAME check OK (os bound, helper present)')
" || exit 1

echo "=== contract tests for this commit ==="
python -m pytest tests/contract/test_sagnac_scale_consistency.py \
  tests/contract/test_readout_observable_coupling.py \
  tests/contract/test_milestone1_answer_decoupling.py -q -p no:cacheprovider 2>&1 | tail -6

cd .. || exit 1
echo "=== stage ==="
git add -A "HENRI V2"
git status --short | head -20

echo "=== commit ==="
git commit -F "$LOCALAPPDATA/Temp/henri_c2_msg.txt" 2>&1 | tail -20
echo "COMMIT_EXIT=$?"
git log --oneline -3
