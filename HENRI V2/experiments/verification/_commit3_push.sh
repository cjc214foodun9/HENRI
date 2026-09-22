#!/usr/bin/env bash
# Commit 3: observational veto wiring. Then push the BRANCH (not main).
set -uo pipefail
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone" || exit 1

echo "=== housekeeping: the untracked helper noted last session ==="
git status --short | grep -E "_final_state.sh" || echo "  (not present)"
rm -f "HENRI V2/experiments/verification/_final_state.sh" 2>/dev/null || true

echo "=== IMPORT + CALL check (py_compile does NOT resolve names) ==="
cd "HENRI V2" || exit 1
python -c "
import sys; sys.path.insert(0,'.')
import sagnac_mcts_planner as m
import henri_grid_observable as g
p = m.SagnacMCTSPlanner(d_model=1024, k_blocks=128, tau_veto=0.35, device='cpu')
assert hasattr(m,'os'), 'os unbound'
assert hasattr(m.SagnacMCTSPlanner,'_norm_consistent_similarity')
assert hasattr(m.SagnacMCTSPlanner,'observational_tau')
assert hasattr(m.SagnacMCTSPlanner,'_demo_observational_stress')
t = p.observational_tau(64)
assert abs(t-0.35) > 0.1, 'tau inherited 0.35'
print(f'  IMPORT+CALL OK; observational_tau(64)={t}')
" || exit 1

echo "=== contract tests for this commit ==="
python -m pytest tests/contract/test_grid_observable_readout.py \
  tests/contract/test_sagnac_scale_consistency.py -q -p no:cacheprovider 2>&1 | tail -4

cd .. || exit 1
echo "=== stage ==="
git add -A "HENRI V2"
git status --short | head -20

echo "=== commit (seal + dependent-pin gates WILL run) ==="
git commit -F "$LOCALAPPDATA/Temp/henri_c3_msg.txt" 2>&1 | tail -12
echo "COMMIT_EXIT=$?"

echo
echo "=== remote ==="
git remote -v | head -4
echo "=== branch state vs origin/main ==="
git log --oneline origin/main..HEAD

echo
echo "=== PUSH THE BRANCH (NOT main) ==="
git push -u origin milestone1-readout-decoupling 2>&1 | tail -15
echo "PUSH_EXIT=$?"
