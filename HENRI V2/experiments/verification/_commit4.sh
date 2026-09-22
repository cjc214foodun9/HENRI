#!/usr/bin/env bash
# Commit 4: production epsilon fix + tau overflow fix + ARC apparatus.
set -uo pipefail
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone" || exit 1

echo "=== IMPORT + CALL check (py_compile cannot resolve names) ==="
cd "HENRI V2" || exit 1
python -c "
import sys; sys.path.insert(0,'.')
from henri_grid_observable import derive_tau
t = derive_tau(4096, 11, 0.01)
assert 0.85 < t['tau_observational'] < 0.95, t
print('  derive_tau(4096,11,0.01) =', round(t['tau_observational'],6))
import sagnac_mcts_planner as m
assert hasattr(m.SagnacMCTSPlanner,'observational_tau')
print('  planner names bound OK')
" || exit 1

echo "=== contract tests for this commit ==="
python -m pytest tests/contract/test_production_sagnac_veto_gate.py \
  tests/contract/test_tau_deployment_lattice.py -q -p no:cacheprovider 2>&1 | tail -6

echo "=== housekeeping: remove throwaway shells already superseded ==="
for f in _scope_sagnac_fix.sh _arc_deps_audit.sh _arcengine_probe.sh; do
  rm -f "experiments/verification/$f" 2>/dev/null || true
done

cd .. || exit 1
echo "=== stage ==="
git add -A "HENRI V2"
git status --short | wc -l

echo "=== commit ==="
git commit -F "$LOCALAPPDATA/Temp/henri_c4_msg.txt" 2>&1 | tail -12
echo "COMMIT_EXIT=$?"
git log --oneline origin/main..HEAD
