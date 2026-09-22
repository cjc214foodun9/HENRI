#!/usr/bin/env bash
# ACTION 4: commit + push the width contract, its tests, and the apparatus.
# Then measure ACTION 5 feasibility (full-scale gating).
#
# SAFETY: explicit paths only. No `git add -A`.
set -uo pipefail

WT="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone"
cd "$WT" || { echo "FATAL cd"; exit 1; }

echo "=== branch / sha before ==="
git rev-parse --abbrev-ref HEAD; git rev-parse --short HEAD

echo "=== contract tests for this commit ==="
cd "HENRI V2" || exit 1
python -m pytest tests/contract/test_sagnac_width_contract.py -q -p no:cacheprovider 2>&1 | tail -12
echo "TEST_RC=$?"

echo "=== IMPORT + BINDING check (py_compile cannot resolve names) ==="
python -c "
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import ast
from pathlib import Path
# binding is asserted on the RUNNER by AST (import-ability needs arc_agi, absent here)
tree = ast.parse(Path('production_arc_run.py').read_text(encoding='utf-8'))
bound = set()
for n in ast.walk(tree):
    if isinstance(n, ast.ImportFrom):
        bound.update(a.asname or a.name for a in n.names)
    elif isinstance(n, ast.Import):
        bound.update((a.asname or a.name).split('.')[0] for a in n.names)
for name in ('SagnacGateUnavailable', 'os'):
    assert name in bound, f'{name} unbound'
print('  runner binding OK:', sorted(b for b in bound if 'Sagnac' in b or b == 'os'))
import sagnac_mcts_planner as sp
assert hasattr(sp, 'SagnacGateUnavailable')
print('  planner class OK; legacy branch:',
      'HENRI_SAGNAC_LEGACY_SCALE' in Path('sagnac_mcts_planner.py').read_text(encoding='utf-8'))
" || { echo "BINDING CHECK FAILED -- ABORT"; exit 1; }

cd .. || exit 1

echo "=== ACTION 5 FEASIBILITY: checkpoint overlay ==="
for t in "$WT/HENRI V2/models/henri_decoder_checkpoint.pt" \
         "C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2/models/henri_decoder_checkpoint.pt"; do
  if [ -f "$t" ]; then
    echo "  PRESENT $t  ($(du -h "$t" | cut -f1))"
  else
    echo "  absent  $t"
  fi
done

echo "=== stage EXPLICIT paths ==="
git add "HENRI V2/sagnac_mcts_planner.py" \
        "HENRI V2/production_arc_run.py" \
        "HENRI V2/tests/contract/test_sagnac_width_contract.py" \
        "HENRI V2/tests/contract/test_tau_deployment_lattice.py" \
        "HENRI V2/henri_grid_observable.py" \
        "HENRI V2/experiments/verification/" 2>&1 | tail -3
git status --short | wc -l

echo "=== commit ==="
git commit -F "$LOCALAPPDATA/Temp/henri_c5_msg.txt" 2>&1 | tail -8
echo "COMMIT_RC=$?"

echo "=== push ==="
git push origin milestone1-readout-decoupling 2>&1 | tail -5
echo "PUSH_RC=$?"
git ls-remote --heads origin milestone1-readout-decoupling | head -2
git log --oneline origin/main..HEAD
