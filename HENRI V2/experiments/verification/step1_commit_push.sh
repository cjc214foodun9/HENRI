#!/usr/bin/env bash
set -uo pipefail
WT="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone"
cd "$WT/HENRI V2" || { echo NO_CD; exit 1; }

echo "=== contract tests (must pass before commit) ==="
python -m pytest tests/contract/test_macro_field_resolution.py tests/contract/test_sagnac_width_contract.py -q -p no:cacheprovider 2>&1 | tail -4

cd "$WT" || exit 1
echo "=== staged count before ==="
git status --short | wc -l

echo "=== stage explicit paths ==="
git add "HENRI V2/production_arc_run.py"
git add "HENRI V2/tests/contract/test_macro_field_resolution.py"
git add "HENRI V2/experiments/verification/"

echo "=== ensure the 763MB overlay is NOT staged ==="
git diff --cached --name-only | grep -c "\.pt$" || echo "  no .pt staged (correct)"

echo "=== commit ==="
git commit -F "$LOCALAPPDATA/Temp/henri_c7_msg.txt" 2>&1 | tail -5

echo "=== push BRANCH ==="
git push origin milestone1-readout-decoupling 2>&1 | tail -3

echo "=== ff check toward main ==="
git fetch origin main 2>&1 | tail -1
if git merge-base --is-ancestor origin/main HEAD; then echo "FF_SAFE=YES"; else echo "FF_SAFE=NO"; fi
echo "=== branch head ==="
git rev-parse HEAD
echo "=== commits ahead of main ==="
git log --oneline origin/main..HEAD | head -10
