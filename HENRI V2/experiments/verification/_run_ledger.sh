#!/usr/bin/env bash
# Full ledger + seal gates on the committed milestone-1 branch.
# Fail-closed: the exit code and the summary are the verdict, not a narrative.
cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2" || exit 1

echo "=== BRANCH / COMMIT ==="
git rev-parse --abbrev-ref HEAD
git log --oneline -1

echo
echo "=== SEAL GATE (all pairs) ==="
python experiments/verification/validate_seal_consistency.py --all --json "$LOCALAPPDATA/Temp/seal_gate.json" 2>&1 | tail -20
echo "SEAL_EXIT=$?"

echo
echo "=== FULL LEDGER ==="
python -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -60
echo "LEDGER_EXIT=${PIPESTATUS[0]}"
