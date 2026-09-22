#!/usr/bin/env bash
# ACTION 4 (continued): commit the verification apparatus + the Action 5 findings.
# Explicit paths only; no `git add -A`.
set -uo pipefail

WT="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone"
cd "$WT" || { echo FATAL; exit 1; }

echo "=== branch/sha ==="
git rev-parse --abbrev-ref HEAD; git rev-parse --short HEAD

echo "=== confirm the staged overlay is IGNORED (must not be committed) ==="
git check-ignore -v "HENRI V2/models/henri_decoder_checkpoint.pt" || \
  echo "  WARNING: checkpoint is NOT ignored; do NOT stage it"

echo "=== stage explicit paths (verification apparatus only) ==="
git add "HENRI V2/experiments/verification/" 2>&1 | tail -2
git status --short | wc -l

echo "=== ensure the 763MB overlay is NOT staged ==="
git diff --cached --name-only | grep -c "\.pt$" || echo "  no .pt staged (correct)"

echo "=== commit ==="
git commit -F "$LOCALAPPDATA/Temp/henri_c6_msg.txt" 2>&1 | tail -8
echo "COMMIT_RC=$?"

echo "=== push ==="
git push origin milestone1-readout-decoupling 2>&1 | tail -4
echo "PUSH_RC=$?"
git log --oneline origin/main..HEAD | head -8
