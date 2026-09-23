#!/usr/bin/env bash
# STEP 1 completion: fast-forward main (FF proven: origin/main is an ancestor of HEAD).
set -uo pipefail
WT="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone"
cd "$WT" || { echo NO_CD; exit 1; }

echo "=== pre-state ==="
git rev-parse --abbrev-ref HEAD
echo "  HEAD      = $(git rev-parse HEAD)"
echo "  origin/main = $(git rev-parse origin/main)"

echo "=== prove fast-forward (old main must be an ancestor of the candidate) ==="
if git merge-base --is-ancestor origin/main HEAD; then
  echo "  FF_SAFE=YES"
else
  echo "  FF_SAFE=NO -> ABORT"; exit 1
fi

echo "=== working tree must be clean apart from ignored overlays ==="
git status --porcelain -uall | wc -l

echo "=== fast-forward push main ==="
git push origin HEAD:main 2>&1 | tail -4

echo "=== reconcile: remote main == local head ? ==="
git fetch origin main 2>&1 | tail -1
R=$(git rev-parse origin/main); L=$(git rev-parse HEAD)
echo "  origin/main = $R"
echo "  local  HEAD = $L"
[ "$R" = "$L" ] && echo "  RECONCILED=YES" || echo "  RECONCILED=NO"

echo "=== main log head ==="
git log --oneline origin/main | head -8

echo "=== overlay still ignored? ==="
git check-ignore -q "HENRI V2/models/henri_decoder_checkpoint.pt" && echo "  ignored (correct)"
