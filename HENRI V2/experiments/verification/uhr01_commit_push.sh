#!/usr/bin/env bash
# UHR-01 COMMIT + PUSH (branch only; main promotion waits for the remote gate).
# Explicit paths, no `git add -A`, never stage the ~763MB .pt overlay.
set -uo pipefail

WT="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone"
MSG="C:/Users/chan/AppData/Local/Temp/henri_uhr01_msg.txt"

cd "$WT" || { echo "FATAL: cannot cd to worktree"; exit 1; }

echo "=== 0. branch + base assertions (fail closed) ==="
BR=$(git rev-parse --abbrev-ref HEAD)
echo "branch=$BR"
case "$BR" in
  carrier/uhr-01-homologous-representation) ;;
  *) echo "FATAL: wrong branch"; exit 1 ;;
esac
if ! git merge-base --is-ancestor adcc24e HEAD; then
  echo "FATAL: adcc24e is not an ancestor of HEAD"; exit 1
fi

echo "=== 1. ensure no .pt is staged (overlay is gitignored, verify anyway) ==="
git diff --cached --name-only | grep -c '\.pt$' | xargs -I{} echo "staged_pt_count={}"

echo "=== 2. stage EXPLICIT paths only ==="
git add -- \
  "HENRI V2/uhr_rfss.py" \
  "HENRI V2/opine_object_mcts.py" \
  "HENRI V2/production_arc_run.py" \
  "HENRI V2/tests/contract/test_uhr01_rfss_homology.py" \
  "HENRI V2/experiments/verification/uhr01_regime.py" \
  "HENRI V2/experiments/verification/uhr01_preregistration.md" || exit 1

echo "=== 3. staged set (must be exactly 6, no .pt) ==="
git diff --cached --name-only
N=$(git diff --cached --name-only | wc -l)
echo "staged_count=$N"
PT=$(git diff --cached --name-only | grep -c '\.pt$' || true)
echo "staged_pt=$PT"
if [ "$PT" != "0" ]; then echo "FATAL: a .pt file is staged"; exit 1; fi

echo "=== 4. commit (default hooks: the seal gate MUST run) ==="
# Deliberately NOT overriding core.hooksPath: forcing an absolute MSYS path there
# can silently DISABLE the pre-commit seal gate -- the exact defect class this
# carrier exists to police. Use the repository's default hook resolution.
git commit -F "$MSG" || {
  echo "COMMIT_FAILED rc=$?"; exit 1; }
COMMIT=$(git rev-parse HEAD)
echo "commit=$COMMIT"

echo "=== 5. push BRANCH only ==="
git push origin HEAD:carrier/uhr-01-homologous-representation 2>&1 | tail -5
echo "push_rc=$?"

echo "=== 6. reconcile: remote branch must equal local HEAD ==="
git fetch origin carrier/uhr-01-homologous-representation 2>&1 | tail -2
RB=$(git rev-parse origin/carrier/uhr-01-homologous-representation 2>/dev/null || echo MISSING)
echo "local_HEAD  =$COMMIT"
echo "remote_branch=$RB"
[ "$COMMIT" = "$RB" ] && echo "RECONCILED=YES" || echo "RECONCILED=NO"

echo "=== 7. main must be UNMOVED (promotion waits for the paired remote gate) ==="
git rev-parse origin/main
