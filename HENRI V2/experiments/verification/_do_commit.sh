#!/usr/bin/env bash
# Commit the Milestone-1 / readout / delta-memory / thermo work on an ISOLATED
# branch. The worktree was created from origin/main and is on a DETACHED HEAD, so
# committing without first creating a branch would leave the commit unreachable.
set -uo pipefail

cd "/c/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone" || exit 1

echo "=== verify the extraction script no longer hardcodes a user path ==="
if grep -n 'Users.chan.Downloads' "HENRI V2/experiments/verification/extract_audit_pdf.py"; then
  echo "FAIL: hardcoded path still present"; exit 1
else
  echo "OK: no hardcoded user path in extract_audit_pdf.py"
fi

echo "=== compile-check every new/changed python file ==="
cd "HENRI V2" || exit 1
FAIL=0
for f in henri_wave_readout.py henri_continuum_memory.py henri_thermo_langevin.py \
         sagnac_mcts_planner.py production_arc_run.py \
         experiments/verification/extract_audit_pdf.py \
         experiments/verification/binding_norm_algebra.py \
         experiments/verification/spec_constant_audit.py \
         experiments/verification/rfss_capacity_sweep.py \
         experiments/verification/readout_delta_sagnac_probe.py \
         experiments/verification/continuum_memory_probe.py \
         tests/contract/test_readout_observable_coupling.py \
         tests/contract/test_milestone1_answer_decoupling.py; do
  if python -m py_compile "$f" 2>/dev/null; then
    echo "  OK   $f"
  else
    echo "  FAIL $f"; FAIL=1
  fi
done
[ "$FAIL" -eq 0 ] || { echo "COMPILE FAILURES"; exit 1; }

cd .. || exit 1

echo "=== move the commit message out of the tree (git log holds it) ==="
MSG="$LOCALAPPDATA/Temp/henri_m1_commit_msg.txt"
mv "HENRI V2/experiments/verification/.commit_msg_milestone1.txt" "$MSG" || exit 1

echo "=== create an isolated branch (HEAD is detached) ==="
git switch -c milestone1-readout-decoupling 2>&1 | tail -2 || exit 1
git rev-parse --abbrev-ref HEAD

echo "=== stage ==="
git add -A "HENRI V2" 2>&1 | tail -3
git status --short | head -40

echo "=== commit (pre-commit seal gate WILL run) ==="
git commit -F "$MSG" 2>&1 | tail -30
echo "COMMIT_EXIT=$?"
echo "=== log ==="
git log --oneline -3
