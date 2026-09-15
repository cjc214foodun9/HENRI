#!/usr/bin/env bash
# BOUNDED COMMIT of the VERIFIED set only. Explicit paths; never `git add -A`.
# Verified at these exact bytes on sm_120 (RTX PRO 6000 Blackwell, cc 12.0,
# torch 2.12.0+cu130) by experiments/performance/verify_phase10e|f|g.sh.
set -uo pipefail
cd "/c/Users/chan/Desktop/HENRI 7B SWARM" || exit 2

FILES=(
  "HENRI V2/o_vsa_torus_encoder.py"
  "HENRI V2/arc_task_functor.py"
  "HENRI V2/experiments/verification/arc_torus_encoder_wiring.py"
  "HENRI V2/experiments/verification/arc_torus_encoder_wiring_observed.json"
  "HENRI V2/experiments/verification/arc_torus_exactness_floor.py"
  "HENRI V2/experiments/verification/arc_torus_exactness_floor_observed.json"
  "HENRI V2/experiments/verification/arc_tensored_carrier_kill.py"
  "HENRI V2/experiments/verification/arc_tensored_carrier_kill_observed.json"
  "HENRI V2/experiments/verification/threefold_grounding_probe.py"
  "HENRI V2/experiments/verification/threefold_grounding_observed.json"
  "HENRI V2/experiments/performance/verify_phase10e.sh"
  "HENRI V2/experiments/performance/verify_phase10f.sh"
  "HENRI V2/experiments/performance/verify_phase10g.sh"
)

echo "### 0. branch / HEAD"
git branch --show-current
git log --oneline -1

echo
echo "### 1. working diff for the edited sources (evidence of what changed)"
git diff --stat HEAD -- "HENRI V2/o_vsa_torus_encoder.py" "HENRI V2/arc_task_functor.py"

echo
echo "### 2. SECRET SCAN on the files to be staged"
HITS=0
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then echo "  MISSING $f"; continue; fi
  n=$(grep -nEi 'aa_[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|postgres://[^"]*:[^"@]*@|API_KEY *= *"[^"]{8,}"|password *= *"[^"]{6,}"' "$f" 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then echo "  SECRET_SUSPECT $f ($n)"; HITS=$((HITS+1)); fi
done
echo "  secret_suspect_files=$HITS"
if [ "$HITS" -gt 0 ]; then echo "ABORT: resolve secrets before committing"; exit 3; fi

echo
echo "### 3. STAGE + CONFIRM EXACTLY WHAT IS STAGED"
git add -- "${FILES[@]}" || exit 4
git diff --cached --name-only
echo "--- staged count ---"
git diff --cached --name-only | wc -l

echo
echo "### 4. residual dirty paths left untouched"
echo "untracked+modified still present: $(git status --porcelain -uall | grep -vc '^[AM] ' || true)"

echo
echo "### NOTE: commit is NOT executed by this script."
echo "### Run the git commit explicitly after inspecting the staged list above."
