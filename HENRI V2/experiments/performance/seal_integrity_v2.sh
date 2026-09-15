#!/usr/bin/env bash
# SEAL INTEGRITY v2 -- PATH-CORRECT.
#
# v1 (seal_integrity_check.sh) produced two FALSE results purely from its own bugs:
#   1. it passed MSYS paths (/c/Users/...) to NATIVE python, which cannot open them,
#      so two VALID json receipts were reported "JSON-BROKEN";
#   2. the same bug made the gate invocation die with
#      `can't open file 'C:\c\Users\...'`, so the gate never ran at all.
# Native tools (git, python) must receive C:/... paths on this host. Bash may use
# either. This script uses native paths for every native-tool argument.
#
# Read-only: it inspects, extracts to TEMP, gates, and reports. It writes nothing
# inside the repository.
set -uo pipefail

ROOT_N="C:/Users/chan/Desktop/HENRI 7B SWARM"
V2N="$ROOT_N/HENRI V2"
T="C:/Users/chan/AppData/Local/Temp/sealhead"

GATE="$V2N/experiments/verification/validate_seal_consistency.py"
CMP="$V2N/experiments/verification/compare_receipt_arms.py"
ER="$V2N/experiments/verification/evaluate_60_task_koopman_gap_observed.json"
DR="$V2N/references/henri_phase10_1_operator_gap_adjudication.md"
FR="$V2N/experiments/verification/koopman_bank_feasibility_observed.json"

# git paths are repo-root-relative
R="HENRI V2/experiments/verification/evaluate_60_task_koopman_gap_observed.json"
D="HENRI V2/references/henri_phase10_1_operator_gap_adjudication.md"
F="HENRI V2/experiments/verification/koopman_bank_feasibility_observed.json"

gitq() { git -c core.quotepath=false -C "$ROOT_N" "$@"; }

echo "=== A. HEAD ==="
gitq rev-parse --short HEAD
gitq log --oneline -1
echo "  staged=$(gitq diff --cached --name-only | wc -l)"
echo "  tracked drift:"
gitq status --porcelain -uall | grep -v '^??' | sed 's/^/    /' || echo "    (none)"

echo
echo "=== B. sealed-file integrity (native paths -> no false alarms) ==="
for f in "$ER" "$FR"; do
  if [ -f "$f" ]; then
    if python -c "import json,sys;json.load(open(sys.argv[1],encoding='utf-8'))" "$f" 2>/dev/null; then
      st="json-ok"
    else
      st="JSON-BROKEN"
    fi
    printf "  %8s B  %-11s %s\n" "$(wc -c < "$f")" "$st" "$(basename "$f")"
  else
    printf "  MISSING               %s\n" "$(basename "$f")"
  fi
done
[ -f "$DR" ] && printf "  %8s B  %-11s %s\n" "$(wc -c < "$DR")" "text" "$(basename "$DR")"

echo
echo "=== C. extract HEAD pair to TEMP ==="
mkdir -p "$T"
gitq show HEAD:"$R" > "$T/receipt_head.json"
gitq show HEAD:"$D" > "$T/doc_head.md"
gitq show HEAD:"$F" > "$T/feas_head.json"
printf "  head receipt %8s B\n  head doc     %8s B\n  head feas    %8s B\n" \
  "$(wc -c < "$T/receipt_head.json")" "$(wc -c < "$T/doc_head.md")" "$(wc -c < "$T/feas_head.json")"

echo
echo "=== D. GATE: COMMITTED seal = HEAD doc vs HEAD receipt  (expect FAIL) ==="
python "$GATE" --doc "$T/doc_head.md" --receipt "$T/receipt_head.json"
echo "  gate_head_rc=$?"

echo
echo "=== E. GATE: WORKING TREE = corrected doc vs receipt  (expect PASS) ==="
python "$GATE" --doc "$DR" --receipt "$ER"
echo "  gate_worktree_rc=$?"

echo
echo "=== F. new contract tests for the gate ==="
cd "$V2N" || exit 2
python -m pytest tests/contract/test_seal_consistency.py -q 2>&1 | tail -10

echo
echo "=== G. drift after this check (must be UNCHANGED: read-only) ==="
gitq status --porcelain -uall | grep -v '^??' | sed 's/^/    /' || echo "    (none)"
echo "### DONE"
