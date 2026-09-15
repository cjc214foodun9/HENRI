#!/usr/bin/env bash
# Install / verify the seal-gate pre-commit hook. Idempotent.
#   install : copies the VERSIONED source into the untracked .git/hooks/
#   verify  : functional test in THREE directions (allow / block / fail-closed)
#
# PATH DISCIPLINE (this bit me earlier this session and produced FALSE results):
#   Native tools (git, python) CANNOT open MSYS paths like /c/Users/... -> they
#   report "can't open file 'C:\c\Users\...'". So python-facing variables use
#   C:/... and bash-facing file operations use /c/.... Both are defined here.
set -uo pipefail

ROOT_N="C:/Users/chan/Desktop/HENRI 7B SWARM"      # native, for git/python
ROOT_M="/c/Users/chan/Desktop/HENRI 7B SWARM"      # msys, for cp/mkdir
V2_N="$ROOT_N/HENRI V2"
V2_M="$ROOT_M/HENRI V2"
TMP_N="C:/Users/chan/AppData/Local/Temp/sealgate"  # native, for python
TMP_M="/c/Users/chan/AppData/Local/Temp/sealgate"  # msys, for bash

SRC_M="$V2_M/scripts/maintenance/pre-commit-seal-gate.sh"
DST_M="$ROOT_M/.git/hooks/pre-commit"

action="${1:-install}"

if [ "$action" = "install" ]; then
  mkdir -p "$ROOT_M/.git/hooks"
  cp "$SRC_M" "$DST_M"
  chmod +x "$DST_M" 2>/dev/null || true
  echo "installed : $DST_M"
  echo "  bytes   : $(wc -c < "$DST_M")   src_bytes: $(wc -c < "$SRC_M")"
  cmp -s "$SRC_M" "$DST_M" && echo "  identical to versioned source: yes" \
                           || echo "  IDENTICAL: NO"
  echo
  echo "=== receipt-filename audit (a wrong name blocks EVERY commit) ==="
  echo "  'observable' (typo): $(grep -c observable "$DST_M" || true)"
  echo "  'observed'   (right): $(grep -c observed "$DST_M" || true)"
  exit 0
fi

# ---------------- verify ----------------
mkdir -p "$TMP_M"
echo "=== hook present? ==="
[ -f "$DST_M" ] || { echo "  MISSING: $DST_M"; exit 1; }
echo "  bytes: $(wc -c < "$DST_M")"
echo "  receipt paths referenced:"
grep -o "evaluate_60_task_koopman_gap_[a-z]*\.json" "$DST_M" | sort -u | sed 's/^/    /'

echo
echo "=== direction A: real pair -> hook must ALLOW (expect rc 0) ==="
bash "$DST_M"; rcA=$?
echo "  rcA=$rcA"

echo
echo "=== direction B: corrupted doc copy -> must BLOCK (expect rc != 0) ==="
cp "$V2_M/references/henri_phase10_1_operator_gap_adjudication.md" "$TMP_M/doc_bad.md"
printf '\n| `koopman_named6@9.99` | +0.1234 | +0.1234 | 0.0000 | 0.0 pct |\n' >> "$TMP_M/doc_bad.md"
HENRI_SEAL_DOC="$TMP_N/doc_bad.md" \
HENRI_SEAL_RECEIPT="$V2_N/experiments/verification/evaluate_60_task_koopman_gap_observed.json" \
HENRI_SEAL_GATE="$V2_N/experiments/verification/validate_seal_consistency.py" \
  bash "$DST_M" > "$TMP_M/block.out" 2>&1
rcB=$?
echo "  rcB=$rcB"
sed 's/^/    /' "$TMP_M/block.out" | head -10
[ "$rcB" -ne 0 ] && echo "  BLOCK direction: WORKS" || echo "  *** BLOCK direction FAILED ***"

echo
echo "=== direction C: missing gate script -> must BLOCK, fail-closed ==="
HENRI_SEAL_GATE="$TMP_N/does_not_exist.py" bash "$DST_M" > "$TMP_M/miss.out" 2>&1
rcC=$?
echo "  rcC=$rcC"
sed 's/^/    /' "$TMP_M/miss.out" | head -3

echo
if [ "$rcA" -eq 0 ] && [ "$rcB" -ne 0 ] && [ "$rcC" -ne 0 ]; then
  echo "RESULT: ALL_THREE_DIRECTIONS_PASS"
else
  echo "RESULT: PROBLEM  (rcA=$rcA rcB=$rcB rcC=$rcC)"
fi
