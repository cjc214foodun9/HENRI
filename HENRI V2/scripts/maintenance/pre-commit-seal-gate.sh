#!/usr/bin/env bash
# PRE-COMMIT: seal-consistency gate ("Phase 10.2 directive 1").
#
# VERSIONED SOURCE. Install with:
#     bash "HENRI V2/scripts/maintenance/install_seal_gate_hook.sh"
# .git/hooks/ is UNTRACKED, so the installer copies this file there. Edit HERE.
#
# Fail-closed: the commit is BLOCKED when the gate script is missing, when the
# sealed pair is missing, or when the doc contradicts its own receipt. Commit
# 99fb88a shipped exactly that contradiction, and it survived visual review plus two
# end-to-end verification scripts. A machine does this check; a human reading does not.
#
# PATH DISCIPLINE (this bug produced FALSE results earlier this session): MSYS git
# returns /c/Users/... from `rev-parse --show-toplevel`, and NATIVE python cannot open
# that form -- it reports `can't open file 'C:\c\Users\...'`. cygpath -m converts to
# C:/Users/... . Verified by the installer's three-direction test.
#
# Overridable so BOTH directions are testable without touching sealed artifacts:
#   HENRI_SEAL_GATE / HENRI_SEAL_DOC / HENRI_SEAL_RECEIPT
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "PRE-COMMIT BLOCKED: not inside a git repository" >&2; exit 1; }

# MSYS path -> native path for the python interpreter.
if command -v cygpath >/dev/null 2>&1; then
  ROOT="$(cygpath -m "$ROOT")"
else
  case "$ROOT" in
    /?/*) ROOT="$(printf '%s' "$ROOT" | sed -E 's|^/([a-zA-Z])/|\1:/|')" ;;
  esac
fi

V2="$ROOT/HENRI V2"
GATE="${HENRI_SEAL_GATE:-$V2/experiments/verification/validate_seal_consistency.py}"
DOC="${HENRI_SEAL_DOC:-$V2/references/henri_phase10_1_operator_gap_adjudication.md}"
RECEIPT="${HENRI_SEAL_RECEIPT:-$V2/experiments/verification/evaluate_60_task_koopman_gap_observed.json}"

for f in "$GATE" "$DOC" "$RECEIPT"; do
  [ -f "$f" ] || { echo "PRE-COMMIT BLOCKED: required file missing: $f" >&2; exit 1; }
done

out="$(python "$GATE" --doc "$DOC" --receipt "$RECEIPT" --quiet 2>&1)"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "PRE-COMMIT BLOCKED: seal-consistency gate FAILED (rc=$rc)." >&2
  printf '%s\n' "$out" | tail -8 >&2
  echo "  Inspect: python \"$GATE\" --doc \"$DOC\" --receipt \"$RECEIPT\"" >&2
  echo "  Intentional bypass only: git commit --no-verify" >&2
  exit 1
fi
echo "pre-commit: seal-consistency gate PASS"
exit 0
