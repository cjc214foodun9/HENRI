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

# Default: cover EVERY sealed pair (validate_seal_consistency.PAIRS). Passing
# --doc/--receipt checks one pair only, which would silently stop covering seals
# added later. The overrides exist for the installer's 3-direction test.
if [ -n "${HENRI_SEAL_DOC:-}" ] && [ -n "${HENRI_SEAL_RECEIPT:-}" ]; then
  ARGS=(--doc "$HENRI_SEAL_DOC" --receipt "$HENRI_SEAL_RECEIPT")
  REQUIRED=("$GATE" "$HENRI_SEAL_DOC" "$HENRI_SEAL_RECEIPT")
else
  ARGS=()
  REQUIRED=("$GATE")
fi

for f in "${REQUIRED[@]}"; do
  [ -f "$f" ] || { echo "PRE-COMMIT BLOCKED: required file missing: $f" >&2; exit 1; }
done

out="$(python "$GATE" ${ARGS[@]+"${ARGS[@]}"} --quiet 2>&1)"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "PRE-COMMIT BLOCKED: seal-consistency gate FAILED (rc=$rc)." >&2
  printf '%s\n' "$out" | tail -8 >&2
  echo "  Inspect: python \"$GATE\" ${ARGS[@]+"${ARGS[@]}"}" >&2
  echo "  Intentional bypass only: git commit --no-verify" >&2
  exit 1
fi
# Report WHAT WAS COVERED. Without this line the hook swallows the gate's coverage
# count, so narrowing PAIRS back to a single pair would be invisible in the commit
# transcript -- the hook would keep printing PASS while silently checking less.
# test_precommit_hook.py::test_hook_covers_every_registered_pair asserts this line.
printf '%s\n' "$out" | grep -E "^RESULT:" >&2 || true
echo "pre-commit: seal-consistency gate PASS"
exit 0
