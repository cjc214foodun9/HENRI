#!/usr/bin/env bash
# UHR-04 RUN #7 - ka59 env-probe paired A/B.
# Pre-registered: experiments/verification/uhr03_preregistration.md (run #7 section).
#
# AMENDMENT 1 (validation-environment migration): ft09 is RETIRED for causal
# evaluation. Measured: 32/32 steps where EVERY action moved exactly 4 cells in
# row 63 (channels 4032..4095) of a 64x64 grid, inter-frame diff bit-identical
# (0.0009765625), zero levels completed. That is a step-indexed progress cursor,
# so a confident 16/16 there proves nothing causal. ka59 is the migration target:
# per-action changed-cell means 15.73 / 17.34 / 18.60 / 11.17 (six distinct).
#
# FAIL-CLOSED: refuses to launch unless the ft09 pin has been fully migrated.
set -uo pipefail

VD="$(cd "$(dirname "$0")" && pwd)"
ARMS="${1:-$VD/uhr03_arms_only.sh}"
HOST="${2:-}"
PORT="${3:-}"

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
  echo "ABORT: usage: uhr04_run7.sh <arms_script> <ssh_host> <ssh_port>"
  exit 1
fi
if [ ! -f "$ARMS" ]; then
  echo "ABORT: arms script not found: $ARMS"
  exit 1
fi

# Guard the ACTIVE export line only. A bare grep also matches the header COMMENT
# that documents the retired pin, which would refuse a correctly migrated script
# (that defect was measured and fixed). `.` accepts the quotes without escaping.
if grep -E "^[[:space:]]*export[[:space:]]+HENRI_SINGLE_ENV=(ft09|.ft09.)" "$ARMS" >/dev/null 2>&1; then
  echo "ABORT: $ARMS still PINS HENRI_SINGLE_ENV=ft09 (retired cursor-band env)."
  exit 1
fi
if ! grep -q 'ENV_ID' "$ARMS"; then
  echo "ABORT: $ARMS has no ENV_ID parameterization; migration incomplete."
  exit 1
fi

echo "RUN7_PRECHECK_PASS  ENV_ID=ka59  host=$HOST port=$PORT"
ENV_ID=ka59 STEPS="${STEPS:-32}" exec bash "$ARMS" "$HOST" "$PORT"
