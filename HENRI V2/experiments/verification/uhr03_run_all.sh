#!/usr/bin/env bash
# UHR-03 DRIVER — arms, then egress, then STOP. The stop is CHAINED, not improvised.
#
# WHY CHAINED: the instance is the clock ($0.68/h). Improvising the stop after the
# arms is the sequence that produced the `tail -1` billing disaster. This driver
# runs the egress+SHA gate and the stop NO MATTER WHAT the arms returned, so a
# failed arm cannot leave the instance burning while a human reads logs.
# `vastai stop` is REVERSIBLE and keeps the disk, so stopping early is never the
# irreversible choice; leaving it running is what costs money irreversibly.
#
# Usage: bash uhr03_run_all.sh <host> <port> <instance-id>
set -uo pipefail
HOST="${1:?host}"; PORT="${2:?port}"; INSTANCE="${3:?instance id}"
HERE="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2/experiments/verification"
OUT="$LOCALAPPDATA/Temp"

echo "########## PHASE 1: ARMS ##########"
bash "$HERE/uhr03_arms_only.sh" "$HOST" "$PORT" 2>&1 | tee "$OUT/uhr03_arms.log"
ARMS_RC=${PIPESTATUS[0]}
echo "ARMS_RC=$ARMS_RC"

echo
echo "########## PHASE 2: EGRESS + STOP (always runs) ##########"
bash "$HERE/uhr03_egress_stop.sh" "$HOST" "$PORT" "$INSTANCE" 2>&1 | tee "$OUT/uhr03_stop.log"
STOP_RC=${PIPESTATUS[0]}
echo "STOP_RC=$STOP_RC"

echo
echo "########## DRIVER SUMMARY ##########"
echo "arms_rc=$ARMS_RC stop_rc=$STOP_RC"
grep -E "^(ARM=|ENV_GATE|DEP_GATE|SMOKE_|EGRESS_|CONFIRMED_STOPPED|WARN_STILL_RUNNING|.* credit =|  id )" \
     "$OUT/uhr03_arms.log" "$OUT/uhr03_stop.log" 2>/dev/null | tail -30
echo "UHR03_RUN_ALL_DONE"
