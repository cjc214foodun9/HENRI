#!/usr/bin/env bash
# ROOT-CAUSE the RuntimeError that made the Sagnac veto fail open on EVERY step.
#
# MEASURED (experiments/verification/_gauntlet_audit.sh): every recorded veto payload
# was {"error": "RuntimeError"} -- 89/89 in run A, 60/60 in run B. So the veto never
# executed; it raised every time and the handler swallowed the message. `engaged=True`
# throughout was therefore a FAIL-OPEN ARTIFACT, not a permissive gate. My epsilon-mode
# probe missed this because it called the function in isolation with MATCHED shapes.
#
# This run captures the real traceback.
set -uo pipefail
TOPDIR="C:/Users/chan/Desktop/HENRI 7B SWARM"
cd "$TOPDIR/HENRI V2" || exit 1
PY="$LOCALAPPDATA/Temp/arc312_env/Scripts/python.exe"

export OPERATION_MODE=offline
export ARC_API_KEY=""
export HENRI_ARC_SAGNAC_VETO=1
export HENRI_ARC_VETO_DEBUG=1
export ENVIRONMENTS_DIR="$TOPDIR/environment_files"

echo "=== gauntlet, 8 steps, traceback ENABLED ==="
timeout 300 "$PY" production_arc_run.py --mode phase823_live_gauntlet --steps 8 2>&1 \
  | grep -vE "SyntaxWarning|invalid escape|^\s+(Computes|W_task|Adapts|2\.|1\.)" \
  > "$LOCALAPPDATA/Temp/veto_tb.txt"
echo "exit=$?"

echo
echo "=== the traceback (first occurrence) ==="
awk '/Traceback \(most recent call last\)/{f=1} f{print} /RuntimeError/{if(f){exit}}' \
  "$LOCALAPPDATA/Temp/veto_tb.txt" | tail -30

echo
echo "=== the error MESSAGE now recorded in telemetry ==="
grep -oE '"error": "[^"]*"' "$LOCALAPPDATA/Temp/veto_tb.txt" | sort | uniq -c | sort -rn | head -5

echo
echo "=== shapes the veto is given (d_model=512 per the run header) ==="
grep -nE "field_to_wave|boundary_batch =|state_wave =|d_model" "$LOCALAPPDATA/Temp/veto_tb.txt" | head -8
