#!/usr/bin/env bash
# FULL-SCALE DIAGNOSTIC RUN: num_blocks=8192, d_model=65536, Zone C surrogate.
#
# ZONE C PATH, READ FROM LIVE CODE (production_arc_run.py:524-537, gate :697-703)
#     HENRI_OFFLINE_DIAG=1 -> dsn = "offline://surrogate"   (in-process, no Postgres)
#     else on CUDA         -> dsn = resolve_zone_c_dsn()    (dev default localhost:5434)
#   The gate RAISES when the live sink is unreachable. The code comment at :525-527
#   states: "Diagnostic only - never score-eligible". So this run answers the WIDTH
#   question ONLY. No capability, score, or benchmark claim follows from it.
#
# ARGUMENT FORM: proven in action5_complete.sh (exit 0, 69 records).
set -uo pipefail
HOST=ssh9.vast.ai; PORT=11444; KEY="$HOME/.ssh/id_ed25519"
SHA=8ba08ddc0dcb797ce2cbd6433cd23b0cf335f1cd
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o IdentitiesOnly=yes -i $KEY"

echo "=== 0. endpoint + instance state ==="
ssh -i "$KEY" $OPTS -p "$PORT" root@"$HOST" 'echo SSH_OK; hostname' 2>&1 | tail -3

ssh -i "$KEY" $OPTS -p "$PORT" root@"$HOST" "SHA=$SHA bash -s" <<'REMOTE'
set -uo pipefail
WT=/workspace/henri-verify; A="$WT/HENRI V2"; PY=/usr/bin/python3
cd "$A" || { echo "FATAL cd $A"; exit 1; }
export PYTHONPATH="$A"

echo "=== 1. tree + overlay re-verified immediately before launch ==="
echo "  HEAD=$(git rev-parse HEAD)"
[ "$(git rev-parse HEAD)" = "$SHA" ] && echo "  SHA_MATCH=YES" || { echo "  SHA_MATCH=NO"; exit 1; }
echo "  status_lines=$(git status --porcelain=v1 -uall | wc -l)"
M="models/henri_decoder_checkpoint.pt"
echo "  overlay_bytes=$(stat -c%s "$M") sha_head=$(sha256sum "$M" | cut -c1-16)"
echo "  coupled_fix: helper=$(grep -c 'def _macro_num_blocks' production_arc_run.py) pad=$(grep -c 'nb = _macro_num_blocks()' production_arc_run.py) store=$(grep -c '_num_channels = _macro_num_blocks()' production_arc_run.py)"

echo
echo "=== 2. CLI options actually supported (no assumed flags) ==="
$PY production_arc_run.py --help 2>&1 | grep -E '^\s+--' | head -18

echo
echo "=== 3. clean slate + detached launch (diagnostic Zone C) ==="
mkdir -p "$WT/telemetry_fs"
pkill -f 'production_arc_run.py' 2>/dev/null && sleep 2 || true
LOG="$WT/fullscale_diag.log"
rm -f "$LOG"
setsid nohup env \
    OPERATION_MODE=offline ARC_API_KEY="" \
    ENVIRONMENTS_DIR="$WT/environment_files" \
    HENRI_TELEMETRY_DIR="$WT/telemetry_fs" \
    HENRI_ARC_SAGNAC_VETO=1 \
    HENRI_MACRO_NUM_CHANNELS=1 \
    HENRI_OFFLINE_DIAG=1 \
    PYTHONPATH="$A" \
    "$PY" production_arc_run.py --mode phase823_live_gauntlet --steps 8 \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "PID=$PID" > "$WT/.fs.pid"
echo "  launched pid=$PID"

echo
echo "=== 4. liveness ==="
for t in 30 60 90 120; do
  sleep 30
  if ! ps -p "$PID" >/dev/null 2>&1; then echo "  t=${t}s EXITED"; break; fi
  echo "  t=${t}s alive log_lines=$(wc -l < "$LOG" 2>/dev/null || echo 0) gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
done
if ps -p "$PID" >/dev/null 2>&1; then echo "  STATUS=STILL_RUNNING"; else echo "  STATUS=EXITED"; fi

echo
echo "=== 5. outcome ==="
echo "  --- steps logged ---"
grep -cE '^  step ' "$LOG" 2>/dev/null | sed 's/^/  steps=/'
grep -m3 -E '^  step ' "$LOG" 2>/dev/null | cut -c1-165
echo "  --- errors / gates ---"
grep -m4 -E 'Error|BLOCKED|Traceback|unavailable|einsum|RuntimeError' "$LOG" 2>/dev/null | cut -c1-175
echo "  --- device/scale header ---"
grep -m2 -E 'device=|scale=' "$LOG" 2>/dev/null | cut -c1-175
echo "  --- opine / sagnac ---"
grep -m4 -E '\[opine\]|sagnac|veto' "$LOG" 2>/dev/null | cut -c1-165
echo "  --- last 16 lines ---"
tail -16 "$LOG" 2>/dev/null | cut -c1-155
echo "  --- telemetry ---"
ls -la "$WT/telemetry_fs" 2>/dev/null | head -6
echo FULLSCALE_DIAG_DONE
REMOTE
echo "  ssh_rc=$?"
