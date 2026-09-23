#!/usr/bin/env bash
# FULL-SCALE RUN at num_blocks=8192, DIAGNOSTIC-LABELED Zone C surrogate.
#
# WHY THIS PATH (read from live code, not assumed)
#   production_arc_run.py:524-537 has exactly two doors on a CUDA host:
#     HENRI_OFFLINE_DIAG=1  -> dsn = "offline://surrogate"   (in-process, no Postgres)
#     else (DEVICE==cuda)   -> dsn = resolve_zone_c_dsn()    (dev default: localhost:5434)
#   with the gate at :697-703 RAISING when the live sink is unreachable.
#   The code comment at :525-527 states plainly: "Diagnostic only - never
#   score-eligible". So this run answers the WIDTH question and nothing about
#   capability. No score claim follows from it.
#
# THE FLAG IS A NO-OP HERE (measured, P2)
#   SCALE["num_blocks"]=8192 on CUDA, so _macro_num_blocks() returns 8192 with the
#   flag ON or OFF -> full-scale behaviour is unchanged by the reduced-scale fix.
#   That is the point: the fix must not perturb production.
set -uo pipefail
HOST=ssh9.vast.ai; PORT=11444; KEY="$HOME/.ssh/id_ed25519"
SHA=8ba08ddc0dcb797ce2cbd6433cd23b0cf335f1cd
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o IdentitiesOnly=yes -i $KEY"

ssh -i "$KEY" $OPTS -p "$PORT" root@"$HOST" "SHA=$SHA bash -s" <<'REMOTE'
set -uo pipefail
WT=/workspace/henri-verify; A="$WT/HENRI V2"; PY=/usr/bin/python3
cd "$A" || exit 1
export PYTHONPATH="$A"

echo "=== A. how does the runner source boundary axioms? ==="
grep -n 'ZONE_C_AXIOM_ENV_FILE' production_arc_run.py | head -6
grep -n 'load_boundary_axioms' production_arc_run.py | head -6
echo "  axiom env files present in tree:"
find "$WT" -maxdepth 3 -name "*boundary_axiom*" -o -maxdepth 3 -name "*axiom*.json" 2>/dev/null | head -8

echo
echo "=== B. tree + overlay re-verified immediately before launch ==="
echo "  HEAD=$(git rev-parse HEAD)"
[ "$(git rev-parse HEAD)" = "$SHA" ] && echo "  SHA_MATCH=YES" || { echo "  SHA_MATCH=NO"; exit 1; }
echo "  status_lines=$(git status --porcelain=v1 -uall | wc -l)"
M="models/henri_decoder_checkpoint.pt"
echo "  overlay_bytes=$(stat -c%s "$M") sha=$(sha256sum "$M" | cut -c1-16)"

echo
echo "=== C. clean slate + launch (detached, diagnostic Zone C) ==="
mkdir -p "$WT/telemetry_fullscale"
pkill -f 'production_arc_run.py' 2>/dev/null && sleep 2 || true
LOG="$WT/fullscale_diag.log"
rm -f "$LOG" "$WT/FULLSCALE_DONE"
setsid nohup env \
    OPERATION_MODE=offline ARC_API_KEY="" \
    ENVIRONMENTS_DIR="$WT/environment_files" \
    HENRI_TELEMETRY_DIR="$WT/telemetry_fullscale" \
    HENRI_ARC_SAGNAC_VETO=1 \
    HENRI_MACRO_NUM_CHANNELS=1 \
    HENRI_OFFLINE_DIAG=1 \
    PYTHONPATH="$A" \
    "$PY" production_arc_run.py --mode phase823_live_gauntlet --envs 1 --steps 8 \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "PID=$PID" > "$WT/.fullscale.pid"
echo "  launched pid=$PID log=$LOG"

echo
echo "=== D. liveness (up to 150s) ==="
for t in 20 40 60 90 120 150; do
  sleep 20
  if ! ps -p "$PID" >/dev/null 2>&1; then
    echo "  t=${t}s process=EXITED"
    break
  fi
  echo "  t=${t}s alive log_lines=$(wc -l < "$LOG" 2>/dev/null || echo 0)  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
done

echo
echo "=== E. outcome ==="
if ps -p "$PID" >/dev/null 2>&1; then
  echo "  STATUS=STILL_RUNNING"
else
  echo "  STATUS=EXITED"
fi
echo "  --- last 22 log lines ---"
tail -22 "$LOG" 2>/dev/null | cut -c1-150
echo "  --- first error, if any ---"
grep -m2 -E 'Error|BLOCKED|Traceback|unavailable|einsum' "$LOG" 2>/dev/null | cut -c1-170
echo "  --- step lines ---"
grep -cE '^  step ' "$LOG" 2>/dev/null | sed 's/^/  steps_logged=/'
grep -m3 -E '^  step ' "$LOG" 2>/dev/null | cut -c1-160
echo "  --- telemetry files ---"
ls -la "$WT/telemetry_fullscale" 2>/dev/null | head -6
echo FULLSCALE_LAUNCH_DONE
REMOTE
echo "  ssh_rc=$?"
