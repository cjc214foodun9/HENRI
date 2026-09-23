#!/usr/bin/env bash
# UHR-03 ARMS — kill-run #1 of 2. Instance already prepared (carrier a0a9e4e +
# 799MB overlay verified sha256 7557238908...).
#
# THE EXPERIMENTAL VARIABLE IS EXACTLY ONE FLAG: HENRI_UHR02_EXTERO_GATE (0 then 1).
#
# DEFECTS THIS VERSION FIXES (all measured, all mine):
#  D1 (cost a full arm-pair) The common env block was passed as an ssh ARGV string.
#     ssh joins argv into ONE line and the remote shell RE-SPLITS it, so `$5`
#     captured only the first token. HENRI_OFFLINE_DIAG therefore never reached the
#     interpreter -> `dsn` fell through to resolve_zone_c_dsn() -> both arms died at
#     production_arc_run.py:722 with psycopg OperationalError 127.0.0.1:5434.
#     Proof: line 720 `if dsn != "offline://surrogate":` guards 722, and line 533
#     maps HENRI_OFFLINE_DIAG -> surrogate. A live DSN in the traceback PROVES the
#     flag was absent. FIX: env is exported INSIDE the quoted heredoc; only
#     space-free single tokens (RWT/WT/ARM/FLAG/STEPS) ride on argv. ENV_GATE now
#     asserts every flag inside the interpreter and fails closed.
#  D2 `if ! cmd 2>&1 | tail -1; then` -- `!` tested TAIL's status (always 0), so
#     DEPS_FAIL was UNREACHABLE. A gate that cannot fail is not a gate. No pipe.
#  D3 `PYTHONPATH=\\$PWD` in double quotes = literal backslash + a LOCALLY expanded
#     path. Replaced with a heredoc where `$PWD` is remote.
#  D4 Extraction used `"key": \{[^}]*\}` on NESTED json, which TRUNCATES at the
#     first inner brace. Replaced with a real json parser (Python).
set -uo pipefail
HOST="${1:?host}"; PORT="${2:?port}"
STEPS="${STEPS:-16}"
KEY="$HOME/.ssh/id_ed25519"
RWT="/workspace/henri-verify"; WT="$RWT/_uhr03"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes
     -o ConnectTimeout=25 -o ServerAliveInterval=15 -i "$KEY" -p "$PORT" "root@$HOST")

echo "STEPS=$STEPS"
echo
echo "=== 0. DEP GATE (arc_agi + torch) — no pipe, so it can actually fail ==="
DEP_OUT=$("${SSH[@]}" bash -s <<'REMOTE'
/usr/bin/python3 -c 'import arc_agi, torch; print("DEP_OK", torch.__version__)' 2>/dev/null
echo "dep_rc=$?"
REMOTE
)
echo "$DEP_OUT"
case "$DEP_OUT" in
  *dep_rc=0*) echo "DEP_GATE_PASS" ;;
  *)          echo "DEP_GATE_FAIL"; exit 1 ;;
esac

echo
echo "=== 1. RUNNER SMOKE: module-level imports execute (line 41 = import arc_agi) ==="
SMOKE_OUT=$("${SSH[@]}" RWT="$RWT" WT="$WT" bash -s <<'REMOTE'
cd "$WT/HENRI V2" || { echo "CD_FAIL $WT"; exit 3; }
export PYTHONPATH="$PWD"
/usr/bin/python3 production_arc_run.py --help >/dev/null 2>&1
echo "RUNNER_IMPORTS_rc=$?"
REMOTE
)
echo "$SMOKE_OUT"
case "$SMOKE_OUT" in
  *RUNNER_IMPORTS_rc=0*) echo "SMOKE_PASS" ;;
  *)                     echo "SMOKE_FAIL"; exit 1 ;;
esac

for ARM in BASELINE RFSS; do
  case "$ARM" in BASELINE) FLAG=0 ;; RFSS) FLAG=1 ;; esac
  echo
  echo "=== 2. ARM=$ARM  HENRI_UHR02_EXTERO_GATE=$FLAG ==="
  "${SSH[@]}" RWT="$RWT" WT="$WT" ARM="$ARM" FLAG="$FLAG" STEPS="$STEPS" bash -s <<'REMOTE'
set -u
D="$RWT/telemetry_uhr03_$ARM"
rm -rf "$D"; mkdir -p "$D"
# --- env INSIDE the quoted heredoc: no local expansion, no remote re-split ---
export HENRI_ARC_SAGNAC_VETO=1
export HENRI_UHR01_RFSS=1
export HENRI_MACRO_NUM_CHANNELS=1
export HENRI_OFFLINE_DIAG=1
export EXTERNAL_OUTCOME_EFE=1
export HENRI_TRACE_UPDATE_GATES=1
export HENRI_UHR02_EXTERO_GATE="$FLAG"
export HENRI_TELEMETRY_DIR="$D"
export EXPECT_FLAG="$FLAG"

echo "--- ENV_GATE (asserted INSIDE the interpreter) ---"
/usr/bin/python3 - <<'PYGATE'
import os, sys
want = {
    "HENRI_ARC_SAGNAC_VETO": "1",
    "HENRI_UHR01_RFSS": "1",
    "HENRI_MACRO_NUM_CHANNELS": "1",
    "HENRI_OFFLINE_DIAG": "1",
    "EXTERNAL_OUTCOME_EFE": "1",
    "HENRI_TRACE_UPDATE_GATES": "1",
    "HENRI_UHR02_EXTERO_GATE": os.environ.get("EXPECT_FLAG", "?"),
}
bad = {k: (os.environ.get(k), v) for k, v in want.items() if os.environ.get(k) != v}
print("ENV_GATE", "PASS" if not bad else "FAIL " + repr(bad))
sys.exit(1 if bad else 0)
PYGATE
_env_rc=$?
[ "$_env_rc" -eq 0 ] || { echo "ENV_GATE_FAIL rc=$_env_rc"; exit 1; }

cd "$WT/HENRI V2" || { echo "CD_FAIL"; exit 1; }
export PYTHONPATH="$PWD"
echo "--- run: phase823_live_gauntlet steps=$STEPS ---"
/usr/bin/python3 production_arc_run.py --mode phase823_live_gauntlet \
    --envs 1 --steps "$STEPS" > "$D/run.log" 2>&1
_rc=$?
echo "ARM=$ARM exit=$_rc   (nonzero => BLOCKED_INFRASTRUCTURE, NO science claim)"
if [ "$_rc" -ne 0 ]; then
  echo "--- run.log HEAD ---"; head -12 "$D/run.log"
  echo "--- run.log TAIL ---"; tail -22 "$D/run.log"
fi
echo "--- files ---"; ls -l "$D"

D="$D" /usr/bin/python3 - <<'PYX'
import json, glob, os
D = os.environ["D"]
fs = sorted(glob.glob(os.path.join(D, "*.jsonl")))
if not fs:
    print("NO_JSONL in", D); raise SystemExit(0)
f = fs[0]
raw = [l for l in open(f) if l.strip()]
print("jsonl=%s lines=%d" % (os.path.basename(f), len(raw)))
rows = []
for l in raw:
    try: rows.append(json.loads(l))
    except Exception: pass
print("parsed records:", len(rows))
keys = set()
for r in rows: keys |= set(r.keys())
p8 = sorted(k for k in keys if k.startswith("phase820"))
print("phase820_* keys:", p8)
print("arbiter keys:", sorted(k for k in keys if "extero" in k or "guard" in k))
for i, r in enumerate(rows):
    p = {k: v for k, v in r.items() if k.startswith("phase820")}
    print("-- step %d: %s" % (i, json.dumps(p, default=str)[:900]))
PYX
REMOTE
done

echo
echo "=== 3. TELEMETRY TREE (remote) ==="
"${SSH[@]}" RWT="$RWT" bash -s <<'REMOTE'
for a in BASELINE RFSS; do
  echo "-- $a --"
  ls -l "$RWT/telemetry_uhr03_$a" 2>/dev/null || echo "  ABSENT"
done
REMOTE
echo "UHR03_ARMS_DONE"
