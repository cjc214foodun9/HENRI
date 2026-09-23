#!/usr/bin/env bash
# UHR-03 ARMS — kill-run #2 of 2. Instance 52289752 (restarted; disk + the 799 MB
# overlay are preserved). The worktree is refreshed to the SHA given on argv.
#
# THE EXPERIMENTAL VARIABLE IS EXACTLY ONE FLAG: HENRI_UHR02_EXTERO_GATE (0 then 1).
#
# AMENDMENT 3 — both changes forced by kill-run #1 MEASUREMENT:
#   (a) HENRI_SINGLE_ENV=ft09  PIN the environment. Kill-run #1 took the API's
#       first-listed env (lp85-305b61c3) and the frame never moved (0/16 probes),
#       so the store only ever saw an identity displacement. ft09 measured 8/8
#       moving in UHR-01 (ft09-0d8bbf25). The env list rotates between runs, so
#       an unpinned run is non-reproducible.
#   (b) min_norm 1e-8 -> 1e-5 in recorded_transition_generators: the old floor sat
#       BELOW the log path's own float32 noise floor (~3.2e-06 for U U^dag), so it
#       admitted an identity's residue as a "recorded transition".
#
# DEFECTS ALREADY FIXED IN THIS FILE (all measured, all mine):
#   D1  env passed as an ssh ARGV string was RE-SPLIT by the remote shell, so
#       HENRI_OFFLINE_DIAG never reached the interpreter and dsn fell through to
#       resolve_zone_c_dsn() -> both arms died at line 722 (psycopg 5434).
#       Env now lives INSIDE the quoted heredoc, asserted by ENV_GATE in-process.
#   D2  `if ! cmd | tail -1` tested TAIL's status (always 0): unreachable gate.
#   D3  the dep gate demanded `import arcade` -> needs an X display (pyglet
#       NoSuchDisplayException headless) and is required by NOTHING (0 hits).
#   D4  nested-JSON regex truncated at the first inner brace -> real parser now.
#
# Usage: bash uhr03_arms_only.sh <host> <port> <sha>
set -uo pipefail
HOST="${1:?host}"; PORT="${2:?port}"
# SHA optional: empty -> use the just-fetched public carrier ref ON THE HOST.
# This keeps the driver (which passes only host+port) correct without a second
# edit, and guarantees the remote worktree matches what I pushed.
SHA="${3:-}"
STEPS="${STEPS:-16}"
KEY="$HOME/.ssh/id_ed25519"
PUBURL="https://github.com/cjc214foodun9/HENRI.git"
RWT="/workspace/henri-verify"; WT="$RWT/_uhr03"
# The git CLONE is $RWT/repo; $WT is a LINKED WORKTREE of it. Fetch must run in
# the clone, not in $RWT (measured: "fatal: not a git repository").
REPO="$RWT/repo"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes
     -o ConnectTimeout=25 -o ServerAliveInterval=15 -i "$KEY" -p "$PORT" "root@$HOST")

echo "SHA=$SHA STEPS=$STEPS"

echo
echo "=== 0. DEP GATE (arc_agi + torch; no pipe, so it can actually fail) ==="
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
echo "=== 1. REFRESH WORKTREE TO THE CARRIER SHA (fetch from the public URL) ==="
"${SSH[@]}" RWT="$RWT" WT="$WT" REPO="$REPO" SHA="$SHA" PUBURL="$PUBURL" bash -s <<'REMOTE'
set -u
cd "$REPO" || exit 2
git fetch "$PUBURL" "carrier/uhr-01-homologous-representation:refs/remotes/pub/uhr01" 2>&1 | tail -2
if [ -z "$SHA" ]; then SHA="$(git rev-parse refs/remotes/pub/uhr01)"; echo "SHA_DEFAULTED_TO=$SHA"; fi
cd "$WT" || { echo "WT_MISSING $WT"; exit 2; }
git checkout -f --detach "$SHA" 2>&1 | tail -2
echo "HEAD=$(git rev-parse HEAD)"
echo "unexpected_sha=$([ "$(git rev-parse HEAD)" = "$SHA" ] && echo NO || echo YES)"
echo "local_mods_lines=$(git status --porcelain=v1 -uall | wc -l)"
O="HENRI V2/models/henri_decoder_checkpoint.pt"
echo "overlay_bytes=$(stat -c '%s' "$O" 2>/dev/null || echo MISSING)"
echo "overlay_sha16=$(sha256sum "$O" 2>/dev/null | cut -c1-16 || echo NA)"
echo "flag_reads=$(grep -c 'HENRI_UHR02_EXTERO_GATE' 'HENRI V2/production_arc_run.py')"
echo "floor_1e-5=$(grep -c 'min_norm: float = 1e-5' 'HENRI V2/uhr02_exteroceptive_gate.py')"
echo "device_fix=$(grep -c 'device=dev' 'HENRI V2/uhr02_exteroceptive_gate.py')"
# FAIL-CLOSED PRECONDITION GATE. Measured defect (kill-run #2 attempt 1): the
# refresh silently no-op'd on stale code and this block only PRINTED
# `unexpected_sha=YES` / `floor_1e-5=0` / `device_fix=0`, then the arms ran
# anyway and burned GPU for zero admissible evidence. A gate that prints but
# does not stop is not a gate. This one EXITS.
G=0
[ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "GATE_FAIL sha_mismatch"; G=1; }
[ "$(grep -c 'min_norm: float = 1e-5' 'HENRI V2/uhr02_exteroceptive_gate.py')" = "1" ] || { echo "GATE_FAIL floor"; G=1; }
[ "$(grep -c 'device=dev' 'HENRI V2/uhr02_exteroceptive_gate.py')" -ge "2" ] || { echo "GATE_FAIL device"; G=1; }
[ "$(stat -c '%s' "$O" 2>/dev/null)" = "799034119" ] || { echo "GATE_FAIL overlay"; G=1; }
if [ "$G" -ne 0 ]; then echo "PRECONDITION_GATE_FAIL: refusing to run arms on stale code"; exit 1; fi
echo "PRECONDITION_GATE_PASS"
REMOTE

echo
echo "=== 2. RUNNER SMOKE: module-level imports execute (line 41 = import arc_agi) ==="
SMOKE_OUT=$("${SSH[@]}" RWT="$RWT" WT="$WT" bash -s <<'REMOTE'
cd "$WT/HENRI V2" || { echo "CD_FAIL"; exit 3; }
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
  echo "=== 3. ARM=$ARM  HENRI_UHR02_EXTERO_GATE=$FLAG ==="
  "${SSH[@]}" RWT="$RWT" WT="$WT" ARM="$ARM" FLAG="$FLAG" STEPS="$STEPS" bash -s <<'REMOTE'
set -u
D="$RWT/telemetry_uhr03_$ARM"
rm -rf "$D"; mkdir -p "$D"
# ---- env INSIDE the quoted heredoc: no local expansion, no remote re-split ----
export HENRI_ARC_SAGNAC_VETO=1
export HENRI_UHR01_RFSS=1
export HENRI_MACRO_NUM_CHANNELS=1
export HENRI_OFFLINE_DIAG=1
export EXTERNAL_OUTCOME_EFE=1
export HENRI_TRACE_UPDATE_GATES=1
export HENRI_SINGLE_ENV=ft09
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
    "HENRI_SINGLE_ENV": "ft09",
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
echo "--- run: phase823_live_gauntlet steps=$STEPS env=ft09 ---"
/usr/bin/python3 production_arc_run.py --mode phase823_live_gauntlet \
    --envs 1 --steps "$STEPS" > "$D/run.log" 2>&1
_rc=$?
echo "ARM=$ARM exit=$_rc   (nonzero => BLOCKED_INFRASTRUCTURE, NO science claim)"
# ENV-NOT-FOUND GUARD. If HENRI_SINGLE_ENV matches nothing the runner prints
# "matched no environment; aborting" and RETURNS 0 with ZERO steps. That silent
# no-op would look like a clean arm. Report it as BLOCKED, never as an arm.
if [ "$_rc" -eq 0 ] && grep -q "matched no environment" "$D/run.log" 2>/dev/null; then
  echo "ARM_BLOCKED_ENV_NOT_FOUND: HENRI_SINGLE_ENV matched no environment; 0 steps ran"
  _rc=90
fi
if [ "$_rc" -eq 0 ] && [ ! -s "$D/run.log" ]; then
  echo "ARM_BLOCKED_EMPTY_LOG: run.log is empty"; _rc=90
fi
if [ "$_rc" -ne 0 ]; then
  echo "--- run.log HEAD ---"; head -10 "$D/run.log"
  echo "--- run.log TAIL ---"; tail -20 "$D/run.log"
fi
echo "--- env + movement witnesses from run.log ---"
grep -E "\[init\]|ingress|in-context|BLOCKED|ENV: " "$D/run.log" | head -8
echo "--- files ---"; ls -l "$D"

D="$D" /usr/bin/python3 - <<'PYX'
import json, glob, os
D = os.environ["D"]
fs = sorted(glob.glob(os.path.join(D, "*.jsonl")))
if not fs:
    print("NO_JSONL in", D); raise SystemExit(0)
f = fs[0]
rows = []
for l in open(f):
    if l.strip():
        try: rows.append(json.loads(l))
        except Exception: pass
print("jsonl=%s lines=%d parsed=%d" % (os.path.basename(f), sum(1 for _ in open(f)), len(rows)))

# --- THE PRECONDITION WITNESS: did the world move at all? ---
fc = sum(1 for r in rows if (r.get("outcome_probe") or {}).get("frame_changed"))
cc = sum(1 for r in rows if ((r.get("outcome_probe") or {}).get("changed_cells") or 0) > 0)
print("MOVEMENT: frame_changed=%d/%d  changed_cells>0=%d/%d" % (fc, len(rows), cc, len(rows)))

envs = sorted({str(r.get("env")) for r in rows if r.get("env")})
print("envs:", envs)
print("store sizes:", sorted({r.get("preference_store_size") for r in rows}))
st = sorted({r.get("status") for r in rows if r.get("status")})
print("statuses:", st)

ui = [r["phase820_update_info"] for r in rows if r.get("phase820_update_info")]
print("update_info n=%d" % len(ui))
for k in ("target_theta_norm", "repeat_count", "stalled", "action", "temperature"):
    print("   %-18s unique=%s" % (k, sorted({repr(u.get(k)) for u in ui})[:4]))

gs = [r["phase820_guard_state"] for r in rows if r.get("phase820_guard_state")]
print("guard_state n=%d" % len(gs))
if gs:
    print("   updated_true=%d" % sum(1 for g in gs if g.get("updated")))
    print("   first=%s" % json.dumps(gs[0]))

ex = [r["phase820_extero_info"] for r in rows if r.get("phase820_extero_info")]
print("extero_info n=%d" % len(ex))
print("   statuses:", sorted({str(e.get("status")) for e in ex}))
for e in ex[:4]:
    print("   ", json.dumps(e, default=str)[:420])
if ex and ex[0].get("status") == "OK":
    ok = [e for e in ex if e.get("status") == "OK"]
    print("   C1 argmin_hits_truth: %d/%d" % (
        sum(1 for e in ok if e.get("argmin_hits_truth")), len(ok)))
    print("   C1 n_recorded values:", sorted({e.get("n_recorded") for e in ok}))
    print("   C2 invalid_minus_own>0: %d/%d" % (
        sum(1 for e in ok if (e.get("invalid_minus_own") or 0) > 0), len(ok)))
    print("   C3 delta_extero distinct:", sorted({e.get("delta_extero") for e in ok})[:8])
    print("   magnitude_only_risk:", sorted({e.get("magnitude_only_risk") for e in ok}))
    print("   role_coherence:", sorted({e.get("role_coherence") for e in ok})[:5])
PYX
REMOTE
done

echo
echo "=== 4. TELEMETRY TREE (remote) ==="
"${SSH[@]}" RWT="$RWT" bash -s <<'REMOTE'
for a in BASELINE RFSS; do
  echo "-- $a --"; ls -l "$RWT/telemetry_uhr03_$a" 2>/dev/null || echo "  ABSENT"
done
REMOTE
echo "UHR03_ARMS_DONE"
