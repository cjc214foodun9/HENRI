#!/usr/bin/env bash
# UHR-03 REMOTE PAIRED A/B — FORM B (exteroceptive comparison domain).
#
# THE EXPERIMENTAL VARIABLE IS EXACTLY ONE FLAG:
#   ARM A: HENRI_UHR02_EXTERO_GATE=0
#   ARM B: HENRI_UHR02_EXTERO_GATE=1
# Everything else is COMMON, including the ATTRIBUTED cause of the empty store:
#   EXTERNAL_OUTCOME_EFE=1      (the dead-flag ancestor found in UHR-03 attribution)
#   HENRI_TRACE_UPDATE_GATES=1  (guard-state receipt, so a silent skip self-reports)
#   HENRI_UHR01_RFSS=1          (holds the UHR-01 veto regime FIXED in both arms,
#                                so a UHR-01 effect cannot be read as a FORM B effect)
#
# Quoting strategy: every remote block is sent as a HEREDOC on stdin
# (`ssh ... bash -s -- args`, quoted delimiter). No nested quoting, so no
# MSYS/bash mangling -- the defect that broke the first draft of this script.
#
# Usage: bash uhr03_remote_ab.sh <host> <port> <sha>
set -uo pipefail
HOST="${1:?host}"; PORT="${2:?port}"; SHA="${3:?sha}"
KEY="$HOME/.ssh/id_ed25519"
RWT="/workspace/henri-verify"
WT="$RWT/_uhr03"
PUBURL="https://github.com/cjc214foodun9/HENRI.git"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes
     -o ConnectTimeout=25 -o ServerAliveInterval=15 -i "$KEY" -p "$PORT" "root@$HOST")

echo "=== 1. PREFLIGHT + MATERIALIZE THE CARRIER AT THE EXACT SHA ==="
"${SSH[@]}" bash -s -- "$RWT" "$WT" "$SHA" "$PUBURL" <<'REMOTE' || { echo "PREFLIGHT_FAIL"; exit 1; }
set -u
RWT="$1"; WT="$2"; SHA="$3"; PUBURL="$4"
set -e
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
/usr/bin/python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
/usr/bin/python3 -c 'import numpy, pytest; print("numpy", numpy.__version__, "pytest", pytest.__version__)'
mkdir -p "$RWT"
if [ ! -d "$RWT/repo/.git" ]; then git clone --quiet "$PUBURL" "$RWT/repo"; fi
cd "$RWT/repo"; git fetch --quiet origin
CUR="$(git -C "$WT" rev-parse HEAD 2>/dev/null || echo none)"
if [ "$CUR" != "$SHA" ]; then
    git worktree prune
    rm -rf "$WT"
    git worktree add --detach "$WT" "$SHA"
fi
cd "$WT"
echo "HEAD=$(git rev-parse HEAD)"
echo "STATUS_LINES=$(git status --porcelain | wc -l)"
test "$(git rev-parse HEAD)" = "$SHA" || { echo SHA_MISMATCH; exit 1; }
test -f "$WT/HENRI V2/production_arc_run.py" || { echo NO_RUNNER; exit 1; }
echo "flag_extero=$(grep -c HENRI_UHR02_EXTERO_GATE "$WT/HENRI V2/production_arc_run.py")"
echo "flag_trace=$(grep -c HENRI_TRACE_UPDATE_GATES "$WT/HENRI V2/production_arc_run.py")"
echo "flag_rfss=$(grep -c HENRI_UHR01_RFSS "$WT/HENRI V2/production_arc_run.py")"
REMOTE

echo
echo "=== 2. DECODER OVERLAY ==="
REMOTE_SHA=$("${SSH[@]}" "if [ -f '$WT/HENRI V2/models/henri_decoder_checkpoint.pt' ]; then sha256sum '$WT/HENRI V2/models/henri_decoder_checkpoint.pt' | cut -c1-16; else echo ABSENT; fi" 2>/dev/null | tail -1)
echo "remote overlay sha16 = $REMOTE_SHA"
if [ "$REMOTE_SHA" != "75572389083455a3" ]; then
    echo "--- transferring the 799,034,119 B overlay ---"
    # DEFECT FIXED HERE: scp to a remote path containing a SPACE ('HENRI V2')
    # word-splits on the remote side. Stage into a no-space path, then mv
    # remotely inside a heredoc block where the space is properly quoted.
    SRC="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2/models/henri_decoder_checkpoint.pt"
    [ -f "$SRC" ] || { echo "LOCAL_OVERLAY_MISSING: $SRC"; exit 1; }
    echo "local: $(stat -c '%s bytes' "$SRC" 2>/dev/null || python -c "import os,sys;print(os.path.getsize(sys.argv[1]),'bytes')" "$SRC")"
    scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes \
        -i "$KEY" -P "$PORT" "$SRC" "root@$HOST:$RWT/_ckpt_stage.pt" || { echo "SCP_FAIL"; exit 1; }
    "${SSH[@]}" bash -s -- "$RWT" "$WT" <<'REMOTE'
set -eu
RWT="$1"; WT="$2"
mkdir -p "$WT/HENRI V2/models"
mv "$RWT/_ckpt_stage.pt" "$WT/HENRI V2/models/henri_decoder_checkpoint.pt"
O="$WT/HENRI V2/models/henri_decoder_checkpoint.pt"
sha256sum "$O"
stat -c '%s bytes' "$O"
echo "verify_prefix=$(sha256sum "$O" | cut -c1-10)"
REMOTE
fi

echo
echo "=== 2b. DEPS: arc_agi + arcade (idempotent, FAIL CLOSED) ==="
"${SSH[@]}" bash -s <<'REMOTE'
set -u
# DEFECT FIXED HERE: uhr03_remote_ab.sh originally omitted this step, so BOTH arms
# died in <1 s with `ModuleNotFoundError: No module named 'arc_agi'` at
# production_arc_run.py:41. That is BLOCKED_INFRASTRUCTURE, not a scientific
# verdict. The contract suites still passed because they need only torch/numpy/
# pytest -- so the suites are NOT a substitute for this check.
if /usr/bin/python3 -c 'import arc_agi, arcade' 2>/dev/null; then
    echo "arc_agi=ALREADY"
else
    echo "installing arc-agi (the blinker RECORD conflict is known)"
    /usr/bin/python3 -m pip install --no-input --disable-pip-version-check \
        --ignore-installed blinker arc-agi 2>&1 | tail -3
fi
if ! /usr/bin/python3 -c 'import arc_agi' 2>/dev/null; then
    echo "DEPS_FAIL: arc_agi still not importable -- aborting before any arm"
    exit 1
fi
/usr/bin/python3 -c 'import arc_agi, arcade, torch; print("deps OK: arc_agi + arcade + torch", torch.__version__)'
REMOTE
[ $? -eq 0 ] || { echo "DEPS_GATE_FAIL"; exit 1; }

echo
echo "=== 3. CONTRACT SUITES ON CUDA ==="
"${SSH[@]}" bash -s -- "$WT" <<'REMOTE'
set -u
WT="$1"
cd "$WT/HENRI V2"
export PYTHONPATH="$PWD"
/usr/bin/python3 -m pytest tests/contract/test_uhr02_exteroceptive_gate.py \
    tests/contract/test_uhr01_rfss_homology.py -q --no-header 2>&1 | tail -4
REMOTE

echo
echo "=== 4. PAIRED A/B (sequential, one GPU, ONE differing flag) ==="
for ARM in BASELINE RFSS; do
  case "$ARM" in BASELINE) FLAG=0 ;; RFSS) FLAG=1 ;; esac
  echo "--- ARM=$ARM HENRI_UHR02_EXTERO_GATE=$FLAG ---"
  "${SSH[@]}" bash -s -- "$RWT" "$WT" "$ARM" "$FLAG" <<'REMOTE'
set -u
RWT="$1"; WT="$2"; ARM="$3"; FLAG="$4"
cd "$WT/HENRI V2"
export PYTHONPATH="$PWD"
D="$RWT/telemetry_uhr03_$ARM"; mkdir -p "$D"
HENRI_TELEMETRY_DIR="$D" \
HENRI_ARC_SAGNAC_VETO=1 \
HENRI_UHR01_RFSS=1 \
HENRI_MACRO_NUM_CHANNELS=1 \
HENRI_OFFLINE_DIAG=1 \
EXTERNAL_OUTCOME_EFE=1 \
HENRI_TRACE_UPDATE_GATES=1 \
HENRI_UHR02_EXTERO_GATE="$FLAG" \
/usr/bin/python3 production_arc_run.py --mode phase823_live_gauntlet \
    --envs 1 --steps 16 > "$D/run.log" 2>&1
echo "ARM=$ARM exit=$?"
ls -l "$D" | tail -5
echo "--- guard/extero/update probes in the JSONL ---"
grep -oE '"(external_outcome_efe|skipped_by|updated)":[^,}]*' "$D"/*.jsonl 2>/dev/null | sort | uniq -c | head -8
grep -oE '"(delta_extero|delta_extero_invalid_min|invalid_minus_own|argmin_hits_truth|n_recorded|role_coherence|relative_group_element)":[^,}]*' "$D"/*.jsonl 2>/dev/null | head -24
echo "--- errors ---"
grep -E 'UNAVAILABLE|einsum|update failed|Traceback|ERROR' "$D/run.log" | head -8
REMOTE
done

echo
echo "=== 5. TELEMETRY TREE (egress pulls this; nothing deleted here) ==="
"${SSH[@]}" bash -s -- "$RWT" <<'REMOTE'
set -u
RWT="$1"
for a in BASELINE RFSS; do
  D="$RWT/telemetry_uhr03_$a"
  echo "-- $a --"
  ls -l "$D" 2>/dev/null | tail -6
done
REMOTE
echo "UHR03_REMOTE_AB_DONE"
