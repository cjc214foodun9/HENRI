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
    mkdir -p "$LOCALAPPDATA/Temp/_uhr03_overlay/models"
    SRC="C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2/models/henri_decoder_checkpoint.pt"
    [ -f "$LOCALAPPDATA/Temp/_uhr03_overlay/models/henri_decoder_checkpoint.pt" ] || cp "$SRC" "$LOCALAPPDATA/Temp/_uhr03_overlay/models/"
    scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes \
        -i "$KEY" -P "$PORT" \
        "$LOCALAPPDATA/Temp/_uhr03_overlay/models/henri_decoder_checkpoint.pt" \
        "root@$HOST:$WT/HENRI V2/models/henri_decoder_checkpoint.pt" || { echo "SCP_FAIL"; exit 1; }
    "${SSH[@]}" "sha256sum '$WT/HENRI V2/models/henri_decoder_checkpoint.pt'; stat -c '%s bytes' '$WT/HENRI V2/models/henri_decoder_checkpoint.pt'"
fi

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
