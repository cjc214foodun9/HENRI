#!/usr/bin/env bash
# UHR-03 REMOTE PAIRED A/B — FORM B (exteroceptive comparison domain).
#
# THE EXPERIMENTAL VARIABLE IS EXACTLY ONE FLAG:
#   ARM A: HENRI_UHR02_EXTERO_GATE=0
#   ARM B: HENRI_UHR02_EXTERO_GATE=1
# Everything else is COMMON, including the attributed cause of the empty store:
#   EXTERNAL_OUTCOME_EFE=1  (the dead-flag ancestor found in UHR-03 attribution)
#   HENRI_TRACE_UPDATE_GATES=1 (guard-state receipt, so a silent skip self-reports)
# Confound control: HENRI_UHR01_RFSS=1 in BOTH arms, so the UHR-01 veto regime is
# held fixed and cannot be mistaken for a FORM B effect.
#
# Usage: bash uhr03_remote_ab.sh <host> <port> <sha>
set -uo pipefail
HOST="${1:?host}"; PORT="${2:?port}"; SHA="${3:?sha}"
KEY="$HOME/.ssh/id_ed25519"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes
     -o ConnectTimeout=25 -o ServerAliveInterval=15 -i "$KEY" -p "$PORT" "root@$HOST")
RWT="/workspace/henri-verify"
WT="$RWT/_uhr03"
PUBURL="https://github.com/cjc214foodun9/HENRI.git"

echo "=== 0. PREFLIGHT (fail closed) ==="
"${SSH[@]}" "bash -lc '
  set -e
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
  /usr/bin/python3 -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())"
  /usr/bin/python3 -c "import numpy, pytest; print(\"numpy\","'"'"'numpy.__version__,"pytest",pytest.__version__)"'"'"'
  mkdir -p \"$RWT\"
'" || { echo "PREFLIGHT_FAIL"; exit 1; }

echo "=== 1. MATERIALIZE THE CARRIER AT THE EXACT SHA ==="
"${SSH[@]}" "bash -lc '
  set -e
  cd \"$RWT\"
  if [ ! -d \"$RWT/repo/.git\" ]; then
      git clone --quiet \"$PUBURL\" \"$RWT/repo\"
  fi
  cd \"$RWT/repo\"
  git fetch --quiet origin
  CUR=\$(git -C \"$WT\" rev-parse HEAD 2>/dev/null || echo none)
  if [ \"\$CUR\" != \"$SHA\" ]; then
      git worktree prune
      rm -rf \"$WT\"
      git worktree add --detach \"$WT\" $SHA
  fi
  cd \"$WT\"
  echo HEAD=\$(git rev-parse HEAD)
  echo STATUS_LINES=\$(git status --porcelain | wc -l)
  test \$(git rev-parse HEAD) = \"$SHA\" || { echo SHA_MISMATCH; exit 1; }
  test -f \"$WT/HENRI V2/production_arc_run.py\" || { echo NO_RUNNER; exit 1; }
  echo flag_ext = \$(grep -c \"HENRI_UHR02_EXTERO_GATE\" \"$WT/HENRI V2/production_arc_run.py\")
  echo flag_trc = \$(grep -c \"HENRI_TRACE_UPDATE_GATES\" \"$WT/HENRI V2/production_arc_run.py\")
'" || { echo "MATERIALIZE_FAIL"; exit 1; }

echo "=== 2. DECODER OVERLAY (799,034,119 B / 7557238908...) ==="
"${SSH[@]}" "bash -lc '
  O=\"$WT/HENRI V2/models/henri_decoder_checkpoint.pt\"
  if [ -f \"\$O\" ]; then
      echo -n \"local  \"; sha256sum \"\$O\" | cut -c1-16; stat -c \"%s bytes\" \"\$O\"
  else
      echo OVERLAY_ABSENT
  fi
'"

echo "=== 3. CONTRACT SUITES ON CUDA ==="
"${SSH[@]}" "bash -lc '
  cd \"$WT/HENRI V2\"
  export PYTHONPATH=\"\$PWD\"
  /usr/bin/python3 -m pytest tests/contract/test_uhr02_exteroceptive_gate.py \
      tests/contract/test_uhr01_rfss_homology.py -q --no-header 2>&1 | tail -4
'"

echo
echo "=== 4. PAIRED A/B (sequential, one GPU, common env, ONE differing flag) ==="
for ARM in BASELINE RFSS; do
  case "$ARM" in
    BASELINE) FLAG=0 ;;
    RFSS)     FLAG=1 ;;
  esac
  echo "--- ARM=$ARM HENRI_UHR02_EXTERO_GATE=$FLAG ---"
  "${SSH[@]}" "bash -lc '
    cd \"$WT/HENRI V2\"
    export PYTHONPATH=\"\$PWD\"
    D=\"$RWT/telemetry_uhr03_$ARM\"; mkdir -p \"\$D\"
    HENRI_TELEMETRY_DIR=\"\$D\" \
    HENRI_ARC_SAGNAC_VETO=1 \
    HENRI_UHR01_RFSS=1 \
    HENRI_MACRO_NUM_CHANNELS=1 \
    HENRI_OFFLINE_DIAG=1 \
    EXTERNAL_OUTCOME_EFE=1 \
    HENRI_TRACE_UPDATE_GATES=1 \
    HENRI_UHR02_EXTERO_GATE=$FLAG \
    /usr/bin/python3 production_arc_run.py --mode phase823_live_gauntlet \
        --envs 1 --steps 16 > \"\$D/run.log\" 2>&1
    echo \"ARM=$ARM exit=\$?\"
    ls -l \"\$D\" | tail -5
    echo \"--- guard / extero / update probes ---\"
    grep -oE \"\\(guard_state|delta_extero|skipped_by\\)[^,}]*\" \"\$D\"/*.jsonl 2>/dev/null | head -6
    echo \"--- veto + errors ---\"
    grep -E \"UNAVAILABLE|einsum|update failed|Traceback|ERROR\" \"\$D/run.log\" | head -8
  '"
done

echo
echo "=== 5. TELEMETRY TREE (egress pulls this; nothing is deleted here) ==="
"${SSH[@]}" "bash -lc '
  for a in BASELINE RFSS; do
    D=\"$RWT/telemetry_uhr03_\$a\"
    echo \"-- \$a --\"
    ls -l \"\$D\" 2>/dev/null | tail -6
  done
'"
