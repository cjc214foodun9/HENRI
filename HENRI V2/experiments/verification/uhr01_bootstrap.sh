#!/usr/bin/env bash
# PARAMETRIZED remote bootstrap for the UHR-01 paired A/B.
# Reusable on a fresh instance when the original host has no capacity.
#   usage: bash uhr01_bootstrap.sh <ssh_host> <ssh_port> [carrier_sha]
set -uo pipefail

HOST="${1:?usage: $0 <ssh_host> <ssh_port> [carrier_sha]}"
PORT="${2:?usage: $0 <ssh_host> <ssh_port> [carrier_sha]}"
SHA="${3:-d825d98aa0538bcbcf1ab895e9110f51ffe1a1be}"
KEY="$HOME/.ssh/id_ed25519"
PUBURL="https://github.com/cjc214foodun9/HENRI.git"
RWT="/workspace/henri-verify"
A="$RWT/HENRI V2"

SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25
     -o IdentitiesOnly=yes -i "$KEY" -p "$PORT" "root@$HOST")

echo "=== 0. SSH GATE ==="
"${SSH[@]}" 'echo SSH_OK; hostname; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader' || exit 1

echo "=== 1. inventory: what does THIS instance already have? ==="
"${SSH[@]}" "bash -lc '
  echo repo=\$(test -d \"$RWT\" && echo PRESENT || echo ABSENT)
  echo overlay=\$(test -f \"$A/models/henri_decoder_checkpoint.pt\" && echo PRESENT || echo ABSENT)
  test -f \"$A/models/henri_decoder_checkpoint.pt\" && sha256sum \"$A/models/henri_decoder_checkpoint.pt\"
  df -h /workspace | tail -1
  /usr/bin/python3 -c \"import torch;print(\\\"torch\\\",torch.__version__,\\\"cuda\\\",torch.cuda.is_available())\" 2>&1 | tail -1
  /usr/bin/python3 -c \"import arc_agi;print(\\\"arc_agi OK\\\")\" 2>&1 | tail -1
'"

echo "=== 2. arc_agi (idempotent, tolerant of the blinker uninstall conflict) ==="
"${SSH[@]}" "bash -lc '
  /usr/bin/python3 -c \"import arc_agi\" 2>/dev/null && { echo arc_agi=ALREADY; exit 0; }
  /usr/bin/python3 -m pip install --no-input --disable-pip-version-check \
      --ignore-installed blinker arc-agi 2>&1 | tail -4
  /usr/bin/python3 -c \"import arc_agi;print(\\\"arc_agi=OK\\\")\" 2>&1 | tail -1
'"

echo "=== 3. clone the PUBLIC url + materialize the carrier SHA (detached, clean) ==="
"${SSH[@]}" "bash -lc '
  if [ ! -d \"$RWT/.git\" ]; then
    git clone \"$PUBURL\" \"$RWT\" 2>&1 | tail -2
  fi
  cd \"$RWT\" || exit 1
  git fetch \"$PUBURL\" carrier/uhr-01-homologous-representation:refs/remotes/pub/uhr01 2>&1 | tail -2
  git worktree prune 2>/dev/null
  rm -rf \"$RWT/_uhr01\" 2>/dev/null
  git worktree add --detach \"$RWT/_uhr01\" $SHA 2>&1 | tail -2
  cd \"$RWT/_uhr01\"
  echo HEAD=\$(git rev-parse HEAD)
  echo status_lines=\$(git status --porcelain=v1 -uall | wc -l)
  echo uhr_rfss=\$(test -f \"HENRI V2/uhr_rfss.py\" && echo PRESENT || echo ABSENT)
  echo opine_delegate=\$(grep -c \"def project_to_boundary_family\" \"HENRI V2/opine_object_mcts.py\")
  echo flag_reads=\$(grep -c \"HENRI_UHR01_RFSS\" \"HENRI V2/production_arc_run.py\")
'"
echo "BOOTSTRAP_DONE"
