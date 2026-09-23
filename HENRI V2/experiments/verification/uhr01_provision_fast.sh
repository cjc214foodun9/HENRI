#!/usr/bin/env bash
# UHR-01 PROVISION (fast steps only) — fresh instance 52182086.
#
# WHY THIS EXISTS
#   uhr01_bootstrap.sh had two defects (found by my own review):
#     (1) it never transfers the 762 MB decoder overlay, and at d_model=65536 the
#         decoder's checkpoint_policy is "required" -> SagnacMCTSPlanner fails to
#         construct -> sagnac_planner=None -> the veto block is SKIPPED. C2 would
#         then report BLOCKED, not measured.
#     (2) it `rm -rf`s the worktree unconditionally, which would DELETE a placed
#         overlay. Code must be materialized FIRST, the overlay placed LAST.
#   This script does the fast, idempotent steps: gate -> inventory -> deps -> code.
#   The 762 MB transfer is a SEPARATE step so a timeout cannot corrupt state.
set -uo pipefail

HOST="${1:?usage: $0 <ssh_host> <ssh_port> [sha]}"
PORT="${2:?usage: $0 <ssh_host> <ssh_port> [sha]}"
SHA="${3:-d825d98aa0538bcbcf1ab895e9110f51ffe1a1be}"
KEY="$HOME/.ssh/id_ed25519"
PUBURL="https://github.com/cjc214foodun9/HENRI.git"
RWT="/workspace/henri-verify"
WT="$RWT/_uhr01"
AC="$WT/HENRI V2"

SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25
     -o IdentitiesOnly=yes -i "$KEY" -p "$PORT" "root@$HOST")

echo "=== 0. SSH GATE (SSH_OK AND a plausible GPU line, or stop) ==="
"${SSH[@]}" 'echo SSH_OK; hostname; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader' || {
  echo "SSH_GATE_FAIL"; exit 1; }

echo
echo "=== 1. INVENTORY (fresh instance: expect no repo, no overlay) ==="
"${SSH[@]}" "bash -lc '
  echo repo=\$(test -d \"$RWT/.git\" && echo PRESENT || echo ABSENT)
  echo worktree=\$(test -d \"$WT\" && echo PRESENT || echo ABSENT)
  echo overlay=\$(test -f \"$AC/models/henri_decoder_checkpoint.pt\" && echo PRESENT || echo ABSENT)
  echo image_models=\$(ls /workspace/HENRI/\"HENRI V2\"/models 2>/dev/null | head -3 | tr \"\\n\" \",\")
  df -h /workspace | tail -1
  /usr/bin/python3 -c \"import torch;print(\\\"torch\\\",torch.__version__,\\\"cuda\\\",torch.cuda.is_available())\" 2>&1 | tail -1
  /usr/bin/python3 -c \"import arc_agi;print(\\\"arc_agi OK\\\")\" 2>&1 | tail -1
'"

echo
echo "=== 2. DEPS: arc_agi + arcade (idempotent; blinker conflict is known) ==="
"${SSH[@]}" "bash -lc '
  /usr/bin/python3 -c \"import arc_agi, arcade\" 2>/dev/null && { echo arc_agi=ALREADY; exit 0; }
  /usr/bin/python3 -m pip install --no-input --disable-pip-version-check \
      --ignore-installed blinker arc-agi 2>&1 | tail -3
  /usr/bin/python3 -c \"import arc_agi,arcade;print(\\\"arc_agi=OK\\\")\" 2>&1 | tail -1
'"

echo
echo "=== 3. CODE at the exact carrier SHA (public URL; configured origin needs creds) ==="
"${SSH[@]}" "bash -lc '
  if [ ! -d \"$RWT/.git\" ]; then git clone \"$PUBURL\" \"$RWT\" 2>&1 | tail -1; fi
  cd \"$RWT\" || exit 1
  git fetch \"$PUBURL\" carrier/uhr-01-homologous-representation:refs/remotes/pub/uhr01 2>&1 | tail -1
  CUR=\$(git -C \"$WT\" rev-parse HEAD 2>/dev/null || echo none)
  if [ \"\$CUR\" != \"$SHA\" ]; then
    echo \"rebuilding worktree (cur=\$CUR)\"
    rm -rf \"$WT\" 2>/dev/null; git worktree prune 2>/dev/null
    git worktree add --detach \"$WT\" $SHA 2>&1 | tail -1
  else
    echo worktree_already_at_sha
  fi
  cd \"$WT\" || exit 1
  echo HEAD=\$(git rev-parse HEAD)
  echo status_lines=\$(git status --porcelain=v1 -uall | wc -l)
  echo uhr_rfss=\$(test -f \"HENRI V2/uhr_rfss.py\" && echo PRESENT || echo ABSENT)
  echo opine_delegate=\$(grep -c \"def project_to_boundary_family\" \"HENRI V2/opine_object_mcts.py\")
  echo flag_decl=\$(grep -c \"HENRI_UHR01_RFSS = os.environ\" \"HENRI V2/production_arc_run.py\")
'"

echo
echo "=== 4. CONTRACT SUITES on the remote interpreter (scaffold cost) ==="
"${SSH[@]}" "bash -lc '
  cd \"$AC\" || exit 1
  export PYTHONPATH=\"\$PWD\"
  for s in tests/contract/test_uhr01_rfss_homology.py tests/contract/test_macro_resolution_coupling.py tests/contract/test_sagnac_width_contract.py; do
    /usr/bin/python3 -m pytest \$s -q --no-header -p no:cacheprovider 2>&1 | tail -2
  done
'"
echo "PROVISION_FAST_DONE"
