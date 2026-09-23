#!/usr/bin/env bash
# UHR-01 PAIRED REMOTE A/B — the pre-registered gate
# (experiments/verification/uhr01_preregistration.md)
#
# DESIGN
#   ARM A (BASELINE): HENRI_UHR01_RFSS unset -> legacy cross-family comparison.
#                     Known signature: delta_axiom ~0.996..0.999, hard_vetoed 8/8.
#   ARM B (RFSS)    : HENRI_UHR01_RFSS=1     -> candidate projected into the axiom
#                     family. Claim: delta_axiom collapses for compliant options.
#   Identical mode, steps, envs, seed. Per-arm telemetry isolation.
#
# FAIL-CLOSED: any nonzero arm exit => BLOCKED_INFRASTRUCTURE, never a science
# verdict (multi-arm kill-matrix rule).
#
# HONESTY LABEL: HENRI_OFFLINE_DIAG=1 selects `offline://surrogate`, which the
# runner's own comment marks DIAGNOSTIC-ONLY. The solve rate from this run is NOT
# score-eligible and is reported as diagnostic evidence only.
set -uo pipefail

HOST="${1:?usage: $0 <ssh_host> <ssh_port>}"
PORT="${2:?usage: $0 <ssh_host> <ssh_port>}"
KEY="$HOME/.ssh/id_ed25519"
CARRIER_SHA="${3:-d825d98aa0538bcbcf1ab895e9110f51ffe1a1be}"
PUBURL="https://github.com/cjc214foodun9/HENRI.git"
RWT="/workspace/henri-verify"

SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25
     -o IdentitiesOnly=yes -i "$KEY" -p "$PORT" "root@$HOST")

echo "=== 0. SSH GATE ==="
"${SSH[@]}" 'echo SSH_OK; hostname; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader' || {
  echo "SSH_GATE_FAIL"; exit 1; }

echo "=== 1. INVENTORY (disk preserved from the prior stop) ==="
"${SSH[@]}" "bash -lc '
  echo worktree=\$(test -d \"$RWT\" && echo PRESENT || echo ABSENT)
  echo overlay=\$(test -f \"/$RWT/HENRI V2/models/henri_decoder_checkpoint.pt\" && echo PRESENT || echo ABSENT)
  test -f \"/$RWT/HENRI V2/models/henri_decoder_checkpoint.pt\" && sha256sum \"/$RWT/HENRI V2/models/henri_decoder_checkpoint.pt\"
  df -h /workspace | tail -1
  /usr/bin/python3 -c \"import torch,arc_agi;print(\\\"torch\\\",torch.__version__,\\\"cuda\\\",torch.cuda.is_available(),\\\"arc_agi OK\\\")\" 2>&1 | tail -2
'"

echo "=== 2. FETCH THE CARRIER TIP (public URL; configured origin needs creds) ==="
"${SSH[@]}" "bash -lc '
  cd \"$RWT\" 2>/dev/null || { echo NO_WORKTREE; exit 1; }
  git fetch \"$PUBURL\" carrier/uhr-01-homologous-representation:refs/remotes/pub/uhr-01 2>&1 | tail -3
  echo fetched=\$?
  git rev-parse refs/remotes/pub/uhr-01
'"

echo "=== 3. MATERIALIZE THE CARRIER AT THE EXACT SHA (detached, clean) ==="
"${SSH[@]}" "bash -lc '
  cd \"$RWT\"
  CUR=\$(git -C \"$RWT/_uhr01\" rev-parse HEAD 2>/dev/null || echo none)
  if [ \"\$CUR\" != \"$CARRIER_SHA\" ]; then
    echo \"rebuilding worktree (cur=\$CUR)\"
    rm -rf \"$RWT/_uhr01\" 2>/dev/null
    git worktree prune 2>/dev/null
    git worktree add --detach \"$RWT/_uhr01\" $CARRIER_SHA 2>&1 | tail -3
  else
    echo \"worktree already at \$CARRIER_SHA -- PRESERVING overlay\"
  fi
  cd \"$RWT/_uhr01\"
  echo HEAD=\$(git rev-parse HEAD)
  echo status_lines=\$(git status --porcelain=v1 -uall | wc -l)
  grep -c \"def project_to_boundary_family\" \"HENRI V2/opine_object_mcts.py\"
  grep -c \"HENRI_UHR01_RFSS\" \"HENRI V2/production_arc_run.py\"
  test -f \"HENRI V2/uhr_rfss.py\" && echo uhr_rfss=PRESENT
  test -f \"HENRI V2/tests/contract/test_uhr01_rfss_homology.py\" && echo tests=PRESENT
  echo overlay=\$(test -f \"HENRI V2/models/henri_decoder_checkpoint.pt\" && echo PRESENT || echo ABSENT)
'"

echo "=== 4. CONTRACT SUITES ON THE REMOTE INTERPRETER ==="
"${SSH[@]}" "bash -lc '
  cd \"$RWT/_uhr01/HENRI V2\"
  export PYTHONPATH=\"\$PWD\"
  for s in tests/contract/test_uhr01_rfss_homology.py tests/contract/test_macro_resolution_coupling.py tests/contract/test_sagnac_width_contract.py; do
    /usr/bin/python3 -m pytest \$s -q --no-header -p no:cacheprovider 2>&1 | tail -2
  done
'"

echo "=== 5. BLOCK-NORM + ADJOINT CHECK ON THE FULL-SCALE AXIOM SHAPE ==="
"${SSH[@]}" "bash -lc '
  cd \"$RWT/_uhr01/HENRI V2\"
  export PYTHONPATH=\"\$PWD\"
  /usr/bin/python3 - <<PY
import math, torch
from uhr_rfss import (project_option_to_boundary_family, block_norm_deviation,
                      adjoint_matrix, axiom_sensitivity)
s3 = math.sqrt(3.0)
b = [[[0,1,0],[1,0,0],[0,0,0]],[[0,-1j,0],[1j,0,0],[0,0,0]],
     [[1,0,0],[0,-1,0],[0,0,0]],[[0,0,1],[0,0,0],[1,0,0]],
     [[0,0,-1j],[0,0,0],[1j,0,0]],[[0,0,0],[0,0,1],[0,1,0]],
     [[0,0,0],[0,0,-1j],[0,1j,0]],
     [[1/s3,0,0],[0,1/s3,0],[0,0,-2/s3]]]
basis = torch.tensor(b, dtype=torch.complex64)
g = torch.Generator().manual_seed(11)
roles = torch.randn(8192, 8, generator=g); roles = roles/roles.norm(dim=-1, keepdim=True)
def gen(a, sc):
    gg = torch.Generator().manual_seed(1000+a)
    th = torch.randn(8, generator=gg)*sc
    return 1j*torch.einsum(\"a,aij->ij\", th.to(basis.dtype), basis)
gens = [gen(i, 0.10) for i in range(4)]
w = project_option_to_boundary_family(gens, basis, roles)
print(\"DEVICE\", w.device, \"SHAPE\", tuple(w.shape), \"DTYPE\", w.dtype, \"COMPLEX\", w.is_complex())
print(\"block_norm_dev\", block_norm_deviation(w))
x = w.flatten().to(torch.float32); y = roles.flatten().to(torch.float32)
cos = float((x*y).sum())/(float(x.norm())*float(y.norm()))
print(\"COMPLIANT delta_axiom=\", 1.0-0.5*(1.0+cos))
A = torch.eye(8)
for gg in gens: A = A @ adjoint_matrix(gg.unsqueeze(0), basis)
print(\"orth_err\", float((A@A.T-torch.eye(8)).abs().max()), \"det\", float(torch.linalg.det(A.double())))
perm = torch.randperm(8192, generator=torch.Generator().manual_seed(3))
ws = project_option_to_boundary_family(gens, basis, roles, role_permutation=perm)
xs = ws.flatten().to(torch.float32)
coss = float((xs*y).sum())/(float(xs.norm())*float(y.norm()))
print(\"INVALID  delta_axiom=\", 1.0-0.5*(1.0+coss))
PY
'"

echo "=== 6. PAIRED A/B (baseline then RFSS; identical seed/steps/envs) ==="
for ARM in BASELINE RFSS; do
  if [ "$ARM" = "RFSS" ]; then FLAG="1"; else FLAG="0"; fi
  echo "--- ARM=$ARM HENRI_UHR01_RFSS=$FLAG ---"
  "${SSH[@]}" "bash -lc '
    cd \"$RWT/_uhr01/HENRI V2\"
    export PYTHONPATH=\"\$PWD\"
    D=\"$RWT/telemetry_uhr01_$ARM\"; mkdir -p \"\$D\"
    HENRI_TELEMETRY_DIR=\"\$D\" \
    HENRI_ARC_SAGNAC_VETO=1 \
    HENRI_UHR01_RFSS=$FLAG \
    HENRI_MACRO_NUM_CHANNELS=1 \
    HENRI_OFFLINE_DIAG=1 \
    /usr/bin/python3 production_arc_run.py --mode phase823_live_gauntlet --envs 1 --steps 8 \
      > \"\$D/run.log\" 2>&1
    echo \"ARM=$ARM exit=\$?\"
    echo \"--- veto payloads ---\"
    grep -o \"delta_axiom[^,}]*\" \"\$D/run.log\" | head -12
    echo \"--- summary lines ---\"
    grep -E \"score|levels_completed|envs_scored|actions:|opine|UNAVAILABLE|einsum\" \"\$D/run.log\" | tail -12
  '"
done

echo "=== 7. EGRESS (pull receipts BEFORE any stop) ==="
mkdir -p "$LOCALAPPDATA/Temp/uhr01_egress"
for ARM in BASELINE RFSS; do
  scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes \
      -i "$KEY" -P "$PORT" -r \
      "root@$HOST:$RWT/telemetry_uhr01_$ARM" \
      "$LOCALAPPDATA/Temp/uhr01_egress/" 2>&1 | tail -2 || echo "SCP_$ARM_FAIL"
done
ls -la "$LOCALAPPDATA/Temp/uhr01_egress" 2>/dev/null
echo "EGRESS_DONE"
