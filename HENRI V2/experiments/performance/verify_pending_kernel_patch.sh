#!/usr/bin/env bash
# Remote verification of the PENDING basal kernel patch (exact bytes, sm_120).
#
# v1 -> git rev-parse guard exited at step 0: the remote code root is a PLAIN
#       DIRECTORY, not a git checkout. (my harness defect)
# v2 -> collection failed with TWO errors:
#         NameError: name 'FUSED_TILE_SIZE' is not defined  (REAL CODE DEFECT:
#            the constant is USED at basal_triton_kernel.py:548 but was not
#            DEFINED anywhere in the repository -- an earlier fuzzy patch
#            consumed the definition line. Import-region patch hazard.)
#         ModuleNotFoundError: koopman_action_ledger  (MY closure defect: the
#            transfer list omitted modules the tests import transitively.)
# v3 -> FUSED_TILE_SIZE defined; FULL transitive dependency closure transferred.
set -uo pipefail

LOCAL="/c/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
RWT="/workspace/phase10/HENRI V2"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p 37414 root@ssh7.vast.ai"

# FULL CLOSURE (measured by AST-import walk of both test modules, recursively)
FILES=(
  "basal_triton_kernel.py"
  "basal_boundary_engine.py"
  "epsilon_band_gate.py"
  "koopman_action_ledger.py"
  "zone_bc_engram_sync.py"
  "hopfield_cleanup.py"
  "product_clifford_product_kernel.py"
  "unified_henri_vla_engine.py"
  "tests/unit/test_basal_triton_kernel.py"
  "tests/unit/test_basal_boundary_bundle.py"
)

echo "### STEP 0: remote root exists"
$SSH "mkdir -p '$RWT' && ls -d '$RWT'" || exit 2

echo
echo "### STEP 1: local SHA-256 (bytes under test)"
for f in "${FILES[@]}"; do
  printf "%s  %s\n" "$(sha256sum "$LOCAL/$f" | cut -d' ' -f1)" "$f"
done

echo
echo "### STEP 2: transfer full closure"
for f in "${FILES[@]}"; do
  d="$RWT/$(dirname "$f")"
  $SSH "mkdir -p '$d'" >/dev/null 2>&1 || exit 2
  cat "$LOCAL/$f" | $SSH "cat > '$RWT/$f'" || exit 2
  echo "sent $f"
done

echo
echo "### STEP 3: remote SHA-256 (MUST equal STEP 1)"
MISMATCH=0
for f in "${FILES[@]}"; do
  L=$(sha256sum "$LOCAL/$f" | cut -d' ' -f1)
  R=$($SSH "sha256sum '$RWT/$f' | cut -d' ' -f1")
  if [ "$L" = "$R" ]; then printf "OK   %s\n" "$f"; else printf "FAIL %s\n  local=%s\n  remote=%s\n" "$f" "$L" "$R"; MISMATCH=1; fi
done
[ "$MISMATCH" -eq 0 ] || { echo "### BLOCKED: transfer SHA mismatch"; exit 3; }

echo
echo "### STEP 4: constant provenance at the EXACT transferred bytes"
$SSH "cd '$RWT' && grep -n 'FUSED_TILE_SIZE = \|BLOCK_SPAN_DEFAULT = \|SPEC_BLOCK_SIZE = ' basal_triton_kernel.py"

echo
echo "### STEP 5: IMPORT PREFLIGHT (the v2 failure point)"
$SSH "cd '$RWT' && PYTHONPATH='HENRI V2' python3 -c \"
import basal_triton_kernel as bk
print('IMPORT_OK FUSED_TILE_SIZE=', bk.FUSED_TILE_SIZE, 'BLOCK_SPAN_DEFAULT=', bk.BLOCK_SPAN_DEFAULT, 'SPEC_BLOCK_SIZE=', bk.SPEC_BLOCK_SIZE)
\" 2>&1 | tail -12"

echo
echo "### STEP 6: basal suite at these exact bytes"
$SSH "cd '$RWT' && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='HENRI V2' timeout 1500 python3 -m pytest tests/unit/test_basal_triton_kernel.py tests/unit/test_basal_boundary_bundle.py -q --tb=line -p no:cacheprovider 2>&1 | tail -30; echo REAL_RC=\${PIPESTATUS[0]}"

echo
echo "### STEP 7: runtime identity"
$SSH "cd '$RWT' && python3 -c \"import torch,triton;print('torch',torch.__version__);print('triton',triton.__version__);print('dev',torch.cuda.get_device_name(0));print('cap',torch.cuda.get_device_capability(0))\""

echo
echo "### DONE"
