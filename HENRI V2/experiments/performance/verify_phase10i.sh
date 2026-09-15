#!/usr/bin/env bash
# PHASE 10I -- functor regularized-diagonal-LS + 60-task gap-closure measurement.
#
# ONE batched remote session (instance 50797414 is billing at $1.43/hr).
# Interpreter is /usr/bin/python3 -- NOT /venv/main/bin/python3 (that path is a
# stale note for a different box and cost one whole run).
set -uo pipefail

LOCAL="/c/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
RWT="/workspace/phase10/HENRI V2"
RVF="experiments/verification"
SSH="ssh -o BatchMode=yes -p 37414 root@ssh7.vast.ai"

# Files that changed this pass + the harness + the new runbook.
FILES=(
  "arc_task_functor.py"
  "adaptive_viscoelastic_thermostat.py"
  "experiments/verification/arc_functor_ls_gap_closure.py"
  "experiments/performance/verify_phase10i.sh"
)

echo "### 1. TRANSFER + SHA-256 RECONCILE"
FAILDEP=0
for f in "${FILES[@]}"; do
  cat "$LOCAL/$f" | timeout 240 $SSH "cat > \"$RWT/$f\"" 2>/dev/null
  L=$(sha256sum "$LOCAL/$f" | awk '{print $1}')
  R=$(timeout 120 $SSH "sha256sum \"$RWT/$f\" 2>/dev/null" 2>/dev/null | awk '{print $1}')
  if [ "$L" = "$R" ]; then
    echo "  OK   $f  ${L:0:16}"
  else
    echo "  FAIL $f  L=${L:0:16} R=${R:0:16}"; FAILDEP=$((FAILDEP+1))
  fi
done
echo "  closure_fail=$FAILDEP"
if [ "$FAILDEP" -ne 0 ]; then echo "ABORT: transfer incomplete"; exit 3; fi

echo
echo "### 2. LIVE CONSTANTS AT THE TRANSFERRED BYTES"
timeout 180 $SSH "cd \"$RWT\" && grep -n 'FUSED_TILE_SIZE\s*=\|BLOCK_SPAN_DEFAULT\s*=' basal_triton_kernel.py; echo '--- stiefel_iters default ---'; grep -n 'stiefel_iters: int' adaptive_viscoelastic_thermostat.py; echo '--- HENRI_FUNCTOR_FIT wired ---'; grep -n 'HENRI_FUNCTOR_FIT\|def compute_optimal_task_functor' arc_task_functor.py; echo '--- ARC corpus ---'; ls /workspace/arcdata/ARC-AGI/data/training/*.json 2>/dev/null | wc -l" 2>/dev/null | tail -20

echo
echo "### 3. GAP-CLOSURE RUN (60 real ARC-AGI-1 tasks)"
timeout 1500 $SSH "cd \"$RWT\" && ARC_CORPUS=/workspace/arcdata/ARC-AGI/data ARC_N_TASKS=60 python3 -u $RVF/arc_functor_ls_gap_closure.py > /root/functor_ls.log 2>&1; echo REAL_RC=\$?; tail -70 /root/functor_ls.log" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun" | tail -85

echo
echo "### 4. PULL RECEIPT"
timeout 180 $SSH "cat \"$RWT/$RVF/arc_functor_ls_gap_closure_observed.json\"" 2>/dev/null > "$LOCAL/$RVF/arc_functor_ls_gap_closure_observed.json"
S=$(wc -c < "$LOCAL/$RVF/arc_functor_ls_gap_closure_observed.json" 2>/dev/null || echo 0)
[ "$S" -gt 200 ] && echo "  PULLED ($S bytes)" || echo "  PULL FAILED ($S bytes)"

echo
echo "### DONE"
