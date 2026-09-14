#!/usr/bin/env bash
# PHASE 10F -- final batched verification.
#   A. encoder wiring gate at CURRENT bytes, threshold CORRECTED (relative, with
#      an inline complex128 arm built from the module's own buffers)
#   B. threefold systemic grounding vs LIVE modules, including the failure mode
#      the document itself names (matter-from-nothing, teleport)
#
# Interpreter on 50797414 is /usr/bin/python3 (torch 2.12.0+cu130, cc 12.0).
# /venv/main/bin/python3 does NOT exist here -- that is a stale note for a
# different instance and cost one wasted run.
set -uo pipefail

LOCAL="/c/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
RWT="/workspace/phase10/HENRI V2"
SSH="ssh -o BatchMode=yes -p 37414 root@ssh7.vast.ai"

FILES=(
  "o_vsa_torus_encoder.py"
  "arc_sagnac_veto.py"
  "adaptive_viscoelastic_thermostat.py"
  "experiments/verification/arc_torus_encoder_wiring.py"
  "experiments/verification/threefold_grounding_probe.py"
)

echo "### 1. TRANSFER + SHA-256 (full closure)"
FAIL=0
for f in "${FILES[@]}"; do
  if [ ! -f "$LOCAL/$f" ]; then echo "  MISSING_LOCAL $f"; FAIL=1; continue; fi
  cat "$LOCAL/$f" | timeout 240 $SSH "cat > \"$RWT/$f\"" 2>/dev/null \
    || { echo "  XFER_FAIL $f"; FAIL=1; continue; }
  L=$(sha256sum "$LOCAL/$f" | awk '{print $1}')
  R=$(timeout 120 $SSH "sha256sum \"$RWT/$f\"" 2>/dev/null | awk '{print $1}')
  [ "$L" = "$R" ] && echo "  OK $f ${L:0:16}" || { echo "  SHA_MISMATCH $f"; FAIL=1; }
done
echo "  closure_fail=$FAIL"

echo
echo "### 2. ENCODER GATE @ CURRENT BYTES (corrected threshold + c128 arm)"
timeout 480 $SSH "cd \"$RWT\" && python3 -u experiments/verification/arc_torus_encoder_wiring.py > /root/w6.log 2>&1; echo REAL_RC=\$?; sed -n '/2b\./,\$p' /root/w6.log" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun"

echo
echo "### 3. THREEFOLD GROUNDING vs LIVE MODULES (doc's own failure modes)"
timeout 540 $SSH "cd \"$RWT\" && python3 -u experiments/verification/threefold_grounding_probe.py > /root/t6.log 2>&1; echo REAL_RC=\$?; cat /root/t6.log" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun" | tail -130

echo
echo "### DONE"
