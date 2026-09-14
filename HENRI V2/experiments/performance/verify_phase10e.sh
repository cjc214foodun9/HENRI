#!/usr/bin/env bash
# PHASE 10E: transfer the FULL dependency closure, re-measure the encoder at its
# CURRENT bytes, resolve the numerical-floor question, and test the threefold
# grounding against live modules.
#
# Defects this run repairs (all mine):
#   * 10b/10c/10d never re-transferred o_vsa_torus_encoder.py after the DC-slot
#     patches, so the wiring gate was measuring the OLD bytes.
#   * arc_sagnac_veto.py / adaptive_viscoelastic_thermostat.py were absent from
#     the remote tree -> ModuleNotFoundError.
#   * the @guard decorator returned the raw function, so nothing was isolated.
set -uo pipefail

LOCAL="/c/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
RWT="/workspace/phase10/HENRI V2"
SSH="ssh -o BatchMode=yes -p 37414 root@ssh7.vast.ai"

# full closure: modules imported by the probes + the probes themselves
FILES=(
  "o_vsa_torus_encoder.py"
  "arc_sagnac_veto.py"
  "adaptive_viscoelastic_thermostat.py"
  "arc_thermostat_shadow.py"
  "experiments/verification/arc_torus_encoder_wiring.py"
  "experiments/verification/arc_torus_exactness_floor.py"
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
echo "### 2. ENCODER AT CURRENT BYTES (wiring gate + kill gate)"
timeout 420 $SSH "cd \"$RWT\" && python3 -u experiments/verification/arc_torus_encoder_wiring.py > /root/w5.log 2>&1; echo REAL_RC=\$?; grep -E 'uniform|identity cos|EXACT|negative|structured=|held_out_mean|identity_mean|ceiling_mean|CARRIER_PROMOTABLE|kill_gate_PASS' /root/w5.log" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun"

echo
echo "### 3. EXACT-ROLL: STRUCTURAL vs float32 ACCUMULATION FLOOR"
timeout 420 $SSH "cd \"$RWT\" && python3 -u experiments/verification/arc_torus_exactness_floor.py > /root/f5.log 2>&1; echo REAL_RC=\$?; cat /root/f5.log" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun" | tail -32

echo
echo "### 4. TENSORED-CARRIER KILL -- carrier-A fit DE-CONTAMINATED (10D log)"
timeout 120 $SSH "sed -n '/B\/C. WAVE-SPACE/,/^  *$/p' /root/kill4.log 2>/dev/null | head -25" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun"

echo
echo "### 5. THREEFOLD GROUNDING vs LIVE MODULES"
timeout 480 $SSH "cd \"$RWT\" && python3 -u experiments/verification/threefold_grounding_probe.py > /root/t5.log 2>&1; echo REAL_RC=\$?; cat /root/t5.log" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun" | tail -95

echo
echo "### DONE"
