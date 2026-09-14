#!/usr/bin/env bash
# PHASE 10G -- re-run the two DEFECTIVE pillars after fixing my own bugs, then
# pull every receipt to local for the record.
#
# Defects fixed (mine, both found by the 10F run):
#   * unit() seeded its generator INSIDE the function -> every "random" draw was
#     the SAME vector -> indep_random cos = 1.0 and a degenerate threshold sweep.
#   * torch.Generator() is a CPU generator; torch.randn(device="cuda",
#     generator=cpu_gen) raises. Needs torch.Generator(device=DEV).
set -uo pipefail

LOCAL="/c/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
RWT="/workspace/phase10/HENRI V2"
RVT="$RWT/experiments/verification"
SSH="ssh -o BatchMode=yes -p 37414 root@ssh7.vast.ai"

echo "### 1. TRANSFER + SHA-256"
f="experiments/verification/threefold_grounding_probe.py"
cat "$LOCAL/$f" | timeout 240 $SSH "cat > \"$RWT/$f\"" 2>/dev/null
L=$(sha256sum "$LOCAL/$f" | awk '{print $1}')
R=$(timeout 120 $SSH "sha256sum \"$RWT/$f\"" 2>/dev/null | awk '{print $1}')
[ "$L" = "$R" ] && echo "  OK $f ${L:0:16}" || echo "  SHA_MISMATCH $f"

echo
echo "### 2. THREEFOLD GROUNDING (all three pillars, defects fixed)"
timeout 540 $SSH "cd \"$RWT\" && python3 -u experiments/verification/threefold_grounding_probe.py > /root/t7.log 2>&1; echo REAL_RC=\$?; cat /root/t7.log" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun" | tail -150

echo
echo "### 3. PULL RECEIPTS TO LOCAL"
mkdir -p "$LOCAL/experiments/verification"
for r in threefold_grounding_observed.json arc_torus_encoder_wiring_observed.json \
         arc_torus_exactness_floor_observed.json arc_tensored_carrier_kill_observed.json; do
  if timeout 180 $SSH "cat \"$RVT/$r\"" 2>/dev/null > "$LOCAL/experiments/verification/$r"; then
    S=$(wc -c < "$LOCAL/experiments/verification/$r")
    if [ "$S" -gt 100 ]; then echo "  PULLED $r ($S bytes)"; else echo "  EMPTY $r"; fi
  else
    echo "  MISSING $r"
  fi
done

echo
echo "### DONE"
