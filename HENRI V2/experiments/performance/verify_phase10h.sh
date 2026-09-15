#!/usr/bin/env bash
# PHASE 10H -- Stiefel reachability: LATENT vs REACHABLE.
# The commit message claimed "callers feeding an unnormalized matrix get NaN".
# That asserted reachability without tracing caller inputs. This run traces it.
set -uo pipefail

LOCAL="/c/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
RWT="/workspace/phase10/HENRI V2"
RVT="$RWT/experiments/verification"
SSH="ssh -o BatchMode=yes -p 37414 root@ssh7.vast.ai"

echo "### 1. TRANSFER + SHA-256"
f="experiments/verification/stiefel_reachability_probe.py"
cat "$LOCAL/$f" | timeout 240 $SSH "cat > \"$RWT/$f\"" 2>/dev/null
L=$(sha256sum "$LOCAL/$f" | awk '{print $1}')
R=$(timeout 120 $SSH "sha256sum \"$RWT/$f\"" 2>/dev/null | awk '{print $1}')
[ "$L" = "$R" ] && echo "  OK $f ${L:0:16}" || echo "  SHA_MISMATCH $f"

echo
echo "### 2. STIEFEL REACHABILITY PROBE"
timeout 480 $SSH "cd \"$RWT\" && python3 -u experiments/verification/stiefel_reachability_probe.py > /root/stiefel.log 2>&1; echo REAL_RC=\$?; cat /root/stiefel.log" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun" | tail -80

echo
echo "### 3. PULL RECEIPT"
timeout 180 $SSH "cat \"$RVT/stiefel_reachability_observed.json\"" 2>/dev/null > "$LOCAL/experiments/verification/stiefel_reachability_observed.json"
S=$(wc -c < "$LOCAL/experiments/verification/stiefel_reachability_observed.json")
[ "$S" -gt 100 ] && echo "  PULLED ($S bytes)" || echo "  PULL FAILED ($S bytes)"

echo
echo "### DONE"
