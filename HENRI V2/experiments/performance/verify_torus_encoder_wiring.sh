#!/usr/bin/env bash
# Remote verification of the TORUS ENCODER REFORM wiring (default-OFF differential).
set -uo pipefail
LOCAL="/c/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
RWT="/workspace/phase10/HENRI V2"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p 37414 root@ssh7.vast.ai"

FILES=(
  "o_vsa_torus_encoder.py"
  "o_vsa_ingress_tokenizer.py"
  "experiments/verification/arc_torus_encoder_wiring.py"
)

echo "### STEP 1: local SHA-256"
for f in "${FILES[@]}"; do
  printf "%s  %s\n" "$(sha256sum "$LOCAL/$f" | cut -d' ' -f1)" "$f"
done

echo
echo "### STEP 2: transfer"
for f in "${FILES[@]}"; do
  d="$RWT/$(dirname "$f")"
  $SSH "mkdir -p '$d'" >/dev/null 2>&1 || exit 2
  cat "$LOCAL/$f" | $SSH "cat > '$RWT/$f'" || exit 2
  echo "sent $f"
done

echo
echo "### STEP 3: remote SHA-256 (MUST equal STEP 1)"
MM=0
for f in "${FILES[@]}"; do
  L=$(sha256sum "$LOCAL/$f" | cut -d' ' -f1)
  R=$($SSH "sha256sum '$RWT/$f' | cut -d' ' -f1")
  if [ "$L" = "$R" ]; then printf "OK   %s\n" "$f"; else printf "FAIL %s\n  local=%s\n  remote=%s\n" "$f" "$L" "$R"; MM=1; fi
done
[ "$MM" -eq 0 ] || { echo "### BLOCKED: SHA mismatch"; exit 3; }

echo
echo "### STEP 4: import preflight"
$SSH "cd '$RWT' && PYTHONPATH='HENRI V2' python3 -c \"
import o_vsa_torus_encoder as t, o_vsa_ingress_tokenizer as tok
print('IMPORT_OK default_enabled=', t.is_enabled(), 'modulus=', t.DEFAULT_MODULUS)
\" 2>&1 | tail -8"

echo
echo "### STEP 5: wiring verification at the exact transferred bytes"
$SSH "cd '$RWT' && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='HENRI V2' ARC_CORPUS=/workspace/arcdata/ARC-AGI/data ARC_N_TASKS=60 python3 -u experiments/verification/arc_torus_encoder_wiring.py > /root/torus.log 2>&1; echo REAL_RC=\$?; echo '=== LOG ==='; cat /root/torus.log"

echo
echo "### DONE"
