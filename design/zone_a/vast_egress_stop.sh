#!/usr/bin/env bash
# EGRESS the real-model receipt, VERIFY it, then STOP the instance.
# Budget rule (skill §7): pull back and verify before stopping. A failed gate
# blocks the stop. Stop is reversible; destroy is not and needs human approval.
set -u
export PATH="$HOME/.local/bin:$PATH"

ID="54185275"
HOST="ssh1.vast.ai"
PORT="38550"
REPO="$HOME/henri-worktrees/phase1-transduction"
DEST="$REPO/design/zone_a/evidence/vast_54185275"
mkdir -p "$DEST"

echo "=== 1. remote sha of receipt ==="
RSH=$(timeout 60 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$PORT" root@"$HOST" \
      'sha256sum /root/henri/real_model_receipt.json | cut -d" " -f1; stat -c%s /root/henri/real_model_receipt.json' 2>/dev/null)
echo "$RSH"
REMOTE_SHA=$(echo "$RSH" | sed -n 1p)
REMOTE_BYTES=$(echo "$RSH" | sed -n 2p)

echo "=== 2. scp pull ==="
timeout 120 scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P "$PORT" \
  root@"$HOST":/root/henri/real_model_receipt.json "$DEST/" 2>&1 | tail -2
LOCAL_SHA=$(sha256sum "$DEST/real_model_receipt.json" | cut -d' ' -f1)
LOCAL_BYTES=$(stat -c%s "$DEST/real_model_receipt.json")
echo "remote sha=$REMOTE_SHA bytes=$REMOTE_BYTES"
echo "local  sha=$LOCAL_SHA bytes=$LOCAL_BYTES"

if [ "$REMOTE_SHA" = "$LOCAL_SHA" ] && [ -n "$REMOTE_SHA" ]; then
  echo "EGRESS_VERIFIED"
else
  echo "EGRESS_FAILED -- refusing to stop"
  exit 1
fi

echo "=== 3. record burn, then STOP ==="
vastai show instance "$ID" --raw 2>/dev/null | python -c "
import sys,json
d=json.load(sys.stdin)
print('  uptime_dph', d.get('dph_total'), 'status', d.get('actual_status'))
print('  gpu', d.get('gpu_name'), 'disk_gb', d.get('disk_space'))
" 2>/dev/null

vastai stop instance "$ID" 2>&1 | tail -3
sleep 6
echo "=== 4. verify stopped ==="
vastai show instance "$ID" --raw 2>/dev/null | python -c "
import sys,json
d=json.load(sys.stdin)
print('  actual_status', d.get('actual_status'), '| intended', d.get('intended_status'))
print('  NOTE: stopped still bills storage at ~\$0.2220/GB/month')
" 2>/dev/null
echo "DONE"
