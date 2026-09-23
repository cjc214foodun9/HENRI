#!/usr/bin/env bash
# SSH GATE for instance 52161444. Success requires SSH_OK + a plausible GPU line.
set -uo pipefail
HOST="ssh9.vast.ai"; PORT="11444"
KEY="$HOME/.ssh/id_ed25519"
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o IdentitiesOnly=yes"
[ -f "$KEY" ] && OPTS="$OPTS -i $KEY"

echo "=== key present: $([ -f "$KEY" ] && echo yes || echo no) ==="
echo "=== GATE 1: SSH identity + GPU ==="
ssh $OPTS -p "$PORT" root@"$HOST" \
  'echo SSH_OK; hostname; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; echo ---; python3 -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())" 2>&1 | tail -2' 2>&1 | tail -12
echo "  rc=$?"
