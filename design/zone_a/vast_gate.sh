#!/usr/bin/env bash
# Wait for instance readiness, then run the SSH gate. Nothing else counts.
set -u
export PATH="$HOME/.local/bin:$PATH"
ID="$1"
MAX="${2:-40}"          # polls of 15s => default ~10 min

for i in $(seq 1 "$MAX"); do
  RAW=$(vastai show instance "$ID" --raw 2>/dev/null)
  ST=$(printf '%s' "$RAW" | python -c "import sys,json;print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null)
  MSG=$(printf '%s' "$RAW" | python -c "import sys,json;print(str(json.load(sys.stdin).get('status_msg'))[:60])" 2>/dev/null)
  echo "[$i] actual_status=$ST status_msg=$MSG"
  if [ "$ST" = "running" ]; then break; fi
  sleep 15
done

HOST=$(printf '%s' "$RAW" | python -c "import sys,json;print(json.load(sys.stdin).get('ssh_host',''))" 2>/dev/null)
PORT=$(printf '%s' "$RAW" | python -c "import sys,json;print(json.load(sys.stdin).get('ssh_port',''))" 2>/dev/null)
GPU=$(printf '%s' "$RAW" | python -c "import sys,json;print(json.load(sys.stdin).get('gpu_name',''))" 2>/dev/null)
echo "ENDPOINT host=$HOST port=$PORT gpu=$GPU"

if [ -z "$HOST" ] || [ -z "$PORT" ]; then echo "NO_ENDPOINT"; exit 2; fi

echo "=== SSH GATE ==="
for attempt in 1 2 3 4 5 6; do
  OUT=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
           -o ConnectTimeout=12 -p "$PORT" root@"$HOST" \
           'echo SSH_OK; hostname; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader' 2>&1)
  echo "--- attempt $attempt ---"; echo "$OUT"
  if printf '%s' "$OUT" | grep -q SSH_OK; then echo "GATE=PASS"; exit 0; fi
  sleep 12
done
echo "GATE=FAIL"
exit 1
