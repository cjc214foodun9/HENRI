#!/usr/bin/env bash
# BOUNDED readiness probe. Hard-capped so it can never hit the tool timeout.
set -u
export PATH="$HOME/.local/bin:$PATH"
ID="${1:-54185275}"
DEADLINE=$(( $(date +%s) + ${2:-100} ))

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  RAW=$(vastai show instance "$ID" --raw 2>/dev/null)
  read -r ST MSG HOST PORT <<<"$(printf '%s' "$RAW" | python -c "
import sys,json
try: d=json.load(sys.stdin)
except Exception: print('PARSE ERR ERR ERR'); raise SystemExit
print(d.get('actual_status','') or '-', (str(d.get('status_msg')) or '-')[:44],
      d.get('ssh_host','') or '-', d.get('ssh_port','') or '-')
" 2>/dev/null)"
  echo "$(date +%H:%M:%S) status=$ST msg=$MSG host=$HOST port=$PORT"
  if [ "$ST" = "running" ] && [ "$HOST" != "-" ] && [ "$HOST" != "ERR" ]; then
    OUT=$(timeout 15 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
             -o ConnectTimeout=8 -p "$PORT" root@"$HOST" \
             'echo SSH_OK; hostname; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader' 2>&1)
    if printf '%s' "$OUT" | grep -q SSH_OK; then
      echo "GATE=PASS"; echo "$OUT"; exit 0
    fi
    echo "  ssh: $(printf '%s' "$OUT" | head -2 | tr '\n' ' ')"
  fi
  sleep 10
done
echo "GATE=TIMEOUT"
exit 1
