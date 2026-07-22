#!/usr/bin/env bash
# Mutagen sync launcher for HENRI V2 → Vast.ai
# Usage: ./scripts/mutagen_setup.sh <INSTANCE_ID> [--detach]
#
# Prerequisites:
#   1. mutagen installed: https://mutagen.io/download
#   2. VASTAI_API_KEY set in environment
#   3. Docker image pushed to ghcr.io/cjc214foodun9/henri-v2-execution:latest

set -e

INSTANCE_ID="${1:?Usage: $0 <instance_id> [--detach]}"
DETACH="${2:-}"

# Extract SSH credentials from Vast
INFO=$(vastai show instance "$INSTANCE_ID" --raw | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')
")
SSH_HOST=$(echo "$INFO" | awk '{print $1}')
SSH_PORT=$(echo "$INFO" | awk '{print $2}')
REMOTE="root@${SSH_HOST}:${SSH_PORT}/workspace/HENRI_V2"
LOCAL="$(dirname "$(cd "$(dirname "$0")" && pwd)")/HENRI V2"

echo "=== HENRI V2 Mutagen Sync ==="
echo "  Instance:  $INSTANCE_ID"
echo "  Remote:    $REMOTE"
echo "  Local:     $LOCAL"
echo ""

# Terminate any existing session with same name
mutagen sync terminate henri-vast 2>/dev/null || true

# Create new sync session
mutagen sync create \
  --name=henri-vast \
  "$LOCAL" "$REMOTE" \
  --ignore ".git, __pycache__, *.pyc, telemetry_logs, .venv, outputs, *.nsys-rep, *.sqlite"

echo ""
echo "Sync active. Changes in $LOCAL propagate to Vast in <100ms."
echo "Monitor:  mutagen sync monitor henri-vast"
echo "Stop:     mutagen sync terminate henri-vast"

if [ "$DETACH" = "--detach" ]; then
  echo "Detached mode — sync runs in background."
else
  echo ""
  echo "Showing live monitor (Ctrl+C to stop monitoring, sync stays active)..."
  mutagen sync monitor henri-vast
fi
