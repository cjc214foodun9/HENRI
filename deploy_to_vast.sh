#!/bin/bash
set -e

# Load environment variables if present
if [ -f ".env" ]; then
    source .env
fi

# Use the provided tokens as fallbacks
HF_READ_TOKEN="${HF_READ_TOKEN:-HFAKUz2BkSyewRWxJX4KZaLy9rs5J5w}"
HF_WRITE_TOKEN="${HF_WRITE_TOKEN:-HFAKhwhMGyeoa5DWtfMyH7Qvp4FRMR6}"

echo "=== Deploying HENRI to Vast.ai ==="
echo "[*] Syncing files to remote host (/root/HENRI)..."
# Ensure the directory exists
ssh -p 20279 -o StrictHostKeyChecking=no root@ssh8.vast.ai "mkdir -p /root/HENRI"

# Rsync the codebase
rsync -avz --exclude '.git' --exclude '.github' --exclude 'venv' -e "ssh -p 20279 -o StrictHostKeyChecking=no" ./ root@ssh8.vast.ai:/root/HENRI/

echo "[*] Starting execution pipeline with port forwarding (-L 8080:localhost:8080)..."
ssh -p 20279 -o StrictHostKeyChecking=no -L 8080:localhost:8080 root@ssh8.vast.ai \
  "cd /root/HENRI && chmod +x ./fix_and_run.sh && export HF_TOKEN=${HF_READ_TOKEN} HF_WRITE_TOKEN=${HF_WRITE_TOKEN} && ./fix_and_run.sh"

echo "=== Deployment and execution completed ==="
