#!/usr/bin/env bash
# REMOTE GROUND-STATE INVENTORY for instance 52161444 (fresh, digest-pinned image).
# Establish what EXISTS before copying 762 MB or assuming a repo layout.
set -uo pipefail
HOST="ssh9.vast.ai"; PORT="11444"; KEY="$HOME/.ssh/id_ed25519"
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o IdentitiesOnly=yes -i $KEY"

ssh $OPTS -p "$PORT" root@"$HOST" 'bash -s' <<'REMOTE'
set -uo pipefail
echo "=== workspace ==="
ls -la /workspace 2>&1 | head -20
echo
echo "=== any HENRI checkout? ==="
find / -maxdepth 4 -type d -name "HENRI*" 2>/dev/null | head -12
echo
echo "=== henri archives / overlays ==="
ls -la /root/henri-archive 2>&1 | head -12
echo "--- manifest ---"
if [ -f /root/henri-archive/manifest.sha256 ]; then head -6 /root/henri-archive/manifest.sha256; fi
echo
echo "=== any decoder checkpoint already present? ==="
find / -maxdepth 6 -name "henri_decoder_checkpoint*.pt" 2>/dev/null | head -6
echo
echo "=== disk ==="
df -h /workspace /root 2>/dev/null | head -6
echo
echo "=== python + key deps (exact interpreter) ==="
for P in /venv/main/bin/python python3 $(command -v python3); do
  [ -x "$P" ] || continue
  echo "--- $P"
  "$P" -c "import sys,torch;print(' set', sys.executable);print(' torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 | tail -3
  "$P" -c "import pytest, numpy; print(' pytest', pytest.__version__, 'numpy', numpy.__version__)" 2>&1 | tail -1
  "$P" -c "import arc_agi; print(' arc_agi', getattr(arc_agi,'__version__','?'))" 2>&1 | tail -1
  break
done
echo
echo "=== git available? ==="
git --version 2>&1 | head -1
echo
echo "=== environment_files present? ==="
ls -la /workspace/HENRI/environment_files 2>/dev/null | head -5 || echo "  (none at /workspace/HENRI/environment_files)"
echo
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,compute_cap --format=csv,noheader
REMOTE
echo "  ssh_rc=$?"
