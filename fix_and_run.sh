#!/bin/bash
set -e

# Default to inherited HF_TOKEN, or fallback if not set
export HF_TOKEN="${HF_TOKEN}"

echo "[*] Installing torch..."
/venv/main/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

echo "[*] Downloading weights..."
/venv/main/bin/python -c '
import os, sys
from huggingface_hub import hf_hub_download
token = os.environ.get("HF_TOKEN")
try:
    hf_hub_download(repo_id="Chandler/HENRI8.6Bswarm", filename="henri_fresh_core.pt", local_dir="/root/HENRI/HENRI_CORE_V1/", token=token)
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
'

export DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/henri
cd /root/HENRI

echo "[*] Running unify_system_integrity.py..."
/venv/main/bin/python /root/HENRI/HENRI_CORE_V1/unify_system_integrity.py || echo "[!] System integrity had warnings"

echo "[*] Seeding axioms..."
/venv/main/bin/python /root/HENRI/HENRI_CORE_V1/seed_universal_axioms.py || echo "[!] Seeding axioms had warnings"

echo "[*] Running benchmark..."
/venv/main/bin/python /root/HENRI/HENRI_CORE_V1/arc_live_benchmark.py > benchmark.log 2>&1 || echo "[!] Benchmark exited with non-zero status"
cat benchmark.log

if [ -n "$HF_WRITE_TOKEN" ]; then
    echo "[*] Uploading weights using HuggingFace Write Token..."
    /venv/main/bin/python upload_weights_to_hf.py --repo "Chandler/HENRI8.6Bswarm" --token "$HF_WRITE_TOKEN"
else
    echo "[!] HF_WRITE_TOKEN is not set. Skipping weight upload."
fi
