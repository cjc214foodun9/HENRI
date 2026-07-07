#!/bin/bash
export HF_TOKEN='${HF_TOKEN}'
/venv/main/bin/python -c '
import os, sys
from huggingface_hub import hf_hub_download
token = os.environ.get("HF_TOKEN")
print("[*] Downloading weights from HuggingFace...")
try:
    hf_hub_download(repo_id="Chandler/HENRI8.6Bswarm", filename="henri_fresh_core.pt", local_dir="/root/HENRI/HENRI_CORE_V1/", token=token)
    print("[SUCCESS] Weights downloaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to download weights: {e}")
    sys.exit(1)
'
export DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/henri
cd /root/HENRI
/venv/main/bin/python /root/HENRI/HENRI_CORE_V1/unify_system_integrity.py
/venv/main/bin/python /root/HENRI/HENRI_CORE_V1/seed_universal_axioms.py
/venv/main/bin/python /root/HENRI/HENRI_CORE_V1/arc_live_benchmark.py
