#!/bin/bash
set -e

# Default to inherited HF_TOKEN, or fallback if not set
export HF_TOKEN="${HF_TOKEN}"

echo "[*] Installing torch..."
/venv/main/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

echo "[*] Downloading weights via wget..."
wget -c --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 --header="Authorization: Bearer ${HF_TOKEN}" -O /root/HENRI/HENRI_CORE_V1/henri_fresh_core.pt "https://huggingface.co/Chandler/HENRI8.6Bswarm/resolve/main/henri_fresh_core.pt"

export DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/henri
cd /root/HENRI

echo "[*] Running unify_system_integrity.py..."
/venv/main/bin/python /root/HENRI/HENRI_CORE_V1/unify_system_integrity.py || echo "[!] System integrity had warnings"

echo "[*] Installing required datasets package..."
/venv/main/bin/python -m pip install datasets --no-cache-dir

echo "[*] Seeding axioms..."
/venv/main/bin/python /root/HENRI/HENRI_CORE_V1/seed_universal_axioms.py || echo "[!] Seeding axioms had warnings"

echo "[*] Running benchmark..."
/venv/main/bin/python /root/HENRI/HENRI_CORE_V1/arc_agi_3_thermodynamic_agent.py > benchmark.log 2>&1 || echo "[!] Benchmark exited with non-zero status"
cat benchmark.log

if [ -n "$HF_WRITE_TOKEN" ]; then
    echo "[*] Uploading weights using HuggingFace Write Token..."
    /venv/main/bin/python upload_weights_to_hf.py --repo "Chandler/HENRI8.6Bswarm" --token "$HF_WRITE_TOKEN"
else
    echo "[!] HF_WRITE_TOKEN is not set. Skipping weight upload."
fi
