import subprocess
import sys
import os

print("[*] Installing torch from high-speed PyTorch CDN...")
subprocess.check_call(["/venv/main/bin/python", "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121", "--no-cache-dir"])

print("[*] Downloading weights from HuggingFace...")
from huggingface_hub import hf_hub_download
token = os.environ.get("HF_TOKEN", "")
hf_hub_download(repo_id="Chandler/HENRI8.6Bswarm", filename="henri_fresh_core.pt", local_dir="/root/HENRI/HENRI_CORE_V1/", token=token)
print("[SUCCESS] Weights downloaded.")

os.environ["DATABASE_URL"] = "postgresql://postgres:password@127.0.0.1:5432/henri"
os.chdir("/root/HENRI")

print("[*] Running unify_system_integrity.py...")
subprocess.check_call(["/venv/main/bin/python", "/root/HENRI/HENRI_CORE_V1/unify_system_integrity.py"])

print("[*] Seeding axioms...")
subprocess.check_call(["/venv/main/bin/python", "/root/HENRI/HENRI_CORE_V1/seed_universal_axioms.py"])

print("[*] Running benchmark...")
subprocess.check_call(["/venv/main/bin/python", "/root/HENRI/HENRI_CORE_V1/arc_live_benchmark.py"])
