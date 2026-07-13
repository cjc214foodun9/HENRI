import subprocess
import sys
import os

print("[*] Installing torch from high-speed PyTorch CDN (Nightly cu124 for RTX 5090 Blackwell Support)...")
subprocess.check_call(["/venv/main/bin/python", "-m", "pip", "install", "--pre", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/nightly/cu124", "--no-cache-dir"])

print("[*] Skipping HF weights download (UnifiedCognitivePipeline initializes natively)")
# from huggingface_hub import hf_hub_download
# token = os.environ.get("HF_TOKEN", "")
# hf_hub_download(repo_id="Chandler/HENRI8.6Bswarm", filename="henri_fresh_core.pt", local_dir="/root/HENRI/HENRI_CORE_V1/", token=token)
# print("[SUCCESS] Weights downloaded.")

os.environ["DATABASE_URL"] = "postgresql://postgres:password@127.0.0.1:5432/henri"
os.chdir("/root/HENRI")

print("[*] Running benchmark (UnifiedCognitivePipeline Natively)...")
subprocess.check_call(["/venv/main/bin/python", "/root/HENRI/HENRI_CORE_V1/arc_live_benchmark.py"])
