import os
import argparse
from huggingface_hub import HfApi

def upload_weights(repo_id: str, token: str):
    print("=== Launching HuggingFace Weight Tunnel ===")
    
    file_path = "HENRI_CORE_V1/henri_fresh_core.pt"
    
    if not os.path.exists(file_path):
        print(f"[!] Error: Weights file not found at {file_path}")
        return
        
    file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
    print(f"[*] Found {file_path} ({file_size_gb:.2f} GB)")
    print(f"[*] Target Repository: {repo_id}")
    
    api = HfApi(token=token)
    
    try:
        # Check if repo exists, if not, create it (private by default to protect proprietary weights)
        print("[*] Validating remote repository...")
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
        
        print("[*] Initiating high-speed multi-part upload to HuggingFace...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo="henri_fresh_core.pt",
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"[SUCCESS] 7GB weights successfully pushed to {repo_id}/henri_fresh_core.pt!")
        print("[*] Future deployments can now download this directly at ~1-5 Gbps!")
    except Exception as e:
        print(f"[!] Critical Error during upload: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload HENRI weights to HuggingFace.")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace Repo ID (e.g., 'username/henri-core')")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace Access Token (Write Permission)")
    
    args = parser.parse_args()
    upload_weights(args.repo, args.token)
