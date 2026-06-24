import torch
import os

checkpoint_path = "henri_core_final.pt"
if not os.path.exists(checkpoint_path):
    checkpoint_path = "/workspace/HENRI/henri_core_final.pt"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        print("Keys:", list(checkpoint.keys()))
        for k, v in checkpoint.items():
            if isinstance(v, dict):
                print(f"  {k} keys (sample):", list(v.keys())[:10])
            else:
                print(f"  {k}: {type(v)}")
    else:
        print("Loaded object type:", type(checkpoint))
else:
    print("Checkpoint not found at", checkpoint_path)
