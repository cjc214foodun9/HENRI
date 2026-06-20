import torch
import sys
import time

sys.path.append('/workspace/HENRI/6')
from henri_core.core import ProprietaryHENRICore

print("Initializing ProprietaryHENRICore...", flush=True)
start_init = time.time()
core = ProprietaryHENRICore(dim=4096, depth=32, num_fluid_states=16)
print(f"ProprietaryHENRICore initialized in {time.time() - start_init:.4f} seconds", flush=True)

print("Moving model to GPU and casting to bfloat16...", flush=True)
start_move = time.time()
core = core.to(device='cuda', dtype=torch.bfloat16)
print(f"Model moved and cast in {time.time() - start_move:.4f} seconds", flush=True)

x = torch.randn(1, 512, 4096, dtype=torch.bfloat16, device='cuda')
y = torch.randn(1, 4096, dtype=torch.bfloat16, device='cuda')

print("Starting forward pass...", flush=True)
start = time.time()
with torch.no_grad():
    out, _ = core(x, y, 0.5)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Forward pass completed in {elapsed:.4f} seconds", flush=True)
