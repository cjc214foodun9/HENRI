import argparse
import sys
import os
sys.path.insert(0, "/workspace/HENRI/6")

import time
import torch
from henri_core.core import ProprietaryHENRICore

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=32)
    return parser.parse_args()

args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SYSTEM] Targeting device: {device} | Depth: {args.depth}")

# Initialize model directly on GPU in bfloat16
print(f"[SYSTEM] Initializing 485M per-expert core model (depth={args.depth})...")
torch.set_default_dtype(torch.bfloat16)
with torch.device(device):
    model = ProprietaryHENRICore(dim=4096, depth=args.depth, num_fluid_states=16, looped_recurrent=False)
model.to(dtype=torch.bfloat16)
model.eval()

# Dummy input wavefront (Batch=8, seq_len=1, dim=4096)
x = torch.randn(8, 1, 4096, device=device, dtype=torch.complex64)
# Normalize to S^4095 hypersphere
x = torch.complex(torch.cos(x.real), torch.sin(x.real))

attractor = torch.randn(8, 4096, device=device, dtype=torch.bfloat16)

print("[WARMUP] Warming up CUDA kernel caches...")
for _ in range(5):
    with torch.no_grad():
        _, _ = model(x, zone_c_attractor=attractor, temperature=0.5)

torch.cuda.synchronize()
print("[BENCHMARK] Executing 20 iterations...")
start_time = time.perf_counter()

num_iterations = 20
for _ in range(num_iterations):
    with torch.no_grad():
        _, _ = model(x, zone_c_attractor=attractor, temperature=0.5)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start_time

batch_size = x.size(0)
total_tokens = batch_size * num_iterations
tps = total_tokens / elapsed

print("=====================================================================")
print(f"  - Total Elapsed: {elapsed:.4f} seconds")
print(f"  - Throughput:    {tps:.2f} tokens/second (Batch size = {batch_size})")
if args.depth < 32:
    projected_tps = tps * (args.depth / 32.0)
    print(f"  - Projected 32-Layer Throughput: {projected_tps:.2f} tokens/second")
print("=====================================================================")
