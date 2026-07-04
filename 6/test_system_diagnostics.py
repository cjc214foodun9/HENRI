import os
import sys
import torch
import pytest

# Insert current file's parent folders to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l3_router_model import L3SwarmRouter

def test_system_graph_profiling():
    print("\n[DIAGNOSTICS] Starting PyTorch Profiling on L3SwarmRouter...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the core router layer
    router = L3SwarmRouter(vocab_size=1000, hidden_dim=256, num_layers=2, num_experts=8, hopfield_dim=4096)
    router.to(device)
    router.eval()
    
    # Prepare dummy inputs (batch=2, seq_len=32)
    tokens = torch.randint(0, 1000, (2, 32), device=device)
    
    # Warm up the model to avoid profiling cold-start/import latency overhead
    with torch.no_grad():
        for _ in range(3):
            _ = router(tokens=tokens)
            
    # Setup trace output path
    archive_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", "diagnostics")
    os.makedirs(archive_dir, exist_ok=True)
    trace_path = os.path.join(archive_dir, "model_trace.pt.trace.json")
    
    # Instrument profiling execution block
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.profiler.record_function("l3_router_forward"):
            with torch.no_grad():
                for _ in range(5):
                    _ = router(tokens=tokens)
                
    # Save trace to disk
    prof.export_chrome_trace(trace_path)
    print(f"[DIAGNOSTICS] Successfully exported trace to: {trace_path}")
    
    # Analyze latency metrics programmatically
    key_averages = prof.key_averages()
    total_cpu_time = sum(item.cpu_time_total for item in key_averages)
    # Average CPU time per call
    avg_cpu_time_us = total_cpu_time / 5.0
    print(f"[DIAGNOSTICS] Total CPU execution time for 5 steps: {total_cpu_time:.2f}us")
    print(f"[DIAGNOSTICS] Average CPU execution time per step: {avg_cpu_time_us:.2f}us")
    
    # Assert latency regression threshold per step (e.g. max 200ms on CPU, faster on GPU)
    max_allowable_latency_us = 200000.0
    assert avg_cpu_time_us < max_allowable_latency_us, f"Latency threshold exceeded: {avg_cpu_time_us}us"
    
    # Verify file was generated and contains trace metadata
    assert os.path.exists(trace_path), "Failed to generate trace file!"
    assert os.path.getsize(trace_path) > 0, "Trace file is empty!"
    print("[DIAGNOSTICS] System execution diagnostics trace PASSED.")
