"""
HENRI V2: Authentic Hardware-Bound Coding Benchmark Runner
Subsystem: Falsifiable Hardware Evaluation Engine

Executes true hardware-bound evaluation on OpenAI HumanEval benchmark items:
  1. Loads raw HumanEval JSONL dataset items from GitHub master branch.
  2. Executes PyTorch/CUDA forward passes over D=65,536 wave hypervectors.
  3. Executes generated code via live isolated exec() and unit test assertions in ExteroceptiveSandboxTransducer.
  4. Measures true PyTorch GPU memory allocation, wall-clock time, and item latency.
  5. Latency is DIAGNOSTIC telemetry only; run validity is set by fidelity
     conditions (checkpoint policy, coverage, provenance, honest eligibility).
"""

import os
import sys
import json
import time
import gzip
import urllib.request
from datetime import datetime, timezone
import torch

repo_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(repo_path)
for p in [repo_path, parent_path, os.path.join(parent_path, "scripts")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from henri_decoder import HENRIUnifiedEgressTransducer
from exteroceptive_sandbox import ExteroceptiveSandboxTransducer
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def load_humaneval_dataset(limit: int = 50) -> list:
    cache_path = os.path.join(repo_path, "data", "HumanEval.jsonl.gz")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if not os.path.exists(cache_path):
        print(f"[BENCHMARK] Downloading official HumanEval dataset from: {HUMANEVAL_URL}")
        urllib.request.urlretrieve(HUMANEVAL_URL, cache_path)

    items = []
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
            if limit and len(items) >= limit:
                break
    return items


def run_authentic_hardware_benchmark(limit: int = 50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("========================================================================")
    print("  HENRI V2: AUTHENTIC HARDWARE-BOUND BENCHMARK RUNNER")
    print("========================================================================")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"Execution Device  : {device.upper()}")
    if device == "cuda":
        print(f"GPU Model Name    : {torch.cuda.get_device_name(0)}")
        print(f"Initial Alloc VRAM: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        torch.cuda.reset_peak_memory_stats(0)

    # Accuracy-first fidelity contract (Class 4 synthesis, 2026-08-20):
    # latency is DIAGNOSTIC telemetry, never a validity gate. Run validity
    # is determined by fidelity conditions: checkpoint policy, candidate
    # coverage, dataset provenance, and honest eligibility telemetry.
    from accuracy_profile import (
        FIDELITY_SCORE_BEARING,
        is_score_promotable,
    )
    execution_profile = FIDELITY_SCORE_BEARING
    score_promotable = is_score_promotable(execution_profile)

    # Initialize PyTorch Neural Egress Unbinder & Sandbox
    transducer = HENRIUnifiedEgressTransducer(
        d_model=65536,
        device=device,
        checkpoint_policy="required",
    )
    sandbox = ExteroceptiveSandboxTransducer(d_model=65536)
    codec = qFHRREpistemicCodec(d_model=65536, device=device)

    items = load_humaneval_dataset(limit=limit)
    item_count = len(items)
    print(f"\n[DATASET] Loaded {item_count} authentic OpenAI HumanEval items.")

    passed_count = 0
    total_time_start = time.perf_counter()
    per_item_latencies = []

    for idx, item in enumerate(items, 1):
        task_id = item["task_id"]
        prompt = item["prompt"]
        entry_point = item["entry_point"]
        test_code = item["test"]

        t_item_start = time.perf_counter()

        # 1. Real PyTorch Forward Pass over D=65,536 Wave Hypervector
        prompt_wave = codec.encode_text(prompt)
        task_op = codec.encode_text("HUMANEVAL_CODING_OPERATOR")
        goal_wave = codec.bind_hadamard(task_op, prompt_wave)

        # Transduce via Neural Unbinder
        generated_text, telem = transducer.decode_wave_to_response(goal_wave, prompt)

        # 2. Live Exteroceptive Sandbox Execution (exec / pytest evaluation)
        full_executable = f"{prompt}\n{generated_text}\n{test_code}\ncheck({entry_point})"
        is_pass, trace_out = sandbox.execute_and_transduce(full_executable, axiom_id=task_id, source_metadata="HumanEval")

        t_item_elapsed = (time.perf_counter() - t_item_start) * 1000.0  # ms
        per_item_latencies.append(t_item_elapsed)

        if is_pass:
            passed_count += 1
            status_str = "PASS"
        else:
            status_str = "FAIL"

        print(f"[{idx:03d}/{item_count:03d}] Task: {task_id:<20} | Status: {status_str} | Item Latency: {t_item_elapsed:7.2f} ms")

    total_time_sec = time.perf_counter() - total_time_start
    avg_latency_ms = (total_time_sec / max(1, item_count)) * 1000.0

    peak_vram_gb = torch.cuda.max_memory_allocated(0) / 1e9 if device == "cuda" else 0.0
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU Host Environment"

    print("\n========================================================================")
    print("                 HARDWARE TELEMETRY ASSERTION REPORT")
    print("========================================================================")
    print(f"GPU Name                  : {gpu_name}")
    print(f"Peak VRAM Allocated       : {peak_vram_gb:.4f} GB")
    print(f"True Wall-Clock Duration  : {total_time_sec:.4f} seconds")
    print(f"Evaluated Item Count      : {item_count} items")
    print(f"Passed Items              : {passed_count} / {item_count} ({(passed_count / max(1, item_count)) * 100.0:.2f}%)")
    print(f"Average Latency Per Item  : {avg_latency_ms:.4f} ms/item")
    print("------------------------------------------------------------------------")

    # Accuracy-first fidelity contract (Class 4 synthesis, 2026-08-20):
    # latency is DIAGNOSTIC telemetry, never a validity gate. A sub-0.5 ms
    # average is a WARNING signal (possible short-circuit), not a run
    # rejection. Run validity is determined by fidelity conditions:
    # checkpoint policy, candidate coverage, dataset provenance, and honest
    # eligibility telemetry.
    if avg_latency_ms < 0.5:
        print(f"LATENCY WARNING: Average latency ({avg_latency_ms:.4f} ms) < 0.5 ms threshold "
              f"— possible short-circuit; inspect candidates and egress path. "
              f"Latency is diagnostic only; validity is set by execution_profile="
              f"{execution_profile}, score_promotable={score_promotable}.")
    else:
        print(f"LATENCY DIAGNOSTIC: {avg_latency_ms:.4f} ms/item (diagnostic only).")
    print("========================================================================")

    # Save raw log output file
    log_dir = os.path.join(repo_path, "telemetry_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    log_file_path = os.path.join(log_dir, f"authentic_run_{ts_str}.log")
    
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("HENRI V2 AUTHENTIC HARDWARE BENCHMARK LOG\n")
        f.write(f"Timestamp          : {ts_str}\n")
        f.write(f"GPU Name           : {gpu_name}\n")
        f.write(f"Peak VRAM (GB)     : {peak_vram_gb:.4f}\n")
        f.write(f"Total Time (sec)   : {total_time_sec:.4f}\n")
        f.write(f"Item Count         : {item_count}\n")
        f.write(f"Passed Count       : {passed_count}\n")
        f.write(f"Avg Latency (ms)   : {avg_latency_ms:.4f}\n")
    
    print(f"\n[TELEMETRY] Raw log written to: {log_file_path}")


if __name__ == "__main__":
    run_authentic_hardware_benchmark(limit=50)
