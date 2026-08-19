"""Phase 8.31 profiler — isolated CUDA timing of the algebraic action head.

Runs on the Vast RTX 5090 at D=65,536, r=128, synthetic demo pairs.
Profiler-first discipline: measure only, no optimization.
Emits a compact JSON receipt to stdout and a torch.profiler trace file.
"""
import argparse
import json
import time

import torch

from algebraic_action_head import AlgebraicActionHeadCalibrator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=65536)
    ap.add_argument("--r", type=int, default=128)
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--trace", default="/root/p831_profiler_trace.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda:0")
    torch.cuda.empty_cache()
    g = torch.Generator(device="cpu").manual_seed(20260819)
    n = args.pairs

    cal = AlgebraicActionHeadCalibrator(d_model=args.d, r_rank=args.r, seed=0)
    # Calibrator is a plain deterministic class (not nn.Module): no .to();
    # all methods follow the input tensor device.

    def _engrams(count: int, seed: int) -> torch.Tensor:
        gg = torch.Generator(device="cpu").manual_seed(seed)
        e = torch.randn(count, args.d, generator=gg)
        return (e / e.norm(dim=1, keepdim=True)).to(dev)

    x = _engrams(n, 1)
    y = _engrams(n, 2)

    # Warmup
    cal.compile_task_operator(x, y, basis_digest="prof", calibration_digest="prof")
    torch.cuda.synchronize()

    # 1) Compile timing
    t0 = time.perf_counter()
    for _ in range(args.iters):
        w = cal.compile_task_operator(x, y, basis_digest="prof", calibration_digest="prof")
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) / args.iters * 1e3

    # 2) Transduce timing (single wave)
    q = _engrams(1, 3)[0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        out = cal.transduce(q)
    torch.cuda.synchronize()
    transduce_ms = (time.perf_counter() - t0) / args.iters * 1e3

    # 3) Batched transduce (B=64)
    qb = _engrams(64, 4)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        outb = cal.transduce(qb)
    torch.cuda.synchronize()
    batch_ms = (time.perf_counter() - t0) / args.iters * 1e3

    # 4) torch.profiler breakdown (compile only)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False) as prof:
        for _ in range(3):
            cal.compile_task_operator(x, y, basis_digest="prof", calibration_digest="prof")
        torch.cuda.synchronize()
    key_averages = prof.key_averages()
    top = sorted(key_averages, key=lambda e: e.self_device_time_total, reverse=True)[:5]
    breakdown = [{"name": k.key, "self_cuda_us": k.self_device_time_total} for k in top]

    props = torch.cuda.get_device_properties(0)
    receipt = {
        "schema_id": "henri.p831-profiler-receipt.v1",
        "d": args.d,
        "r": args.r,
        "pairs": args.pairs,
        "iters": args.iters,
        "device": props.name,
        "gpu_mem_total_mib": round(props.total_memory / 2**20),
        "compile_ms_per_op": round(compile_ms, 4),
        "transduce_ms_per_wave": round(transduce_ms, 4),
        "transduce_b64_ms_per_batch": round(batch_ms, 4),
        "top_cuda_kernels_us": breakdown,
        "status": "OK",
    }
    print(json.dumps(receipt, indent=2))
    prof.export_chrome_trace(args.trace)
    print(f"trace={args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
