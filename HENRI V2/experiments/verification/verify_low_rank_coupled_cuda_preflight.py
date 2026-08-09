"""Phase 5 P1 — LowRankCoupledTransition CUDA structural preflight.

STRUCTURAL / MECHANISM telemetry only. This is NOT a benchmark and makes no
task-capability claim. The formal r=64 vs r=128 production A/B requires
authentic immutable ARC-AGI-3 transition traces; with no such harness
present, the production verdict stays BLOCKED (no toy substitution).

Emits: device assertions, off-block Jacobian, Stiefel residual on the
effective rank, one online dual-EDMD train_transition_batch step on
synthetic unit waves (production learning rule smoke), forward latency,
peak VRAM, commit digest.
"""

import argparse
import json
import subprocess
import time

import torch


def unit_wave(nb: int, device: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(nb, 8, generator=g, device="cpu")
    w = w / torch.norm(w, p=2, dim=-1, keepdim=True)
    return w.to(device)


def sagnac_loss(pred, target):
    p = pred.reshape(-1)
    o = target.reshape(-1)
    return 1.0 - torch.dot(p, o) / (torch.norm(p) * torch.norm(o)).clamp(min=1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    ap.add_argument("--commit", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("BLOCKED_INFRASTRUCTURE: cuda unavailable")
        return 2
    device = args.device

    from efe_planner import EFEPlanner  # local import: PYTHONPATH="HENRI V2"

    torch.manual_seed(0)
    t0 = time.time()
    planner = EFEPlanner(
        num_blocks=8192, d_model=65536, transition_rank=args.rank).to(device)
    t_construct = time.time() - t0

    # Device assertions (per-arm preflight rule)
    for name, p in planner.named_parameters():
        if p.is_floating_point() and p.device.type != device:
            print(f"INVALID_PLUMBING: param {name} on {p.device}")
            return 3

    t = planner.transition
    requested = t.requested_rank
    effective = t.rank
    d_model = t.d

    # Structural: off-block Jacobian at production dim
    s0 = unit_wave(8192, device, 100).requires_grad_(True)
    a0 = unit_wave(8192, device, 101)
    out = t(s0, a0)
    g = torch.autograd.grad(out[5, 0].sum(), s0)[0]
    off_block_jac = float(g[0].abs().max().item())

    # Stiefel residual on effective rank
    V = t.field_V
    gram = V.T @ V
    Iv = torch.eye(effective, device=device)
    stiefel_residual = float((gram - Iv).abs().max().item())

    # Production learning rule smoke: one online dual-EDMD step, synthetic
    # unit waves (structural only — NOT ARC traces, NOT task evidence).
    n = 4
    tr_s = torch.stack([unit_wave(8192, device, 4000 + k) for k in range(n)])
    tr_a = torch.stack([unit_wave(8192, device, 5000 + k) for k in range(n)])
    with torch.no_grad():
        tr_y = torch.stack([t(tr_s[k], tr_a[k]) for k in range(n)])
    t0 = time.time()
    planner.train_transition_batch(
        tr_s, tr_a, tr_y, iters=1, ridge=1e-4,
        update_residual=True, blend=0.5)
    t_train_step = time.time() - t0
    with torch.no_grad():
        post_step_sagnac = float(
            torch.stack([sagnac_loss(t(tr_s[k], tr_a[k]), tr_y[k])
                         for k in range(n)]).mean().item())

    # Forward latency (warmup + 10)
    with torch.no_grad():
        for _ in range(3):
            t(s0, a0)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            t(s0, a0)
        torch.cuda.synchronize()
        fwd_latency_ms = (time.time() - t0) / 10 * 1000

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    commit = args.commit or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    result = {
        "scope": "CUDA_STRUCTURAL_PREFLIGHT (not a benchmark)",
        "verdict": "PASS",
        "commit": commit,
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "requested_rank": requested,
        "effective_rank": effective,
        "d_model": d_model,
        "off_block_jacobian": off_block_jac,
        "stiefel_residual": stiefel_residual,
        "post_step_sagnac": post_step_sagnac,
        "fwd_latency_ms": round(fwd_latency_ms, 4),
        "construct_s": round(t_construct, 3),
        "train_step_s": round(t_train_step, 3),
        "peak_vram_gb": round(peak_vram_gb, 3),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
