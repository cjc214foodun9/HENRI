"""F2-M3 real-scale CUDA smoke — calibrate + snap at D=65536, V=32000 on RTX 5090.

Usage (remote, from repo root):
  env HENRI_F2_EGRESS=1 PYTHONPATH="HENRI V2" /venv/main/bin/python \
      "HENRI V2/experiments/verification/f2_egress_smoke.py" --N 2048 --D 65536 --V 32000

Prints a JSON receipt: status, shapes, codebook_bytes, times, finiteness.
Software-sanity on CPU; the CUDA smoke is the memory-guard + engagement boundary.
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--D", type=int, default=65536)
    ap.add_argument("--V", type=int, default=32000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from f2_egress_codebook import F2HopfieldEgressCodebook, get_f2_egress

    assert get_f2_egress(d_model=args.D, vocab_size=args.V) is not None, "HENRI_F2_EGRESS must be 1"

    t0 = time.perf_counter()
    X = torch.randn(args.N, args.D, dtype=torch.float32, device=device)
    y = torch.randint(0, min(args.V, 64), (args.N,), device=device)
    Y = torch.nn.functional.one_hot(y, num_classes=args.V).to(torch.float32)
    t_gen = time.perf_counter() - t0

    cb = F2HopfieldEgressCodebook(d_model=args.D, vocab_size=args.V, beta=8.0, ridge_lambda=1e-3)
    t0 = time.perf_counter()
    cb.calibrate(X, Y)
    t_cal = time.perf_counter() - t0
    torch.cuda.synchronize() if device == "cuda" else None

    t0 = time.perf_counter()
    z, logits = cb.snap(X[:64], return_logits=True)
    t_snap = time.perf_counter() - t0
    torch.cuda.synchronize() if device == "cuda" else None

    receipt = {
        "status": "ENGAGED" if cb.M is not None else "BLOCKED",
        "device": device,
        "shapes": {"M": list(cb.M.shape), "z": list(z.shape), "logits": list(logits.shape)},
        "codebook_bytes": cb.codebook_bytes(),
        "codebook_bytes_gib": round(cb.codebook_bytes() / 1024**3, 3),
        "finite": bool(torch.isfinite(cb.M).all().item() and torch.isfinite(z).all().item()),
        "time_gen_s": round(t_gen, 3),
        "time_calibrate_s": round(t_cal, 3),
        "time_snap_s": round(t_snap, 3),
        "beta": cb.beta,
        "ridge_lambda": cb.ridge_lambda,
    }
    print(json.dumps(receipt, indent=2))
    assert receipt["finite"], "non-finite codebook/snap"
    assert receipt["codebook_bytes"] < 16 * 1024**3, "memory guard exceeded"
    print("F2_SMOKE_OK")


if __name__ == "__main__":
    main()
