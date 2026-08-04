"""Run the pre-registered projective-vs-flattened Hopfield CUDA matrix.

This is an execution artifact generator, not a benchmark evaluator. It fails
closed without CUDA and writes one JSON result only after the measured cells
complete. The flattened control is the existing ContinuousHopfieldCleanup;
it is not the paper's spherical vector Hopfield implementation.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

# The harness lives below the flat import boundary. Add the active-code root
# explicitly so the command works from either the repository root or this
# experiments directory without relying on PYTHONPATH side effects.
ACTIVE_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(ACTIVE_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACTIVE_CODE_ROOT))

from hopfield_cleanup import ContinuousHopfieldCleanup
from projective_hopfield import ProjectiveHopfieldCleanup


def _parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected one or more positive integers")
    return values


def _parse_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0.0 or item >= 1.0 for item in values):
        raise argparse.ArgumentTypeError("corruptions must be in [0,1)")
    return values


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _unit_complex(
    shape: tuple[int, ...], *, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    real = torch.randn(shape, generator=generator, device="cpu").to(device)
    imag = torch.randn(shape, generator=generator, device="cpu").to(device)
    value = torch.complex(real, imag)
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-12)


def _flatten_complex(value: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(value).reshape(value.shape[0], -1).to(torch.float32)


def _measure_projective(
    memories: torch.Tensor,
    cues: torch.Tensor,
    targets: torch.Tensor,
    sweeps: int,
) -> dict[str, float | int]:
    module = ProjectiveHopfieldCleanup(qudit_dim=memories.shape[-1]).to(memories.device)
    module.store_memories(memories)
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = module.retrieve(cues, sweeps=sweeps)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    predicted = result.selected_memory_index
    correct = (predicted == targets).to(torch.float32)
    norm_error = (torch.linalg.vector_norm(result.state, dim=-1) - 1.0).abs().amax()
    return {
        "top1_rate": float(correct.mean().item()),
        "correct": int(correct.sum().item()),
        "attempted": int(correct.numel()),
        "latency_ms": elapsed_ms,
        "mean_spectral_gap": float(result.spectral_gap.mean().item()),
        "min_spectral_gap": float(result.spectral_gap.min().item()),
        "max_hermitian_residual": float(result.hermitian_residual.max().item()),
        "max_local_norm_error": float(norm_error.item()),
        "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(memories.device)),
    }


def _measure_flattened_control(
    memories: torch.Tensor,
    cues: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    flattened_memories = _flatten_complex(memories)
    flattened_cues = _flatten_complex(cues)
    module = ContinuousHopfieldCleanup(dim=flattened_memories.shape[-1]).to(memories.device)
    module.store_engrams(flattened_memories)
    torch.cuda.synchronize()
    start = time.perf_counter()
    _, predicted, _ = module.hard_retrieve(flattened_cues)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    correct = (predicted == targets).to(torch.float32)
    return {
        "top1_rate": float(correct.mean().item()),
        "correct": int(correct.sum().item()),
        "attempted": int(correct.numel()),
        "latency_ms": elapsed_ms,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to produce a CPU experiment artifact")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    cells: list[dict[str, object]] = []
    for load in args.loads:
        memories = _unit_complex(
            (load, args.neurons, args.qudit_dim), generator=generator, device=device
        )
        for corruption in args.corruptions:
            targets = torch.arange(args.repeats, device=device) % load
            clean = memories[targets]
            noise = _unit_complex(
                tuple(clean.shape), generator=generator, device=device
            )
            cues = clean + corruption * noise
            cues = cues / torch.linalg.vector_norm(cues, dim=-1, keepdim=True).clamp_min(1e-12)
            torch.cuda.reset_peak_memory_stats(device)
            projective = _measure_projective(memories, cues, targets, args.sweeps)
            flattened = _measure_flattened_control(memories, cues, targets)
            cells.append(
                {
                    "load": load,
                    "corruption": corruption,
                    "projective": projective,
                    "flattened_control": flattened,
                }
            )
    return {
        "schema_id": "henri.projective-hopfield-cuda-matrix.v1",
        "status": "PASS",
        "evidence_class": "component_execution",
        "commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "loads": args.loads,
        "corruptions": args.corruptions,
        "repeats": args.repeats,
        "neurons": args.neurons,
        "qudit_dim": args.qudit_dim,
        "sweeps": args.sweeps,
        "cells": cells,
        "limitations": [
            "flattened control is not the paper's spherical vector Hopfield baseline",
            "component execution is not external task outcome evidence",
            "no production caller or Zone C consumer is exercised",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loads", type=_parse_ints, default=[4, 8, 16])
    parser.add_argument("--corruptions", type=_parse_floats, default=[0.10, 0.30, 0.50])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=16)
    parser.add_argument("--qudit-dim", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.neurons < 2 or args.qudit_dim < 2 or args.sweeps < 1:
        parser.error("repeats, neurons, qudit-dim, and sweeps must be positive; neurons and qudit-dim >= 2")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "cells": len(result["cells"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
