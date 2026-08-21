"""Default-OFF loader for the Phase 2 fused CUDA Sagnac scorer.

The extension is an implementation sidecar for the live AST candidate scoring
consumer. It is not evidence that SagnacMCTSPlanner controls ARC action choice.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


class SagnacCudaUnavailable(RuntimeError):
    """Raised when the explicitly requested fused CUDA path cannot load."""


_EXTENSION_LOADED = False


def enabled() -> bool:
    return os.environ.get("HENRI_SAGNAC_CUDA", "0") == "1"


def _source_path() -> Path:
    return Path(__file__).resolve().parent / "cuda" / "sagnac_mcts_cuda_core.cu"


def load_extension(*, verbose: bool = False) -> None:
    """Build and load the registered torch operator once.

    The caller must opt in with ``HENRI_SAGNAC_CUDA=1``. Build failures are
    typed and fail closed; the wrapper does not silently fall back after opt-in.
    """
    global _EXTENSION_LOADED
    if _EXTENSION_LOADED:
        return
    if not torch.cuda.is_available():
        raise SagnacCudaUnavailable("HENRI_SAGNAC_CUDA=1 requires CUDA")
    source = _source_path()
    if not source.is_file():
        raise SagnacCudaUnavailable(f"CUDA source missing: {source}")

    from torch.utils.cpp_extension import load

    build_dir = os.environ.get("HENRI_SAGNAC_CUDA_BUILD_DIR")
    kwargs = {
        "name": "henri_sagnac_mcts_cuda_v1",
        "sources": [str(source)],
        "extra_cuda_cflags": ["-O3", "--use_fast_math"],
        "with_cuda": True,
        "is_python_module": False,
        "verbose": verbose,
    }
    if build_dir:
        kwargs["build_directory"] = build_dir
    try:
        load(**kwargs)
    except Exception as exc:  # noqa: BLE001 - convert to typed boundary
        raise SagnacCudaUnavailable(f"CUDA extension build/load failed: {exc}") from exc
    if not hasattr(torch.ops, "henri") or not hasattr(torch.ops.henri, "sagnac_mcts"):
        raise SagnacCudaUnavailable("torch.ops.henri.sagnac_mcts was not registered")
    _EXTENSION_LOADED = True


def batched_mean_phase_cosine_cuda(
    candidates: torch.Tensor,
    codebook: torch.Tensor,
    *,
    verbose: bool = False,
) -> torch.Tensor:
    """Compute production-equivalent phase cosine scores through the CUDA op."""
    if not candidates.is_cuda or not codebook.is_cuda:
        raise SagnacCudaUnavailable("fused scorer requires CUDA tensors")
    if candidates.dtype != torch.uint8 or codebook.dtype != torch.uint8:
        raise TypeError("fused scorer requires uint8 phase tensors")
    if candidates.ndim != 2 or codebook.ndim != 2:
        raise ValueError("expected candidates [C,D] and codebook [N,D]")
    if candidates.shape[1] != codebook.shape[1]:
        raise ValueError("candidate and codebook dimensions must match")
    if candidates.numel() == 0 or codebook.numel() == 0:
        raise ValueError("candidate and codebook tensors must be non-empty")
    candidates = candidates.contiguous()
    codebook = codebook.contiguous()
    load_extension(verbose=verbose)
    return torch.ops.henri.sagnac_mcts(candidates, codebook)
