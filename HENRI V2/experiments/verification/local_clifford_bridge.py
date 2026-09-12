"""8-channel local Clifford block reduction for the wave -> d64 bridges.

WHY THIS EXISTS
---------------
Stage 2 (metric locality) of HENRI-ARCH-2026-CARRIER-AUDIT-AND-PHYSICAL-ML-GAPS.

The four wave->descriptor bridges currently collapse a 65536-dimension wave by
global mean pooling:

    pooled = w.view(16, 4096).mean(dim=0)      # [16, 4096] -> [4096]

That averages ACROSS the 16 blocks, so a wave and its block-permuted twin map to
the same descriptor. Measured (verify_stage2_task_dependence.py, CPU):

    task           mean pooling AUC   Clifford AUC
    both              0.9292 PASS      1.0000 PASS
    channel_only      1.0000 PASS      0.9939 PASS
    block_only        0.5014 FAIL      0.9959 PASS

Block-permutation cosine: 1.0000 for mean pooling, -0.1024 for Clifford blocks.

WHAT THIS FUNCTION DOES
-----------------------
Reduces the wave LOCALLY. No reduction ever crosses a block boundary or a
channel boundary:

    65536 = [16 blocks, 8 channels, 512]
              |
              |  per-channel local mean, kernel 16, stride 16   (LOCAL only)
              v
           [16 blocks, 8 channels, 32]   = 4096   <- same budget as mean pooling
              |
              |  Cl(3,0) sign binding across the 8 channels    (Clifford)
              v
           [4096]  flattened, block-major

Block identity survives because the 16 blocks are CONCATENATED, never summed.
Channel identity survives because channels are bound, never averaged.

HONEST BOUNDARY
---------------
The output width (4096) is deliberately identical to the mean-pooling output, so
this is a drop-in swap at the same call site with the same downstream scale
(consumers apply F.normalize(...) * 64.0). The Cl(3,0) sign binding is a
VSA-style bind: it keeps opposite-signed channels from cancelling. That binding
is HYPOTHESIS, not proven optimal. The locality property it buys is measured.

Evidence class for the numbers above: OBSERVED.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_BLOCKS = 16
BLOCK_LEN = 4096
CHANNELS = 8
PER_CHANNEL = BLOCK_LEN // CHANNELS          # 512
LOCAL_WINDOW = 16                            # local reduction window
PER_BLOCK_OUT = CHANNELS * (PER_CHANNEL // LOCAL_WINDOW)   # 8 * 32 = 256
OUT_LEN = NUM_BLOCKS * PER_BLOCK_OUT         # 16 * 256 = 4096

# Cl(3,0) basis sign pattern {1, e1, e2, e3, e12, e13, e23, e123}.
# e_i * e_i = +1 (scalar), e_i * e_j = +-e_ij (bivector) for i != j.
CL_SIGN = (1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0)


def local_clifford_pool(wave, out_len: int = OUT_LEN) -> torch.Tensor:
    """Reduce a 65536-dimension wave to `out_len` while preserving locality.

    Args:
        wave: 65536-element wave (tensor or array-like). Padded/truncated to
              65536 exactly as the existing bridges do.
        out_len: output width. Default 4096, matching the mean-pooling output.

    Returns:
        [out_len] float32 tensor. NOT normalized and NOT scaled; the caller
        applies the existing F.normalize(...) * 64.0 step so the operating
        point is unchanged.
    """
    w = torch.as_tensor(wave, dtype=torch.float32).reshape(-1)
    total = NUM_BLOCKS * BLOCK_LEN
    if w.numel() < total:
        w = F.pad(w, (0, total - w.numel()))
    else:
        w = w[:total]

    # [16, 8, 512] -- block axis and channel axis both kept.
    x = w.view(NUM_BLOCKS, CHANNELS, PER_CHANNEL)

    # Local mean INSIDE each (block, channel) row. Kernel and stride are both
    # LOCAL_WINDOW, so no window ever spans a channel or a block boundary.
    y = x.reshape(NUM_BLOCKS * CHANNELS, 1, PER_CHANNEL)
    y = F.avg_pool1d(y, kernel_size=LOCAL_WINDOW, stride=LOCAL_WINDOW)
    y = y.reshape(NUM_BLOCKS, CHANNELS, PER_CHANNEL // LOCAL_WINDOW)

    # Cl(3,0) sign binding across channels. Averaging would cancel
    # opposite-signed channels; this bind preserves them.
    # sign must be shaped [1, CHANNELS, 1] so it broadcasts over the CHANNEL
    # axis, not the trailing spatial axis.
    sign = torch.tensor(CL_SIGN, dtype=y.dtype, device=y.device)
    sign = sign.view(1, CHANNELS, 1)
    y = y * sign

    flat = y.reshape(-1)                      # [4096], block-major
    if out_len == flat.numel():
        return flat
    if out_len < flat.numel():
        return flat[:out_len]
    reps = -(-out_len // flat.numel())        # ceil division
    return flat.repeat(reps)[:out_len]


def is_enabled() -> bool:
    """The flag that selects this bridge over the global mean-pooling default."""
    import os
    return os.environ.get("HENRI_LOCAL_CLIFFORD_BRIDGE") == "1"
