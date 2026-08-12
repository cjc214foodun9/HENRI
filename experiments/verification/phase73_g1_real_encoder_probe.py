"""Phase 7.3 G1 real-encoder probe (split candidate, D=65,536).

Re-runs the accepted G1 gate against the REAL production encoder path
(flag -> constructor field -> encode_grid computational branch -> output),
NOT the masked replica used in the original packet probe.

Measured per pre-registered gate (phase7_3_kill_gates.md):
  - byte identity: legacy vs explicit-default encoder, max tensor diff == 0.0
  - same-sum degeneracy: cos((1,2),(2,1)) < 0.5 gate (was ~1.0 collinear)
  - max cross-cosine over all 36 single-pixel waves < 0.5
  - LUT argmax recovery via production fractional_unbind_coordinate: 72/72
  - empty-foreground fail-closed: all-zero grid under bg_mask raises ValueError
"""
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "HENRI V2")
from henri_vision_encoder import HENRIVisionEncoder
from arc_phase_map import fractional_unbind_coordinate

D = 65536
K = 8192
DIM = 6


def build(kind, bg_mask=True):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return HENRIVisionEncoder(
        d_model=D, k_blocks=K, device=dev,
        spatial_basis_kind=kind, bg_mask=bg_mask,
    )


def wave(enc, grid):
    return enc.encode_grid(grid).to(torch.float32)


def single_pixel(r, c, color=5):
    g = [[0] * DIM for _ in range(DIM)]
    g[r][c] = color
    return g


def main():
    results = {}

    # 1. Byte identity: legacy vs explicit default (D=65,536, real encoder).
    legacy = HENRIVisionEncoder(d_model=D, k_blocks=K, device="cpu")
    defaulted = HENRIVisionEncoder(
        d_model=D, k_blocks=K, device="cpu",
        spatial_basis_kind="default", bg_mask=False,
    )
    grid = [[0, 0, 5, 0, 3, 0],
            [0, 0, 0, 12, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]
    with torch.no_grad():
        w_legacy = legacy.encode_grid(grid)
        w_default = defaulted.encode_grid(grid)
    max_diff = float(torch.max(torch.abs(w_legacy - w_default)).item())
    results["byte_identity_max_diff"] = max_diff

    # 2. G1 gate on real encoder for incommensurate and random.
    for kind in ("incommensurate", "random"):
        enc = build(kind)
        with torch.no_grad():
            waves = {k: wave(enc, single_pixel(*k)) for k in
                     ((r, c) for r in range(DIM) for c in range(DIM))}
            same_sum = float(torch.dot(waves[(1, 2)], waves[(2, 1)]).item())
            keys = list(waves)
            max_pair = 0.0
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    c = float(torch.vdot(waves[keys[i]], waves[keys[j]]).abs().item())
                    if c > max_pair:
                        max_pair = c
            correct = total = 0
            for r in range(DIM):
                for c in range(DIM):
                    for color in (5, 12):
                        w = wave(enc, single_pixel(r, c, color))
                        xr, yr, _ = fractional_unbind_coordinate(
                            w, enc, color, DIM, device=enc.device
                        )
                        total += 1
                        if (xr, yr) == (r, c):
                            correct += 1
            results[f"{kind}_same_sum_cos"] = same_sum
            results[f"{kind}_max_pair_cos"] = max_pair
            results[f"{kind}_recovery"] = f"{correct}/{total}"

    # 3. Empty-foreground fail-closed.
    try:
        enc = build("incommensurate")
        wave(enc, [[0] * DIM for _ in range(DIM)])
        results["empty_foreground_fail_closed"] = "NOT_RAISED"
    except ValueError:
        results["empty_foreground_fail_closed"] = "RAISED"

    for k, v in results.items():
        print(f"G1[{k}] = {v}")
    ok = (
        results["byte_identity_max_diff"] == 0.0
        and all(results[f"{k}_same_sum_cos"] < 0.5 for k in ("incommensurate", "random"))
        and all(results[f"{k}_max_pair_cos"] < 0.5 for k in ("incommensurate", "random"))
        and all(results[f"{k}_recovery"] == "72/72" for k in ("incommensurate", "random"))
        and results["empty_foreground_fail_closed"] == "RAISED"
    )
    print("G1_REAL_ENCODER_PASS" if ok else "G1_REAL_ENCODER_FAIL")


if __name__ == "__main__":
    main()
