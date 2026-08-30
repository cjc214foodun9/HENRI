"""Capture the LEGACY (pre-wiring) compile_task_functor baseline constants.

Run BEFORE wiring HENRI_F7_AFFINE into arc_task_functor.py. The output
values are pinned in test_f7_affine_egress.py::test_c5_differential.
The fixture matches the F6 capture (4 grid pairs, D=128 mock tokenizer).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arc_task_functor import compile_task_functor  # noqa: E402


class MockTok:
    def __init__(self, D: int = 128):
        self.D = D

    def encode_spatial_grid(self, grid):
        g = torch.Generator()
        seed = int(hashlib.sha256(np.asarray(grid).tobytes()).hexdigest()[:8], 16)
        g.manual_seed(seed)
        return torch.randn(1, self.D, generator=g)


def main() -> None:
    assert "HENRI_F7_AFFINE" not in os.environ, "capture must run flag-unset (pre-wiring)"
    pairs = [
        (np.zeros((2, 2), dtype=np.int64), np.ones((2, 2), dtype=np.int64)),
        (np.ones((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64)),
        (np.full((2, 2), 2, dtype=np.int64), np.full((2, 2), 3, dtype=np.int64)),
        (np.full((2, 2), 3, dtype=np.int64), np.full((2, 2), 2, dtype=np.int64)),
    ]
    res = compile_task_functor(pairs, MockTok(), device="cpu", task_id="f7-baseline")
    print(json.dumps({
        "status": res.status,
        "w_task_sha256": res.w_task_sha256,
        "held_out_cos": res.held_out_cos,
        "identity_cos": res.identity_cos,
        "goal_wave_sha256": res.goal_wave_sha256,
        "pairs_digest": res.pairs_digest,
    }, indent=2))


if __name__ == "__main__":
    main()
