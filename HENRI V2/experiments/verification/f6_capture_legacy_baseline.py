"""Capture the LEGACY (pre-wiring) compile_task_functor baseline constants.

Run BEFORE wiring HENRI_F6_FUNCTOR into arc_task_functor.py. The output
values are pinned in test_f6_adaptive_functor.py::test_c7 as the
default-OFF differential reference (F5 G6 precedent: flag unset =>
byte-identical legacy path).
"""
import sys
from pathlib import Path

import numpy as np
import torch

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from arc_task_functor import compile_task_functor


class MockTok:
    def __init__(self, D: int):
        self.D = D

    def encode_spatial_grid(self, grid):
        g = torch.Generator().manual_seed(99)
        return torch.rand(1, self.D, generator=g)


def main() -> None:
    D = 128
    tok = MockTok(D)
    pairs = [
        (np.zeros((2, 2), dtype=np.int64), np.ones((2, 2), dtype=np.int64)),
        (np.ones((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64)),
        (np.full((2, 2), 2, dtype=np.int64), np.full((2, 2), 3, dtype=np.int64)),
        (np.full((2, 2), 3, dtype=np.int64), np.full((2, 2), 2, dtype=np.int64)),
    ]
    r = compile_task_functor(pairs, tok, device="cpu", task_id="t")
    print(f"status={r.status}")
    print(f"w_task_sha256={r.w_task_sha256}")
    print(f"held_out_cos={r.held_out_cos}")
    print(f"identity_cos={r.identity_cos}")
    print(f"reason={r.reason}")


if __name__ == "__main__":
    main()
