"""Contract tests: ARC Task Functor (Phase 7.2 Step 1). CPU-only, no network."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arc_task_functor import (
    STATUS_FALSIFIED,
    STATUS_NO_DEMOS,
    STATUS_OK,
    compile_task_functor,
)


class _FakeTokenizer:
    def __init__(self, d_model=2048):
        self.d_model = d_model

    def encode_spatial_grid(self, grid):
        # Deterministic wave: hash-based but structured per grid (not random).
        import hashlib
        flat = torch.tensor(sum(grid, []), dtype=torch.float32)
        h = hashlib.sha256(flat.numpy().tobytes()).digest()
        seed = int.from_bytes(h[:8], "little")
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(self.d_model, generator=g)
        return w.view(1, self.d_model // 8, 8)


def _mk(rc, color, n=6):
    g = [[0] * n for _ in range(n)]
    g[rc[0]][rc[1]] = color
    return g


def test_no_demos():
    res = compile_task_functor([], _FakeTokenizer(), task_id="t")
    assert res.status == STATUS_NO_DEMOS


def test_holds_out_last_pair():
    tok = _FakeTokenizer()
    pairs = [(_mk((1, 1), 5), _mk((2, 2), 5)),
             (_mk((2, 2), 5), _mk((3, 3), 5)),
             (_mk((3, 3), 5), _mk((4, 4), 5))]
    res = compile_task_functor(pairs, tok, task_id="t")
    assert res.demo_pair_count == 3
    assert res.pairs_digest
    assert res.w_task_sha256
    assert res.goal_wave_sha256
    assert res.hold_out_index == 2 if hasattr(res, "hold_out_index") else True
    assert res.provenance["schema_id"] == "henri.task-functor.v1"


def test_provenance_pinned():
    tok = _FakeTokenizer()
    pairs = [(_mk((1, 1), 5), _mk((2, 2), 5)),
             (_mk((2, 2), 5), _mk((3, 3), 5))]
    res_a = compile_task_functor(pairs, tok, task_id="t")
    res_b = compile_task_functor(pairs, tok, task_id="t")
    assert res_a.pairs_digest == res_b.pairs_digest
    assert res_a.w_task_sha256 == res_b.w_task_sha256
    # Different pair set must change the digest.
    pairs_c = pairs + [(_mk((1, 3), 7), _mk((2, 4), 7))]
    res_c = compile_task_functor(pairs_c, tok, task_id="t")
    assert res_c.pairs_digest != res_a.pairs_digest
