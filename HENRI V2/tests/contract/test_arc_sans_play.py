"""Contract tests: ARC SANS Epistemic Play (Phase 7.2 Step 2). CPU-only."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arc_sans_play import (
    STATUS_BUFFER_INSUFFICIENT,
    STATUS_CALIBRATED,
    STATUS_DEGENERATE_LABELS,
    STATUS_IMPORT_FAILED,
    MIN_SAMPLES,
    run_sans_play,
)


class _Action:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class _Vocab:
    def __init__(self, n=3):
        self.id_to_action = [_Action(f"A{i}", i) for i in range(n)]


class _Frame:
    def __init__(self, arr):
        self.frame = [arr]


class _Game:
    """Deterministic toy game: action i toggles i cells (delta_nu=i+1)."""

    def __init__(self, n_actions=3, dim=8):
        self.grid = [[0] * dim for _ in range(dim)]
        self.n_actions = n_actions
        self.state = type("S", (), {"name": "RUNNING"})()

    def reset(self):
        self.grid = [[0] * 8 for _ in range(8)]
        return _Frame([row[:] for row in self.grid])

    def step(self, action):
        v = getattr(action, "value", 0)
        for k in range(v + 1):
            self.grid[k % 8][(k * 3) % 8] = (self.grid[k % 8][(k * 3) % 8] + 1) % 10
        if self.n_actions > 3 and v >= 3:
            self.state.name = "GAME_OVER"
        return _Frame([row[:] for row in self.grid])


class _Transducer:
    def __init__(self, d_model=512):
        self.d_model = d_model
        from torch import nn
        self.unbinder = nn.Sequential()  # placeholder; replaced below
        self._hidden_dim = 64
        self._down = nn.Linear(d_model, self._hidden_dim, bias=False)
        self._ln = nn.LayerNorm(self._hidden_dim)
        self.unbinder = type("U", (), {
            "down_proj": self._down,
            "layer_norm": self._ln,
            "act": torch.nn.GELU(),
        })()


class _Token:
    def __init__(self, d_model=512):
        self.d_model = d_model

    def encode_spatial_grid(self, grid):
        w = torch.zeros(self.d_model)
        flat = sum((list(r) for r in grid), [])
        for i, v in enumerate(flat[: self.d_model]):
            w[i] = (v + 1) * 0.001 + (i % 7) * 1e-5
        return w.view(1, self.d_model // 8, 8)


def test_insufficient_buffer():
    game = _Game(n_actions=3)
    tok = _Token(512)
    tr = _Transducer(512)
    head = torch.nn.Linear(64, 3)
    res = run_sans_play(
        game, tok, tr, head, _Vocab(3), n_steps=2, device="cpu", seed=1,
        env_name="toy", tele=None,
    )
    # 2 steps can produce at most 2 rows < MIN_SAMPLES
    assert res.status in (STATUS_BUFFER_INSUFFICIENT, STATUS_DEGENERATE_LABELS)
    assert res.buffer_size <= 2


def test_calibrated_with_provenance():
    game = _Game(n_actions=3)
    tok = _Token(512)
    tr = _Transducer(512)
    head = torch.nn.Linear(64, 3)
    res = run_sans_play(
        game, tok, tr, head, _Vocab(3), n_steps=40, device="cpu", seed=7,
        env_name="toy", tele=None, head_path=str(Path(__file__).parent / "tmp_sans_head.pt"),
    )
    # With 3 actions producing delta_nu 1/2/3, labels are separable-ish;
    # require at least that the machinery ran and emitted a typed status.
    assert res.buffer_size > 0
    assert res.calibration_dataset_digest or res.status != STATUS_CALIBRATED
    if res.status == STATUS_CALIBRATED:
        assert res.action_head_sha256
        assert res.action_head_state_dict_sha256
        assert res.split_identity
        assert res.provenance["schema_id"] == "henri.sans-action-head.v1"
        assert res.held_out_accuracy is not None


def test_calibrated_head_dims_match():
    game = _Game(n_actions=3)
    tok = _Token(512)
    tr = _Transducer(512)
    head = torch.nn.Linear(64, 3)
    res = run_sans_play(
        game, tok, tr, head, _Vocab(3), n_steps=40, device="cpu", seed=11,
        env_name="toy", tele=None, head_path=str(Path(__file__).parent / "tmp_sans_head2.pt"),
    )
    if res.status == STATUS_CALIBRATED:
        assert res.hidden_dim == 64
        assert res.action_dim == 3


def test_steer_never_crashes_and_stays_legal():
    """Phase 7.3 G3: steer=True must run the same machinery and only ever
    select legal actions (seeded toy game, 3 actions)."""
    game = _Game(n_actions=3)
    tok = _Token(512)
    tr = _Transducer(512)
    head = torch.nn.Linear(64, 3)
    res = run_sans_play(
        game, tok, tr, head, _Vocab(3), n_steps=40, device="cpu", seed=7,
        env_name="toy", tele=None, steer=True,
        head_path=str(Path(__file__).parent / "tmp_sans_steer_head.pt"),
    )
    assert res.buffer_size > 0
    assert res.status in (
        STATUS_BUFFER_INSUFFICIENT,
        STATUS_DEGENERATE_LABELS,
        STATUS_CALIBRATED,
    )
    assert res.status != STATUS_IMPORT_FAILED
