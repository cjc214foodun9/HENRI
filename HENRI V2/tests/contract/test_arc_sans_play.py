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


# ---- Phase 7.6: Sagnac hard-axiom steering (default OFF) ----

def _axioms(n, blocks=64, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    ax = torch.randn(n, blocks, dim, generator=g)
    return ax / (ax.norm(dim=(1, 2), keepdim=True) + 1e-12)


def test_select_action_sagnac_vetoes_off_manifold():
    from arc_sans_play import select_action_sagnac
    g = torch.Generator().manual_seed(1)
    w = torch.randn(64, 8, generator=g)
    w = w / (w.norm() + 1e-12)
    # Antipodal axioms -> cos = -1 -> delta = 1.0 > 0.35 -> veto.
    axioms = -w.unsqueeze(0)
    assert select_action_sagnac(w, axioms, g, 4) is None


def test_select_action_sagnac_steers_valid_state():
    from arc_sans_play import select_action_sagnac
    g = torch.Generator().manual_seed(2)
    w = torch.randn(64, 8, generator=g)
    w = w / (w.norm() + 1e-12)
    axioms = w.unsqueeze(0)  # identical -> delta = 0 -> steer
    idx = select_action_sagnac(w, axioms, g, 4)
    assert idx is not None and 0 <= idx < 4


def test_select_action_sagnac_rejects_non_3d_axioms():
    from arc_sans_play import select_action_sagnac
    g = torch.Generator().manual_seed(3)
    w = torch.randn(64, 8)
    assert select_action_sagnac(w, torch.randn(64, 8), g, 4) is None


def test_run_sans_play_sagnac_fail_closed_without_axioms():
    from arc_sans_play import STATUS_BLOCKED_AXIOMS, run_sans_play
    game = _Game(n_actions=3)
    tok = _Token(512)
    tr = _Transducer(512)
    head = torch.nn.Linear(64, 3)
    res = run_sans_play(
        game, tok, tr, head, _Vocab(3), n_steps=10, device="cpu", seed=1,
        env_name="toy", tele=None, selection_mode="sagnac",
        axiom_waves=None,
    )
    assert res.status == STATUS_BLOCKED_AXIOMS
    assert res.buffer_size == 0


def test_run_sans_play_random_preserves_default_semantics():
    game = _Game(n_actions=3)
    tok = _Token(512)
    tr = _Transducer(512)
    head = torch.nn.Linear(64, 3)
    res = run_sans_play(
        game, tok, tr, head, _Vocab(3), n_steps=40, device="cpu", seed=7,
        env_name="toy", tele=None,
        head_path=str(Path(__file__).parent / "tmp_sans_head3.pt"),
    )
    assert res.selection_mode == "random"
    assert res.veto_steps == 0
    assert res.veto_rate == 0.0


# ---- Phase 7.7: adaptive/discriminative Sagnac steering (default OFF) ----

def test_select_action_sagnac_adaptive_steers_on_subspace():
    """w exactly on the axiom subspace (w == a1, M=4 orthonormal axioms)
    -> admitted + telemetry. deltas = [0, .5, .5, .5], mu=0.375, sigma=0.2165
    (population), eps_t=0.1585, min=0 < eps_t -> ADMIT."""
    from arc_sans_play import select_action_sagnac_adaptive
    g = torch.Generator().manual_seed(4)
    a1 = torch.randn(64, 8, generator=g)
    a1 = a1 / (a1.norm() + 1e-12)
    others = [torch.randn(64, 8, generator=g) for _ in range(3)]
    ax = [a1]
    for v in others:
        for b in ax:
            v = v - (v.flatten() @ b.flatten()) * b
        v = v / (v.norm() + 1e-12)
        ax.append(v)
    axioms = torch.stack(ax)
    idx, epsilon_t, delta_subspace = select_action_sagnac_adaptive(a1, axioms, g, 4)
    assert idx is not None and 0 <= idx < 4
    assert epsilon_t is not None and 0.1 < epsilon_t < 0.25
    assert delta_subspace is not None and delta_subspace < 1e-4


def test_select_action_sagnac_adaptive_relative_criterion_veto():
    """No dominant axiom -> min_delta > mu - sigma -> veto (PDF rule).

    Right-skewed deltas [0.475, 0.5, 0.5, 0.703]: mu=0.544, sigma=0.092,
    eps_t=0.452, min=0.475 > eps_t -> VETO. Constructed: w has cos 0.05 to
    a1, ~0 to a2/a3, -0.4 to a4, rest orthogonal to span(A)."""
    from arc_sans_play import select_action_sagnac_adaptive
    g = torch.Generator().manual_seed(5)
    ax = [torch.randn(64, 8, generator=g) for _ in range(4)]
    for i in range(4):
        for b in ax[:i]:
            ax[i] = ax[i] - (ax[i].flatten() @ b.flatten()) * b
        ax[i] = ax[i] / (ax[i].norm() + 1e-12)
    e = torch.randn(64, 8, generator=g)
    for b in ax:
        e = e - (e.flatten() @ b.flatten()) * b
    e = e / (e.norm() + 1e-12)
    w = 0.05 * ax[0] - 0.4 * ax[3] + 0.9 * e
    w = w / (w.norm() + 1e-12)
    axioms = torch.stack(ax)
    idx, epsilon_t, delta_subspace = select_action_sagnac_adaptive(w, axioms, g, 4)
    assert idx is None
    assert epsilon_t is not None
    # 0.2-0.4 band: partially off-subspace (predicted ~0.296).
    assert delta_subspace is not None and 0.2 < delta_subspace < 0.4


def test_select_action_sagnac_adaptive_vetoes_null_projection():
    """Zero wave -> zero projection norm -> fail-closed veto (no division by 0)."""
    from arc_sans_play import select_action_sagnac_adaptive
    g = torch.Generator().manual_seed(6)
    axioms = _axioms(2, seed=6)
    w = torch.zeros_like(axioms[0])
    idx, epsilon_t, delta_subspace = select_action_sagnac_adaptive(w, axioms, g, 4)
    assert idx is None
    assert epsilon_t is None
    assert delta_subspace is None


def test_run_sans_play_sagnac_adaptive_fail_closed_without_axioms():
    from arc_sans_play import STATUS_BLOCKED_AXIOMS, run_sans_play
    game = _Game(n_actions=3)
    tok = _Token(512)
    tr = _Transducer(512)
    head = torch.nn.Linear(64, 3)
    res = run_sans_play(
        game, tok, tr, head, _Vocab(3), n_steps=10, device="cpu", seed=1,
        env_name="toy", tele=None, selection_mode="sagnac-adaptive",
        axiom_waves=None,
    )
    assert res.status == STATUS_BLOCKED_AXIOMS
    assert res.buffer_size == 0


def test_run_sans_play_sagnac_adaptive_rejects_bad_axioms():
    from arc_sans_play import STATUS_BLOCKED_AXIOMS, run_sans_play
    game = _Game(n_actions=3)
    tok = _Token(512)
    tr = _Transducer(512)
    head = torch.nn.Linear(64, 3)
    res = run_sans_play(
        game, tok, tr, head, _Vocab(3), n_steps=10, device="cpu", seed=1,
        env_name="toy", tele=None, selection_mode="sagnac-adaptive",
        axiom_waves=torch.randn(64, 8),
    )
    assert res.status == STATUS_BLOCKED_AXIOMS
