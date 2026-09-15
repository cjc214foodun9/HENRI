"""Contract tests: ARC Task Functor (Phase 7.2 Step 1). CPU-only, no network."""

import os
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


# --- Phase 10.2 directive 2a: production default LOCKED to diag_ls ----------
def _pairs():
    return [(_mk((1, 1), 5), _mk((2, 2), 5)),
            (_mk((2, 2), 5), _mk((3, 3), 5))]


def _clear(monkeypatch):
    for k in ("HENRI_FUNCTOR_FIT", "HENRI_F7_AFFINE", "HENRI_F6_FUNCTOR",
              "HENRI_ENCODER_TORUS"):
        monkeypatch.delenv(k, raising=False)


def test_default_fit_is_diag_ls(monkeypatch):
    """No HENRI_FUNCTOR_FIT -> regularized per-slot DIAGONAL least-squares.

    Phase 10.2 directive 4.2 locks this. The Koopman falsification (best held-out
    +0.3917 < diag_ls +0.4215 and < identity +0.4033) is why the default must not
    drift onto the coupled family: a coupled default silently loses ~0.03 held-out.
    """
    _clear(monkeypatch)
    res = compile_task_functor(_pairs(), _FakeTokenizer(), task_id="t")
    fit = res.provenance["fit"]
    assert fit["mode"] == "diag_ls", fit
    assert fit["operator_family"] == "per_slot_diagonal_ridge_ls", fit
    # The ridge must be REPORTED relative to the excitation, so a ridge that is
    # numerically absent is visible. An absolute lambda alone hides that.
    assert "reg_lambda" in fit
    assert "ridge_rel_to_mean_excitation" in fit, fit


def test_unknown_fit_mode_falls_back_to_diag_ls(monkeypatch):
    """An unrecognised value must fail SAFE onto the production default."""
    _clear(monkeypatch)
    monkeypatch.setenv("HENRI_FUNCTOR_FIT", "definitely_not_a_mode")
    res = compile_task_functor(_pairs(), _FakeTokenizer(), task_id="t")
    assert res.provenance["fit"]["mode"] == "diag_ls"


def test_koopman_is_opt_in_and_fails_closed(monkeypatch):
    """The Koopman arm must be unreachable without explicit opt-in, and must fail
    closed when its encoder prerequisite is absent rather than silently degrading
    to a diagonal operator."""
    _clear(monkeypatch)
    monkeypatch.setenv("HENRI_FUNCTOR_FIT", "koopman_8")
    with pytest.raises(RuntimeError, match="BLOCKED_NO_TORUS_ENCODER"):
        compile_task_functor(_pairs(), _FakeTokenizer(), task_id="t")


def test_directive_alias_koopman_maps_to_koopman_8(monkeypatch):
    """Phase 10.2 names the arm `koopman`; the live arm is `koopman_8`. The alias
    must resolve to the REAL Koopman path. Proven by the SAME fail-closed error the
    canonical name raises: if the alias degraded to diag_ls, this would not raise."""
    _clear(monkeypatch)
    monkeypatch.setenv("HENRI_FUNCTOR_FIT", "koopman")
    with pytest.raises(RuntimeError, match="BLOCKED_NO_TORUS_ENCODER"):
        compile_task_functor(_pairs(), _FakeTokenizer(), task_id="t")
