"""Contract tests for the Phase 7 ARC semantic action head.

CPU-only by design: the action-head checkpoint contract mirrors the
decoder checkpoint contract (policy=required|disabled; "disabled" never
deserializes a production artifact). All heads here are tiny in-memory
stubs (d_model=64, d_hidden=16, |A|=6).
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from arc_action_head import (
    ActionHead,
    ActionHeadError,
    ActionHeadState,
    decode_action_head,
    load_action_head,
    unbinder_hidden,
)
from arc_egress_contract import ActionEgressVocabulary, flatten_uwe


class StubUnbinder(nn.Module):
    def __init__(self, d_model: int = 64, d_hidden: int = 16):
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.layer_norm = nn.LayerNorm(d_hidden)
        self.act = nn.GELU()


class StubTransducer:
    def __init__(self, d_model: int = 64, d_hidden: int = 16):
        self.d_model = d_model
        self.unbinder = StubUnbinder(d_model, d_hidden)


class StubAction:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value


def make_vocab() -> ActionEgressVocabulary:
    actions = [StubAction(f"ACTION{i}", i) for i in (1, 2, 3, 4, 6, 7)]
    return ActionEgressVocabulary(StubAction, actions)


def make_wave(d_model: int = 64) -> torch.Tensor:
    nb = d_model // 8
    return torch.randn(nb, 8, dtype=torch.float32)


def make_head(d_hidden: int = 16, n_actions: int = 6) -> ActionHead:
    return ActionHead(d_hidden=d_hidden, n_actions=n_actions)


def make_checkpoint(head: ActionHead, digest: str = "calib-digest-0001"):
    return {
        "schema_id": "henri.action-head.v1",
        "d_model": 64,
        "hidden_dim": head.d_hidden,
        "action_dim": head.n_actions,
        "calibration_dataset_digest": digest,
        "state_dict": head.state_dict(),
    }


def test_policy_disabled_never_touches_artifact():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "henri_action_head.pt")
        head = make_head()
        state = load_action_head(head, path, policy="disabled")
        assert state.action_head_load_status == "SKIPPED_POLICY_DISABLED"
        assert state.trained_action_head_active is False
        assert state.action_head_sha256 is None
        assert not os.path.exists(path)


def test_required_missing_checkpoint_raises():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "missing.pt")
        head = make_head()
        with pytest.raises(ActionHeadError):
            load_action_head(head, path, policy="required")


def test_valid_checkpoint_loads_with_provenance():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "henri_action_head.pt")
        head = make_head()
        torch.save(make_checkpoint(head), path)
        loaded = make_head()
        state = load_action_head(
            loaded,
            path,
            policy="required",
            expected_hidden=16,
            expected_actions=6,
        )
        assert state.action_head_load_status == "LOADED"
        assert state.trained_action_head_active is True
        assert state.action_head_sha256
        assert state.action_head_state_dict_sha256
        assert state.hidden_dim == 16
        assert state.action_dim == 6
        assert state.calibration_dataset_digest == "calib-digest-0001"
        # Strict load actually copied weights.
        assert torch.allclose(loaded.head.weight, head.head.weight)
        assert torch.allclose(loaded.head.bias, head.head.bias)


def test_incompatible_hidden_dim_raises():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "henri_action_head.pt")
        head = make_head()
        torch.save(make_checkpoint(head), path)
        with pytest.raises(ActionHeadError):
            load_action_head(
                make_head(), path, policy="required",
                expected_hidden=32, expected_actions=6,
            )


def test_trained_active_requires_calibration_digest():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "henri_action_head.pt")
        head = make_head()
        ckpt = make_checkpoint(head)
        ckpt["calibration_dataset_digest"] = ""
        torch.save(ckpt, path)
        state = load_action_head(make_head(), path, policy="required")
        assert state.action_head_load_status == "LOADED"
        assert state.trained_action_head_active is False


def test_decode_action_head_returns_legal_action():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "henri_action_head.pt")
        head = make_head()
        torch.save(make_checkpoint(head), path)
        loaded = make_head()
        state = load_action_head(
            loaded, path, policy="required",
            expected_hidden=16, expected_actions=6,
        )
        vocab = make_vocab()
        transducer = StubTransducer()
        wave = make_wave()
        res = decode_action_head(
            transducer, wave, loaded, vocab,
            device="cpu", require_loaded=True, head_state=state,
        )
        assert res.action_name.startswith("ACTION")
        assert res.action_index in range(6)
        assert 0.0 <= res.entropy_bits <= 1.0
        assert len(res.top3) == 3
        assert res.action_probs.shape == (6,)


def test_decode_requires_loaded_head():
    vocab = make_vocab()
    transducer = StubTransducer()
    wave = make_wave()
    state = ActionHeadState(action_head_policy="required")
    with pytest.raises(ActionHeadError):
        decode_action_head(
            transducer, wave, make_head(), vocab,
            device="cpu", require_loaded=True, head_state=state,
        )


def test_unbinder_hidden_shape_and_flatten_boundary():
    transducer = StubTransducer()
    wave = make_wave()
    flat = flatten_uwe(wave, 64)
    assert tuple(flat.shape) == (1, 64)
    h = unbinder_hidden(transducer, flat, device="cpu")
    assert tuple(h.shape) == (1, 16)


def test_head_forward_shape_guard():
    head = make_head()
    with pytest.raises(ActionHeadError):
        head(torch.randn(1, 32))
