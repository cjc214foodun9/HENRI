"""Phase 6 contract tests: ARC egress transducer wiring (CPU, checkpoint disabled).

Proves, on CPU with fake environments and a contract-compatible stub
transducer:
1. flatten_uwe maps [num_blocks, 8] -> [1, D] exactly and fails closed on
   illegal shapes.
2. ActionEgressVocabulary is deterministic, deduplicated, and sorted.
3. decode_action_egress returns a LEGAL action inside the allowed set, with
   top3 and both entropies; never a code-token position.
4. decode_action_egress fails closed when the transducer is not LOADED.
5. decode_action_egress fails closed when the logit vocabulary is smaller
   than the action vocabulary.
6. adapt_sgld_from_demos raises NoDemonstrationsError on empty demos and
   returns None when steps <= 0.
7. reset_decoder_optimizer installs a fresh AdamW optimizer.
8. Default path stability: no egress flag set -> no egress imports at module
   level (the contract module is importable standalone on CPU).
"""

import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "HENRI V2"))

from arc_egress_contract import (  # noqa: E402
    ActionEgressVocabulary,
    EgressFailClosedError,
    NoDemonstrationsError,
    adapt_sgld_from_demos,
    decode_action_egress,
    flatten_uwe,
    reset_decoder_optimizer,
)

D_MODEL = 64  # 8 blocks x 8 channels (reduced CPU scale)


class _FakeAction:
    def __init__(self, name):
        self.name = name
        self.value = 1

    def __repr__(self):
        return f"GameAction.{self.name}"


ACTIONS = [_FakeAction("ACTION6"), _FakeAction("ACTION1"), _FakeAction("ACTION3")]


class _FakeUnbinder(torch.nn.Module):
    """Mini unbinder matching the contract surface (no production weights)."""

    def __init__(self, d_model=D_MODEL, d_hidden=8, vocab_size=32000):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.down_proj = torch.nn.Linear(d_model, d_hidden, bias=False)
        self.layer_norm = torch.nn.LayerNorm(d_hidden)
        self.act = torch.nn.GELU()
        self.lm_head = torch.nn.Linear(d_hidden, vocab_size, bias=False)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(self, wave, w_task=None):
        return self.lm_head(self.act(self.layer_norm(self.down_proj(wave))))

    def adapt_in_context_sgld_wave(self, active, target, steps=3, seed=0, **kw):
        return {
            "adapt_protocol": "wave_soft_targets_scheduled_sgld",
            "steps": steps,
            "avg_loss": 0.0,
            "yield_events": 0,
            "demo_pair_count": 0,
        }


class _FakeTransducer:
    def __init__(self, d_model=D_MODEL, status="LOADED"):
        self.d_model = d_model
        self.checkpoint_load_status = status
        self.checkpoint_sha256 = "c0ffee" * 8
        self.checkpoint_state_dict_sha256 = "b00b5" * 12 + "beef"
        self.unbinder = _FakeUnbinder(d_model=d_model)


class _FakeTokenizer:
    def __init__(self, blocks=8):
        self.blocks = blocks

    def encode_spatial_grid(self, grid):
        # deterministic [blocks, 8] wave
        w = torch.linspace(-1, 1, self.blocks * 8).reshape(self.blocks, 8)
        return w.unsqueeze(0)


def _wave(blocks=8):
    return torch.linspace(-1, 1, blocks * 8).reshape(blocks, 8)


def test_flatten_uwe_exact():
    w = _wave()
    flat = flatten_uwe(w, D_MODEL)
    assert flat.shape == (1, D_MODEL)
    assert torch.allclose(flat[0], w.reshape(-1))
    assert flat.dtype == torch.float32


def test_flatten_uwe_fails_on_illegal_shapes():
    with pytest.raises(EgressFailClosedError):
        flatten_uwe(torch.zeros(4, 4), D_MODEL)
    with pytest.raises(EgressFailClosedError):
        flatten_uwe(torch.zeros(8, 8), 512)  # numel mismatch
    with pytest.raises(EgressFailClosedError):
        flatten_uwe(torch.zeros(8), D_MODEL)  # rank 1


def test_vocabulary_deterministic_dedup_sorted():
    v = ActionEgressVocabulary(_FakeAction, ACTIONS)
    v2 = ActionEgressVocabulary(_FakeAction, list(reversed(ACTIONS)))
    assert v.n_actions == 3
    assert [a.name for a in v.actions] == ["ACTION1", "ACTION3", "ACTION6"]
    assert v.id_to_action == v2.id_to_action
    with pytest.raises(EgressFailClosedError):
        ActionEgressVocabulary(_FakeAction, [ACTIONS[0], ACTIONS[0]])


def test_decode_returns_legal_action_and_entropies():
    t = _FakeTransducer(status="LOADED")
    v = ActionEgressVocabulary(_FakeAction, ACTIONS)
    r = decode_action_egress(t, _wave(), v, device="cpu", require_loaded=True)
    assert r.action in ACTIONS
    assert r.action_index in v.id_to_action
    assert len(r.top3) == 3
    assert all(isinstance(n, str) and isinstance(p, float) for n, p in r.top3)
    assert 0.0 <= r.entropy_bits <= 1.0  # normalized action entropy
    assert r.token_entropy_bits > 0.0    # full-vocab diagnostic
    assert r.action_probs.shape == (3,)


def test_decode_fails_closed_when_not_loaded():
    t = _FakeTransducer(status="SKIPPED_NO_CHECKPOINT")
    v = ActionEgressVocabulary(_FakeAction, ACTIONS)
    with pytest.raises(EgressFailClosedError):
        decode_action_egress(t, _wave(), v, device="cpu", require_loaded=True)


def test_decode_fails_closed_when_vocab_smaller_than_actions():
    t = _FakeTransducer(d_model=D_MODEL, status="LOADED")
    # Force logits smaller than the action vocab by using a stub unbinder
    # whose vocab is 2 while we request 3 actions.
    class _TinyUnbinder(_FakeUnbinder):
        def __init__(self, d_model):
            super().__init__(d_model=d_model, vocab_size=2)

    t.unbinder = _TinyUnbinder(D_MODEL)
    v = ActionEgressVocabulary(_FakeAction, ACTIONS)
    with pytest.raises(EgressFailClosedError):
        decode_action_egress(t, _wave(), v, device="cpu", require_loaded=True)


def test_sgld_requires_demos():
    t = _FakeTransducer(status="LOADED")
    tok = _FakeTokenizer()
    with pytest.raises(NoDemonstrationsError):
        adapt_sgld_from_demos(t, [], tok, steps=10)
    assert adapt_sgld_from_demos(t, [], tok, steps=0) is None


def test_sgld_runs_on_demos():
    t = _FakeTransducer(status="LOADED")
    tok = _FakeTokenizer()
    demos = [([[0, 1], [1, 0]], [[1, 0], [0, 1]]), ([[0, 0], [0, 0]], [[1, 1], [1, 1]])]
    m = adapt_sgld_from_demos(t, demos, tok, steps=4, seed=7)
    assert m is not None
    assert m["demo_pair_count"] == 2
    assert m["steps"] == 4


def test_reset_decoder_optimizer_fresh():
    t = _FakeTransducer(status="LOADED")
    old = t.unbinder.optimizer
    assert reset_decoder_optimizer(t) is True
    assert t.unbinder.optimizer is not old


def test_contract_module_imports_without_arcengine():
    # The contract module must not require arcengine (CPU-safe import).
    import arc_egress_contract  # noqa: F401
    assert hasattr(arc_egress_contract, "decode_action_egress")
