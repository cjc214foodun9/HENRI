"""ARC Egress Transducer Contract — Phase 6 (bounded, default-off).

Bridges the continuous UWE wave action path ([num_blocks, 8]) to the
HENRIUnifiedEgressTransducer neural decoder head, producing a COMPLETE
(GameAction, data) action tuple with fail-closed semantics.

Contracts (per Phase 6 task packet and henri-agent-integration):
1. An ARC action is (GameAction, data), not a bare enum. Coordinate-bearing
   actions (ACTION6) receive screen-space payloads via arc_action_payloads.
2. The transducer consumes FLAT [D] waves; ARC candidates are [num_blocks, 8]
   Clifford UWE. A deterministic row-major reshape is the ONLY boundary
   mapping (documented; never a lossy projection).
3. The 32k code-token vocabulary is NOT an action vocabulary. The
   action-legal vocabulary occupies the FIRST N logit positions in a
   deterministic order over the environment's allowed actions. Positions
   >= N are code tokens and are never interpreted as actions.
4. Missing checkpoint / decode failure / illegal shape raise typed
   EgressFailClosedError. There is NO silent bare-enum fallback.
5. SGLD adaptation requires in-context demonstration pairs (X_i, Y_i).
   Absent demos raise NoDemonstrationsError (emitted as
   BLOCKED_NO_DEMONSTRATIONS by the caller). Labels are never bootstrapped
   and pseudo-demonstrations are never fabricated.
6. Adaptation uses the corrected protocol (adapt_in_context_sgld_wave):
   frozen soft targets snapshot before adaptation, scheduled thermal noise
   T(t)=T0(1+0.05t)^-0.55, unit-normalized Langevin increments, Cholesky
   Stiefel retraction, Sagnac term L = CE + 0.25 * (1 - cos(p, p_target)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


class EgressFailClosedError(RuntimeError):
    """Raised when the egress transducer cannot legally produce an action."""


class NoDemonstrationsError(RuntimeError):
    """Raised when SGLD adaptation is requested without demo pairs."""


@dataclass(frozen=True)
class EgressDecodeResult:
    action: object
    action_index: int
    action_name: str
    action_logits: torch.Tensor  # [N] action-legal logits
    action_probs: torch.Tensor   # [N] softmax over action-legal logits
    top3: List[Tuple[str, float]]
    entropy_bits: float          # action-legal logit entropy
    token_entropy_bits: float    # full-vocab logit entropy (diagnostic)


class ActionEgressVocabulary:
    """Deterministic id <-> GameAction mapping over the allowed action set.

    The action-legal logits are the first N positions of the decoder's logit
    vector. Positions >= N are code tokens and are never interpreted as
    actions (fail-closed by construction).
    """

    def __init__(self, action_enum_class: object, allowed_actions: Sequence):
        self.action_enum_class = action_enum_class
        seen = set()
        for a in allowed_actions:
            if not hasattr(a, "name"):
                raise EgressFailClosedError(f"action {a!r} has no .name")
            if a in seen:
                raise EgressFailClosedError(f"duplicate action in allowed set: {a!r}")
            seen.add(a)
        self.actions: List = sorted(set(allowed_actions), key=lambda a: a.name)
        if len(self.actions) != len(set(allowed_actions)):
            raise EgressFailClosedError("allowed set collapsed under sorting")
        self.id_to_action: Dict[int, object] = {
            i: a for i, a in enumerate(self.actions)
        }
        self.action_to_id: Dict[object, int] = {
            a: i for i, a in enumerate(self.actions)
        }

    @property
    def n_actions(self) -> int:
        return len(self.actions)


def flatten_uwe(wave: torch.Tensor, d_model: int) -> torch.Tensor:
    """Deterministic [num_blocks, 8] -> [1, D] row-major flatten.

    D must equal num_blocks * 8 (65536 at CUDA scale, 512 at reduced CPU
    scale). Any other shape raises EgressFailClosedError.
    """
    if wave.dim() != 2 or wave.shape[-1] != 8:
        raise EgressFailClosedError(
            f"expected [num_blocks, 8] UWE, got {tuple(wave.shape)}"
        )
    flat = wave.reshape(-1)
    if flat.numel() != d_model:
        raise EgressFailClosedError(
            f"flatten numel {flat.numel()} != d_model {d_model}"
        )
    return flat.unsqueeze(0).to(torch.float32)


def _full_vocab_entropy(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    return float(-(p * torch.log2(p + 1e-12)).sum().item())


def decode_action_egress(
    transducer: object,
    predicted_wave: torch.Tensor,
    vocab: ActionEgressVocabulary,
    device: str = "cpu",
    require_loaded: bool = True,
) -> EgressDecodeResult:
    """Fail-closed action decode through the transducer head.

    Args:
        transducer: HENRIUnifiedEgressTransducer (or contract-compatible stub).
        predicted_wave: [num_blocks, 8] chosen candidate wave.
        vocab: deterministic action-legal vocabulary for this environment.
        device: target device for the decode.
        require_loaded: if True, raise unless checkpoint_load_status == "LOADED".

    Returns:
        EgressDecodeResult with the legal action, logits, probs, top3,
        action entropy and full-vocab token entropy.
    """
    status = getattr(transducer, "checkpoint_load_status", "SKIPPED_POLICY_DISABLED")
    if require_loaded and status != "LOADED":
        raise EgressFailClosedError(
            f"transducer not LOADED (status={status})"
        )
    d_model = getattr(transducer, "d_model", None)
    if d_model is None:
        raise EgressFailClosedError("transducer has no d_model")
    flat = flatten_uwe(predicted_wave, d_model)
    unbinder = getattr(transducer, "unbinder", None)
    if unbinder is None:
        raise EgressFailClosedError("transducer has no unbinder head")
    with torch.no_grad():
        logits = unbinder.forward(flat.to(device)).to(torch.float32)  # [1, V]
    if logits.shape[-1] < vocab.n_actions:
        raise EgressFailClosedError(
            f"logit vocab {logits.shape[-1]} smaller than action vocab {vocab.n_actions}"
        )
    action_logits = logits[0, : vocab.n_actions]
    probs = torch.softmax(action_logits, dim=-1)
    idx = int(torch.argmax(action_logits).item())
    action = vocab.id_to_action[idx]
    entropy_bits = entropy_bits_of(probs)  # normalized in [0, 1]
    k = min(3, vocab.n_actions)
    top_idx = torch.topk(action_logits, k=k).indices.tolist()
    top3 = [(vocab.id_to_action[i].name, float(probs[i])) for i in top_idx]
    return EgressDecodeResult(
        action=action,
        action_index=idx,
        action_name=action.name,
        action_logits=action_logits.detach().cpu(),
        action_probs=probs.detach().cpu(),
        top3=top3,
        entropy_bits=entropy_bits,
        token_entropy_bits=_full_vocab_entropy(logits[0]),
    )


def adapt_sgld_from_demos(
    transducer: object,
    demo_pairs: Sequence[Tuple],
    tokenizer: object,
    device: str = "cpu",
    steps: int = 500,
    seed: int = 0,
    d_model: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """Online test-time SGLD adaptation on in-context demo pairs.

    Encodes each (input, output) grid through the production tokenizer path
    (encode_spatial_grid), flattens [num_blocks, 8] -> [D], and runs
    adapt_in_context_sgld_wave (corrected protocol).

    Returns None when steps <= 0. Raises NoDemonstrationsError when
    demo_pairs is empty.
    """
    if steps <= 0:
        return None
    if not demo_pairs:
        raise NoDemonstrationsError(
            "no in-context demonstration pairs available for SGLD adaptation"
        )
    unbinder = getattr(transducer, "unbinder", None)
    if unbinder is None or not hasattr(unbinder, "adapt_in_context_sgld_wave"):
        raise EgressFailClosedError("transducer unbinder lacks adapt_in_context_sgld_wave")
    encode = getattr(tokenizer, "encode_spatial_grid", None)
    if encode is None:
        raise EgressFailClosedError("tokenizer lacks encode_spatial_grid")
    active: List[torch.Tensor] = []
    target: List[torch.Tensor] = []
    for x, y in demo_pairs:
        _x = x.tolist() if hasattr(x, "tolist") else x
        _y = y.tolist() if hasattr(y, "tolist") else y
        wx = encode(_x).squeeze(0)
        wy = encode(_y).squeeze(0)
        if wx.dim() == 2 and wx.shape[-1] == 8:
            wx = wx.reshape(-1)
        if wy.dim() == 2 and wy.shape[-1] == 8:
            wy = wy.reshape(-1)
        active.append(wx)
        target.append(wy)
    A = torch.stack(active).to(device).to(torch.float32)
    T_ = torch.stack(target).to(device).to(torch.float32)
    metrics = unbinder.adapt_in_context_sgld_wave(A, T_, steps=steps, seed=seed)
    if isinstance(metrics, dict):
        metrics["demo_pair_count"] = len(demo_pairs)
        metrics["steps"] = steps
    return metrics


def reset_decoder_optimizer(transducer: object) -> bool:
    """Per-episode decoder reset: fresh AdamW optimizer state.

    Prevents SGLD adaptation from leaking across episodes. Returns True when
    the reset happened.
    """
    unbinder = getattr(transducer, "unbinder", None)
    if unbinder is None or not hasattr(unbinder, "parameters"):
        return False
    unbinder.optimizer = torch.optim.AdamW(
        unbinder.parameters(), lr=1e-3, weight_decay=1e-4
    )
    return True


def entropy_bits_of(probs: torch.Tensor) -> float:
    """Normalized entropy in bits over the action-legal distribution."""
    if probs.numel() <= 1:
        return 0.0
    h = float(-(probs * torch.log2(probs + 1e-12)).sum().item())
    return h / math.log2(probs.numel()) if probs.numel() > 1 else h
