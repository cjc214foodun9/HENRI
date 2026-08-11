"""ARC Semantic Action Head — Phase 7 (bounded, default-off).

Decouples the ARC action space from the legacy 32k text-token vocabulary.
A dedicated Linear(2048 -> |A|) projection maps the transducer unbinder's
intermediate feature states (d_hidden=2048) directly to the bounded set of
legal ARC operations.

Contracts (per Phase 7 task packet and henri-agent-integration):
1. The action head is a SEPARATE projection from the token lm_head. It is
   never initialized and labeled "calibrated": trained_action_head_active
   is True only when a checkpoint with provenance (file sha256, state-dict
   sha256, exact dims, calibration_dataset_digest) loaded successfully.
2. checkpoint-policy discipline: "required" for production egress /
   score-bearing evaluation (missing/corrupt/incompatible raises typed
   ActionHeadError), "disabled" for reduced CPU tests (never deserialize
   the production artifact).
3. Absent action-head checkpoint -> eligibility False. First-N token-logit
   relabeling is diagnostic plumbing only and never flips eligibility.
4. Coordinate-bearing ACTION6 still requires screen-space payload
   calibration (arc_action_payloads); enum classification alone is not
   sufficient for ACTION6.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn


class ActionHeadError(RuntimeError):
    """Raised when the action head cannot legally produce an action."""


@dataclass
class ActionHeadState:
    """Deterministic telemetry record for the action head."""

    action_head_policy: str = "disabled"
    action_head_load_status: str = "SKIPPED_POLICY_DISABLED"
    action_head_sha256: Optional[str] = None
    action_head_state_dict_sha256: Optional[str] = None
    input_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    action_dim: Optional[int] = None
    trained_action_head_active: bool = False
    calibration_dataset_digest: Optional[str] = None
    action_head_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @staticmethod
    def fingerprint_keys() -> Sequence[str]:
        return (
            "action_head_policy",
            "action_head_load_status",
            "action_head_sha256",
            "action_head_state_dict_sha256",
            "input_dim",
            "hidden_dim",
            "action_dim",
            "trained_action_head_active",
            "calibration_dataset_digest",
            "action_head_path",
        )


class ActionHead(nn.Module):
    """Linear projection d_hidden=2048 -> |A| legal ARC actions.

    Consumes the unbinder's intermediate feature states (post down_proj,
    layer_norm, GELU). It is deliberately small (single Linear) so its
    calibration is auditable and its weights are provenance-tracked.
    """

    def __init__(self, d_hidden: int = 2048, n_actions: int = 6):
        super().__init__()
        if d_hidden <= 0 or n_actions <= 0:
            raise ActionHeadError(
                f"invalid dims d_hidden={d_hidden} n_actions={n_actions}"
            )
        self.d_hidden = d_hidden
        self.n_actions = n_actions
        self.head = nn.Linear(d_hidden, n_actions, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [*, d_hidden] -> action logits [*, n_actions]."""
        if hidden.shape[-1] != self.d_hidden:
            raise ActionHeadError(
                f"hidden dim {hidden.shape[-1]} != d_hidden {self.d_hidden}"
            )
        return self.head(hidden)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in (
        "d_model",
        "hidden_dim",
        "action_dim",
        "calibration_dataset_digest",
        "schema_id",
    ):
        if k in metadata:
            out[k] = metadata[k]
    return out


def load_action_head(
    head: ActionHead,
    checkpoint_path: str,
    *,
    policy: str = "required",
    expected_hidden: Optional[int] = None,
    expected_actions: Optional[int] = None,
    calibration_dataset_digest: Optional[str] = None,
) -> ActionHeadState:
    """Strict, provenance-checked load of the action-head checkpoint.

    Raises ActionHeadError for missing/corrupt/incompatible artifacts when
    policy == "required". Returns a SKIPPED_POLICY_DISABLED state when
    policy == "disabled" without touching the artifact.
    """
    state = ActionHeadState(action_head_policy=policy)
    if policy == "disabled":
        return state
    if policy not in ("required", "auto"):
        raise ActionHeadError(f"unknown action_head_policy: {policy!r}")

    path = Path(checkpoint_path)
    state.action_head_path = str(path)
    if not path.is_file():
        if policy == "required":
            raise ActionHeadError(f"action head checkpoint missing: {path}")
        state.action_head_load_status = "SKIPPED_NO_CHECKPOINT"
        return state

    raw = path.read_bytes()
    state.action_head_sha256 = _sha256_bytes(raw)

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - torch load failure
        if policy == "required":
            raise ActionHeadError(f"action head corrupt: {exc}") from exc
        state.action_head_load_status = "LOAD_FAILED"
        return state

    if not isinstance(ckpt, dict):
        if policy == "required":
            raise ActionHeadError("action head checkpoint is not a dict")
        state.action_head_load_status = "LOAD_FAILED"
        return state

    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", None))
    if sd is None:
        if policy == "required":
            raise ActionHeadError("action head checkpoint has no state_dict")
        state.action_head_load_status = "LOAD_FAILED"
        return state

    required_keys = {"head.weight", "head.bias"}
    if not required_keys.issubset(sd.keys()):
        missing = required_keys - set(sd.keys())
        if policy == "required":
            raise ActionHeadError(
                f"action head state_dict missing keys: {sorted(missing)}"
            )
        state.action_head_load_status = "LOAD_FAILED"
        return state

    w = sd["head.weight"]
    b = sd["head.bias"]
    if w.dim() != 2 or b.dim() != 1 or w.shape[0] != b.shape[0]:
        raise ActionHeadError("action head weight/bias shape mismatch")
    n_actions, hidden = w.shape
    if expected_hidden is not None and hidden != expected_hidden:
        raise ActionHeadError(
            f"action head hidden {hidden} != expected {expected_hidden}"
        )
    if expected_actions is not None and n_actions != expected_actions:
        raise ActionHeadError(
            f"action head actions {n_actions} != expected {expected_actions}"
        )

    # Exact-dimension strict load; never strict=False to hide a mismatch.
    head.head.weight.data.copy_(w)
    head.head.bias.data.copy_(b)

    state.action_head_load_status = "LOADED"
    state.input_dim = int(ckpt.get("d_model", w.shape[1]))
    state.hidden_dim = int(hidden)
    state.action_dim = int(n_actions)
    state.calibration_dataset_digest = str(
        ckpt.get(
            "calibration_dataset_digest",
            calibration_dataset_digest or "",
        )
    )
    # Trained-active requires provenance: artifact hash + digest + exact dims.
    state.trained_action_head_active = bool(
        state.action_head_sha256
        and state.calibration_dataset_digest
        and head.d_hidden == state.hidden_dim
        and head.n_actions == state.action_dim
    )
    # Deterministic state-dict hash (sorted keys, canonical tensor bytes).
    state.action_head_state_dict_sha256 = _state_dict_sha256(sd)
    return state


def _state_dict_sha256(sd: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def unbinder_hidden(
    transducer: object, flat_wave: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """Extract the [1, d_hidden] intermediate feature state.

    Mirrors HENRINeuralEgressUnbinder.forward up to the vocabulary head:
    normalize -> down_proj -> layer_norm -> GELU. No lm_head call.
    """
    unbinder = getattr(transducer, "unbinder", None)
    if unbinder is None or not hasattr(unbinder, "down_proj"):
        raise ActionHeadError("transducer has no unbinder head")
    with torch.no_grad():
        x = flat_wave.to(device).to(torch.float32)
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-8
        unit = x / norm
        h = unbinder.down_proj(unit)
        h = unbinder.layer_norm(h)
        h = unbinder.act(h)
    return h


def decode_action_head(
    transducer: object,
    predicted_wave: torch.Tensor,
    action_head: ActionHead,
    vocab: object,
    *,
    device: str = "cpu",
    require_loaded: bool = True,
    head_state: Optional[ActionHeadState] = None,
) -> Any:
    """Fail-closed action decode through the calibrated action head.

    Returns a dataclass-shaped result (action, action_index, action_name,
    action_logits, action_probs, top3, entropy_bits) compatible with
    EgressDecodeResult consumers.
    """
    if head_state is not None:
        if require_loaded and head_state.action_head_load_status != "LOADED":
            raise ActionHeadError(
                f"action head not LOADED (status={head_state.action_head_load_status})"
            )
        if require_loaded and not head_state.trained_action_head_active:
            raise ActionHeadError(
                "action head not trained-active (no calibration provenance)"
            )
    d_model = getattr(transducer, "d_model", None)
    if d_model is None:
        raise ActionHeadError("transducer has no d_model")

    from arc_egress_contract import flatten_uwe

    flat = flatten_uwe(predicted_wave, d_model)
    hidden = unbinder_hidden(transducer, flat, device=device)
    with torch.no_grad():
        logits = action_head(hidden).to(torch.float32)  # [1, |A|]
    if logits.shape[-1] != action_head.n_actions:
        raise ActionHeadError(
            f"head logits {logits.shape[-1]} != n_actions {action_head.n_actions}"
        )
    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.argmax(logits, dim=-1).item())
    if idx >= len(vocab.id_to_action):
        raise ActionHeadError(
            f"action index {idx} outside legal vocab "
            f"({len(vocab.id_to_action)} actions)"
        )
    action = vocab.id_to_action[idx]

    from arc_egress_contract import _full_vocab_entropy, entropy_bits_of

    # Diagnostic parity: token entropy over the full 32k vocab via the
    # unbinder's lm_head (extra no-grad pass; only meaningful for TOKEN_HEAD
    # comparison, never for action semantics). Guarded for stubs without lm_head.
    _unb = getattr(transducer, "unbinder", None)
    if _unb is not None and hasattr(_unb, "lm_head"):
        with torch.no_grad():
            full_logits = _unb.forward(flat.to(device)).to(torch.float32)
        token_entropy_bits = _full_vocab_entropy(full_logits[0])
    else:
        token_entropy_bits = None

    k = min(3, action_head.n_actions)
    top_idx = torch.topk(logits[0], k=k).indices.tolist()
    top3 = [(vocab.id_to_action[i].name, float(probs[0, i])) for i in top_idx]
    return type(
        "ActionHeadDecodeResult",
        (),
        {
            "action": action,
            "action_index": idx,
            "action_name": action.name,
            "action_logits": logits[0].detach().cpu(),
            "action_probs": probs[0].detach().cpu(),
            "top3": top3,
            "entropy_bits": entropy_bits_of(probs[0]),
            "token_entropy_bits": token_entropy_bits,
        },
    )()
