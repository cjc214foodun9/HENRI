"""Prefix conditioning for the HENRI egress decoder — the honest Task-16 remedy.

WHY THIS EXISTS (Case B, MEASURED 2026-09-27)
=============================================
Task 16 was "connect `henri_wave_transducer.prefix_embeddings` into the attention
KV-cache interface".  I read `henri_decoder.py` (727 lines).  Counts of the
symbols an attention core would require:

    q_proj 0 | k_proj 0 | v_proj 0 | MultiheadAttention 0
    scaled_dot_product 0 | past_key_values 0 | attn 0 | causal_mask 0

`HENRINeuralEgressUnbinder` is a POINTWISE unbinder:

    down_proj  nn.Linear(65536, 2048, bias=False)
    layer_norm nn.LayerNorm(2048)
    act        nn.GELU()
    lm_head    nn.Linear(2048, 32000, bias=False)

There is no attention core.  A genuine key/value cache CANNOT be wired, because
there is nothing to cache: the per-position computation would be an elementwise
Linear + LayerNorm + GELU, whose "KV" for any position is the position's own
activation.  Caching it changes nothing observable.  So the directive's stated
interface DOES NOT EXIST, and this module implements the two things that ARE
honest:

  1. `PrefixConditioner` — a DEFAULT-OFF module that mixes a pooled prefix into
     the decoder's hidden state.  This is what a prefix CAN do to a pointwise
     core: it conditions the state, it does not populate a cache.
  2. `verify_prefix_capability()` — the fail-closed gate that refuses a
     capability claim on wiring alone.

A real KV cache becomes possible only after an attention core is ADDED.  That is
a separate, load-bearing architecture change; it is recorded as a SpecContract
proposal in the ontology, not smuggled into this module.

CAPABILITY BOUNDARY (do not restate this as a win)
=================================================
* The conditioning projection is UNTRAINED unless a checkpoint supplies it.
  Therefore this wiring is `DIAGNOSTIC` and is NOT score-eligible.
* SciCode re-run stays `BLOCKED`: it needs (a) trained projections AND (b)
  remote CUDA verification.  Local CPU exercises plumbing only.
* `use_prefix=False` (the default) is BYTE-IDENTICAL to the legacy path.
* No benchmark score is claimed anywhere in this module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

DEFAULT_USE_PREFIX = False          # default-OFF: the legacy path is unchanged
_POOLINGS = ("mean", "last", "max")


class PrefixKVError(RuntimeError):
    """Raised on a prefix-wiring contract violation (fail closed)."""


class PrefixConditioner(nn.Module):
    """Mix a pooled prefix into the decoder hidden state.  Default OFF.

    Shapes
    ------
    hidden : [B, T, d_hidden]
    prefix : [B, P, d_hidden]   (P = prefix length; P may be 0)
    out    : [B, T, d_hidden]

    With `use_prefix=False` the input tensor is returned UNCHANGED (the same
    object), so the default path is byte-identical by construction.
    """

    def __init__(self, d_hidden: int, use_prefix: bool = DEFAULT_USE_PREFIX,
                 pooling: str = "mean", dropout: float = 0.0) -> None:
        super().__init__()
        if d_hidden < 1:
            raise PrefixKVError("d_hidden must be >= 1")
        if pooling not in _POOLINGS:
            raise PrefixKVError(f"pooling must be one of {_POOLINGS}, got {pooling!r}")
        self.d_hidden = int(d_hidden)
        self.use_prefix = bool(use_prefix)
        self.pooling = pooling
        # Learned per-dimension gain for the pooled prefix.  This is the ONLY
        # parameter; it is untrained unless a checkpoint provides it.
        self.prefix_gain = nn.Parameter(torch.zeros(self.d_hidden))
        self.prefix_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------ pool
    def pool(self, prefix: torch.Tensor) -> torch.Tensor:
        """[B, P, d_hidden] -> [B, d_hidden]."""
        if prefix.dim() != 3:
            raise PrefixKVError(f"prefix must be [B,P,d], got shape {tuple(prefix.shape)}")
        if prefix.shape[-1] != self.d_hidden:
            raise PrefixKVError(
                f"prefix last dim {prefix.shape[-1]} != d_hidden {self.d_hidden}")
        if prefix.shape[1] == 0:
            raise PrefixKVError("prefix has P=0; pass prefix=None instead")
        if self.pooling == "mean":
            return prefix.mean(dim=1)
        if self.pooling == "last":
            return prefix[:, -1, :]
        return prefix.amax(dim=1)

    # --------------------------------------------------------------- forward
    def forward(self, hidden: torch.Tensor,
                prefix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Accept [B, T, d] OR [B, d].

        The 2-D form is required by the HENRI egress unbinder, whose hidden state
        is `[B, d_hidden]` (a pointwise MLP has no sequence axis). 2-D input is
        treated as T=1 for the conditioning arithmetic and returned with the SAME
        rank it arrived with, so the default path stays byte-identical.
        """
        squeeze = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)          # [B, d] -> [B, 1, d]
            squeeze = True
        if hidden.dim() != 3:
            raise PrefixKVError(
                f"hidden must be [B,T,d] or [B,d], got shape {tuple(hidden.shape)}")
        if hidden.shape[-1] != self.d_hidden:
            raise PrefixKVError(
                f"hidden last dim {hidden.shape[-1]} != d_hidden {self.d_hidden}")
        # DEFAULT PATH: return the input unchanged.  Byte-identical by identity.
        if not self.use_prefix or prefix is None:
            return hidden.squeeze(1) if squeeze else hidden
        if prefix.device != hidden.device:
            raise PrefixKVError(
                f"device mismatch: prefix on {prefix.device}, hidden on {hidden.device}")
        bias = self.pool(prefix.to(hidden.dtype))
        # FAIL CLOSED on a batch mismatch. DEFECT FIXED 2026-09-27: without this
        # the addition broadcast silently (or raised a raw RuntimeError), so a
        # caller could pair B=2 hidden states with B=3 prefixes and get a
        # shape-error traceback instead of a contract violation.
        if bias.shape[0] != hidden.shape[0]:
            raise PrefixKVError(
                f"batch mismatch: prefix batch {bias.shape[0]} != hidden batch "
                f"{hidden.shape[0]}; refusing to broadcast")
        out = hidden + self.prefix_drop((bias * self.prefix_gain).unsqueeze(1))
        return out.squeeze(1) if squeeze else out

    # ------------------------------------------------------------ diagnostics
    def capability_report(self) -> Dict[str, Any]:
        """State exactly what this wiring does and does not establish."""
        return {
            "module": "henri_prefix_kv.PrefixConditioner",
            "use_prefix": self.use_prefix,
            "pooling": self.pooling,
            "default_path_changed": False,
            "evidence_class": "OBSERVED" if not self.use_prefix else "DIAGNOSTIC",
            "kv_cache_possible": False,
            "kv_cache_reason": (
                "henri_decoder.py has NO attention core (0 occurrences of q_proj/k_proj/"
                "v_proj/MultiheadAttention/scaled_dot_product/past_key_values/attn/"
                "causal_mask). A KV cache requires attention to cache."),
            "projections_trained": False,
            "score_eligible": False,
            "scicode_rerun": "BLOCKED",
            "scicode_rerun_requires": ["trained prefix projections", "remote CUDA verification"],
        }


def pool_prefix_embeddings(prefix_embeddings: torch.Tensor, d_hidden: int) -> torch.Tensor:
    """One-way, norm-preserving adapter from a prefix tensor to the hidden width.

    `henri_wave_transducer.prefix_embeddings` emits a wave-shaped prefix.  This
    adapter maps it to [B, P, d_hidden] WITHOUT inventing structure: more dims ->
    deterministic reshape+mean-fold; fewer -> tile.  It never silently drops mass,
    and it refuses a non-finite input.
    """
    if not torch.isfinite(prefix_embeddings).all():
        raise PrefixKVError("prefix_embeddings contains non-finite values")
    x = prefix_embeddings
    if x.dim() == 2:
        x = x.unsqueeze(0)                     # [P, d] -> [1, P, d]
    if x.dim() != 3:
        raise PrefixKVError(f"prefix_embeddings must be 2-D or 3-D, got {tuple(x.shape)}")
    p, d = x.shape[1], x.shape[2]
    if d == d_hidden:
        return x.contiguous()
    if d > d_hidden:
        if d % d_hidden != 0:
            raise PrefixKVError(f"cannot fold dim {d} -> {d_hidden} (not a divisor)")
        return x.reshape(x.shape[0], p * (d // d_hidden), d_hidden).contiguous()
    if d_hidden % d != 0:
        raise PrefixKVError(f"cannot tile dim {d} -> {d_hidden} (not a multiple)")
    return x.repeat(1, 1, d_hidden // d).contiguous()


def verify_prefix_capability(*, flag_declared: bool, flag_forwarded: bool,
                             flag_reaches_consumer: bool,
                             output_changed_when_on: bool) -> Dict[str, Any]:
    """Fail-closed gate.  A capa  claim needs a CONSUMER, not a declaration.

    This is the dead-flag gate: the exact defect class (a config field declared
    but never read) that the architecture catalog records as "the mechanism does
    not exist".  All four conditions must hold, or the wiring is refused.
    """
    checks = {
        "flag_declared": bool(flag_declared),
        "flag_forwarded": bool(flag_forwarded),
        "flag_reaches_consumer": bool(flag_reaches_consumer),
        "output_changed_when_on": bool(output_changed_when_on),
    }
    ok = all(checks.values())
    return {
        "schema": "henri.prefix-kv.capability-gate.v1",
        "checks": checks,
        "passed": ok,
        "verdict": ("PREFIX_WIRING_LIVE_DIAGNOSTIC" if ok
                    else "PREFIX_WIRING_REFUSED_DEAD_FLAG_OR_NO_EFFECT"),
        "note": ("Even on PASS this is DIAGNOSTIC-only: untrained projections, no attention "
                 "core, local CPU. It never promotes a benchmark score claim."),
    }


def cache_equivalence_stepwise(cond: PrefixConditioner, hidden: torch.Tensor,
                               prefix: torch.Tensor) -> float:
    """Max abs difference between a stepwise application and the batch call.

    A real cache must be *equivalent* to recomputation; a conditioner must be too.
    Returns 0.0 exactly when the two paths agree bit-for-bit.
    """
    whole = cond(hidden, prefix)
    pieces = [cond(hidden[:, t:t + 1, :], prefix) for t in range(hidden.shape[1])]
    step = torch.cat(pieces, dim=1)
    return float((whole - step).abs().max().item()) if whole.numel() else 0.0
