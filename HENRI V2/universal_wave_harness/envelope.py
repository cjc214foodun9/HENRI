"""Universal Evidence Envelope — WavePacket contract (U1).

Canonical wave boundary: real [num_blocks, 8] (Cl(3,0) blocks, D = num_blocks*8).
Every packet carries mandatory provenance: source-byte sha, encoder id + sha,
layout, norm rule, causal availability, leakage class, evaluator isolation.
Raw source strings/bytes are NEVER part of a packet and never persist to Zone C.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Tuple

import torch

CANONICAL_NUM_BLOCKS = 8192
CANONICAL_BLOCK_DIM = 8
SCHEMA_ID = "henri.wavepacket.v1"

EVALUATOR_ONLY_FIELDS = frozenset({
    "answer", "reference", "gold", "rubric", "rubric_json", "rubric_pretty",
    "correct_answer", "label", "target",
})


class UnsupportedModalityError(RuntimeError):
    """Fail-closed: no typed adapter exists for the requested modality."""


class EvaluatorLeakError(RuntimeError):
    """Fail-closed: evaluator-only content reached a model-facing object."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def assert_no_evaluator_fields(obj: dict) -> None:
    """Structural quarantine: reject evaluator-only fields in model-facing objects."""
    present = sorted(set(obj.keys()) & EVALUATOR_ONLY_FIELDS)
    if present:
        raise EvaluatorLeakError(
            f"evaluator-only fields in model-facing object: {present}")


@dataclass(frozen=True)
class WavePacket:
    schema_id: str = SCHEMA_ID
    modality: str = ""
    media_type: str = ""
    source_uri: str = ""
    source_sha256: str = ""
    item_id: str = ""
    encoder_id: str = ""
    encoder_sha256: str = ""
    wave: Any = field(repr=False, default=None)
    row_norms: Tuple[float, ...] = field(default=(), repr=False)
    layout: str = "clifford_blocks"
    dtype: str = ""
    device: str = ""
    norm_rule: str = "per_block_unit_rows"
    quant_rule: str = "none"
    causal_availability: str = ""
    leakage_class: str = "model_facing"
    evaluator_isolation: bool = True
    created_utc: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_utc", self.created_utc or _utcnow())
        self._reject_evaluator_fields()
        if self.wave is not None:
            self._validate_wave()

    def _reject_evaluator_fields(self) -> None:
        present = sorted(set(asdict(self).keys()) & EVALUATOR_ONLY_FIELDS)
        if present:
            raise EvaluatorLeakError(
                f"evaluator-only fields present in model-facing packet: {present}")

    def _validate_wave(self) -> None:
        t = self.wave
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"wave must be torch.Tensor, got {type(t).__name__}")
        if tuple(t.shape) != (CANONICAL_NUM_BLOCKS, CANONICAL_BLOCK_DIM):
            raise ValueError(
                f"wave shape {tuple(t.shape)} != canonical "
                f"{(CANONICAL_NUM_BLOCKS, CANONICAL_BLOCK_DIM)}")
        norms = t.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
            raise ValueError("per-block unit-row norm contract violated")
        object.__setattr__(self, "row_norms", tuple(float(x) for x in norms))
        object.__setattr__(self, "dtype", str(t.dtype))
        object.__setattr__(self, "device", str(t.device))

    def to_dict(self, include_wave_sha: bool = False) -> dict:
        d = asdict(self)
        d.pop("wave", None)
        d["row_norms"] = {
            "count": len(self.row_norms),
            "min": min(self.row_norms) if self.row_norms else None,
            "max": max(self.row_norms) if self.row_norms else None,
        }
        if include_wave_sha and self.wave is not None:
            d["wave_sha256"] = hashlib.sha256(
                self.wave.detach().cpu().numpy().tobytes()).hexdigest()
        return d

    def provenance_digest(self) -> str:
        d = self.to_dict(include_wave_sha=False)
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
