"""Typed text ingress adapter (U2) — verified structured codec protocol.

Uses the goal-adapter Channel T codec: circular-mean phasor bundling with
odd position keys (q_char*(2p+1)) mod 256; per-block unit rows; gates
C1-C5+C7 verified 2026-08-25. NEVER qFHRREpistemicCodec.encode_text
(random-ring, non-compositional; measured 2026-08-02 run20).
"""
from __future__ import annotations

from typing import Optional

import torch

from ..envelope import CANONICAL_BLOCK_DIM, CANONICAL_NUM_BLOCKS, WavePacket

try:
    from henri_goal_adapter import HenriPromptCodec
except Exception:  # pragma: no cover - import failure must be loud at use
    HenriPromptCodec = None  # type: ignore

CODEC_ID = "henri_prompt_qfhrr.v1"


class TextWaveAdapter:
    modality = "text"
    media_type = "text/plain"

    def __init__(self, device: Optional[str] = None) -> None:
        if HenriPromptCodec is None:
            raise ImportError("henri_goal_adapter.HenriPromptCodec unavailable")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.codec = HenriPromptCodec(
            num_blocks=CANONICAL_NUM_BLOCKS,
            block_dim=CANONICAL_BLOCK_DIM,
            device=self.device,
        )
        self.encoder_sha256 = self.codec.manifest_sha256()

    def encode(
        self,
        text: str,
        *,
        source_uri: str = "",
        source_sha256: str = "",
        item_id: str = "",
        causal_availability: str = "query_only",
    ) -> WavePacket:
        wave = self.codec.encode_prompt(text)
        return WavePacket(
            modality=self.modality,
            media_type=self.media_type,
            source_uri=source_uri,
            source_sha256=source_sha256,
            item_id=item_id,
            encoder_id=CODEC_ID,
            encoder_sha256=self.encoder_sha256,
            wave=wave,
            causal_availability=causal_availability,
            leakage_class="model_facing",
            evaluator_isolation=True,
        )
