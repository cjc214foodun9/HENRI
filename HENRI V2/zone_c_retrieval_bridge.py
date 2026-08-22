# -*- coding: utf-8 -*-
"""Phase 8.37 — default-OFF pgvector retrieval bridge (component C).

Implements HENRI-ANALYSIS-SOTA-BOTTLENECKS-2026 §3.2 "Bridge Wave-JEPA
Planning with Zone C Factual Retrieval": a zero-entropy retrieval
baseplate over the live pgvector engram store.

Contract:
- Default-OFF: the bridge does nothing unless HENRI_ZONEC_BRIDGE == '1'.
  No production consumer activates it; benchmarks/tests opt in.
- Retrieval path: encode query wave -> semantic_projection ->
  TimescaleZoneCStore.query_engrams (HNSW <=> over
  phylogenetic_engrams_65536.semantic_index) -> top-k engram waves.
- Fail-closed: missing flag => bridge disabled (no connect); DB error =>
  typed error; nothing falls back to a surrogate.
- Zero-entropy baseplate claim: retrieval is exact store lookup; the
  bridge never fabricates or generates candidates.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch

from zone_c_segment_cache import (
    TimescaleZoneCStore,
    semantic_projection,
)

BRIDGE_FLAG = "HENRI_ZONEC_BRIDGE"
DEFAULT_TOP_K = 5
DEFAULT_MAX_AGE_HOURS = 24.0 * 365.0  # effectively unbounded (engram store)


def bridge_enabled_from_env(env: Optional[Dict[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    return env.get(BRIDGE_FLAG, "0") == "1"


class ZoneCRetrievalBridge:
    """Default-OFF pgvector retrieval bridge for Wave-JEPA planning."""

    def __init__(
        self,
        dsn: str,
        num_blocks: int = 8192,
        top_k: int = DEFAULT_TOP_K,
        max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
        enabled: Optional[bool] = None,
    ) -> None:
        self.num_blocks = num_blocks
        self.top_k = top_k
        self.max_age_hours = max_age_hours
        self.enabled = bridge_enabled_from_env() if enabled is None else enabled
        if not self.enabled:
            self.store = None
            return
        self.store = TimescaleZoneCStore(dsn=dsn, num_blocks=num_blocks)

    def retrieve(
        self,
        query_wave: torch.Tensor,
        top_k: Optional[int] = None,
        domain_family: Optional[str] = None,
    ) -> List[Tuple[torch.Tensor, float, float]]:
        """Top-k (engram_wave, similarity, age_hours) for the query wave.

        CLASS49 Gate 4: domain_family restricts the candidate pool so action
        tasks never retrieve AST-family engrams (and vice versa).
        Returns [] when the bridge is disabled (default-OFF, no connect).
        Raises DatabaseConnectionError when enabled but the store is down.
        """
        if not self.enabled:
            return []
        if self.store is None:
            raise RuntimeError("bridge enabled but store not constructed")
        k = top_k or self.top_k
        return self.store.query_engrams(query_wave, k, self.max_age_hours,
                                        domain_family)


def retrieve_for_planning(
    query_wave: torch.Tensor,
    dsn: str,
    num_blocks: int = 8192,
    top_k: int = DEFAULT_TOP_K,
    env: Optional[Dict[str, str]] = None,
) -> List[Tuple[torch.Tensor, float, float]]:
    """Stateless helper: construct + retrieve in one call (still default-OFF)."""
    bridge = ZoneCRetrievalBridge(
        dsn=dsn, num_blocks=num_blocks, top_k=top_k,
        enabled=bridge_enabled_from_env(env))
    return bridge.retrieve(query_wave, top_k)


def probe_projection(
    wave: torch.Tensor,
    dim: int = 2000,
) -> torch.Tensor:
    """Expose the semantic projection for diagnostics/tests (same code path)."""
    return semantic_projection(wave)
