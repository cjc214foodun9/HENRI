# -*- coding: utf-8 -*-
"""CLASS49 unit tests — Zone C attribution guard + namespace family isolation.

Packet: HENRI-PACKET-CLASS49-ATTRIBUTION-SAGNAC-2026.
Covers:
- canonical_domain_family mapping over live domain vocabulary
- assert_attribution fail-closed raises (Gate 1)
- SegmentCache.retrieve domain_family filtering on the surrogate store
  (Gate 4 behavior on the in-process backend; Timescale path verified in
  dev-Docker smoke).
"""
import pytest
import torch

from zone_c_segment_cache import (
    SegmentCache,
    assert_attribution,
    canonical_domain_family,
)


def test_canonical_family_live_vocabulary():
    assert canonical_domain_family("arc3/ar25-0c556536") == "action"
    assert canonical_domain_family("ar25-0c556536:ACTION6") == "action"
    assert canonical_domain_family(
        "arc3/ar25-0c556536/field_channel_consolidated") == "action"
    assert canonical_domain_family("arc3/bp35-0a0ad940") == "action"
    assert canonical_domain_family("code") == "ast"
    assert canonical_domain_family("math") == "ast"
    assert canonical_domain_family("unclassified") == "general"


def test_attribution_guard_fails_closed():
    with pytest.raises(ValueError, match="ATTRIBUTION_VIOLATION"):
        assert_attribution("", "A", "abc123")
    with pytest.raises(ValueError, match="ATTRIBUTION_VIOLATION"):
        assert_attribution("run1", "legacy_unattributed", "abc123")
    with pytest.raises(ValueError, match="ATTRIBUTION_VIOLATION"):
        assert_attribution("run1", "A", "untracked")
    # Valid attribution passes without exception.
    assert_attribution("run1", "A", "abc123") is None


def test_surrogate_retrieve_filters_family():
    cache = SegmentCache.connect("offline://surrogate", num_blocks=8192)
    w = torch.randn(8192, 8, dtype=torch.float32)
    w = w / torch.norm(w, p=2, dim=-1, keepdim=True)
    cache.checkpoint(w, "arc3/env_a", 0.9)
    cache.checkpoint(w, "code", 0.9)

    res_action = cache.retrieve(w, domain_family="action")
    assert res_action["hits"] == 1

    res_all = cache.retrieve(w)
    assert res_all["hits"] == 2
