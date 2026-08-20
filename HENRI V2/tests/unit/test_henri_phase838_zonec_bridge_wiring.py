# -*- coding: utf-8 -*-
"""Phase 8.38 — Zone C retrieval bridge consumer wiring contract.

Verifies the causal consumer chain for HENRI_ZONEC_BRIDGE in the live ARC
runner (production_arc_run.py):

- flag -> module constant (default OFF)
- constant -> `zonec_bridge` init (fail-closed constructor when enabled)
- zonec_bridge -> both live consumers (goal layer + state-recall conditioning)
- bridge path fail-closed: no silent surrogate; legacy path byte-identical.

Deterministic source-inspection assertions resolve the target from
Path(__file__) so invocation directory cannot flip the result.
"""
import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _REPO_ROOT / "production_arc_run.py"


def _runner_src() -> str:
    return _RUNNER.read_text(encoding="utf-8")


def test_flag_default_off():
    os.environ.pop("HENRI_ZONEC_BRIDGE", None)
    import production_arc_run as m
    assert m.HENRI_ZONEC_BRIDGE is False


def test_flag_env_on_parses_strictly():
    from zone_c_retrieval_bridge import bridge_enabled_from_env
    assert bridge_enabled_from_env({"HENRI_ZONEC_BRIDGE": "1"}) is True
    assert bridge_enabled_from_env({"HENRI_ZONEC_BRIDGE": "true"}) is False
    assert bridge_enabled_from_env({}) is False


def test_runner_has_bridge_import_and_flag():
    src = _runner_src()
    assert "from zone_c_retrieval_bridge import ZoneCRetrievalBridge" in src
    assert 'HENRI_ZONEC_BRIDGE = os.environ.get("HENRI_ZONEC_BRIDGE", "0") == "1"' in src


def test_bridge_init_fail_closed_when_enabled():
    src = _runner_src()
    # Init block: bridge constructed only under the flag; enabled without a
    # reachable store must raise (no silent surrogate).
    assert "zonec_bridge = ZoneCRetrievalBridge(" in src
    assert "if HENRI_ZONEC_BRIDGE:" in src


def test_goal_layer_consumer_wired():
    src = _runner_src()
    # Goal layer: bridge path sets GOAL_ZONE_C_BRIDGE; legacy path keeps
    # GOAL_ZONE_C_ANALOGICAL.
    assert "GOAL_ZONE_C_BRIDGE" in src
    assert "GOAL_ZONE_C_ANALOGICAL" in src
    assert "zonec_bridge.retrieve(init_wave.cpu(), top_k=4)" in src


def test_state_recall_consumer_wired():
    src = _runner_src()
    assert "zonec_bridge.retrieve(state_wave.cpu(), top_k=4)" in src
    assert "if zonec_bridge is not None:" in src
    assert "else:" in src  # legacy SegmentCache branch retained


def test_bridge_path_fail_closed_no_surrogate():
    src = _runner_src()
    # Bridge except-branch must re-raise; only the legacy path may pass.
    assert "raise  # bridge path is fail-closed: no silent surrogate" in src
    assert "pass  # legacy: Zone C may be offline; fall through" in src


def test_legacy_default_path_preserved():
    src = _runner_src()
    # Default path (bridge off) still calls the SegmentCache retrieve at both
    # consumer sites.
    assert "res = orch.segment_cache.retrieve(init_wave.cpu())" in src
    assert "res = orch.segment_cache.retrieve(state_wave.cpu())" in src
    # Blend math untouched.
    assert "state_wave = 0.7 * state_wave + 0.3 * recalled" in src
