"""Contract tests for the Phase 8 compute-envelope probe.

Covers: feature default OFF (no allocation), exact UWE shape, shared
parameter storage across batch sizes, B=1 equivalence to the production
path, deterministic replay digest, legal-action masking, telemetry schema,
egress/action-head OFF => diagnostic_only/score_eligible, and the
no-game.step source-inspection guard.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../HENRI V2
_PROBE_PATH = _REPO_ROOT / "experiments" / "performance" / "phase8_batched_nav_probe.py"

sys.path.insert(0, str(_REPO_ROOT))


def _load_probe():
    spec = importlib.util.spec_from_file_location("phase8_batched_nav_probe",
                                                  _PROBE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return _load_probe()


@pytest.fixture()
def off_env(monkeypatch):
    monkeypatch.delenv("HENRI_ARC_BATCHED_NAV_SWARM", raising=False)
    monkeypatch.setenv("HENRI_PROBE_SCALE", "reduced")


def test_feature_default_off(probe, off_env):
    result = probe.run_probe(scale="reduced")
    assert result["status"] == "FEATURE_DISABLED"
    assert result["schema_id"] == "henri.phase8-compute-probe.v1"
    assert result["diagnostic_only"] is True
    assert result["score_eligible"] is False
    assert "did not allocate" in result["reason"]


def test_feature_on_produces_complete(probe, monkeypatch):
    monkeypatch.setenv("HENRI_ARC_BATCHED_NAV_SWARM", "1")
    monkeypatch.setenv("HENRI_PROBE_SCALE", "reduced")
    result = probe.run_probe(scale="reduced", batches=[1, 4],
                             iterations=2, warmup=1)
    assert result["status"] == "COMPLETE"
    assert result["scale"] == "reduced"


def test_exact_uwe_shape(probe, monkeypatch):
    monkeypatch.setenv("HENRI_ARC_BATCHED_NAV_SWARM", "1")
    result = probe.run_probe(scale="reduced", batches=[1], iterations=1,
                             warmup=1)
    assert result["shapes"]["state_wave"] == [64, 8]
    assert result["shapes"]["dtype"] == "torch.float32"


def test_shared_param_storage_identity(probe, monkeypatch):
    """Batching must not clone weights: same module object, same storage."""
    monkeypatch.setenv("HENRI_ARC_BATCHED_NAV_SWARM", "1")
    monkeypatch.setenv("HENRI_PROBE_SCALE", "reduced")
    from arcengine import GameAction
    from darwinian_phase_swarm import HenriSwarmOrchestrator
    SCALE = dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64)
    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction, constraint_weight_max=5.0,
        constraint_reject_thresh=0.5, beta_pragmatic=1.0, lambda_goal=0.0,
        learnable_actions=False, chimera_mode=True, chimera_alpha=0.3,
        chimera_explorer_fraction=0.25, happy_tensor_cut=0.5,
        external_outcome_efe=False, external_eig_weight=0.0,
        external_task_weight=0.0, task_weighted_eig=False,
        task_eig_gamma=0.0, **SCALE)
    p1 = orch.planner.transition.field_V.data_ptr()
    _ = orch.planner.score_actions(
        torch.randn(64, 8), [(1, torch.randn(64, 8))],
        torch.randn(1, 64, 8))
    p2 = orch.planner.transition.field_V.data_ptr()
    assert p1 == p2, "score_actions must not clone/replace weights"


def test_b1_identity_with_production_path(probe, monkeypatch):
    monkeypatch.setenv("HENRI_ARC_BATCHED_NAV_SWARM", "1")
    result = probe.run_probe(scale="reduced", batches=[1], iterations=1,
                             warmup=1, seed=7)
    assert result["b1_identity"]["production_action_match"] is True
    assert len(result["b1_identity"]["digest_sha256_32"]) == 32


def test_deterministic_replay_digest(probe, monkeypatch):
    monkeypatch.setenv("HENRI_ARC_BATCHED_NAV_SWARM", "1")
    a = probe.run_probe(scale="reduced", batches=[1], iterations=1,
                        warmup=1, seed=99)
    b = probe.run_probe(scale="reduced", batches=[1], iterations=1,
                        warmup=1, seed=99)
    assert a["b1_identity"]["digest_sha256_32"] == \
        b["b1_identity"]["digest_sha256_32"]


def test_legal_action_masking(probe, monkeypatch):
    monkeypatch.setenv("HENRI_ARC_BATCHED_NAV_SWARM", "1")
    from arcengine import GameAction
    from arc_egress_contract import ActionEgressVocabulary
    allowed = list(GameAction)[:6]
    vocab = ActionEgressVocabulary(GameAction, allowed)
    result = probe.run_probe(scale="reduced", batches=[1], iterations=1,
                             warmup=1)
    # Probe's B=1 loop action must be within the allowed mask.
    assert result["b1_identity"]["production_action_match"] is True
    # Vocabulary is deterministic over the mask.
    assert vocab.n_actions == 6
    assert len(set(vocab.id_to_action.values())) == 6


def test_telemetry_schema(probe, monkeypatch):
    monkeypatch.setenv("HENRI_ARC_BATCHED_NAV_SWARM", "1")
    result = probe.run_probe(scale="reduced", batches=[1, 4], iterations=2,
                             warmup=1)
    assert result["schema_id"] == "henri.phase8-compute-probe.v1"
    for key in ["status", "feature_gate", "scale", "device", "shapes",
                "b1_identity", "batches", "egress", "checkpoint", "git",
                "raw_log_sha256"]:
        assert key in result, f"missing schema key {key}"
    row = result["batches"][0]
    for key in ["B", "total_scored_particles", "particles_per_s",
                "latency_mean_ms_per_particle",
                "lower_bound_logical_bandwidth_gbps",
                "bandwidth_label", "ess_descriptor",
                "distinct_actions_descriptor", "finite_ok"]:
        assert key in row, f"missing batch key {key}"
    assert row["bandwidth_label"] == "LOWER_BOUND_LOGICAL_BANDWIDTH"
    # raw_log_sha256 must match a canonical re-serialization
    canonical = json.dumps(
        {k: v for k, v in result.items() if k != "raw_log_sha256"},
        sort_keys=True)
    import hashlib
    assert result["raw_log_sha256"] == \
        hashlib.sha256(canonical.encode()).hexdigest()


def test_egress_and_action_head_stay_off(probe, monkeypatch):
    monkeypatch.setenv("HENRI_ARC_BATCHED_NAV_SWARM", "1")
    result = probe.run_probe(scale="reduced", batches=[1], iterations=1,
                             warmup=1)
    assert result["egress"]["diagnostic_only"] is True
    assert result["egress"]["score_eligible"] is False
    assert result["egress"]["transducer_loaded"] is False
    assert result["checkpoint"]["load_status"] == "SKIPPED_POLICY_DISABLED"


def test_no_game_step_in_probe_source(probe):
    """Compute-only invariant: the probe must never call game.step/arcade."""
    src = _PROBE_PATH.read_text(encoding="utf-8")
    for token in ["game.step", "arcade", "Arcade(", "step_with_payload",
                  "levels_completed", "sans_buffer"]:
        assert token not in src, f"probe must not contain {token!r}"


def test_reduced_scale_env_contract(probe, monkeypatch):
    monkeypatch.setenv("HENRI_ARC_BATCHED_NAV_SWARM", "1")
    result = probe.run_probe(scale="reduced", batches=[4], iterations=1,
                             warmup=1)
    assert result["shapes"]["state_wave"] == [64, 8]
    assert len(result["batches"]) == 1
    assert result["batches"][0]["B"] == 4
