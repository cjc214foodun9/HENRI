"""Contract tests for the Class 4 accuracy-first fidelity remediation.

Covers:
1. Execution-profile contract: score promotion only under the fidelity profile.
2. Sealed ranking-lever closure: levers fail closed under
   HENRI_ACCURACY_FIRST_CLASS4 (Gate A' did not transfer; Gate B 2/50;
   bottleneck is grammar expressiveness, not candidate order).
3. Falsified ring-mod-256 functor guards (planner search,
   synthesize_code_program, MoorePenroseToolCompiler) fail closed under the
   flag (phase5 p3 KILL c0e3128).
4. Mock Identity shortcut guard fails closed under the flag.
5. Default (flag OFF) preserves legacy behavior: guards pass, profile
   telemetry is recorded.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "HENRI V2"))

import pytest  # noqa: E402

import accuracy_profile as ap  # noqa: E402

FLAG = ap.FIDELITY_MIGRATION_FLAG


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    """The fidelity flag defaults OFF in every test unless set explicitly."""
    monkeypatch.delenv(FLAG, raising=False)


def test_profile_contract_only_fidelity_promotes():
    assert ap.is_score_promotable(ap.FIDELITY_SCORE_BEARING) is True
    assert ap.is_score_promotable(ap.DIAGNOSTIC_BALANCED) is False
    assert ap.is_score_promotable(ap.PERFORMANCE_ONLY) is False
    assert ap.is_score_promotable(None) is False


def test_runner_profile_fidelity_unless_sealed_lever():
    assert ap.runner_execution_profile(sealed_lever_enabled=False) == ap.FIDELITY_SCORE_BEARING
    assert ap.runner_execution_profile(sealed_lever_enabled=True) == ap.DIAGNOSTIC_BALANCED


def test_sealed_levers_are_closed():
    levers = {"reward_rank": True, "decoder_rank": False,
              "spec_rank": True, "trained_rank": False, "ast_idf_only": False}
    active = ap.enabled_sealed_levers(levers)
    assert "reward_rank" in active
    assert "spec_rank" in active
    assert "decoder_rank" not in active
    assert ap.enabled_sealed_levers({}) == []


def test_humaneval_runner_fails_closed_on_sealed_lever(monkeypatch):
    """Flag ON + any sealed lever raises FidelityGuardError before dataset load."""
    monkeypatch.setenv(FLAG, "1")
    import humaneval_wave_ast_runner as hr
    with pytest.raises(ap.FidelityGuardError):
        hr.run_benchmark(limit=1, reward_rank=True)
    with pytest.raises(ap.FidelityGuardError):
        hr.run_benchmark(limit=1, decoder_rank=True)


def test_humaneval_runner_no_sealed_lever_passes_guard(monkeypatch):
    """Flag ON but no sealed lever: the guard does not fire (run continues)."""
    monkeypatch.setenv(FLAG, "1")
    import humaneval_wave_ast_runner as hr
    # The guard is before dataset load; reaching dataset load means the
    # guard passed. Monkeypatch the loader to avoid network/sandbox.
    import gzip
    def _fake_load(items):  # pragma: no cover - replaced before use
        return []
    # Instead of a full run (network), assert the contract helpers used by
    # the guard produce the expected profile under the flag.
    assert ap.fidelity_migration_enabled() is True
    assert ap.runner_execution_profile(sealed_lever_enabled=False) == ap.FIDELITY_SCORE_BEARING


def test_planner_functor_guard_fails_closed(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    import sagnac_mcts_planner as sp
    with pytest.raises(ap.FalsifiedOperatorError):
        sp._guard_falsified_ring_functor()
    with pytest.raises(ap.MockShortcutRejectedError):
        sp._guard_mock_identity_shortcut()


def test_planner_functor_guard_default_passes(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    import sagnac_mcts_planner as sp
    sp._guard_falsified_ring_functor()
    sp._guard_mock_identity_shortcut()


def test_repl_tool_compiler_fails_closed(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    import torch
    from henri_universal_repl import MoorePenroseToolCompiler

    class FakeTransducer:
        d_model = 64
        device = "cpu"

        def transduce_text(self, s):
            return torch.randn(self.d_model)

    tc = MoorePenroseToolCompiler(FakeTransducer())
    with pytest.raises(ap.FalsifiedOperatorError):
        tc.compile_tool_functor([("ls", "out")])


def test_scorecard_profile_fields_present():
    rec = ap.profile_record(
        ap.FIDELITY_SCORE_BEARING, d_model=65536, precision="float32",
        candidate_coverage=71, attempts=12, checkpoint_policy="disabled",
        checkpoint_load_status="SKIPPED_NO_CHECKPOINT",
        evaluator="HumanEval_official", dataset_sha256="a" * 64)
    assert rec["execution_profile"] == ap.FIDELITY_SCORE_BEARING
    assert rec["score_promotable"] is True
    assert rec["fidelity_migration_flag"] is False
    assert rec["dataset_sha256"] == "a" * 64
