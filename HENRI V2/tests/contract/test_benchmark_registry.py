import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from henri_benchmark_registry import (
    ARCEpisodeTrace,
    BenchmarkRecord,
    BenchmarkRegistry,
    RunEvidence,
    load_benchlm_index,
    validate_score_eligibility,
)


def test_benchlm_import_is_index_only_and_hashes_source(tmp_path):
    source = tmp_path / "benchmarks.json"
    payload = [{"id": "arc-agi-3", "name": "ARC-AGI-3", "category": "reasoning"}]
    raw = json.dumps(payload).encode()
    source.write_bytes(raw)
    registry = load_benchlm_index(source)
    assert registry.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert registry.records[0].adapter_status == "BLOCKED"
    assert registry.records[0].benchlm_metadata["category"] == "reasoning"


def test_registry_rejects_metadata_promotion_without_evidence():
    with pytest.raises(ValueError):
        BenchmarkRecord(
            benchmark_id="arc-agi-3", display_name="ARC-AGI-3",
            adapter_status="ADAPTER_READY",
        )


def test_run_accounting_and_hashes_are_strict():
    with pytest.raises(ValueError):
        RunEvidence(
            schema_id="henri.run-evidence.v1", status="OBSERVED", run_id="r",
            commit_sha256="a" * 40, command="pytest", benchmark_id="arc-agi-3",
            device="cuda", torch_version="2", item_count=2, attempted_count=1,
            passed_count=1, failed_count=1, execution_error_count=0, vetoed_count=0,
        )


def test_incomplete_evidence_is_blocked():
    registry = BenchmarkRegistry(
        schema_id="henri.benchmark-registry.v1", benchlm_source_uri="x",
        retrieved_at_utc=None, source_sha256="b" * 64, source_root_type="list",
        records=[BenchmarkRecord(benchmark_id="arc-agi-3", display_name="ARC-AGI-3")],
    )
    evidence = RunEvidence(
        schema_id="henri.run-evidence.v1", status="OBSERVED", run_id="r",
        commit_sha256="a" * 40, command="run", benchmark_id="arc-agi-3",
        device="cuda", torch_version="2", item_count=20, attempted_count=20,
        passed_count=0, failed_count=20, execution_error_count=0, vetoed_count=0,
    )
    eligible, reasons = validate_score_eligibility(evidence, registry)
    assert not eligible
    assert "PRIMARY_SOURCE_NOT_VERIFIED" in reasons
    assert "ARTIFACT_DIGEST_MISSING" in reasons


def test_arc_trace_rejects_bad_hash():
    with pytest.raises(ValueError):
        ARCEpisodeTrace(
            schema_id="henri.arc-episode-trace.v1", episode_id="e",
            commit_sha256="a" * 40, task_input_sha256="bad", dataset_sha256="b" * 64,
            split_id="train", task_specific_persistence_preexisting=False,
            demo_pair_count=1, candidate_count=1, veto_count=0,
            evaluator_reached=False, exact_pass=None, evaluator_status="NOT_REACHED",
        )


def test_arc_trace_rejects_preexisting_task_state():
    with pytest.raises(ValueError):
        ARCEpisodeTrace(
            schema_id="henri.arc-episode-trace.v1", episode_id="e",
            commit_sha256="a" * 40, task_input_sha256="a" * 64, dataset_sha256="b" * 64,
            split_id="test", task_specific_persistence_preexisting=True,
            demo_pair_count=1, candidate_count=1, veto_count=0,
            evaluator_reached=True, exact_pass=False, evaluator_status="EVALUATED",
        )


def test_synthetic_and_transport_evidence_cannot_promote():
    registry = BenchmarkRegistry(
        schema_id="henri.benchmark-registry.v1", benchlm_source_uri="x",
        retrieved_at_utc=None, source_sha256="b" * 64, source_root_type="list",
        records=[BenchmarkRecord(
            benchmark_id="arc-agi-3", display_name="ARC-AGI-3",
            canonical_source="https://example.invalid/dataset",
            dataset_sha256="c" * 64, evaluator_id="official", evaluator_version="1",
            evaluator_sha256="d" * 64,
        )],
    )
    evidence = RunEvidence(
        schema_id="henri.run-evidence.v1", status="OBSERVED", run_id="r",
        commit_sha256="a" * 40, command="run", benchmark_id="arc-agi-3",
        dataset_source="generated template fixture", dataset_sha256="c" * 64,
        evaluator_id="official", evaluator_version="1", evaluator_sha256="d" * 64,
        device="cuda", torch_version="2", item_count=20, attempted_count=19,
        passed_count=0, failed_count=19, execution_error_count=1, vetoed_count=0,
        checkpoint_load_status="SKIPPED_POLICY_DISABLED", trained_decoder_active=False,
        raw_stdout_sha256="e" * 64, raw_stderr_sha256="f" * 64,
        item_results_sha256="1" * 64, artifact_paths=["run.log"],
    )
    eligible, reasons = validate_score_eligibility(evidence, registry)
    assert not eligible
    assert "SYNTHETIC_DATASET_DETECTED" in reasons
    assert "EXECUTION_ERROR_PRESENT" in reasons
    assert "TRAINED_DECODER_NOT_LOADED" in reasons


def test_decoder_checkpoint_contract_loads_matching_state(tmp_path):
    import torch
    from henri_decoder import HENRIUnifiedEgressTransducer

    source = HENRIUnifiedEgressTransducer(
        d_model=8, hidden_dim=4, vocab_size=6, device="cpu", checkpoint_policy="disabled"
    )
    checkpoint = tmp_path / "decoder.pt"
    torch.save(source.unbinder.state_dict(), checkpoint)
    loaded = HENRIUnifiedEgressTransducer(
        d_model=8, hidden_dim=4, vocab_size=6, device="cpu",
        checkpoint_path=str(checkpoint), checkpoint_policy="required"
    )
    assert loaded.checkpoint_load_status == "LOADED"
    assert loaded.checkpoint_sha256
    assert loaded.checkpoint_state_dict_sha256
    assert loaded.checkpoint_telemetry()["trained_decoder_active"] is True


def test_decoder_skips_auto_discovered_production_checkpoint_for_reduced_runtime():
    from henri_decoder import HENRIUnifiedEgressTransducer

    decoder = HENRIUnifiedEgressTransducer(
        d_model=8, hidden_dim=4, vocab_size=6, device="cpu", checkpoint_policy="auto"
    )
    assert decoder.checkpoint_load_status == "SKIPPED_INCOMPATIBLE_ARCHITECTURE"
    assert decoder.checkpoint_telemetry()["decoder_state"] == "UNTRAINED_DECODER"


def test_decoder_required_policy_raises_on_architecture_mismatch(tmp_path):
    import torch
    from henri_decoder import DecoderCheckpointCompatibilityError, HENRIUnifiedEgressTransducer

    source = HENRIUnifiedEgressTransducer(
        d_model=8, hidden_dim=4, vocab_size=6, device="cpu", checkpoint_policy="disabled"
    )
    checkpoint = tmp_path / "decoder.pt"
    torch.save(source.unbinder.state_dict(), checkpoint)
    with pytest.raises(DecoderCheckpointCompatibilityError, match="incompatible"):
        HENRIUnifiedEgressTransducer(
            d_model=9, hidden_dim=4, vocab_size=6, device="cpu",
            checkpoint_path=str(checkpoint), checkpoint_policy="required"
        )


def test_decoder_disabled_policy_does_not_inspect_checkpoint(tmp_path):
    from henri_decoder import HENRIUnifiedEgressTransducer

    decoder = HENRIUnifiedEgressTransducer(
        d_model=8, hidden_dim=4, vocab_size=6, device="cpu",
        checkpoint_path=str(tmp_path / "missing.pt"), checkpoint_policy="disabled"
    )
    assert decoder.checkpoint_load_status == "SKIPPED_POLICY_DISABLED"
    assert decoder.checkpoint_sha256 is None


def test_decoder_required_policy_rejects_corrupt_checkpoint(tmp_path):
    from henri_decoder import DecoderCheckpointCompatibilityError, HENRIUnifiedEgressTransducer

    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    with pytest.raises(DecoderCheckpointCompatibilityError):
        HENRIUnifiedEgressTransducer(
            d_model=8, hidden_dim=4, vocab_size=6, device="cpu",
            checkpoint_path=str(checkpoint), checkpoint_policy="required"
        )


def test_score_gate_requires_evaluated_adapter_and_loaded_checkpoint_digest():
    registry = BenchmarkRegistry(
        schema_id="henri.benchmark-registry.v1", benchlm_source_uri="x",
        retrieved_at_utc=None, source_sha256="b" * 64, source_root_type="list",
        records=[BenchmarkRecord(
            benchmark_id="arc-agi-3", display_name="ARC-AGI-3",
            adapter_status="EVALUATED",
            canonical_source="https://example.invalid/dataset",
            dataset_sha256="c" * 64, evaluator_id="official", evaluator_version="1",
            evaluator_sha256="d" * 64,
        )],
    )
    evidence = RunEvidence(
        schema_id="henri.run-evidence.v1", status="OBSERVED", run_id="r",
        commit_sha256="a" * 40, command="run", benchmark_id="arc-agi-3",
        dataset_source="https://example.invalid/dataset", dataset_sha256="c" * 64,
        evaluator_id="official", evaluator_version="1", evaluator_sha256="d" * 64,
        checkpoint_load_status="LOADED", trained_decoder_active=True,
        device="cuda", torch_version="2", item_count=20, attempted_count=20,
        passed_count=20, failed_count=0, execution_error_count=0, vetoed_count=0,
        raw_stdout_sha256="e" * 64, raw_stderr_sha256="f" * 64,
        item_results_sha256="1" * 64, artifact_paths=["run.log"],
    )
    eligible, reasons = validate_score_eligibility(evidence, registry)
    assert not eligible
    assert "CHECKPOINT_DIGEST_MISSING" in reasons


def test_score_gate_rejects_default_blocked_adapter():
    registry = BenchmarkRegistry(
        schema_id="henri.benchmark-registry.v1", benchlm_source_uri="x",
        retrieved_at_utc=None, source_root_type="list", source_sha256="b" * 64,
        records=[BenchmarkRecord(
            benchmark_id="arc-agi-3", display_name="ARC-AGI-3",
            canonical_source="https://example.invalid/dataset",
            dataset_sha256="c" * 64, evaluator_id="official", evaluator_version="1",
            evaluator_sha256="d" * 64,
        )],
    )
    evidence = RunEvidence(
        schema_id="henri.run-evidence.v1", status="OBSERVED", run_id="r",
        commit_sha256="a" * 40, command="run", benchmark_id="arc-agi-3",
        dataset_source="https://example.invalid/dataset", dataset_sha256="c" * 64,
        evaluator_id="official", evaluator_version="1", evaluator_sha256="d" * 64,
        checkpoint_sha256="e" * 64, checkpoint_load_status="LOADED",
        trained_decoder_active=True, device="cuda", torch_version="2",
        item_count=20, attempted_count=20, passed_count=20, failed_count=0,
        execution_error_count=0, vetoed_count=0, raw_stdout_sha256="f" * 64,
        raw_stderr_sha256="1" * 64, item_results_sha256="2" * 64,
        artifact_paths=["run.log"],
    )
    eligible, reasons = validate_score_eligibility(evidence, registry)
    assert not eligible
    assert "ADAPTER_NOT_EVALUATED" in reasons
