import pytest

from henri_benchmark_registry import (
    BenchmarkRecord,
    BenchmarkRegistry,
    RunEvidence,
    validate_score_eligibility,
)


HASH = "b" * 64


def _registry() -> BenchmarkRegistry:
    return BenchmarkRegistry(
        schema_id="henri.benchmark-registry.v1",
        benchlm_source_uri="file:///benchmarks.json",
        retrieved_at_utc=None,
        source_sha256=HASH,
        source_root_type="list",
        records=[
            BenchmarkRecord(
                benchmark_id="held-out-code",
                display_name="Held-out Code",
                canonical_source="https://canonical.example/dataset",
                dataset_sha256=HASH,
                evaluator_id="exact-tests",
                evaluator_version="1.0.0",
                evaluator_sha256=HASH,
                adapter_status="EVALUATED",
                block_reason=None,
            )
        ],
    )


def _evidence(checkpoint_sha256: str | None) -> RunEvidence:
    return RunEvidence(
        schema_id="henri.run-evidence.v1",
        status="OBSERVED",
        run_id="run-p0-contract",
        commit_sha256="a" * 40,
        command="python evaluator.py --split test",
        benchmark_id="held-out-code",
        dataset_source="https://canonical.example/dataset",
        dataset_sha256=HASH,
        evaluator_id="exact-tests",
        evaluator_version="1.0.0",
        evaluator_sha256=HASH,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_load_status="LOADED",
        trained_decoder_active=True,
        device="cuda:0",
        torch_version="2.x",
        cuda_version="12.x",
        item_count=20,
        attempted_count=20,
        passed_count=20,
        failed_count=0,
        execution_error_count=0,
        vetoed_count=0,
        raw_stdout_sha256=HASH,
        raw_stderr_sha256=HASH,
        item_results_sha256=HASH,
        artifact_paths=["artifacts/items.jsonl"],
        grader_mode="exact_tests",
        declared_split_count=20,
    )


def test_loaded_checkpoint_without_digest_is_rejected():
    eligible, reasons = validate_score_eligibility(_evidence(None), _registry())
    assert not eligible
    assert "CHECKPOINT_DIGEST_MISSING" in reasons


def test_complete_evidence_bundle_is_eligible():
    eligible, reasons = validate_score_eligibility(_evidence(HASH), _registry())
    assert eligible
    assert reasons == []
