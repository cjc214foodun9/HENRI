from __future__ import annotations

import json
from pathlib import Path

import pytest

from aa_v41_index_validator import validate_manifest
from henri_benchmark_registry import BenchmarkRecord, BenchmarkRegistry, RunEvidence, validate_score_eligibility
from mbpp_contamination_scan import scan
from mbpp_heldout_pilot import extract_code_blocks, load_items, validate_static_bundle
from mbpp_secure_executor import SandboxUnavailable, SecurePythonSandbox

ROOT = Path(__file__).resolve().parents[2]


def test_aa_index_is_metadata_only_and_current_family_count_is_nine():
    path = ROOT / "data/official_benchmarks/staged_eval_suites/aa_v41_manifest.json"
    result = validate_manifest(path)
    assert result["schema_id"] == "henri.aa-v41-index.v1"
    assert result["evaluation_family_count"] == 9
    assert result["task_data_present"] is False
    assert result["composite_status"] == "BLOCKED"


def test_mbpp_source_and_split_are_immutable():
    manifest, items = validate_static_bundle()
    assert manifest["source_commit"] == "8dadc6c56e2c2e51a9dd7e0d4bf2840922b4b6c0"
    assert len(items) == 500
    assert [int(item["task_id"]) for item in items] == list(range(11, 511))
    assert manifest["prompt_contract"]["reference_code_exposed"] is False


def test_completion_extraction_matches_pinned_contract():
    assert extract_code_blocks("```python\nprint(1)\n```") == "print(1)"
    assert extract_code_blocks("plain text") == ""


def test_contamination_scan_detects_exact_task_text(tmp_path: Path):
    items = load_items()
    exposed = tmp_path / "prior_run.jsonl"
    exposed.write_text(json.dumps({"task_id": items[0]["task_id"], "text": items[0]["text"]}) + "\n", encoding="utf-8")
    result = scan(
        ROOT / "data/official_benchmarks/canonical/mbpp/mbpp.jsonl",
        [tmp_path],
        [],
    )
    assert result["status"] == "BLOCKED_TASK_EXPOSURE"
    assert result["matches"]


def _evidence(**overrides):
    values = dict(
        schema_id="henri.run-evidence.v1",
        status="OBSERVED",
        run_id="test-run",
        commit_sha256="0" * 40,
        command="pilot",
        benchmark_id="mbpp_google_test_v1",
        dataset_source="https://example.invalid/mbpp.jsonl",
        dataset_sha256="1" * 64,
        evaluator_id="evaluator",
        evaluator_version="commit",
        evaluator_sha256="2" * 64,
        checkpoint_sha256="3" * 64,
        checkpoint_load_status="LOADED",
        trained_decoder_active=True,
        device="cuda:0",
        torch_version="test",
        cuda_version="13.0",
        item_count=500,
        attempted_count=500,
        passed_count=400,
        failed_count=100,
        execution_error_count=0,
        vetoed_count=0,
        raw_stdout_sha256="4" * 64,
        raw_stderr_sha256="5" * 64,
        item_results_sha256="6" * 64,
        artifact_paths=["run_evidence.json"],
        grader_mode="isolated_assertion_execution_pinned_mbpp_pass_at_1",
        declared_split_count=500,
    )
    values.update(overrides)
    return RunEvidence(**values)


def _registry(**overrides):
    values = dict(
        benchmark_id="mbpp_google_test_v1",
        display_name="MBPP",
        family="coding",
        canonical_source="https://example.invalid/mbpp.jsonl",
        official_split="11..510",
        evaluator_id="evaluator",
        evaluator_version="commit",
        evaluator_sha256="2" * 64,
        dataset_sha256="1" * 64,
        adapter_status="EVALUATED",
        block_reason=None,
    )
    values.update(overrides)
    return BenchmarkRegistry(
        schema_id="henri.benchmark-registry.v1",
        benchlm_source_uri="https://example.invalid/index",
        retrieved_at_utc=None,
        source_sha256="7" * 64,
        source_root_type="dict",
        records=[BenchmarkRecord(**values)],
    )


def test_score_gate_requires_exact_evaluator_and_source_equality():
    evidence = _evidence()
    eligible, reasons = validate_score_eligibility(evidence, _registry(), minimum_items=500)
    assert eligible is True
    assert reasons == []

    mismatch, _ = validate_score_eligibility(
        _evidence(evaluator_version="different"), _registry(), minimum_items=500
    )
    assert mismatch is False


def test_secure_executor_never_falls_back_to_unsandboxed_execution():
    if __import__("os").name != "posix":
        with pytest.raises(SandboxUnavailable):
            SecurePythonSandbox()
