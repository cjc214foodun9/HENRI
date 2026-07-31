"""Index-only benchmark registry and fail-closed evidence promotion.

BenchLM metadata identifies benchmarks. It does not provide task data or prove
that an evaluator is available. This module never downloads data or creates
tasks. It only parses a supplied index and validates evidence before promotion.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT = re.compile(r"^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$")
SYNTHETIC_MARKERS = ("synthetic", "generated", "template", "placeholder", "smoke")
GENERIC_GRADER_MARKERS = ("substring", "contains_answer", "boxed_marker", "generic")


class BenchmarkRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    benchmark_id: str
    display_name: str
    family: str | None = None
    benchlm_metadata: dict[str, Any] = Field(default_factory=dict)
    canonical_source: str | None = None
    license: str | None = None
    official_split: str | None = None
    evaluator_id: str | None = None
    evaluator_version: str | None = None
    evaluator_sha256: str | None = None
    dataset_sha256: str | None = None
    adapter_status: Literal["BLOCKED", "ADAPTER_READY", "EVALUATED"] = "BLOCKED"
    block_reason: str | None = "PRIMARY_SOURCE_NOT_VERIFIED"

    @field_validator("dataset_sha256", "evaluator_sha256")
    @classmethod
    def valid_optional_hash(cls, value: str | None) -> str | None:
        if value is not None and not HEX64.fullmatch(value):
            raise ValueError("hash must be a 64-character SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def prevent_metadata_promotion(self) -> "BenchmarkRecord":
        if self.adapter_status != "BLOCKED":
            required = (self.canonical_source, self.dataset_sha256, self.evaluator_id,
                        self.evaluator_version, self.evaluator_sha256)
            if any(value is None for value in required):
                raise ValueError("adapter cannot be promoted without source, dataset, evaluator, and digests")
        return self


class BenchmarkRegistry(BaseModel):
    schema_id: Literal["henri.benchmark-registry.v1"]
    benchlm_source_uri: str
    retrieved_at_utc: str | None
    source_sha256: str
    source_root_type: Literal["dict", "list"]
    records: list[BenchmarkRecord]

    @field_validator("source_sha256")
    @classmethod
    def valid_source_hash(cls, value: str) -> str:
        if not HEX64.fullmatch(value):
            raise ValueError("source_sha256 must be a 64-character SHA-256 hex digest")
        return value


class RunEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["henri.run-evidence.v1"]
    status: Literal["OBSERVED", "BLOCKED", "INVALID", "EXECUTION_ERROR", "SCHEMA_INVALID"]
    run_id: str
    commit_sha256: str
    command: str
    benchmark_id: str
    dataset_source: str | None = None
    dataset_sha256: str | None = None
    evaluator_id: str | None = None
    evaluator_version: str | None = None
    evaluator_sha256: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_load_status: Literal[
        "LOADED", "SKIPPED_POLICY_DISABLED", "SKIPPED_NO_CHECKPOINT",
        "SKIPPED_INCOMPATIBLE_ARCHITECTURE", "FAILED_REQUIRED_CHECKPOINT",
        "FAILED_CORRUPT_CHECKPOINT"
    ] | None = None
    trained_decoder_active: bool | None = None
    device: str
    torch_version: str
    cuda_version: str | None = None
    item_count: int = Field(ge=0)
    attempted_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    execution_error_count: int = Field(ge=0)
    vetoed_count: int = Field(ge=0)
    raw_stdout_sha256: str | None = None
    raw_stderr_sha256: str | None = None
    item_results_sha256: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    limitations: str = ""
    grader_mode: str | None = None
    synthetic_source: bool = False
    task_leakage_detected: bool = False
    declared_split_count: int | None = Field(default=None, ge=1)

    @field_validator("commit_sha256")
    @classmethod
    def valid_commit(cls, value: str) -> str:
        if not COMMIT.fullmatch(value):
            raise ValueError("commit_sha256 must be a 40- or 64-character hex SHA")
        return value

    @field_validator("dataset_sha256", "evaluator_sha256", "checkpoint_sha256",
                     "raw_stdout_sha256", "raw_stderr_sha256", "item_results_sha256")
    @classmethod
    def valid_hash(cls, value: str | None) -> str | None:
        if value is not None and not HEX64.fullmatch(value):
            raise ValueError("artifact hashes must be 64-character SHA-256 hex digests")
        return value

    @model_validator(mode="after")
    def validate_accounting(self) -> "RunEvidence":
        if self.passed_count + self.failed_count != self.attempted_count:
            raise ValueError("passed_count + failed_count must equal attempted_count")
        if self.attempted_count + self.execution_error_count + self.vetoed_count != self.item_count:
            raise ValueError("item accounting does not reconcile to item_count")
        return self


class ARCEpisodeTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["henri.arc-episode-trace.v1"]
    episode_id: str
    commit_sha256: str
    task_input_sha256: str
    dataset_sha256: str
    split_id: str
    seed: int | None = None
    task_specific_persistence_preexisting: bool
    demo_pair_count: int = Field(ge=0)
    object_count: int | None = Field(default=None, ge=0)
    wave_norm_max_error: float | None = None
    task_operator_sha256: str | None = None
    task_operator_norm: float | None = None
    candidate_count: int = Field(ge=0)
    action_entropy: float | None = None
    min_sagnac_delta: float | None = None
    veto_count: int = Field(ge=0)
    veto_reasons: dict[str, int] = Field(default_factory=dict)
    evaluator_reached: bool
    external_state_delta: float | None = None
    exact_pass: bool | None = None
    evaluator_status: str
    ingress_ms: float | None = None
    induction_ms: float | None = None
    planning_ms: float | None = None
    evaluation_ms: float | None = None
    limitations: str = ""

    @field_validator("task_input_sha256", "dataset_sha256", "task_operator_sha256")
    @classmethod
    def valid_arc_hash(cls, value: str) -> str:
        if not HEX64.fullmatch(value):
            raise ValueError("ARC hashes must be 64-character SHA-256 hex digests")
        return value

    @model_validator(mode="after")
    def reject_preexisting_task_state(self) -> "ARCEpisodeTrace":
        if self.task_specific_persistence_preexisting:
            raise ValueError("pre-existing task-specific state invalidates unseen-task evaluation")
        return self


def _record_candidates(root: Any) -> list[dict[str, Any]]:
    if isinstance(root, list):
        return [item for item in root if isinstance(item, dict)]
    if not isinstance(root, dict):
        raise ValueError("BenchLM index root must be a JSON object or array")
    for key in ("benchmarks", "data", "results", "items"):
        value = root.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [root]


def _stable_id(item: dict[str, Any], index: int) -> str:
    for key in ("benchmark_id", "id", "slug", "name", "title"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace(" ", "-")
    return f"benchlm-record-{index:04d}"


def load_benchlm_index(path: str | Path, source_uri: str = "https://benchlm.ai/data/benchmarks.json") -> BenchmarkRegistry:
    """Read a supplied BenchLM JSON file and preserve its metadata.

    This function performs no network access, dataset download, task generation,
    evaluator loading, or scorecard writing.
    """
    source_path = Path(path)
    raw = source_path.read_bytes()
    root = json.loads(raw)
    candidates = _record_candidates(root)
    records: list[BenchmarkRecord] = []
    for index, item in enumerate(candidates, 1):
        name = next((item.get(k) for k in ("display_name", "name", "title")
                     if isinstance(item.get(k), str) and item.get(k).strip()),
                    f"BenchLM record {index}")
        family = item.get("family") if isinstance(item.get("family"), str) else None
        records.append(BenchmarkRecord(
            benchmark_id=_stable_id(item, index),
            display_name=name,
            family=family,
            benchlm_metadata=item,
            block_reason="PRIMARY_SOURCE_NOT_VERIFIED",
        ))
    return BenchmarkRegistry(
        schema_id="henri.benchmark-registry.v1",
        benchlm_source_uri=source_uri,
        retrieved_at_utc=None,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_root_type="list" if isinstance(root, list) else "dict",
        records=records,
    )


def detect_synthetic_content(*values: str | None) -> bool:
    text = " ".join(value.lower() for value in values if value)
    return any(marker in text for marker in SYNTHETIC_MARKERS)


def validate_score_eligibility(evidence: RunEvidence, registry: BenchmarkRegistry,
                               minimum_items: int = 20) -> tuple[bool, list[str]]:
    """Return eligibility and refusal reasons immediately before score promotion."""
    reasons: list[str] = []
    record = next((r for r in registry.records if r.benchmark_id == evidence.benchmark_id), None)
    if record is None:
        reasons.append("BENCHMARK_NOT_REGISTERED")
    else:
        if record.adapter_status != "EVALUATED":
            reasons.append("ADAPTER_NOT_EVALUATED")
        if not record.canonical_source:
            reasons.append("PRIMARY_SOURCE_NOT_VERIFIED")
        if not record.dataset_sha256:
            reasons.append("DATASET_DIGEST_MISSING")
        if not record.evaluator_id:
            reasons.append("EVALUATOR_ID_MISSING")
        if not record.evaluator_sha256:
            reasons.append("EVALUATOR_DIGEST_MISSING")
    if evidence.status != "OBSERVED":
        reasons.append("STATUS_NOT_OBSERVED")
    if evidence.checkpoint_load_status != "LOADED" or evidence.trained_decoder_active is not True:
        reasons.append("TRAINED_DECODER_NOT_LOADED")
    if evidence.checkpoint_load_status == "LOADED" and not evidence.checkpoint_sha256:
        reasons.append("CHECKPOINT_DIGEST_MISSING")
    if not evidence.dataset_source or not evidence.dataset_sha256:
        reasons.append("DATASET_EVIDENCE_MISSING")
    if not evidence.evaluator_id or not evidence.evaluator_version or not evidence.evaluator_sha256:
        reasons.append("EVALUATOR_EVIDENCE_MISSING")
    if not evidence.raw_stdout_sha256 or not evidence.raw_stderr_sha256 or not evidence.item_results_sha256:
        reasons.append("ARTIFACT_DIGEST_MISSING")
    if not evidence.artifact_paths:
        reasons.append("ARTIFACT_PATHS_MISSING")
    if evidence.item_count < minimum_items:
        reasons.append("INSUFFICIENT_SAMPLE_SIZE")
    if evidence.declared_split_count is not None and evidence.item_count != evidence.declared_split_count:
        reasons.append("PARTIAL_SPLIT_NOT_DECLARED")
    if evidence.synthetic_source or detect_synthetic_content(evidence.dataset_source, evidence.command,
                                                              evidence.grader_mode):
        reasons.append("SYNTHETIC_DATASET_DETECTED")
    if evidence.grader_mode and any(marker in evidence.grader_mode.lower() for marker in GENERIC_GRADER_MARKERS):
        reasons.append("GENERIC_GRADER_DETECTED")
    if evidence.task_leakage_detected:
        reasons.append("TASK_LEAKAGE_DETECTED")
    if evidence.execution_error_count or evidence.vetoed_count:
        reasons.append("EXECUTION_ERROR_PRESENT")
    if record and evidence.dataset_sha256 and record.dataset_sha256 and evidence.dataset_sha256 != record.dataset_sha256:
        reasons.append("DATASET_DIGEST_MISMATCH")
    return not reasons, reasons