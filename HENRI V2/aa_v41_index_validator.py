"""Fail-closed validator for the official Artificial Analysis v4.1 index metadata.

This module validates an index and provenance boundary. It never loads task data,
creates benchmark items, invokes a model, or computes an external score.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_ID = "henri.aa-v41-index.v1"
EXPECTED_FAMILIES = {
    "gdpval_aa_v2",
    "tau3_banking",
    "terminalbench_v2_1",
    "scicode",
    "aa_lcr",
    "aa_omniscience",
    "hle",
    "gpqa_diamond",
    "critpt",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AAIndexContractError(ValueError):
    """Raised when AA index metadata cannot support an adapter boundary."""


def _require(value: Any, field: str) -> None:
    if value is None or value == "":
        raise AAIndexContractError(f"{field}: missing")


def validate_manifest(raw: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Validate the current AA v4.1 index mapping or JSON file path."""
    if isinstance(raw, (str, Path)):
        raw = json.loads(Path(raw).read_text(encoding="utf-8"))
    if raw.get("schema_id") != SCHEMA_ID:
        raise AAIndexContractError("schema_id: unexpected")
    if raw.get("index_id") != "artificial_analysis_intelligence_index_v4_1":
        raise AAIndexContractError("index_id: unexpected")
    if raw.get("evaluation_family_count") != 9:
        raise AAIndexContractError("evaluation_family_count: must equal official nine")
    for field in (
        "official_index_uri",
        "official_methodology_uri",
        "retrieved_at_utc",
        "official_index_page_sha256",
        "official_methodology_page_sha256",
    ):
        _require(raw.get(field), field)
    for field in ("official_index_page_sha256", "official_methodology_page_sha256"):
        if not HEX64.fullmatch(str(raw[field])):
            raise AAIndexContractError(f"{field}: invalid SHA-256")

    evaluations = raw.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 9:
        raise AAIndexContractError("evaluations: must contain exactly nine families")
    ids = {item.get("benchmark_id") for item in evaluations}
    if ids != EXPECTED_FAMILIES:
        raise AAIndexContractError(
            f"evaluations: family mismatch; expected={sorted(EXPECTED_FAMILIES)} observed={sorted(ids)}"
        )

    for item in evaluations:
        benchmark_id = item.get("benchmark_id", "<missing>")
        if item.get("adapter_status") != "BLOCKED":
            raise AAIndexContractError(f"{benchmark_id}: unverified index entry must remain BLOCKED")
        _require(item.get("official_page_uri"), f"{benchmark_id}.official_page_uri")
        _require(item.get("official_metric"), f"{benchmark_id}.official_metric")
        if any(key in item for key in ("file_path", "task_file", "items", "tasks")):
            raise AAIndexContractError(f"{benchmark_id}: index must not contain task data or task paths")
        for key in ("canonical_dataset_uri", "dataset_sha256", "evaluator_id", "evaluator_version", "evaluator_sha256"):
            if item.get(key) not in (None, ""):
                raise AAIndexContractError(
                    f"{benchmark_id}: unverified adapter field {key} must remain null"
                )

    legacy = raw.get("legacy_or_additional_evaluations")
    if not isinstance(legacy, list):
        raise AAIndexContractError("legacy_or_additional_evaluations: missing list")
    for item in legacy:
        if item.get("adapter_status") != "EXCLUDED_FROM_V41_COMPOSITE":
            raise AAIndexContractError(
                f"{item.get('benchmark_id', '<missing>')}: legacy entry must be excluded"
            )

    if raw.get("task_data_present") is not False:
        raise AAIndexContractError("task_data_present: must be false")
    if raw.get("composite_status") != "BLOCKED":
        raise AAIndexContractError("composite_status: must remain BLOCKED")

    return {
        "schema_id": SCHEMA_ID,
        "evaluation_family_count": len(evaluations),
        "blocked_family_count": sum(item["adapter_status"] == "BLOCKED" for item in evaluations),
        "legacy_count": len(legacy),
        "task_data_present": False,
        "composite_status": "BLOCKED",
    }


def validate_manifest_file(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return validate_manifest(json.loads(source.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_manifest_file(args.path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
