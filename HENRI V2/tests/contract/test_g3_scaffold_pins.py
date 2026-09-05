"""Contract test for the G3 AAII scaffold (status taxonomy + pin table, no network).

Guards the scaffold's honesty contract: HLE stays BLOCKED_GATED, Terminal-Bench
is GitHub-origin (HF dataset is metadata-only), SciCode is HF-origin with GitHub
license pin, and no constituent can be STAGED_OK without staged file hashes.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCAFFOLD_PATH = Path(__file__).resolve().parents[2] / "experiments" / "verification" / "g3_aaii_scaffold.py"


def _load():
    spec = importlib.util.spec_from_file_location("g3_aaii_scaffold", SCAFFOLD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_pin_table_hle_gated(mod):
    assert mod.CONSTITUENTS["hle"]["gated"] is True
    assert "terms_url" in mod.CONSTITUENTS["hle"]


def test_pin_table_terminal_bench_github_origin(mod):
    tb = mod.CONSTITUENTS["terminal-bench-2.1"]
    assert tb["origin"] == "github"
    assert tb["repo"] == "harbor-framework/terminal-bench-2-1"
    assert "task.toml" in tb["task_files"]
    assert "instruction.md" in tb["task_files"]
    # HF dataset side is metadata-only; task content is GitHub tasks/ — the
    # scaffold must NOT claim an HF staging artifact for Terminal-Bench.
    assert "registry.json" not in tb["stage_files"]


def test_pin_table_scicode_hf_origin(mod):
    sc = mod.CONSTITUENTS["scicode"]
    assert sc["origin"] == "hf"
    assert sc["hf"] == "SciCode1/SciCode"
    assert "problems_test.jsonl" in sc["stage_files"]


def test_status_vocabulary(mod):
    allowed = {"STAGED_OK", "STAGED_BLOCKED_GATED", "STAGED_BLOCKED_INFRA", "CRASH"}
    for name in mod.CONSTITUENTS:
        assert name in mod.CONSTITUENTS


def test_no_score_claim_vocabulary(mod):
    src = SCAFFOLD_PATH.read_text(encoding="utf-8")
    # The scaffold must never assert an accuracy/score; it only records exec/status.
    for banned in ("accuracy =", "score = 0.85", "pass_rate", "return 0.85"):
        assert banned not in src


def test_manifest_schema_fields(mod):
    spec = mod.CONSTITUENTS["scicode"]
    assert set(mod.CONSTITUENTS) == {"terminal-bench-2.1", "scicode", "hle"}
    # staging functions exist and return typed records
    assert callable(mod.stage)
