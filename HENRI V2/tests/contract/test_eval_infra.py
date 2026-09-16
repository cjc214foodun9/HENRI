"""Contract tests for the S0 evidence infrastructure (henri_eval_infra).

These are S0's own gate. Each test asserts a MEASURABLE property, and every
invariant that exists because of a session-measured defect carries a NEGATIVE
CONTROL that must fail it.

Measured defects this module exists to prevent:
  * two runs sharing one output filename destroyed an earlier verdict;
  * aggregate-only telemetry lost every row when aggregation crashed;
  * a text manifest hashed with raw bytes did not reproduce from a fresh clone
    (core.autocrlf=true strips CR on commit);
  * a published receipt hash described the scratch file, not the committed blob.
"""
from __future__ import annotations

import json
import pytest

from henri_eval_infra import (
    RUN_ITEM_FIELDS,
    STATUS_VALUES,
    ItemLedger,
    assert_sagnac,
    build_run_receipt,
    reconcile,
    run_id_new,
    run_output_dir,
    scan_contamination,
    sha256_bytes,
    sha256_text_lf,
)


def _good_row(item_id: str = "i0", **over):
    row = {
        "item_id": item_id,
        "prompt_sha256": "a" * 64,
        "raw_stdout_sha256": "b" * 64,
        "raw_stderr_sha256": "c" * 64,
        "status": "PASSED",
        "elapsed_ms": 1.5,
        "tool_calls": 0,
        "retries": 0,
        "candidate_scores": [],
        "sagnac_delta": 0.5,
        "token_top1": 7,
        "token_entropy": 1.0,
    }
    row.update(over)
    return row


# --- reconciliation arithmetic ---------------------------------------------

def test_reconcile_valid():
    r = reconcile(item_count=4, attempted=3, passed=2, failed=1, execution_errors=1)
    assert r["valid"] is True and r["problems"] == []


def test_reconcile_detects_passed_plus_failed_mismatch():
    r = reconcile(item_count=4, attempted=3, passed=1, failed=1, execution_errors=1)
    assert r["valid"] is False
    assert any("attempted" in p for p in r["problems"])


def test_reconcile_detects_attempted_plus_errors_mismatch():
    r = reconcile(item_count=5, attempted=3, passed=2, failed=1, execution_errors=1)
    assert r["valid"] is False
    assert any("item_count" in p for p in r["problems"])


def test_reconcile_rejects_negative_counts():
    r = reconcile(item_count=4, attempted=3, passed=4, failed=-1, execution_errors=1)
    assert r["valid"] is False


# --- digests: LF-canonical text vs raw bytes -------------------------------

def test_text_lf_digest_equals_lf_bytes_and_differs_from_crlf_bytes():
    text_crlf = "a\r\nb\r\n"
    assert sha256_text_lf(text_crlf) == sha256_bytes(b"a\nb\n")
    assert sha256_text_lf(text_crlf) != sha256_bytes(b"a\r\nb\r\n")


def test_raw_bytes_digest_is_stable_for_identical_input():
    assert sha256_bytes(b"x") == sha256_bytes(b"x")
    assert sha256_bytes(b"x") != sha256_bytes(b"y")


# --- run identity and unique output paths ----------------------------------

def test_run_id_is_unique_across_calls():
    ids = {run_id_new() for _ in range(50)}
    assert len(ids) == 50


def test_run_output_dir_is_unique_and_refuses_reuse(tmp_path):
    d1 = run_output_dir(tmp_path, "a" * 40, "sci", "run-1")
    assert d1.is_dir()
    # NEGATIVE CONTROL: reusing the same key must RAISE, not silently overwrite.
    with pytest.raises(FileExistsError):
        run_output_dir(tmp_path, "a" * 40, "sci", "run-1")
    d2 = run_output_dir(tmp_path, "a" * 40, "sci", "run-2")
    assert d1 != d2 and d2.is_dir()


def test_run_output_dir_sanitises_hostile_benchmark_id(tmp_path):
    d = run_output_dir(tmp_path, "b" * 40, "aa/brief:case*?", "r")
    assert d.is_dir()
    assert "/" not in d.name and ":" not in d.name and "*" not in d.name


# --- ItemLedger -------------------------------------------------------------

def test_ledger_appends_and_round_trips(tmp_path):
    led = ItemLedger(tmp_path / "items.jsonl")
    led.append(_good_row("i0"))
    led.append(_good_row("i1"))
    assert led.count == 2
    rows = list(led.rows())
    assert [r["item_id"] for r in rows] == ["i0", "i1"]
    assert led.sha256() == sha256_bytes((tmp_path / "items.jsonl").read_bytes())


def test_ledger_rejects_missing_field(tmp_path):
    led = ItemLedger(tmp_path / "items.jsonl")
    with pytest.raises(ValueError):
        led.append({"item_id": "bad"})
    assert led.count == 0


@pytest.mark.parametrize("missing", RUN_ITEM_FIELDS)
def test_ledger_requires_every_declared_field(tmp_path, missing):
    led = ItemLedger(tmp_path / f"{missing}.jsonl")
    row = _good_row()
    del row[missing]
    with pytest.raises(ValueError):
        led.append(row)


def test_ledger_rejects_unknown_status(tmp_path):
    led = ItemLedger(tmp_path / "items.jsonl")
    with pytest.raises(ValueError):
        led.append(_good_row(status="MADE_UP"))


@pytest.mark.parametrize("status", sorted(STATUS_VALUES))
def test_ledger_accepts_every_declared_status(tmp_path, status):
    led = ItemLedger(tmp_path / f"{status}.jsonl")
    led.append(_good_row(status=status))
    assert led.count == 1


@pytest.mark.parametrize("bad", [2.0000001, -0.0001, 3.0, 9.9])
def test_ledger_rejects_sagnac_out_of_bounds(tmp_path, bad):
    led = ItemLedger(tmp_path / f"s{abs(hash(bad))}.jsonl")
    with pytest.raises(ValueError):
        led.append(_good_row(sagnac_delta=bad))


@pytest.mark.parametrize("ok", [0.0, 2.0, 1.0])
def test_ledger_accepts_sagnac_boundaries(tmp_path, ok):
    led = ItemLedger(tmp_path / f"ok{ok}.jsonl")
    led.append(_good_row(sagnac_delta=ok))
    assert led.count == 1


def test_ledger_accepts_none_sagnac(tmp_path):
    led = ItemLedger(tmp_path / "none.jsonl")
    led.append(_good_row(sagnac_delta=None))
    assert led.count == 1


def test_assert_sagnac_bounds():
    assert assert_sagnac(1.0) == 1.0
    with pytest.raises(ValueError):
        assert_sagnac(2.5)
    with pytest.raises(ValueError):
        assert_sagnac(-0.5)


# --- contamination scanner (K-D) -------------------------------------------

def test_scanner_flags_benchmark_task_row():
    rows = [{"prompt": "solve this scicode problem", "item_id": "17"}]
    r = scan_contamination(rows)
    assert r["contaminated"] is True
    assert r["hits"] and r["rows_scanned"] == 1


def test_scanner_ignores_clean_rows():
    rows = [{"prompt": "zone c engram prior", "note": "no identifier"}]
    assert scan_contamination(rows)["contaminated"] is False


def test_scanner_ignores_marker_without_item_id():
    # A document that merely NAMES a benchmark is not a task row.
    rows = [{"prompt": "the scicode methodology page describes grading",
             "note": "no numeric item id"}]
    assert scan_contamination(rows)["contaminated"] is False


def test_scanner_handles_empty_input():
    r = scan_contamination([])
    assert r["contaminated"] is False and r["rows_scanned"] == 0


def test_scanner_handles_non_dict_rows():
    r = scan_contamination(["not a dict", 42, None])
    assert r["contaminated"] is False


# --- run receipt ------------------------------------------------------------

def test_receipt_is_observed_when_accounting_is_valid(tmp_path):
    led = ItemLedger(tmp_path / "items.jsonl")
    led.append(_good_row())
    rec = build_run_receipt(run_id="r", commit_sha="c" * 40, benchmark_id="sci",
                            item_count=1, attempted=1, passed=1, failed=0,
                            execution_errors=0, ledger=led)
    assert rec["schema_id"] == "henri.run-evidence.v1"
    assert rec["status"] == "OBSERVED"
    assert rec["accounting"]["valid"] is True
    assert rec["item_results_sha256"] == led.sha256()
    # Field names must match the pre-existing MBPP run_evidence.json receipts.
    for key in ("benchmark_id", "commit_sha256", "item_count", "attempted_count",
                "failed_count", "execution_error_count", "dataset_source",
                "dataset_sha256", "run_id"):
        assert key in rec


def test_receipt_is_invalid_when_accounting_breaks(tmp_path):
    rec = build_run_receipt(run_id="r", commit_sha="c" * 40, benchmark_id="sci",
                            item_count=9, attempted=1, passed=1, failed=0,
                            execution_errors=0)
    assert rec["status"] == "INVALID"
    assert rec["accounting"]["problems"]


def test_receipt_serialises_to_json(tmp_path):
    rec = build_run_receipt(run_id="r", commit_sha="c" * 40, benchmark_id="sci",
                            item_count=0, attempted=0, passed=0, failed=0,
                            execution_errors=0)
    json.dumps(rec)  # must not raise (tensors/objects would break json.dump)
