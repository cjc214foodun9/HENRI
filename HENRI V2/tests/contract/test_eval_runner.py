"""Contract tests for the live AAII evaluation runner (henri_eval_runner.py).

WHAT THESE TESTS ARE FOR
    A gate that has never been observed to FAIL is not a gate, it is a comment.
    Every pilot gate therefore has an explicit negative control: the test
    breaks the gate's input on purpose and asserts that the chain REFUSES,
    names the gate, halts (later gates do not run) and never calls the adapter.

    The plumbing run is the anti-vacuity control in the other direction: if the
    harness cannot complete a 12-item synthetic run with zero infrastructure
    errors, then a real run's failure cannot be attributed to the benchmark.

RUN
    cd 'HENRI V2' && PYTHONPATH='HENRI V2' python -m pytest \
        tests/contract/test_eval_runner.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import henri_eval_runner as R
from henri_eval_infra import run_output_dir, sha256_bytes, sha256_text_lf

#: Field names read off the eight populated
#: ``Drive_Telemetry/mbpp_run*/run_evidence.json`` receipts of the previous
#: evidence surface. The new runner must keep emitting all of them.
PUBLISHED_RECEIPT_FIELDS = (
    "artifact_paths", "attempted_count", "benchmark_id", "checkpoint_load_status",
    "checkpoint_sha256", "command", "commit_sha256", "cuda_version",
    "dataset_sha256", "dataset_source", "declared_split_count", "device",
    "evaluator_id", "evaluator_sha256", "evaluator_version",
    "execution_error_count", "failed_count", "grader_mode", "item_count",
    "item_results_sha256",
)

PLUMBING_N = 12


# --- helpers -----------------------------------------------------------------

def lf_digest(path: Path) -> str:
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


@pytest.fixture(scope="module")
def plumbing_run(tmp_path_factory):
    """One canonical clean plumbing run, reused by the read-only assertions."""
    root = tmp_path_factory.mktemp("plumbing")
    return R.run_plumbing_mode(root=root, n=PLUMBING_N)


def run_dir_name(root: Path, cfg: R.RunConfig) -> str:
    """The canonical (commit_sha, benchmark_id, run_id) directory name."""
    probe = run_output_dir(root / "name-probe", cfg.commit_sha, cfg.benchmark_id, cfg.run_id)
    return probe.name


def fresh_plumbing_cfg(root: Path, n: int = 4, **over) -> R.RunConfig:
    cfg = R.build_plumbing_config(root, n=n)
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


class SpyAdapter:
    """Counts every call so a test can prove the adapter was never invoked."""

    adapter_id = "spy-adapter-v1"
    requires_code_execution = False

    def __init__(self, n: int = 4, yields: int | None = None,
                 fail_at: int | None = None, sagnac: float | None = 0.25,
                 status: str = "PASSED", requires_code_execution: bool = False):
        self.requires_code_execution = requires_code_execution
        self.n = n
        self._yields = n if yields is None else yields
        self.fail_at = fail_at
        self.sagnac = sagnac
        self.status = status
        self.evaluate_calls = 0
        self.items_calls = 0

    def item_count(self) -> int:
        return self.n

    def items(self):
        self.items_calls += 1
        for i in range(self._yields):
            yield {"item_id": f"spy-{i:03d}",
                   "prompt": f"spy probe prompt {i:03d} padded past the guard length"}

    def evaluate(self, item):
        if self.fail_at is not None and self.evaluate_calls == self.fail_at:
            self.evaluate_calls += 1
            raise RuntimeError("spy adapter exploded on purpose")
        self.evaluate_calls += 1
        return R.AdapterOutcome(status=self.status, raw_stdout="ok",
                                sagnac_delta=self.sagnac)


def assert_fail_closed(cfg: R.RunConfig, adapter, gate_name: str,
                       gates_run: int) -> dict:
    """Assert a gate failure halts, names the gate, and executes nothing."""
    with pytest.raises(R.GateFailure) as excinfo:
        R.run_evaluation(cfg, adapter)
    failure = excinfo.value
    assert failure.gate == gate_name, f"halted at {failure.gate}, expected {gate_name}"
    assert str(failure).startswith(f"PILOT GATE FAILED: {gate_name}: "), str(failure)
    assert failure.reason, "a gate failure must carry a reason"
    assert isinstance(failure.evidence, dict) and failure.evidence
    assert adapter.evaluate_calls == 0, "items executed despite a gate failure"

    run_dirs = [p for p in cfg.output_root.iterdir()
                if p.is_dir() and p.name.endswith(f"__{cfg.run_id}")]
    assert len(run_dirs) == 1, run_dirs
    run_dir = run_dirs[0]
    gate_receipt = json.loads((run_dir / "gate_receipt.json").read_text(encoding="utf-8"))
    assert gate_receipt["halted_at"] == gate_name
    assert gate_receipt["passed"] is False
    assert len(gate_receipt["gates"]) == gates_run, (
        "the chain did not halt: later gates executed after the failure")
    assert [g["gate"] for g in gate_receipt["gates"]][-1] == gate_name
    assert not (run_dir / "item_ledger.jsonl").exists(), "no items may be persisted"
    blocked = json.loads((run_dir / "run_receipt.json").read_text(encoding="utf-8"))
    assert blocked["status"] == "BLOCKED"
    assert blocked["blocked_by_gate"] == gate_name
    assert blocked["item_count"] == 0 and blocked["attempted_count"] == 0
    return failure.evidence


def make_ctx(tmp_path: Path, **over) -> R.PilotContext:
    base = dict(
        run_id="ctx-run", run_dir=tmp_path / "run", commit_sha="a" * 40,
        benchmark_id="ctx-bench", adapter_id="ctx-adapter",
        bundle_manifest=tmp_path / "no_manifest.json",
        dataset_path=None, dataset_sha256=None, store_paths=(),
        checkpoint_policy="disabled",
        checkpoint_skip_reason="test declares no trained decoder",
        decoder_source_paths=(), sandbox_mode="plumbing",
        adapter_requires_execution=False,
    )
    base.update(over)
    return R.PilotContext(**base)


def write_manifest(directory: Path, entries: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / "static_bundle.json"
    p.write_text(json.dumps({"schema_id": R.SCHEMA_STATIC_BUNDLE, "files": entries},
                            indent=2, sort_keys=True) + "\n",
                 encoding="utf-8", newline="\n")
    return p


# --- positive: the whole chain runs clean ------------------------------------

def test_plumbing_mode_runs_the_whole_chain_with_zero_infrastructure_errors(plumbing_run):
    assert plumbing_run.status == "OBSERVED"
    assert [g.gate for g in plumbing_run.gate_results] == list(R.GATE_ORDER)
    assert all(g.passed for g in plumbing_run.gate_results)

    receipt = plumbing_run.receipt
    assert receipt["accounting"]["valid"] is True
    assert receipt["accounting"]["problems"] == []
    assert receipt["run_error"] is None
    assert receipt["item_count"] == PLUMBING_N <= R.PLUMBING_MAX_ITEMS
    assert receipt["attempted_count"] + receipt["execution_error_count"] == \
        receipt["item_count"]
    assert receipt["passed_count"] + receipt["failed_count"] == receipt["attempted_count"]
    assert receipt["checkpoint_load_status"] == "LOADED"
    assert receipt["trained_decoder_active"] is True
    assert receipt["synthetic_source"] is True
    assert receipt["adaptation_enabled"] is False and receipt["run_index"] == 0
    # the arithmetic must be exercised by a real mix, not all-passes
    breakdown = receipt["outcome_breakdown"]
    assert breakdown.get("PASSED", 0) > 0
    assert breakdown.get("FAILED", 0) > 0
    assert breakdown.get("EXECUTION_ERROR", 0) > 0

    for name in ("gate_receipt.json", "item_ledger.jsonl", "raw_stdout.jsonl",
                 "raw_stderr.jsonl", "run_receipt.json"):
        assert (plumbing_run.run_dir / name).is_file(), name
    ledger_rows = list(plumbing_run.ledger_path.open(encoding="utf-8"))
    assert len(ledger_rows) == PLUMBING_N
    assert all(json.loads(line) for line in ledger_rows)


def test_gate_results_are_typed_and_carry_evidence(plumbing_run):
    assert len(plumbing_run.gate_results) == len(R.GATE_ORDER)
    for result in plumbing_run.gate_results:
        assert isinstance(result, R.GateResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.reason, str) and result.reason
        assert isinstance(result.evidence, dict) and result.evidence

    receipt = json.loads((plumbing_run.run_dir / "gate_receipt.json").read_text(
        encoding="utf-8"))
    assert receipt["schema_id"] == "henri.pilot-gate-chain.v1"
    assert receipt["order"] == list(R.GATE_ORDER)
    assert receipt["halted_at"] is None and receipt["passed"] is True


def test_chain_order_is_the_canonical_order():
    chain = R.build_pilot_gate_chain()
    assert tuple(g.name for g in chain) == R.GATE_ORDER
    assert R.GATE_ORDER == (
        "static_bundle_digest", "contamination_scan", "checkpoint_provenance",
        "decoder_fallback_source", "sandbox_availability")


# --- negative control per gate -----------------------------------------------

def test_gate1_negative_control_digest_mismatch_halts(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g1")
    adapter = SpyAdapter(n=3)
    # tamper AFTER the manifest pinned the digest (the classic silent drift)
    (tmp_path / "root" / "_plumbing_bundle" / "eval_adapter.py").write_text(
        '"""tampered"""\n', encoding="utf-8", newline="\n")
    evidence = assert_fail_closed(cfg, adapter, "static_bundle_digest", gates_run=1)
    assert evidence["mismatches"][0]["problem"] == "DIGEST_MISMATCH"


def test_gate1_negative_control_dataset_must_be_pinned_raw(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g1b")
    bundle = tmp_path / "root" / "_plumbing_bundle"
    doc = json.loads(cfg.bundle_manifest.read_text(encoding="utf-8"))
    for entry in doc["files"]:
        if entry["path"].endswith(".jsonl"):
            entry["digest_mode"] = "text_lf"          # LF digest still matches...
            entry["sha256"] = lf_digest(bundle / entry["path"])
    cfg.bundle_manifest.write_text(json.dumps(doc), encoding="utf-8", newline="\n")
    evidence = assert_fail_closed(cfg, SpyAdapter(n=3), "static_bundle_digest", gates_run=1)
    assert "hint" in evidence


def test_gate1_negative_control_missing_manifest(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g1c",
                             bundle_manifest=tmp_path / "nope.json")
    assert_fail_closed(cfg, SpyAdapter(n=3), "static_bundle_digest", gates_run=1)


def test_gate2_negative_control_contaminated_store_halts(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g2")
    store = cfg.store_paths[0]
    with store.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"item_id": "mbpp-7",
                                 "prompt": "mbpp task row leaked into the store"}) + "\n")
        handle.flush()
    evidence = assert_fail_closed(cfg, SpyAdapter(n=3), "contamination_scan", gates_run=2)
    assert evidence["contamination_hit_count"] >= 1


def test_gate2_negative_control_missing_store_halts(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g2b",
                             store_paths=(tmp_path / "no_such_store.jsonl",))
    assert_fail_closed(cfg, SpyAdapter(n=3), "contamination_scan", gates_run=2)


def test_gate3_negative_control_provenance_digest_mismatch(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g3")
    prov = json.loads(cfg.checkpoint_provenance.read_text(encoding="utf-8"))
    prov["checkpoint_sha256"] = "f" * 64
    cfg.checkpoint_provenance.write_text(json.dumps(prov), encoding="utf-8", newline="\n")
    assert_fail_closed(cfg, SpyAdapter(n=3), "checkpoint_provenance", gates_run=3)


def test_gate3_negative_control_checkpoint_trained_on_eval_split(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g3b")
    prov = json.loads(cfg.checkpoint_provenance.read_text(encoding="utf-8"))
    prov["dataset_sha256"] = "e" * 64          # different split than the run's dataset
    cfg.checkpoint_provenance.write_text(json.dumps(prov), encoding="utf-8", newline="\n")
    assert_fail_closed(cfg, SpyAdapter(n=3), "checkpoint_provenance", gates_run=3)


def test_gate3_negative_control_required_checkpoint_missing(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g3c",
                             checkpoint_path=tmp_path / "absent.pt")
    assert_fail_closed(cfg, SpyAdapter(n=3), "checkpoint_provenance", gates_run=3)


def test_gate4_negative_control_fallback_return_in_except(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g4")
    stub = tmp_path / "fallback_decoder.py"
    stub.write_text(
        "def decode(x):\n"
        "    try:\n"
        "        return x + 1\n"
        "    except Exception:\n"
        "        return True   # fabricates a PASS out of a fault\n",
        encoding="utf-8", newline="\n")
    cfg.decoder_source_paths = (stub,)
    evidence = assert_fail_closed(cfg, SpyAdapter(n=3), "decoder_fallback_source",
                                  gates_run=4)
    assert evidence["findings"][0]["status"] == "FABRICATED_CONSTANT_RETURN"


def test_gate4_negative_control_named_fallback_function(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g4b")
    stub = tmp_path / "named_fallback.py"
    stub.write_text("def stub_answer(*args):\n    return True\n",
                    encoding="utf-8", newline="\n")
    cfg.decoder_source_paths = (stub,)
    evidence = assert_fail_closed(cfg, SpyAdapter(n=3), "decoder_fallback_source",
                                  gates_run=4)
    assert evidence["findings"][0]["status"] == "FABRICATED_FALLBACK_FUNCTION"


def test_gate4_negative_control_unparseable_source(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g4c")
    broken = tmp_path / "broken_decoder.py"
    broken.write_text("def decode(:\n", encoding="utf-8", newline="\n")
    cfg.decoder_source_paths = (broken,)
    assert_fail_closed(cfg, SpyAdapter(n=3), "decoder_fallback_source", gates_run=4)


def test_gate5_negative_control_unknown_sandbox_mode(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g5",
                             sandbox_mode="none")
    adapter = SpyAdapter(n=3, requires_code_execution=True)
    evidence = assert_fail_closed(cfg, adapter, "sandbox_availability", gates_run=5)
    assert "allowed" in evidence


def test_gate5_negative_control_plumbing_mode_refuses_executing_adapter(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g5b")
    adapter = SpyAdapter(n=3, requires_code_execution=True)
    evidence = assert_fail_closed(cfg, adapter, "sandbox_availability", gates_run=5)
    assert evidence["adapter_id"] == "spy-adapter-v1"


def test_gate5_negative_control_no_posix_boundary_on_this_host(tmp_path):
    import os
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="g5c",
                             sandbox_mode="namespace")
    adapter = SpyAdapter(n=3, requires_code_execution=True)
    if os.name != "posix":
        evidence = assert_fail_closed(cfg, adapter, "sandbox_availability", gates_run=5)
        assert evidence["sandbox_mode"] == "namespace"
    else:  # pragma: no cover - only on a POSIX host with a broken unshare
        with pytest.raises(R.GateFailure):
            R.run_evaluation(cfg, adapter)


def test_negative_control_exists_for_every_gate(tmp_path):
    """Table-driven coverage: no gate ships without a fail-closed control."""
    cases = {
        "static_bundle_digest": lambda root: fresh_plumbing_cfg(
            root, n=2, run_id="nc1", bundle_manifest=root / "missing.json"),
        "contamination_scan": lambda root: fresh_plumbing_cfg(
            root, n=2, run_id="nc2", store_paths=(root / "missing_store.jsonl",)),
        "checkpoint_provenance": lambda root: fresh_plumbing_cfg(
            root, n=2, run_id="nc3", checkpoint_path=root / "missing.pt"),
        "decoder_fallback_source": lambda root: fresh_plumbing_cfg(
            root, n=2, run_id="nc4", decoder_source_paths=()),
        "sandbox_availability": lambda root: fresh_plumbing_cfg(
            root, n=2, run_id="nc5", sandbox_mode="bogus"),
    }
    assert set(cases) == set(R.GATE_ORDER)
    for index, gate_name in enumerate(R.GATE_ORDER, start=1):
        root = tmp_path / gate_name
        root.mkdir(parents=True, exist_ok=True)
        cfg = cases[gate_name](root)
        if gate_name == "sandbox_availability":
            adapter = SpyAdapter(n=2, requires_code_execution=True)
        else:
            adapter = SpyAdapter(n=2, requires_code_execution=True)
        assert_fail_closed(cfg, adapter, gate_name, gates_run=index)


# --- gate-level unit checks (the gates are testable in isolation) ------------

def test_lf_and_raw_digest_rules_are_both_honoured(tmp_path):
    crlf = tmp_path / "manifest.txt"
    crlf.write_bytes(b"alpha\r\nbeta\r\n")
    assert sha256_bytes(crlf.read_bytes().replace(b"\r\n", b"\n")) == \
        sha256_text_lf(crlf.read_text(encoding="utf-8"))
    assert sha256_bytes(crlf.read_bytes()) != lf_digest(crlf)

    gate = R.StaticBundleDigestGate()
    # text_lf mode tolerates an autocrlf rewrite; raw mode does not
    ok_manifest = write_manifest(tmp_path / "ok", [
        {"path": "../manifest.txt", "digest_mode": "text_lf",
         "sha256": lf_digest(crlf)}])
    ctx = make_ctx(tmp_path, bundle_manifest=ok_manifest)
    assert gate.run(ctx).passed is True

    bad_manifest = write_manifest(tmp_path / "bad", [
        {"path": "../manifest.txt", "digest_mode": "raw", "sha256": lf_digest(crlf)}])
    result = gate.run(make_ctx(tmp_path, bundle_manifest=bad_manifest))
    assert result.passed is False and result.reason == "DIGEST_MISMATCH"


def test_unknown_digest_mode_is_refused(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8", newline="\n")
    manifest = write_manifest(tmp_path / "m", [
        {"path": "../a.txt", "digest_mode": "sha1", "sha256": lf_digest(f)}])
    result = R.StaticBundleDigestGate().run(make_ctx(tmp_path, bundle_manifest=manifest))
    assert result.passed is False and result.reason == "UNKNOWN_DIGEST_MODE"


def test_contamination_gate_requires_a_declared_target(tmp_path):
    result = R.ContaminationScanGate().run(make_ctx(tmp_path))
    assert result.passed is False and result.reason == "NO_SCAN_TARGET_DECLARED"


def test_contamination_gate_rejects_unparseable_rows(tmp_path):
    store = tmp_path / "store.jsonl"
    store.write_text("{not json}\n", encoding="utf-8", newline="\n")
    result = R.ContaminationScanGate().run(make_ctx(tmp_path, store_paths=(store,)))
    assert result.passed is False and result.reason == "STORE_ROW_UNPARSEABLE"


def test_contamination_gate_accepts_a_clean_store(tmp_path):
    store = tmp_path / "store.jsonl"
    store.write_text(json.dumps({"record_id": "zonec-clean-0",
                                 "content": "documentation prose, no benchmark rows"}) + "\n",
                     encoding="utf-8", newline="\n")
    result = R.ContaminationScanGate().run(make_ctx(tmp_path, store_paths=(store,)))
    assert result.passed is True and result.evidence["rows_scanned"] == 1


def test_contamination_gate_ignores_prose_that_merely_names_a_benchmark(tmp_path):
    """The documented K-D rule: marker AND an id field carrying a digit."""
    store = tmp_path / "store.jsonl"
    store.write_text(json.dumps({
        "record_id": "zonec-doc-0",
        "content": "this changelog mentions mbpp and humaneval as excluded",
        "note": "no item identifier here"}) + "\n",
        encoding="utf-8", newline="\n")
    result = R.ContaminationScanGate().run(make_ctx(tmp_path, store_paths=(store,)))
    assert result.passed is True


def test_decoder_gate_allows_reraise_and_flags_silent_pass(tmp_path):
    ok = tmp_path / "ok_decoder.py"
    ok.write_text(
        "def decode(x):\n"
        "    try:\n"
        "        return x + 1\n"
        "    except ValueError:\n"
        "        raise\n",
        encoding="utf-8", newline="\n")
    gate = R.DecoderFallbackSourceGate()
    assert gate.run(make_ctx(tmp_path, decoder_source_paths=(ok,))).passed is True

    silent = tmp_path / "silent_decoder.py"
    silent.write_text(
        "def decode(x):\n"
        "    try:\n"
        "        return x + 1\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8", newline="\n")
    result = gate.run(make_ctx(tmp_path, decoder_source_paths=(silent,)))
    assert result.passed is False
    assert result.evidence["findings"][0]["status"] == "SILENT_EXCEPT_SWALLOW"


def test_checkpoint_gate_requires_a_stated_reason_when_disabled(tmp_path):
    result = R.CheckpointProvenanceGate().run(
        make_ctx(tmp_path, checkpoint_skip_reason="   "))
    assert result.passed is False
    assert result.reason == "CHECKPOINT_DISABLED_WITHOUT_REASON"


# --- ledger / aggregation ordering ------------------------------------------

def test_per_item_rows_reach_disk_before_the_next_item_runs(tmp_path):
    """Proof, not assertion: the adapter reads the ledger mid-run."""
    root = tmp_path / "root"
    cfg = fresh_plumbing_cfg(root, n=8, run_id="observed")
    run_dir = root / run_dir_name(root, cfg)

    class Observing(R.SyntheticPlumbingAdapter):
        def _observe(self):
            if not self.ledger_path.exists():
                self.observed_row_counts.append(0)
                self.complete_lines.append(True)
                return
            text = self.ledger_path.read_text(encoding="utf-8")
            self.observed_row_counts.append(len(text.splitlines()))
            self.complete_lines.append(text.endswith("\n"))

    adapter = Observing(n=8)
    adapter.ledger_path = run_dir / "item_ledger.jsonl"
    adapter.complete_lines = []
    result = R.run_evaluation(cfg, adapter)

    assert result.status == "OBSERVED"
    assert adapter.observed_row_counts == list(range(8)), (
        "item i must see exactly i persisted rows before it executes")
    assert all(adapter.complete_lines)
    assert result.receipt["items_executed"] == 8
    assert len(list(result.ledger_path.open(encoding="utf-8"))) == 8
    # aggregation reads the ledger back: the receipt digest IS the file digest
    assert result.receipt["item_results_sha256"] == sha256_bytes(
        result.ledger_path.read_bytes())


def test_ledger_rows_carry_every_required_telemetry_field(plumbing_run):
    from henri_eval_infra import RUN_ITEM_FIELDS
    rows = list(plumbing_run.ledger_path.open(encoding="utf-8"))
    assert rows
    for line in rows:
        row = json.loads(line)
        missing = [f for f in RUN_ITEM_FIELDS if f not in row]
        assert not missing, missing
        assert 0.0 <= row["sagnac_delta"] <= 2.0
        assert row["status"] in {"PASSED", "FAILED", "EXECUTION_ERROR"}


# --- reconciliation ----------------------------------------------------------

def test_reconciliation_flags_invalid_run_when_ledger_is_short(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="short")
    adapter = SpyAdapter(n=5, yields=4)          # declares 5, yields 4
    result = R.run_evaluation(cfg, adapter)

    assert result.status == "INVALID"
    assert result.receipt["item_count"] == 5
    assert result.receipt["attempted_count"] == 4
    assert result.receipt["attempted_count"] + \
        result.receipt["execution_error_count"] != result.receipt["item_count"]
    problems = result.receipt["accounting"]["problems"]
    assert any("item_count" in p for p in problems), problems
    # an INVALID receipt cannot be loaded as RunEvidence at all: the promoted
    # schema itself refuses non-reconciling arithmetic.
    from pydantic import ValidationError
    from henri_benchmark_registry import RunEvidence
    projected = {k: v for k, v in result.receipt.items()
                 if k in RunEvidence.model_fields}
    with pytest.raises(ValidationError):
        RunEvidence(**projected)


def test_adapter_exception_yields_invalid_run_never_a_silent_pass(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="boom")
    adapter = SpyAdapter(n=6, fail_at=2)
    result = R.run_evaluation(cfg, adapter)

    assert result.status == "INVALID"
    assert result.receipt["run_error"] and "RuntimeError" in result.receipt["run_error"]
    assert result.receipt["attempted_count"] == 2 < result.receipt["item_count"] == 6
    problems = result.receipt["accounting"]["problems"]
    assert any(p.startswith("RUN_ABORTED") for p in problems), problems
    assert any("item_count" in p for p in problems)


def test_out_of_range_sagnac_delta_is_refused(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="sagnac")
    adapter = SpyAdapter(n=2, sagnac=2.5)
    result = R.run_evaluation(cfg, adapter)
    assert result.status == "INVALID"
    assert "outside [0,2]" in (result.receipt["run_error"] or "")
    assert result.receipt["attempted_count"] == 0


def test_declared_split_smaller_than_run_is_refused(tmp_path):
    cfg = fresh_plumbing_cfg(tmp_path / "root", n=3, run_id="split",
                             declared_split_count=2)
    result = R.run_evaluation(cfg, SpyAdapter(n=3))
    assert result.status == "INVALID"
    assert any("declared_split_count" in p for p in result.receipt["accounting"]["problems"])


# --- output directories -------------------------------------------------------

def test_run_directory_is_unique_per_commit_benchmark_and_run_id(tmp_path):
    root = tmp_path / "root"
    cfg = fresh_plumbing_cfg(root, n=2, run_id="fixed-run")
    first = R.run_evaluation(cfg, SpyAdapter(n=2))
    assert cfg.benchmark_id in first.run_dir.name
    assert cfg.commit_sha[:12] in first.run_dir.name
    assert "fixed-run" in first.run_dir.name

    # same key again -> refuse to overwrite an existing run's evidence
    cfg2 = fresh_plumbing_cfg(root, n=2, run_id="fixed-run")
    with pytest.raises(FileExistsError):
        R.run_evaluation(cfg2, SpyAdapter(n=2))

    # a new run_id gets its own directory, and the first one survives untouched
    cfg3 = fresh_plumbing_cfg(root, n=2, run_id="second-run")
    second = R.run_evaluation(cfg3, SpyAdapter(n=2))
    assert second.run_dir != first.run_dir
    assert first.run_dir.is_dir() and second.run_dir.is_dir()
    assert (first.run_dir / "run_receipt.json").is_file()
    assert len(list(root.glob("plumbing-synthetic-v1__*"))) == 2


def test_first_run_is_frozen_and_refuses_adaptation(tmp_path):
    root = tmp_path / "root"
    cfg = fresh_plumbing_cfg(root, n=2, run_id="frozen", adaptation_enabled=True)
    with pytest.raises(R.FrozenFirstRunViolation):
        R.run_evaluation(cfg, SpyAdapter(n=2))
    assert not list(root.glob("plumbing-synthetic-v1__*")), (
        "the refusal must happen before any run directory is created")

    cfg2 = fresh_plumbing_cfg(root, n=2, run_id="adapted", run_index=1,
                              adaptation_enabled=True)
    result = R.run_evaluation(cfg2, SpyAdapter(n=2))
    assert result.status == "OBSERVED"
    assert result.receipt["adaptation_enabled"] is True
    assert result.receipt["run_index"] == 1


# --- task rows must never reach disk ----------------------------------------

def test_no_benchmark_task_row_is_persisted_anywhere(plumbing_run):
    prompts = [f"synthetic plumbing item {i:03d}: no benchmark data"
               for i in range(PLUMBING_N)]
    for artifact in plumbing_run.run_dir.rglob("*"):
        if not artifact.is_file():
            continue
        blob = artifact.read_bytes().decode("utf-8", errors="replace")
        for prompt in prompts:
            assert prompt not in blob, f"prompt text persisted in {artifact}"
    for line in plumbing_run.ledger_path.open(encoding="utf-8"):
        row = json.loads(line)
        assert "prompt" not in row
        assert len(row["prompt_sha256"]) == 64


def test_task_row_persistence_guard_actually_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prompt = "a synthetic plumbing item long enough to be a task row"
    (run_dir / "leak.jsonl").write_text(json.dumps({"prompt": prompt}) + "\n",
                                        encoding="utf-8", newline="\n")
    with pytest.raises(R.TaskRowPersistenceViolation):
        R.assert_no_task_rows_persisted(run_dir, [prompt])
    assert R.assert_no_task_rows_persisted(run_dir, ["short"])["prompts_checked"] == 0


def test_leaked_prompt_turns_the_run_invalid(tmp_path):
    """Negative control for the task-row guard: an adapter that echoes the
    prompt into stdout must NOT produce an OBSERVED run."""
    root = tmp_path / "root"
    cfg = fresh_plumbing_cfg(root, n=2, run_id="leak")

    class Echoing(SpyAdapter):
        def evaluate(self, item):
            self.evaluate_calls += 1
            return R.AdapterOutcome(status="PASSED", raw_stdout=item["prompt"],
                                    sagnac_delta=0.1)

    result = R.run_evaluation(cfg, Echoing(n=2))
    assert result.status == "INVALID"
    assert result.receipt["accounting"]["valid"] is False
    assert any("TASK_ROW_PERSISTED" in p
               for p in result.receipt["accounting"]["problems"])
    assert "violations" in result.receipt["task_row_persistence"] or \
        "error" in result.receipt["task_row_persistence"]


def test_clean_run_reports_a_task_row_guard_pass(tmp_path):
    root = tmp_path / "root"
    cfg = fresh_plumbing_cfg(root, n=2, run_id="clean")
    result = R.run_evaluation(cfg, SpyAdapter(n=2))
    assert result.status == "OBSERVED"
    report = result.receipt["task_row_persistence"]
    assert report["prompts_checked"] == 2
    assert report["violations"] == []
    assert report["files_checked"] >= 5


# --- receipt schema compatibility -------------------------------------------

def test_receipt_keeps_every_published_mbpp_field_name(plumbing_run):
    receipt = plumbing_run.receipt
    missing = [f for f in PUBLISHED_RECEIPT_FIELDS if f not in receipt]
    assert not missing, missing
    assert receipt["schema_id"] == "henri.run-evidence.v1"


def test_receipt_validates_against_the_existing_run_evidence_schema(plumbing_run):
    from henri_benchmark_registry import RunEvidence
    projected = {k: v for k, v in plumbing_run.receipt.items()
                 if k in RunEvidence.model_fields}
    model = RunEvidence(**projected)
    assert model.status == "OBSERVED"
    assert model.item_count == PLUMBING_N
    assert model.passed_count + model.failed_count == model.attempted_count
    assert model.attempted_count + model.execution_error_count + model.vetoed_count == \
        model.item_count
    assert model.item_results_sha256 == plumbing_run.receipt["item_results_sha256"]


def test_plumbing_receipt_can_never_be_promoted_to_a_score(plumbing_run):
    from henri_benchmark_registry import BenchmarkRecord, BenchmarkRegistry, \
        RunEvidence, validate_score_eligibility
    registry = BenchmarkRegistry(
        schema_id="henri.benchmark-registry.v1", benchlm_source_uri="x",
        retrieved_at_utc=None, source_sha256="b" * 64, source_root_type="list",
        records=[BenchmarkRecord(benchmark_id=plumbing_run.receipt["benchmark_id"],
                                 display_name="plumbing")])
    projected = {k: v for k, v in plumbing_run.receipt.items()
                 if k in RunEvidence.model_fields}
    eligible, reasons = validate_score_eligibility(RunEvidence(**projected), registry)
    assert not eligible
    assert "SYNTHETIC_DATASET_DETECTED" in reasons
    # the record exists but is BLOCKED: metadata alone can never promote a score
    assert "ADAPTER_NOT_EVALUATED" in reasons
    assert "PRIMARY_SOURCE_NOT_VERIFIED" in reasons


def test_returned_receipt_equals_the_receipt_on_disk(plumbing_run):
    """A reported number that is not the number on disk is not evidence."""
    on_disk = json.loads((plumbing_run.run_dir / "run_receipt.json").read_text(
        encoding="utf-8"))
    assert on_disk == plumbing_run.receipt
    assert on_disk["status"] == "OBSERVED"
    assert on_disk["task_row_persistence"]["files_checked"] >= 5


def test_plumbing_output_root_is_hermetic_under_a_temp_root(tmp_path):
    root = tmp_path / "plumbing_root"
    result = R.run_plumbing_mode(root=root, n=6)
    assert result.status == "OBSERVED"
    assert result.receipt["item_count"] == 6
    assert str(root) in str(result.run_dir)


def test_plumbing_adapter_item_count_is_capped(tmp_path):
    with pytest.raises(ValueError):
        R.SyntheticPlumbingAdapter(n=R.PLUMBING_MAX_ITEMS + 1)
    with pytest.raises(ValueError):
        R.build_plumbing_config(tmp_path / "root", n=R.PLUMBING_MAX_ITEMS + 1)
