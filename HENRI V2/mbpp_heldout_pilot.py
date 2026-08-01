"""MBPP Google test-split pilot with immutable provenance and fail-closed scoring.

This module does not download data, use reference solutions, adapt online, retry
outputs, or turn infrastructure success into task correctness. Remote execution
requires CUDA, the exact checkpoint, a POSIX network-disabled sandbox, and a
clean contamination scan.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from henri_benchmark_registry import BenchmarkRecord, BenchmarkRegistry, RunEvidence, validate_score_eligibility
from mbpp_contamination_scan import scan as run_contamination_scan
from mbpp_secure_executor import SandboxUnavailable, SecurePythonSandbox


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data/official_benchmarks/mbpp_google_test_v1_manifest.json"
SOURCE_PATH = ROOT / "data/official_benchmarks/canonical/mbpp/mbpp.jsonl"
CHECKPOINT_PROVENANCE_PATH = ROOT / "data/official_benchmarks/mbpp_henri_checkpoint_provenance_v1.json"
PROMPT_CONTRACT_PATH = ROOT / "data/official_benchmarks/evaluators/mbpp_henri_prompt_contract_v1.json"
FEWSHOT_CONTRACT_PATH = ROOT / "data/official_benchmarks/evaluators/mbpp_henri_fewshot10_contract_v1.json"
EXEMPLAR_IDS = list(range(1, 11))
DECODER_PATH = ROOT / "henri_decoder.py"

CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n?(.*?)\n?```", re.DOTALL)
FALLBACK_MARKER = "def solution():\n    return True"
FALLBACK_SOURCE_MARKER = r"def solution():\n    return True"


class PilotBlocked(RuntimeError):
    """The run cannot produce a valid external outcome."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_lf_path(path: Path) -> str:
    """Hash canonical LF bytes for text artifacts.

    Manifest digests are computed over the canonical LF forms of text files.
    Windows git checkouts (core.autocrlf) materialize CRLF working copies;
    hashing raw working-tree bytes would make the check platform-dependent.
    """
    raw = path.read_bytes().replace(bytes((13, 10)), b"\n")
    return sha256_bytes(raw)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_items() -> list[dict[str, Any]]:
    items = []
    for line in SOURCE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            task_id = int(item["task_id"])
            if 11 <= task_id <= 510:
                items.append(item)
    items.sort(key=lambda item: int(item["task_id"]))
    if len(items) != 500 or [int(item["task_id"]) for item in items] != list(range(11, 511)):
        raise PilotBlocked("MBPP_TEST_SPLIT_INVALID")
    return items


def load_exemplars() -> list[dict[str, Any]]:
    """Load the paper-sanctioned few-shot exemplars (task_id 1..10).

    Exemplars are the MBPP paper's own few-shot set; they are distinct from
    the heldout 11..510 and are used only to compile W_task at test time.
    """
    items = []
    for line in SOURCE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if int(item["task_id"]) in EXEMPLAR_IDS:
                items.append(item)
    items.sort(key=lambda item: int(item["task_id"]))
    if len(items) != 10 or [int(item["task_id"]) for item in items] != EXEMPLAR_IDS:
        raise PilotBlocked("MBPP_EXEMPLAR_SPLIT_INVALID")
    for ex in items:
        if not isinstance(ex.get("text"), str) or not isinstance(ex.get("code"), str):
            raise PilotBlocked(f"MBPP_EXEMPLAR_SCHEMA_INVALID:{ex.get('task_id')}")
    return items


def validate_static_bundle(egress_path: str = "zero_shot") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(MANIFEST_PATH)
    contract_key = "zero_shot" if egress_path == "legacy" else "henri_fewshot10"
    contract_info = (manifest.get("prompt_contracts") or {}).get(contract_key)
    if contract_info is None:
        raise PilotBlocked(f"PROMPT_CONTRACT_NOT_IN_MANIFEST:{contract_key}")
    contract_path = ROOT / contract_info["artifact"]
    prompt_contract = load_json(contract_path)
    if contract_info["sha256"] != sha256_lf_path(contract_path):
        raise PilotBlocked("PROMPT_CONTRACT_DIGEST_MISMATCH")
    checkpoint_provenance = load_json(CHECKPOINT_PROVENANCE_PATH)
    if manifest["source_sha256"] != sha256_path(SOURCE_PATH):
        raise PilotBlocked("DATASET_DIGEST_MISMATCH")
    evaluator = manifest["evaluator"]
    evaluator_dir = ROOT / "data/official_benchmarks/evaluators/lm-evaluation-harness/mbpp"
    parts = []
    for name in ("mbpp.yaml", "utils.py"):
        path = evaluator_dir / name
        parts.append(name.encode() + b"\0" + path.read_bytes().replace(bytes((13, 10)), b"\n"))
    if evaluator["bundle_sha256"] != sha256_bytes(b"\0".join(parts)):
        raise PilotBlocked("EVALUATOR_BUNDLE_DIGEST_MISMATCH")
    if manifest["checkpoint_provenance_artifact"] != str(CHECKPOINT_PROVENANCE_PATH.relative_to(ROOT)).replace("\\", "/"):
        raise PilotBlocked("CHECKPOINT_PROVENANCE_PATH_MISMATCH")
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_provenance.get("checkpoint_sha256", "")):
        raise PilotBlocked("CHECKPOINT_PROVENANCE_INVALID")
    if int(checkpoint_provenance.get("expected_bytes", 0)) <= 0:
        raise PilotBlocked("CHECKPOINT_PROVENANCE_SIZE_INVALID")
    items = load_items()
    if manifest["item_count"] != len(items):
        raise PilotBlocked("MANIFEST_ITEM_COUNT_MISMATCH")
    if prompt_contract["reference_code_exposed"] or prompt_contract["online_adaptation"] or prompt_contract["zone_c_task_persistence"]:
        raise PilotBlocked("REFERENCE_CODE_EXPOSURE_OR_ONLINE_ADAPTATION_ENABLED")
    if contract_key == "zero_shot":
        if prompt_contract["num_fewshot"] != 0:
            raise PilotBlocked("REFERENCE_CODE_EXPOSURE_ENABLED")
    else:
        if prompt_contract["num_fewshot"] != 10 or list(prompt_contract["exemplar_ids"]) != EXEMPLAR_IDS:
            raise PilotBlocked("FEWSHOT_CONTRACT_INVALID")
    return manifest, items


def render_prompt(item: dict[str, Any]) -> str:
    tests = item.get("test_list")
    if not isinstance(item.get("text"), str) or not isinstance(tests, list) or len(tests) < 3:
        raise PilotBlocked(f"MBPP_ITEM_SCHEMA_INVALID:{item.get('task_id')}")
    return (
        "You are an expert Python programmer, and here is your task: "
        + item["text"]
        + " Your code should pass these tests:\n\n"
        + "\n".join(str(test) for test in tests[:3])
        + "\n[BEGIN]\n"
    )


def extract_code_blocks(text: str) -> str:
    matches = CODE_BLOCK_RE.findall(text)
    if not matches:
        text_without_lang = re.sub(r"```python", "```", text)
        matches = CODE_BLOCK_RE.findall(text_without_lang)
    return matches[0] if matches else ""


def validate_candidate(code: str) -> None:
    if not code.strip():
        raise PilotBlocked("MODEL_OUTPUT_EMPTY")
    if FALLBACK_MARKER in code:
        raise PilotBlocked("DECODER_FALLBACK_OUTPUT_REACHED")
    try:
        ast.parse(code, filename="<mbpp_generated>")
    except SyntaxError as exc:
        raise PilotBlocked(f"MODEL_OUTPUT_SYNTAX_PATH_INVALID:{exc.msg}") from exc


def checkpoint_preflight(path: Path, provenance: dict[str, Any]) -> str:
    if not path.exists():
        raise PilotBlocked("CHECKPOINT_MISSING")
    digest = sha256_path(path)
    if digest != provenance["checkpoint_sha256"]:
        raise PilotBlocked("CHECKPOINT_DIGEST_MISMATCH")
    if path.stat().st_size != provenance["expected_bytes"]:
        raise PilotBlocked("CHECKPOINT_SIZE_MISMATCH")
    return digest


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_registry(manifest: dict[str, Any], evaluated: bool) -> BenchmarkRegistry:
    evaluator = manifest["evaluator"]
    return BenchmarkRegistry(
        schema_id="henri.benchmark-registry.v1",
        benchlm_source_uri=manifest["source_uri"],
        retrieved_at_utc=None,
        source_sha256=manifest["source_sha256"],
        source_root_type="dict",
        records=[BenchmarkRecord(
            benchmark_id=manifest["benchmark_id"],
            display_name=manifest["display_name"],
            family="coding",
            canonical_source=manifest["source_uri"],
            official_split=manifest["official_split_rule"],
            evaluator_id=evaluator["evaluator_id"],
            evaluator_version=evaluator["evaluator_version"],
            evaluator_sha256=evaluator["bundle_sha256"],
            dataset_sha256=manifest["source_sha256"],
            adapter_status="EVALUATED" if evaluated else "ADAPTER_READY",
            block_reason=None if evaluated else "REMOTE_RUN_PENDING",
        )],
    )


def _artifact_rel(path: Path) -> str:
    """Repo-relative artifact path when inside ROOT, absolute path otherwise."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def build_evidence(
    manifest: dict[str, Any],
    output_dir: Path,
    status: str,
    checkpoint_status: str,
    checkpoint_sha: str | None,
    attempted: int,
    passed: int,
    failed: int,
    execution_errors: int,
    raw_stdout_sha: str,
    raw_stderr_sha: str,
    item_results_sha: str,
    limitations: str,
) -> RunEvidence:
    evaluator = manifest["evaluator"]
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch_version = "UNAVAILABLE"
        cuda_version = None
        device = "unavailable"
    return RunEvidence(
        schema_id="henri.run-evidence.v1",
        status=status,
        run_id=output_dir.name,
        commit_sha256=git_commit(),
        command=" ".join(sys.argv),
        benchmark_id=manifest["benchmark_id"],
        dataset_source=manifest["source_uri"],
        dataset_sha256=manifest["source_sha256"],
        evaluator_id=evaluator["evaluator_id"],
        evaluator_version=evaluator["evaluator_version"],
        evaluator_sha256=evaluator["bundle_sha256"],
        checkpoint_sha256=checkpoint_sha,
        checkpoint_load_status=checkpoint_status,
        trained_decoder_active=checkpoint_status == "LOADED",
        device=device,
        torch_version=torch_version,
        cuda_version=cuda_version,
        item_count=500,
        attempted_count=attempted,
        passed_count=passed,
        failed_count=failed,
        execution_error_count=execution_errors,
        vetoed_count=0,
        raw_stdout_sha256=raw_stdout_sha,
        raw_stderr_sha256=raw_stderr_sha,
        item_results_sha256=item_results_sha,
        artifact_paths=[_artifact_rel(path) for path in output_dir.glob("*")],
        limitations=limitations,
        grader_mode="isolated_assertion_execution_pinned_mbpp_pass_at_1",
        synthetic_source=False,
        task_leakage_detected=False,
        declared_split_count=500,
    )


def blocked_bundle(manifest: dict[str, Any], output_dir: Path, reason: str, checkpoint_status: str) -> RunEvidence:
    write_jsonl(output_dir / "raw_stdout.jsonl", [])
    write_jsonl(output_dir / "raw_stderr.jsonl", [{"run_error": reason}])
    write_jsonl(output_dir / "item_results.jsonl", [])
    raw_stdout_sha = sha256_path(output_dir / "raw_stdout.jsonl")
    raw_stderr_sha = sha256_path(output_dir / "raw_stderr.jsonl")
    item_results_sha = sha256_path(output_dir / "item_results.jsonl")
    evidence = build_evidence(
        manifest, output_dir, "BLOCKED", checkpoint_status, None,
        attempted=0, passed=0, failed=0, execution_errors=500,
        raw_stdout_sha=raw_stdout_sha, raw_stderr_sha=raw_stderr_sha,
        item_results_sha=item_results_sha, limitations=reason,
    )
    write_json(output_dir / "run_evidence.json", evidence.model_dump(mode="json"))
    return evidence


def run_pilot(output_dir: Path, checkpoint_path: Path, scan_root: Path, preflight_only: bool = False, sandbox_mode: str = "namespace", egress_path: str = "henri") -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        manifest, items = validate_static_bundle(egress_path)
        provenance = load_json(CHECKPOINT_PROVENANCE_PATH)
        source_exclusions = [
            SOURCE_PATH,
            MANIFEST_PATH,
            PROMPT_CONTRACT_PATH,
            CHECKPOINT_PROVENANCE_PATH,
            output_dir,
        ]
        scan_result = run_contamination_scan(SOURCE_PATH, [scan_root], source_exclusions)
        write_json(output_dir / "contamination_scan.json", scan_result)
        if scan_result["matches"]:
            return {"status": "BLOCKED", "reason": "TASK_EXPOSURE_MATCH", "evidence": blocked_bundle(manifest, output_dir, "TASK_EXPOSURE_MATCH", "BLOCKED_PREFLIGHT")}
        checkpoint_sha = checkpoint_preflight(checkpoint_path, provenance)
        if FALLBACK_SOURCE_MARKER in DECODER_PATH.read_text(encoding="utf-8"):
            return {"status": "BLOCKED", "reason": "DECODER_FALLBACK_PATH_PRESENT", "evidence": blocked_bundle(manifest, output_dir, "DECODER_FALLBACK_PATH_PRESENT", "FAILED_MODEL_PATH_PREFLIGHT")}
        try:
            sandbox = SecurePythonSandbox(mode=sandbox_mode)
        except SandboxUnavailable as exc:
            return {"status": "BLOCKED", "reason": str(exc), "evidence": blocked_bundle(manifest, output_dir, str(exc), "BLOCKED_PREFLIGHT")}
        probe_result = sandbox.execute("print(41 + 1)")
        if probe_result.status != "PASS":
            return {"status": "BLOCKED", "reason": f"SANDBOX_PROBE_FAILED:{probe_result.status}:{probe_result.stderr.strip()[:200]}", "evidence": blocked_bundle(manifest, output_dir, f"SANDBOX_PROBE_FAILED:{probe_result.status}", "BLOCKED_PREFLIGHT")}
        if egress_path == "henri":
            load_exemplars()  # validate exemplar split/schema before any run
        if preflight_only:
            return {"status": "PREFLIGHT_PASS", "reason": "STATIC_AND_SANDBOX_PREFLIGHT_ONLY", "checkpoint_sha256": checkpoint_sha}

        import torch
        if not torch.cuda.is_available():
            return {"status": "BLOCKED", "reason": "CUDA_REQUIRED", "evidence": blocked_bundle(manifest, output_dir, "CUDA_REQUIRED", "LOADED")}
        from henri_decoder import HENRIUnifiedEgressTransducer
        from zone_c_epistemic_axiom_harness import HolographicTaskFunctorCompiler, qFHRREpistemicCodec

        transducer = HENRIUnifiedEgressTransducer(d_model=65536, device="cuda", checkpoint_path=str(checkpoint_path))
        codec = qFHRREpistemicCodec(d_model=65536, device="cuda")

        # HENRI path: compile W_task online from the paper-sanctioned exemplars
        # (X_i = rendered prompt, Y_i = reference_solution). Zero pretraining;
        # W_task is an input-side task operator, not model parameter adaptation.
        w_task_ring = None
        w_task_vector = None
        if egress_path == "henri":
            exemplars = load_exemplars()
            task_compiler = HolographicTaskFunctorCompiler(codec)
            demo_pairs = [
                (codec.encode_text(render_prompt(ex)), codec.encode_text(ex["code"]))
                for ex in exemplars
            ]
            w_task_ring = task_compiler.compile_functor(demo_pairs)
            w_task_vector = (w_task_ring.to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0).to("cuda")
        stdout_records = []
        stderr_records = []
        item_records = []
        started = time.perf_counter()
        passed = 0
        failed = 0
        execution_errors = 0
        for item in items:
            task_id = int(item["task_id"])
            try:
                prompt = render_prompt(item)
                prompt_wave = codec.encode_text(prompt)
                if egress_path == "henri":
                    goal_wave = codec.bind_hadamard(w_task_ring, prompt_wave)
                else:
                    task_operator = codec.encode_text("MBPP_CODING_OPERATOR")
                    goal_wave = codec.bind_hadamard(task_operator, prompt_wave)
                response, telemetry = transducer.decode_wave_to_response(goal_wave, prompt, w_task=w_task_vector)
                code = extract_code_blocks(response)
                validate_candidate(code)
                result = sandbox.execute(code + "\n" + "\n".join(item["test_list"]))
                is_pass = result.status == "PASS"
                passed += int(is_pass)
                failed += int(not is_pass)
                stdout_records.append({"task_id": task_id, "stdout": result.stdout})
                stderr_records.append({"task_id": task_id, "stderr": result.stderr})
                item_records.append({
                    "task_id": task_id,
                    "split": "test",
                    "source_sha256": manifest["source_sha256"],
                    "rendered_prompt_sha256": sha256_bytes(prompt.encode()),
                    "model_output_sha256": sha256_bytes(response.encode()),
                    "postprocessed_output_sha256": sha256_bytes(code.encode()),
                    "pass": is_pass,
                    "failure_reason": None if is_pass else result.status,
                    "runtime_ms": result.runtime_ms,
                    "telemetry": telemetry,
                })
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                # Item-level fail-closed: an invalid-AST decode, empty output,
                # or sandbox refusal records as an execution error for this
                # item and the run continues. Any execution error blocks score
                # promotion; it is never an observed task outcome.
                execution_errors += 1
                stdout_records.append({"task_id": task_id, "stdout": ""})
                stderr_records.append({"task_id": task_id, "stderr": f"{type(exc).__name__}: {exc}"})
                item_records.append({
                    "task_id": task_id,
                    "split": "test",
                    "source_sha256": manifest["source_sha256"],
                    "rendered_prompt_sha256": None,
                    "model_output_sha256": None,
                    "postprocessed_output_sha256": None,
                    "pass": False,
                    "failure_reason": f"EXECUTION_ERROR:{type(exc).__name__}:{exc}",
                    "runtime_ms": None,
                    "telemetry": {},
                })
        elapsed = time.perf_counter() - started
        write_jsonl(output_dir / "raw_stdout.jsonl", stdout_records)
        write_jsonl(output_dir / "raw_stderr.jsonl", stderr_records)
        write_jsonl(output_dir / "item_results.jsonl", item_records)
        evidence = build_evidence(
            manifest, output_dir, "OBSERVED", "LOADED", checkpoint_sha,
            attempted=passed + failed, passed=passed, failed=failed, execution_errors=execution_errors,
            raw_stdout_sha=sha256_path(output_dir / "raw_stdout.jsonl"),
            raw_stderr_sha=sha256_path(output_dir / "raw_stderr.jsonl"),
            item_results_sha=sha256_path(output_dir / "item_results.jsonl"),
            limitations=f"Public MBPP operational holdout; egress_path={egress_path}; sandbox_mode={sandbox_mode}; elapsed_sec={elapsed:.6f}",
        )
        registry = make_registry(manifest, evaluated=True)
        eligible, reasons = validate_score_eligibility(evidence, registry, minimum_items=500)
        write_json(output_dir / "run_evidence.json", evidence.model_dump(mode="json"))
        if not eligible:
            return {"status": "OBSERVED", "score_eligible": False, "reason": "SCORE_PROMOTION_BLOCKED:" + ",".join(reasons), "evidence": evidence}
        return {"status": "OBSERVED", "score_eligible": True, "evidence": evidence}
    except PilotBlocked as exc:
        manifest = load_json(MANIFEST_PATH)
        evidence = blocked_bundle(manifest, output_dir, str(exc), "FAILED_REQUIRED_CHECKPOINT")
        return {"status": "BLOCKED", "reason": str(exc), "evidence": evidence}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/henri_decoder_checkpoint.pt")
    parser.add_argument("--scan-root", type=Path, default=ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--sandbox-mode", choices=["namespace", "container-rlimit"], default="namespace")
    parser.add_argument("--egress-path", choices=["legacy", "henri"], default="henri")
    args = parser.parse_args()
    result = run_pilot(args.output_dir, args.checkpoint, args.scan_root, args.preflight_only, args.sandbox_mode, args.egress_path)
    print(json.dumps({"status": result["status"], "reason": result.get("reason"), "score_eligible": result.get("score_eligible", False)}, sort_keys=True))
    return 0 if result["status"] in {"OBSERVED", "PREFLIGHT_PASS", "BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
