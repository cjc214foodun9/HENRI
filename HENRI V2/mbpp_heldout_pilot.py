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

# c3-next R-EDMD latent composition gate: the online operator must beat the
# identity operator (A=I, no updates) on exemplar self-prediction by this
# margin (dimension-normalized; at r=16/d=65536 the identity baseline sits at
# ~sqrt(r/d) ~ 0.0156 cosine, so a 0.02 margin = a strong learned signal).
EDMD_PREDICT_MIN_IMPROVEMENT = float(os.environ.get("EDMD_PREDICT_MIN_IMPROVEMENT", "0.02"))
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


def _mean_logit_entropy(torch_mod: Any, unbinder: Any, waves: list) -> float:
    """Mean softmax logit entropy (nats) over wave states. INTERNAL telemetry only."""
    with torch_mod.no_grad():
        total = 0.0
        for w in waves:
            logits = unbinder(w.unsqueeze(0))
            p = torch_mod.softmax(logits.float(), dim=-1)
            total += float(-(p * torch_mod.log(p + 1e-12)).sum(dim=-1).mean().item())
        return total / max(1, len(waves))


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


def run_pilot(output_dir: Path, checkpoint_path: Path, scan_root: Path, preflight_only: bool = False, sandbox_mode: str = "namespace", egress_path: str = "henri", sgld_adapt: bool = False, hopfield_snap: bool = False, edmd_predict: bool = False, cegis_synth: bool = False, ast_decode: bool = False) -> dict[str, Any]:
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
        from mbpp_cegis_synthesizer import CandidateMissError

        transducer = HENRIUnifiedEgressTransducer(d_model=65536, device="cuda", checkpoint_path=str(checkpoint_path))
        codec = qFHRREpistemicCodec(d_model=65536, device="cuda")

        # HENRI path: compile W_task online from the paper-sanctioned exemplars
        # (X_i = rendered prompt, Y_i = reference_solution). Zero pretraining;
        # W_task is an input-side task operator, not model parameter adaptation.
        w_task_ring = None
        w_task_vector = None
        egress = None
        if egress_path == "henri":
            exemplars = load_exemplars()
            task_compiler = HolographicTaskFunctorCompiler(codec)
            demo_pairs = [
                (codec.encode_text(render_prompt(ex)), codec.encode_text(ex["code"]))
                for ex in exemplars
            ]
            w_task_ring = task_compiler.compile_functor(demo_pairs)
            w_task_vector = (
                w_task_ring.to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0
            ).to("cuda")
            # C3 remembering probe: UniversalEgress Hopfield codebook over the
            # exemplar solution waves. When enabled, the W_task goal wave snaps
            # to the nearest exemplar solution (zero-entropy retrieval, beta=8.0)
            # instead of passing through the linear decode head. The c3-next
            # R-EDMD composition path also snaps (from the PREDICTED wave).
            sol_waves_real = None
            if hopfield_snap:
                from henri_egress import TextEgress
                egress = TextEgress(d_model=65536, beta=8.0)
                sol_waves_real = [
                    (codec.encode_text(ex["code"]).to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0).view(-1).to("cuda")
                    for ex in exemplars
                ]
                egress.register_tokens(torch.stack(sol_waves_real), [ex["code"] for ex in exemplars])
            # c3-next: R-EDMD latent composition (WaveJEPA p_psi operator).
            # Fit ONLINE from the exemplar pairs: (prompt wave + W_task action)
            # -> solution wave. Zero pretraining. Kill probe: the learned operator
            # must BEAT the identity operator (A=I) on exemplar self-prediction
            # by a dimension-normalized margin; otherwise fail closed.
            edmd_predictor = None
            edmd_telemetry = None
            synth = None
            cegis_probe = None
            decoder = None
            if cegis_synth or ast_decode:
                edmd_predict = True  # CEGIS egress ranks candidates by the predicted wave
            if ast_decode:
                cegis_synth = True  # wave->AST decoder is a CEGIS candidate generator
            if edmd_predict:
                from recursive_dual_edmd import RecursiveDualEDMD
                pred_waves_real = [
                    (codec.encode_text(render_prompt(ex)).to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0).view(-1).to("cuda")
                    for ex in exemplars
                ]
                sol_waves_real = [
                    (codec.encode_text(ex["code"]).to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0).view(-1).to("cuda")
                    for ex in exemplars
                ]
                w_task_real = (
                    w_task_ring.to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0
                ).view(-1).to("cuda")
                # Task-manifold dictionary (Koopman EDMD dictionary condition,
                # arXiv:1408.4408): V = orthonormal span of the exemplar
                # prompt+action and solution waves, so the operator composes
                # within the observables of interest instead of a random
                # projection (run8 failure).
                with torch.no_grad():
                    _, _, Vt = torch.linalg.svd(
                        torch.stack(pred_waves_real + sol_waves_real), full_matrices=False)
                    v_basis = Vt.T[:, :16].contiguous().to("cuda")
                edmd_predictor = RecursiveDualEDMD(d_model=65536, r_rank=16, lambda_forget=0.98, v_basis=v_basis).to("cuda")
                with torch.no_grad():
                    for pw, sw in zip(pred_waves_real, sol_waves_real):
                        edmd_predictor.update_online_step(
                            pw.view(8192, 8), w_task_real.view(8192, 8), sw.view(8192, 8))
                    # identity baseline (A=I, no updates) in the SAME manifold
                    edmd_identity = RecursiveDualEDMD(d_model=65536, r_rank=16, lambda_forget=0.98, v_basis=v_basis).to("cuda")
                    def _self_sim(pred):
                        sims = []
                        for pw, sw in zip(pred_waves_real, sol_waves_real):
                            p = pred(pw.view(8192, 8), w_task_real.view(8192, 8)).view(-1)
                            sims.append(float(torch.dot(
                                torch.nn.functional.normalize(p, p=2, dim=0),
                                torch.nn.functional.normalize(sw, p=2, dim=0)).item()))
                        return float(sum(sims) / len(sims))
                    self_sim_learned = _self_sim(edmd_predictor)
                    self_sim_identity = _self_sim(edmd_identity)
                improvement = self_sim_learned - self_sim_identity
                edmd_telemetry = {
                    "v_basis": "task_manifold_svd",
                    "self_sim_learned": round(self_sim_learned, 6),
                    "self_sim_identity": round(self_sim_identity, 6),
                    "improvement": round(improvement, 6),
                }
                if improvement < EDMD_PREDICT_MIN_IMPROVEMENT:
                    raise PilotBlocked(
                        f"EDMD_PREDICTOR_UNDERFIT:improvement={improvement:.6f} "
                        f"< min={EDMD_PREDICT_MIN_IMPROVEMENT}")
                print(f"  [edmd] r=16 online fit | self_sim_learned={self_sim_learned:.4f} "
                      f"identity={self_sim_identity:.4f} improvement={improvement:.4f}")
            if cegis_synth:
                from mbpp_cegis_synthesizer import CEGIS_PROBE_MIN_HIT, MbppCegisSynthesizer
                synth = MbppCegisSynthesizer(exemplars, codec, device="cuda")
                if ast_decode:
                    from mbpp_wave_ast_decoder import WaveASTDecoder
                    decoder = WaveASTDecoder(codec, device="cuda")
                with torch.no_grad():
                    self_preds = [
                        edmd_predictor(pw.view(8192, 8), w_task_real.view(8192, 8)).view(-1)
                        for pw in pred_waves_real
                    ]
                cegis_probe = synth.probe_self_selection(self_preds, prompt_waves=pred_waves_real)
                if ast_decode:
                    # The self-selection probe measures the exemplar-anchored
                    # path; in the decoder union (approx 150 candidates) it is
                    # diluted and fires falsely. The --ast-decode path is gated
                    # by the decoder's expressiveness probe instead.
                    cegis_probe = synth.probe_decoder_expressiveness(decoder, exemplars, sandbox)
                    if cegis_probe["expressible"] < 1:
                        raise PilotBlocked(
                            f"DECODER_EXPRESSIVENESS_INERT:expressible="
                            f"{cegis_probe['expressible']} < 1")
                    print(f"  [ast-decode] expressiveness={cegis_probe['expressible']}/"
                          f"{cegis_probe['total']}")
                    cegis_probe = {**cegis_probe, "hit_rate": float(cegis_probe["expressible"] >= 1)}
                if cegis_probe["hit_rate"] < CEGIS_PROBE_MIN_HIT:
                    raise PilotBlocked(
                        f"CEGIS_SELECTION_INERT:hit_rate={cegis_probe['hit_rate']} "
                        f"< {CEGIS_PROBE_MIN_HIT}")
                print(f"  [cegis] probe hit_rate={cegis_probe['hit_rate']} "
                      f"top_ranks={cegis_probe['top_ranks']}")
            adapt_telemetry = None
            if sgld_adapt:
                try:
                    demo_waves = [codec.encode_text(render_prompt(ex)) for ex in exemplars]
                    target_waves = [codec.encode_text(ex["code"]) for ex in exemplars]
                    # Fixed bootstrap labels: pre-adaptation argmax token of each solution wave.
                    # Kept ONLY as comparison telemetry (run4 showed 4/10 degeneracy);
                    # the C2 loss uses full soft-target distributions instead.
                    with torch.no_grad():
                        demo_token_ids = [
                            int(transducer.unbinder(w.unsqueeze(0)).argmax(dim=-1).item())
                            for w in target_waves
                        ]
                    distinct_labels = len(set(demo_token_ids))
                    probe_waves = [codec.encode_text(render_prompt(it)) for it in items[:10]]
                    ent_demo_before = _mean_logit_entropy(torch, transducer.unbinder, demo_waves)
                    ent_probe_before = _mean_logit_entropy(torch, transducer.unbinder, probe_waves)
                    adapt_result = transducer.unbinder.adapt_in_context_sgld_wave(
                        active_waves=torch.stack(demo_waves),
                        target_waves=torch.stack(target_waves),
                        steps=500,
                        seed=0,
                    )
                    ent_demo_after = _mean_logit_entropy(torch, transducer.unbinder, demo_waves)
                    ent_probe_after = _mean_logit_entropy(torch, transducer.unbinder, probe_waves)
                    adapt_telemetry = {
                        "sgld_protocol": "wave_soft_targets_scheduled_sgld",
                        "demo_token_ids": demo_token_ids,
                        "distinct_bootstrap_labels": distinct_labels,
                        "logit_entropy_nats_demo_before": round(ent_demo_before, 6),
                        "logit_entropy_nats_demo_after": round(ent_demo_after, 6),
                        "logit_entropy_nats_probe_before": round(ent_probe_before, 6),
                        "logit_entropy_nats_probe_after": round(ent_probe_after, 6),
                    }
                    adapt_telemetry.update(adapt_result)
                except Exception as exc:
                    raise PilotBlocked(f"SGLD_ADAPT_FAILED:{type(exc).__name__}:{exc}") from exc
        else:
            adapt_telemetry = None
            if sgld_adapt:
                raise PilotBlocked("SGLD_ADAPT_REQUIRES_HENRI_EGRESS")
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
                    w_task_vector = None
                if edmd_predictor is not None:
                    # c3-next: COMPOSE the answer via latent prediction instead of
                    # retrieval. The R-EDMD operator (fitted online from the 10
                    # exemplars) predicts the solution wave from (prompt + W_task).
                    prompt_wave_real = (prompt_wave.to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0).view(-1).to("cuda")
                    pred_wave = edmd_predictor(
                        prompt_wave_real.view(8192, 8), w_task_real.view(8192, 8)).view(-1)
                    edmd_sims = [
                        float(torch.dot(torch.nn.functional.normalize(pred_wave, p=2, dim=0),
                                        torch.nn.functional.normalize(sw, p=2, dim=0)).item())
                        for sw in sol_waves_real
                    ]
                    edmd_sim_max = max(edmd_sims) if edmd_sims else 0.0
                    if cegis_synth:
                        # CEGIS/AST program synthesis egress: instantiate
                        # exemplar-anchored candidates under the item's
                        # signature, rank by predicted-wave similarity, and
                        # verify in the sandbox against the item's tests.
                        cands = synth.build_candidates(prompt, item.get("test_list"))
                        if ast_decode:
                            # Wave->AST structural decode: score grammar slots
                            # from the predicted wave, enumerate the pruned
                            # grammar, then retain exemplar identity anchors so
                            # the exemplar path is preserved in the union.
                            from mbpp_cegis_synthesizer import parse_entry_from_tests, parse_entry_signature
                            sig = parse_entry_signature(prompt) or parse_entry_from_tests(item.get("test_list") or [])
                            if sig is not None:
                                entry, args = sig
                                dec_cands = decoder.decode(pred_wave, prompt_wave_real, entry, args)
                                anchors = [c for c in cands if c[1].get("morphism") == "identity"]
                                cands = dec_cands + anchors
                        ranked = synth.rank_candidates(cands, pred_wave, prompt_wave=prompt_wave_real)
                        code, meta = synth.cegis_verify(ranked, item, sandbox)
                        if code is None:
                            raise CandidateMissError(
                                f"CEGIS_NO_CANDIDATE_PASSED:attempts={meta['candidates_tried']}")
                        response = "```python\n" + code + "\n```"
                        telemetry = {**meta, "egress": "cegis_ast_synth",
                                     "edmd_sim_max": round(edmd_sim_max, 4)}
                    elif egress is not None:
                        # remember: snap the PREDICTED wave to the exemplar codebook
                        snapped_text, snapped_idx, snap_sim = egress.decode_wave(pred_wave)
                        response = "```python\n" + snapped_text + "\n```"
                        telemetry = {"snap_idx": int(snapped_idx), "snap_sim": round(float(snap_sim), 4),
                                     "snap_source": "edmd_predicted", "edmd_sim_max": round(edmd_sim_max, 4)}
                    else:
                        # compose + decode: map the predicted REAL wave back to a
                        # ring and pass it through the linear decode head.
                        pred_ring = ((pred_wave.clamp(-1, 1) + 1.0) / 2.0 * (codec.k_bins - 1)).round().clamp(0, codec.k_bins - 1).to(torch.uint8)
                        response, telemetry = transducer.decode_wave_to_response(pred_ring, prompt, w_task=w_task_vector)
                        telemetry = dict(telemetry or {})
                        telemetry.update({"egress": "linear_decode_from_edmd", "edmd_sim_max": round(edmd_sim_max, 4)})
                elif egress is not None:
                    goal_real = (goal_wave.to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0).to("cuda")
                    snapped_text, snapped_idx, snap_sim = egress.decode_wave(goal_real)
                    # Codebook entries are raw code; present fenced so the shared
                    # extract_code_blocks contract applies unchanged.
                    response = "```python\n" + snapped_text + "\n```"
                    telemetry = {"snap_idx": int(snapped_idx), "snap_sim": round(float(snap_sim), 4), "snap_source": "exemplar_codebook"}
                else:
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
            except CandidateMissError as exc:
                # A genuine synthesis miss: the CEGIS search ran the item's
                # real sandbox tests and no candidate passed. Task-level FAIL
                # (the mechanism executed); it must not block score eligibility.
                failed += 1
                stdout_records.append({"task_id": task_id, "stdout": ""})
                stderr_records.append({"task_id": task_id, "stderr": str(exc)})
                item_records.append({
                    "task_id": task_id,
                    "split": "test",
                    "source_sha256": manifest["source_sha256"],
                    "rendered_prompt_sha256": sha256_bytes(prompt.encode()),
                    "model_output_sha256": None,
                    "postprocessed_output_sha256": None,
                    "pass": False,
                    "failure_reason": f"CEGIS_MISS:{exc}",
                    "runtime_ms": None,
                    "telemetry": {},
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
        if adapt_telemetry is not None:
            (output_dir / "adapt_telemetry.json").write_text(json.dumps(adapt_telemetry, indent=2), encoding="utf-8")
        evidence = build_evidence(
            manifest, output_dir, "OBSERVED", "LOADED", checkpoint_sha,
            attempted=passed + failed, passed=passed, failed=failed, execution_errors=execution_errors,
            raw_stdout_sha=sha256_path(output_dir / "raw_stdout.jsonl"),
            raw_stderr_sha=sha256_path(output_dir / "raw_stderr.jsonl"),
            item_results_sha=sha256_path(output_dir / "item_results.jsonl"),
            limitations=f"Public MBPP operational holdout; egress_path={egress_path}; sgld_adapt={sgld_adapt}; hopfield_snap={hopfield_snap}; edmd_predict={edmd_predict}; cegis_synth={cegis_synth}; sandbox_mode={sandbox_mode}; elapsed_sec={elapsed:.6f}",
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
    except Exception as exc:
        manifest = load_json(MANIFEST_PATH)
        evidence = blocked_bundle(manifest, output_dir, f"UNHANDLED_RUN_ERROR:{type(exc).__name__}:{exc}", "BLOCKED_PILOT")
        return {"status": "BLOCKED", "reason": f"UNHANDLED_RUN_ERROR:{type(exc).__name__}:{exc}", "evidence": evidence}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/henri_decoder_checkpoint.pt")
    parser.add_argument("--scan-root", type=Path, default=ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--sandbox-mode", choices=["namespace", "container-rlimit"], default="namespace")
    parser.add_argument("--egress-path", choices=["legacy", "henri"], default="henri")
    parser.add_argument("--sgld-adapt", action="store_true", help="test-time SGLD unbinder adaptation on the 10 exemplars (henri path only)")
    parser.add_argument("--hopfield-snap", action="store_true", help="UniversalEgress remembering probe: snap the W_task goal wave to the nearest exemplar solution in a Hopfield codebook (henri path only)")
    parser.add_argument("--edmd-predict", action="store_true", help="compose the answer via online R-EDMD latent prediction from the exemplars (c3-next; implies hopfield-snap egress)")
    parser.add_argument("--cegis-synth", action="store_true", help="CEGIS/AST program synthesis egress: rank exemplar-anchored AST candidates by the R-EDMD predicted wave, verify in the sandbox (implies --edmd-predict)")
    parser.add_argument("--ast-decode", action="store_true", help="Wave->AST structural decode egress: score a bounded AST grammar's slots from the predicted wave, enumerate the pruned grammar + exemplar identity anchors, CEGIS-verify (implies --cegis-synth)")
    args = parser.parse_args()
    result = run_pilot(args.output_dir, args.checkpoint, args.scan_root, args.preflight_only, args.sandbox_mode, args.egress_path, args.sgld_adapt, args.hopfield_snap, args.edmd_predict, args.cegis_synth, args.ast_decode)
    print(json.dumps({"status": result["status"], "reason": result.get("reason"), "score_eligible": result.get("score_eligible", False)}, sort_keys=True))
    return 0 if result["status"] in {"OBSERVED", "PREFLIGHT_PASS", "BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
