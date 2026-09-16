"""Live AAII evaluation runner: pluggable adapter + fail-closed pilot gate chain.

WHY THIS MODULE EXISTS (session-measured defects, 2026-09-15/16)

1. ``run_evaluation`` refuses to call the adapter until five pilot gates pass
   IN ORDER. Every one of the quarantined evaluators in
   ``_archive/invalid_evaluators/`` could start executing items before it had
   proved that its inputs were the inputs it claimed. A run that discovers a
   digest mismatch after 500 items have executed has already burned the GPU
   window and, worse, has already produced numbers that look reportable.
   DEFECT PREVENTED: evidence produced from an unverified bundle.

2. Per-item telemetry goes to an append-only JSONL ledger, flushed AND fsync'd
   per row, BEFORE any aggregation reads it. The aggregation step reads the
   ledger BACK FROM DISK and derives passed/failed/execution_errors from the
   persisted rows, never from an in-memory tally.
   DEFECT PREVENTED: a crash between "the arms executed" and "telemetry was
   written" losing the evidence, and a tally that can disagree with the file.

3. ``reconcile`` (henri_eval_infra) is applied to the ledger-derived counts
   against the adapter's DECLARED item count. A mismatch is INVALID_RUN, never
   a warning: ``passed + failed == attempted`` and
   ``attempted + execution_errors == item_count``.
   DEFECT PREVENTED: reporting a subset of a split as if it were the split.

4. Output goes to a unique directory keyed to (commit_sha, benchmark_id,
   run_id) via ``run_output_dir``, which uses ``exist_ok=False``.
   DEFECT PREVENTED: a later run overwriting an earlier run's verdict.

5. Every gate returns a typed ``(passed, reason, evidence)`` result. A failure
   names the gate and HALTS: later gates do not run, the adapter is never
   called, and the run directory still receives ``gate_receipt.json`` plus a
   BLOCKED ``run_receipt.json``. Silent partial success is impossible.

6. ``run_plumbing_mode`` exercises the WHOLE chain (all five gates, ledger,
   reconciliation, receipt) on a synthetic adapter of <= 16 items so the
   harness itself can be validated with zero infrastructure errors. The
   plumbing receipt is deliberately NOT score-eligible: it is annotated
   ``synthetic_source=True`` and its dataset_source names the synthetic origin,
   so ``validate_score_eligibility`` rejects it. A harness validation can never
   be laundered into a benchmark score.

SCHEMA RELATIONSHIP
    This module extends, and does not replace, the three schemas declared in
    ``henri_benchmark_registry`` (henri.benchmark-registry.v1,
    henri.run-evidence.v1, henri.arc-episode-trace.v1). It emits
    ``henri.run-evidence.v1`` with the same field names as the populated MBPP
    receipts, plus a ``henri.pilot-gate-chain.v1`` gate receipt.

WHAT THIS MODULE DOES NOT DO
    It never reads benchmark task rows from disk. The adapter supplies items;
    the dataset file is pinned by digest only. Prompts reach the disk as
    ``prompt_sha256`` and nothing else.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence, runtime_checkable

from henri_eval_infra import (
    ItemLedger,
    STATUS_VALUES,
    reconcile,
    run_id_new,
    run_output_dir,
    scan_contamination,
    sha256_bytes,
    sha256_text_lf,
)

# --- schema / constant identity ---------------------------------------------

SCHEMA_GATE_CHAIN = "henri.pilot-gate-chain.v1"
SCHEMA_STATIC_BUNDLE = "henri.static-bundle.v1"
SCHEMA_CHECKPOINT_PROVENANCE = "henri.checkpoint-provenance.v1"
RUN_EVIDENCE_SCHEMA = "henri.run-evidence.v1"

#: Canonical gate order. The chain stops at the first failure, so the ORDER is
#: part of the contract: cheap static checks run before expensive probes, and
#: the sandbox probe runs last because it is the only gate that touches the OS.
GATE_ORDER: tuple[str, ...] = (
    "static_bundle_digest",
    "contamination_scan",
    "checkpoint_provenance",
    "decoder_fallback_source",
    "sandbox_availability",
)

DIGEST_MODES: tuple[str, ...] = ("text_lf", "raw")

#: Plumbing mode may never run more items than this, so harness validation
#: cannot quietly grow into an evaluation run.
PLUMBING_MAX_ITEMS = 16


class EvalRunnerError(RuntimeError):
    """Base class for runner-level refusals."""


class FrozenFirstRunViolation(EvalRunnerError):
    """Run index 0 is the frozen first run; adaptation must be off."""


class TaskRowPersistenceViolation(EvalRunnerError):
    """A benchmark task row reached an artifact on disk."""


# --- typed gate abstraction --------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    """The ONLY shape a gate may return: (passed, reason, evidence)."""

    gate: str
    passed: bool
    reason: str
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"gate": self.gate, "passed": bool(self.passed),
                "reason": self.reason, "evidence": dict(self.evidence)}


class GateFailure(EvalRunnerError):
    """Raised when a pilot gate fails. Names the gate; halts the chain.

    The message is intentionally machine-parsable: ``PILOT GATE FAILED:
    <gate>: <reason>``.
    """

    def __init__(self, result: GateResult):
        super().__init__(f"PILOT GATE FAILED: {result.gate}: {result.reason}")
        self.result = result
        self.gate = result.gate
        self.reason = result.reason
        self.evidence = dict(result.evidence)


@dataclass
class PilotContext:
    """Everything the gates are allowed to look at.

    Gates receive this and return a GateResult. They do not mutate it and they
    do not reach outside it (no network, no global state), which is what makes
    a negative control possible: give a gate a broken input, watch it fail.
    """

    run_id: str
    run_dir: Path
    commit_sha: str
    benchmark_id: str
    adapter_id: str
    bundle_manifest: Path
    dataset_path: Path | None = None
    dataset_sha256: str | None = None
    store_paths: tuple[Path, ...] = ()
    checkpoint_policy: str = "disabled"
    checkpoint_path: Path | None = None
    checkpoint_sha256_expected: str | None = None
    checkpoint_provenance: Path | None = None
    checkpoint_skip_reason: str | None = None
    decoder_source_paths: tuple[Path, ...] = ()
    sandbox_mode: str = "none"
    adapter_requires_execution: bool = True


class PilotGate:
    """Typed gate abstraction.

    A gate is a named, side-effect-free check with a single entry point::

        run(ctx) -> GateResult(passed, reason, evidence)

    ``passed=False`` MUST name a refusal reason and MUST carry the evidence
    that produced the refusal.
    """

    name: str = "unnamed-gate"

    def run(self, ctx: PilotContext) -> GateResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def _ok(self, reason: str, **evidence: Any) -> GateResult:
        return GateResult(self.name, True, reason, evidence)

    def _fail(self, reason: str, **evidence: Any) -> GateResult:
        return GateResult(self.name, False, reason, evidence)


def _lf_bytes_digest(path: Path) -> str:
    """Canonical-LF digest of a text artifact.

    This repository sets core.autocrlf=true, so a scratch file with CRLF
    endings and its committed blob with LF endings have DIFFERENT raw digests.
    Text contracts hash the LF-normalized BYTES. This is the byte-level form of
    ``henri_eval_infra.sha256_text_lf`` and a test asserts the two agree.
    """
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


# --- gate 1: static bundle digest -------------------------------------------

class StaticBundleDigestGate(PilotGate):
    """Recompute every pinned digest before an item runs.

    Manifest schema ``henri.static-bundle.v1``::

        {"schema_id": "henri.static-bundle.v1",
         "files": [{"path": "eval_adapter.py", "digest_mode": "text_lf",
                    "sha256": "<64 hex>"}, ...]}

    ``digest_mode`` is explicit per file because the two rules are different on
    purpose: text artifacts hash canonical LF bytes, dataset .jsonl and binary
    artifacts hash RAW bytes. The run's dataset MUST be pinned with
    ``digest_mode="raw"``; pinning it as text would let a CRLF rewrite pass.
    """

    name = "static_bundle_digest"

    def run(self, ctx: PilotContext) -> GateResult:
        manifest_path = Path(ctx.bundle_manifest)
        if not manifest_path.is_file():
            return self._fail("MANIFEST_MISSING", manifest=str(manifest_path))
        try:
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return self._fail("MANIFEST_UNPARSEABLE", manifest=str(manifest_path),
                              error=f"{type(exc).__name__}: {exc}")
        if not isinstance(doc, dict) or doc.get("schema_id") != SCHEMA_STATIC_BUNDLE:
            return self._fail("MANIFEST_SCHEMA_MISMATCH",
                              expected=SCHEMA_STATIC_BUNDLE,
                              found=doc.get("schema_id") if isinstance(doc, dict) else None)
        entries = doc.get("files")
        if not isinstance(entries, list) or not entries:
            return self._fail("BUNDLE_EMPTY", manifest=str(manifest_path))

        checked: list[dict[str, Any]] = []
        mismatches: list[dict[str, Any]] = []
        pinned_raw: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                return self._fail("MANIFEST_ENTRY_NOT_OBJECT", entry=repr(entry)[:160])
            rel = entry.get("path")
            if not isinstance(rel, str) or not rel.strip():
                return self._fail("MANIFEST_ENTRY_WITHOUT_PATH")
            target = Path(rel)
            if not target.is_absolute():
                target = manifest_path.parent / target
            mode = entry.get("digest_mode")
            if mode not in DIGEST_MODES:
                return self._fail("UNKNOWN_DIGEST_MODE", path=str(target),
                                  digest_mode=mode, allowed=list(DIGEST_MODES))
            expected = str(entry.get("sha256", "")).lower()
            if not target.is_file():
                mismatches.append({"path": str(target), "problem": "FILE_MISSING",
                                   "expected": expected, "actual": None})
                continue
            actual = (_lf_bytes_digest(target) if mode == "text_lf"
                      else sha256_bytes(target.read_bytes()))
            resolved = str(target.resolve())
            if mode == "raw":
                pinned_raw.add(resolved)
            checked.append({"path": resolved, "digest_mode": mode,
                            "sha256": actual, "matched": actual == expected})
            if actual != expected:
                mismatches.append({"path": resolved, "problem": "DIGEST_MISMATCH",
                                   "digest_mode": mode, "expected": expected,
                                   "actual": actual})

        evidence: dict[str, Any] = {
            "manifest": str(manifest_path),
            "manifest_sha256": _lf_bytes_digest(manifest_path),
            "files_checked": len(checked),
            "files_declared": len(entries),
            "digests": checked,
            "mismatches": mismatches,
        }
        if mismatches:
            return self._fail("DIGEST_MISMATCH", **evidence)

        if ctx.dataset_path is not None:
            dataset_resolved = str(Path(ctx.dataset_path).resolve())
            if dataset_resolved not in pinned_raw:
                return self._fail(
                    "DATASET_NOT_PINNED_RAW",
                    dataset=dataset_resolved,
                    hint="the dataset .jsonl must be listed with digest_mode='raw'",
                    **evidence)
            raw_digest = sha256_bytes(Path(ctx.dataset_path).read_bytes())
            evidence["dataset_sha256"] = raw_digest
            if ctx.dataset_sha256 and ctx.dataset_sha256 != raw_digest:
                return self._fail("DATASET_DIGEST_DRIFT",
                                  expected=ctx.dataset_sha256,
                                  actual=raw_digest, **evidence)

        return self._ok(f"{len(checked)} pinned artifacts reproduced their digest",
                        **evidence)


# --- gate 2: contamination scan ---------------------------------------------

class ContaminationScanGate(PilotGate):
    """K-D: no benchmark task row may sit in a store before evaluation runs.

    Uses ``henri_eval_infra.scan_contamination``. Fail-closed on unreadable
    input: a store we cannot parse cannot be certified clean, so an unparseable
    row is a REFUSAL, not a skipped row.
    """

    name = "contamination_scan"

    def __init__(self, max_rows: int = 200_000):
        self.max_rows = int(max_rows)

    def _store_files(self, path: Path) -> list[Path]:
        if path.is_dir():
            return sorted(p for p in path.rglob("*") if p.is_file()
                          and p.suffix.lower() in (".jsonl", ".json"))
        return [path]

    def run(self, ctx: PilotContext) -> GateResult:
        if not ctx.store_paths:
            # Fail closed: "no store was scanned" is not evidence of cleanliness.
            return self._fail("NO_SCAN_TARGET_DECLARED",
                              hint="declare at least one Zone C / model store path")
        rows: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        unparsed: list[dict[str, Any]] = []
        for store in ctx.store_paths:
            store_path = Path(store)
            if not store_path.exists():
                return self._fail("STORE_PATH_MISSING", store=str(store_path))
            for f in self._store_files(store_path):
                raw = f.read_text(encoding="utf-8", errors="replace")
                n_file = 0
                for lineno, line in enumerate(raw.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    n_file += 1
                    try:
                        parsed = json.loads(line)
                    except Exception as exc:
                        unparsed.append({"file": str(f), "line": lineno,
                                         "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    if isinstance(parsed, dict):
                        rows.append(parsed)
                    elif isinstance(parsed, list):
                        rows.extend(r for r in parsed if isinstance(r, dict))
                    if len(rows) >= self.max_rows:
                        break
                sources.append({"store": str(store_path), "file": str(f),
                                "rows_read": n_file})
                if len(rows) >= self.max_rows:
                    break

        evidence: dict[str, Any] = {"sources": sources,
                                    "unparseable_rows": unparsed[:20],
                                    "unparseable_row_count": len(unparsed)}
        if unparsed:
            return self._fail("STORE_ROW_UNPARSEABLE",
                              hint="an unreadable store cannot be certified clean",
                              **evidence)

        report = scan_contamination(rows)
        evidence.update({"rows_scanned": report["rows_scanned"],
                         "contamination_hits": report["hits"][:20],
                         "contamination_hit_count": len(report["hits"])})
        if report["contaminated"]:
            return self._fail("CONTAMINATION_DETECTED", **evidence)
        return self._ok(f"{report['rows_scanned']} store rows scanned; no benchmark task rows",
                        **evidence)


# --- gate 3: checkpoint provenance preflight --------------------------------

class CheckpointProvenanceGate(PilotGate):
    """Prove the decoder checkpoint is readable, pinned, and off-split.

    Fail-closed rules:
      * policy ``required`` and no usable checkpoint  -> REFUSE
      * policy ``disabled`` with no stated reason      -> REFUSE (silent
        untrained-decoder runs were a measured defect)
      * checkpoint bytes drift from the expected digest -> REFUSE
      * provenance sidecar missing/mismatched          -> REFUSE
      * provenance dataset digest != run dataset digest -> REFUSE: a checkpoint
        trained on the evaluation split would make the run meaningless
      * torch cannot load the file as a state dict     -> REFUSE
    """

    name = "checkpoint_provenance"
    POLICIES = ("required", "disabled")

    def run(self, ctx: PilotContext) -> GateResult:
        policy = ctx.checkpoint_policy
        if policy not in self.POLICIES:
            return self._fail("UNKNOWN_CHECKPOINT_POLICY", policy=policy,
                              allowed=list(self.POLICIES))

        if policy == "disabled":
            if not (ctx.checkpoint_skip_reason or "").strip():
                return self._fail("CHECKPOINT_DISABLED_WITHOUT_REASON",
                                  hint="state why the trained decoder is not loaded")
            return self._ok("checkpoint policy disabled with a stated reason",
                            policy="disabled", reason=ctx.checkpoint_skip_reason,
                            checkpoint_load_status="SKIPPED_POLICY_DISABLED",
                            trained_decoder_active=False)

        if ctx.checkpoint_path is None:
            return self._fail("CHECKPOINT_REQUIRED_BUT_MISSING")
        ckpt = Path(ctx.checkpoint_path)
        if not ckpt.is_file():
            return self._fail("CHECKPOINT_FILE_MISSING", checkpoint=str(ckpt))

        digest = sha256_bytes(ckpt.read_bytes())
        evidence: dict[str, Any] = {"checkpoint": str(ckpt),
                                    "checkpoint_sha256": digest,
                                    "policy": policy,
                                    "declared_sha256": ctx.checkpoint_sha256_expected}
        if ctx.checkpoint_sha256_expected and \
                str(ctx.checkpoint_sha256_expected).lower() != digest:
            return self._fail("CHECKPOINT_DIGEST_MISMATCH", **evidence)

        if ctx.checkpoint_provenance is None:
            return self._fail("CHECKPOINT_PROVENANCE_MISSING", **evidence)
        prov_path = Path(ctx.checkpoint_provenance)
        if not prov_path.is_file():
            return self._fail("CHECKPOINT_PROVENANCE_MISSING",
                              provenance=str(prov_path), **evidence)
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return self._fail("CHECKPOINT_PROVENANCE_UNPARSEABLE",
                              provenance=str(prov_path),
                              error=f"{type(exc).__name__}: {exc}", **evidence)
        if not isinstance(prov, dict) or \
                prov.get("schema_id") != SCHEMA_CHECKPOINT_PROVENANCE:
            return self._fail("CHECKPOINT_PROVENANCE_SCHEMA_MISMATCH",
                              expected=SCHEMA_CHECKPOINT_PROVENANCE,
                              found=prov.get("schema_id") if isinstance(prov, dict) else None,
                              **evidence)
        evidence["provenance"] = {"path": str(prov_path),
                                  "dataset_sha256": prov.get("dataset_sha256"),
                                  "step": prov.get("step"),
                                  "commit_sha256": prov.get("commit_sha256")}
        if str(prov.get("checkpoint_sha256", "")).lower() != digest:
            return self._fail("PROVENANCE_DIGEST_MISMATCH",
                              provenance_sha256=prov.get("checkpoint_sha256"),
                              actual_sha256=digest, **evidence)
        if ctx.dataset_sha256 and \
                str(prov.get("dataset_sha256", "")).lower() != str(ctx.dataset_sha256).lower():
            return self._fail(
                "CHECKPOINT_PROVENANCE_DATASET_MISMATCH",
                provenance_dataset_sha256=prov.get("dataset_sha256"),
                run_dataset_sha256=ctx.dataset_sha256,
                hint="a checkpoint trained on the evaluation split invalidates the run",
                **evidence)

        try:
            import torch
        except Exception as exc:
            return self._fail("CHECKPOINT_PROBE_IMPORT_FAILED",
                              error=f"{type(exc).__name__}: {exc}", **evidence)
        try:
            loaded = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        except Exception as exc:
            return self._fail("CHECKPOINT_UNREADABLE",
                              error=f"{type(exc).__name__}: {exc}", **evidence)
        if not isinstance(loaded, dict) or not loaded:
            return self._fail("CHECKPOINT_NOT_A_STATE_DICT",
                              loaded_type=type(loaded).__name__, **evidence)

        evidence["tensors"] = len(loaded)
        evidence["checkpoint_load_status"] = "LOADED"
        evidence["trained_decoder_active"] = bool(prov.get("trained_decoder_active"))
        return self._ok("checkpoint bytes, provenance, and dataset split all agree",
                        **evidence)


# --- gate 4: decoder fallback source scan -----------------------------------

#: Function names that announce a fabricated default answer.
_FALLBACK_NAME_RE = re.compile(r"(fallback|stub|mock|fake|dummy|placeholder)", re.I)


class DecoderFallbackSourceGate(PilotGate):
    """Static scan: a decoder must never fabricate a task outcome.

    A decoder that answers ``return True`` when its path fails turns an
    infrastructure fault into a PASS. This gate refuses two AST shapes:

      * an ``except`` handler whose entire body returns a literal constant
        (silent swallow -> fabricated answer), and
      * a function whose name announces a fallback and whose body is a single
        literal return.

    Handlers that ``raise`` are fine. An unparseable declared source is a
    REFUSAL: absence of findings cannot be proved for a file we cannot read.
    """

    name = "decoder_fallback_source"
    CONSTANT_RETURN_STATUSES = ("FABRICATED_CONSTANT_RETURN", "FABRICATED_FALLBACK_FUNCTION")

    @staticmethod
    def _is_literal_return(node: ast.AST) -> bool:
        return isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)

    @classmethod
    def _scan_tree(cls, tree: ast.AST, path: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                body = [s for s in node.body if not isinstance(s, ast.Expr)
                        or not isinstance(s.value, ast.Constant)]
                if body and all(cls._is_literal_return(s) for s in body):
                    consts = [repr(getattr(s, "value", None).value)
                              if isinstance(getattr(s, "value", None), ast.Constant) else None
                              for s in body]
                    findings.append({
                        "status": "FABRICATED_CONSTANT_RETURN", "file": path,
                        "line": getattr(node, "lineno", None),
                        "handler": type(node.type).__name__ if node.type else "bare",
                        "returned": consts,
                        "why": "an except handler that returns a literal turns a "
                               "fault into an answer",
                    })
                elif any(isinstance(s, ast.Pass) for s in node.body[:1]) and \
                        all(isinstance(s, ast.Pass) for s in node.body):
                    findings.append({
                        "status": "SILENT_EXCEPT_SWALLOW", "file": path,
                        "line": getattr(node, "lineno", None),
                        "handler": type(node.type).__name__ if node.type else "bare",
                        "why": "a bare pass handler hides decoder faults",
                    })
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _FALLBACK_NAME_RE.search(node.name):
                    continue
                body = [s for s in node.body
                        if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)]
                if len(body) == 1 and cls._is_literal_return(body[0]):
                    findings.append({
                        "status": "FABRICATED_FALLBACK_FUNCTION", "file": path,
                        "line": node.lineno, "function": node.name,
                        "why": f"{node.name}() returns a literal as a default answer",
                    })
        return findings

    def run(self, ctx: PilotContext) -> GateResult:
        if not ctx.decoder_source_paths:
            return self._fail("NO_DECODER_SOURCE_DECLARED",
                              hint="declare the decoder modules to scan")
        findings: list[dict[str, Any]] = []
        scanned: list[str] = []
        for src in ctx.decoder_source_paths:
            p = Path(src)
            if not p.is_file():
                findings.append({"status": "SOURCE_MISSING", "file": str(p)})
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text, filename=str(p))
            except SyntaxError as exc:
                findings.append({"status": "SOURCE_UNPARSEABLE", "file": str(p),
                                 "line": exc.lineno, "why": str(exc)})
                continue
            scanned.append(str(p.resolve()))
            findings.extend(self._scan_tree(tree, str(p.resolve())))
        evidence = {"sources_scanned": scanned, "findings": findings[:20],
                    "finding_count": len(findings)}
        if findings:
            return self._fail("DECODER_FALLBACK_DETECTED", **evidence)
        return self._ok(f"{len(scanned)} decoder sources scanned; no fabricated fallback",
                        **evidence)


# --- gate 5: sandbox availability -------------------------------------------

class SandboxAvailabilityGate(PilotGate):
    """Prove an isolated execution boundary exists before candidate code runs.

    ``plumbing`` is the one mode that asserts NO boundary, and it is only legal
    for an adapter that declares ``requires_code_execution = False`` (the
    synthetic plumbing adapter never executes candidate code, so there is no
    boundary to provide). The gate refuses plumbing mode for any adapter that
    would execute code, so the escape hatch cannot leak into a real run.

    ``namespace`` requires POSIX + a live ``unshare`` probe.
    ``container-rlimit`` requires POSIX and an explicit caller declaration.
    Anything else REFUSES.
    """

    name = "sandbox_availability"

    def __init__(self, probe_timeout_sec: float = 20.0):
        self.probe_timeout_sec = float(probe_timeout_sec)

    def run(self, ctx: PilotContext) -> GateResult:
        mode = ctx.sandbox_mode
        if mode == "plumbing":
            if ctx.adapter_requires_execution:
                return self._fail(
                    "PLUMBING_MODE_REJECTS_EXECUTING_ADAPTER",
                    adapter_id=ctx.adapter_id,
                    hint="plumbing mode asserts no execution boundary; an adapter "
                         "that executes candidate code needs a real sandbox")
            return self._ok("plumbing mode: adapter executes no candidate code, "
                            "no boundary required",
                            sandbox_mode="plumbing", candidate_code_executed=False)
        if mode not in ("namespace", "container-rlimit"):
            return self._fail("UNKNOWN_SANDBOX_MODE", sandbox_mode=mode,
                              allowed=["namespace", "container-rlimit", "plumbing"])
        if os.name != "posix":
            return self._fail("SANDBOX_UNAVAILABLE", sandbox_mode=mode,
                              os_name=os.name,
                              hint="isolated execution requires a POSIX host")
        if mode == "namespace":
            unshare = shutil.which("unshare")
            if not unshare:
                return self._fail("SANDBOX_UNAVAILABLE", sandbox_mode=mode,
                                  hint="unshare is not installed")
            probe = [unshare, "--user", "--map-root-user", "--net", "--pid",
                     "--fork", "true"]
            try:
                proc = subprocess.run(probe, capture_output=True, text=True,
                                      timeout=self.probe_timeout_sec)
            except Exception as exc:
                return self._fail("SANDBOX_PROBE_ERROR", sandbox_mode=mode,
                                  error=f"{type(exc).__name__}: {exc}")
            if proc.returncode != 0:
                return self._fail("SANDBOX_PROBE_FAILED", sandbox_mode=mode,
                                  returncode=proc.returncode,
                                  stderr=(proc.stderr or "")[:400])
            return self._ok("namespace isolation probe succeeded",
                            sandbox_mode=mode, unshare=unshare,
                            probe_returncode=proc.returncode)
        return self._ok("container-rlimit surrogate boundary declared by caller",
                        sandbox_mode=mode, network_isolation=False)


# --- the chain ---------------------------------------------------------------

def build_pilot_gate_chain() -> tuple[PilotGate, ...]:
    """The canonical chain, in the canonical order. Never reorder casually."""
    return (
        StaticBundleDigestGate(),
        ContaminationScanGate(),
        CheckpointProvenanceGate(),
        DecoderFallbackSourceGate(),
        SandboxAvailabilityGate(),
    )


def run_pilot_gate_chain(ctx: PilotContext,
                         gates: Sequence[PilotGate] | None = None
                         ) -> tuple[list[GateResult], GateResult | None]:
    """Run gates IN ORDER. Halt at the first failure.

    Returns ``(results, failure)`` where ``results`` holds ONLY the gates that
    actually ran (through and including the failing gate). A caller can
    therefore assert that a failure at gate *i* produced exactly *i+1* results
    and that the gates after it never executed.
    """
    chain = tuple(gates) if gates is not None else build_pilot_gate_chain()
    results: list[GateResult] = []
    for gate in chain:
        result = gate.run(ctx)
        if result.gate != gate.name:
            raise EvalRunnerError(
                f"gate {gate.name!r} returned a result naming {result.gate!r}")
        if not isinstance(result.passed, bool) or not isinstance(result.reason, str) \
                or not isinstance(result.evidence, dict):
            raise EvalRunnerError(
                f"gate {gate.name!r} returned a malformed GateResult")
        results.append(result)
        if not result.passed:
            return results, result
    return results, None


# --- adapter protocol ---------------------------------------------------------

@dataclass(frozen=True)
class AdapterOutcome:
    """One item's measured outcome, supplied by the adapter."""

    status: str
    raw_stdout: str = ""
    raw_stderr: str = ""
    candidate: str | None = None
    elapsed_ms: float = 0.0
    tool_calls: int = 0
    retries: int = 0
    candidate_scores: tuple[float, ...] = ()
    sagnac_delta: float | None = None
    token_top1: str | None = None
    token_entropy: float | None = None


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """A pluggable benchmark adapter.

    ``item_count()`` is the DECLARED size of the run. Reconciliation compares
    the persisted ledger against this number, so an adapter that yields fewer
    items than it declared produces an INVALID_RUN rather than a quiet subset.
    """

    adapter_id: str
    requires_code_execution: bool

    def item_count(self) -> int: ...

    def items(self) -> Iterable[dict[str, Any]]: ...

    def evaluate(self, item: dict[str, Any]) -> AdapterOutcome: ...


# --- run configuration -------------------------------------------------------

@dataclass
class RunConfig:
    benchmark_id: str
    commit_sha: str
    output_root: Path
    bundle_manifest: Path
    adapter_id: str = "unspecified-adapter"
    dataset_path: Path | None = None
    dataset_source: str | None = None
    store_paths: tuple[Path, ...] = ()
    checkpoint_policy: str = "disabled"
    checkpoint_path: Path | None = None
    checkpoint_sha256_expected: str | None = None
    checkpoint_provenance: Path | None = None
    checkpoint_skip_reason: str | None = "no trained decoder declared for this run"
    decoder_source_paths: tuple[Path, ...] = ()
    sandbox_mode: str = "none"
    evaluator_id: str = "unspecified-evaluator"
    evaluator_version: str = "0"
    evaluator_source: Path | None = None
    grader_mode: str = "unspecified"
    declared_split_count: int | None = None
    run_id: str | None = None
    run_index: int = 0
    adaptation_enabled: bool = False
    synthetic_source: bool = False
    command: str = ""
    extra_limitations: str = ""
    gate_chain: tuple[PilotGate, ...] | None = None


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    status: str
    receipt: dict[str, Any]
    gate_results: list[GateResult]
    ledger_path: Path | None
    blocked_by_gate: str | None = None

    @property
    def accounting(self) -> dict[str, Any]:
        return dict(self.receipt.get("accounting", {}))


# --- helpers -----------------------------------------------------------------

def torch_environment() -> dict[str, Any]:
    """Device / torch / cuda facts for the receipt. Never raises."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is present on this host
        return {"device": "cpu", "torch_version": "unavailable",
                "cuda_version": None, "cuda_available": False,
                "torch_error": f"{type(exc).__name__}: {exc}"}
    cuda = bool(torch.cuda.is_available())
    return {"device": "cuda:0" if cuda else "cpu",
            "torch_version": str(torch.__version__),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": cuda}


def _write_json_atomic(path: Path, obj: dict[str, Any]) -> Path:
    """Write JSON with a tmp+replace so a crash cannot leave half a receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def assert_no_task_rows_persisted(run_dir: Path,
                                  prompts: Sequence[str],
                                  min_len: int = 24) -> dict[str, Any]:
    """Refuse to certify a run whose artifacts contain benchmark task text.

    Prompt text is only ever meant to reach disk as ``prompt_sha256``. This
    walks every artifact written under ``run_dir`` and searches for the raw
    prompt strings.
    """
    needles = [p for p in prompts if isinstance(p, str) and len(p) >= min_len]
    violations: list[dict[str, Any]] = []
    files_checked = 0
    for f in sorted(run_dir.rglob("*")):
        if not f.is_file():
            continue
        files_checked += 1
        blob = f.read_bytes().decode("utf-8", errors="replace")
        for i, needle in enumerate(needles):
            if needle in blob:
                violations.append({"artifact": str(f), "prompt_index": i,
                                   "prompt_sha256": sha256_text_lf(needle)})
                break
    report = {"files_checked": files_checked, "prompts_checked": len(needles),
              "violations": violations}
    if violations:
        raise TaskRowPersistenceViolation(
            f"benchmark task rows persisted in {len(violations)} artifact(s): "
            f"{[v['artifact'] for v in violations]}")
    return report


def classify_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Derive the accounting FROM THE PERSISTED ROWS, never an in-memory tally.

    Any status that is not PASSED and not EXECUTION_ERROR counts as FAILED for
    Contract A arithmetic; the exact breakdown is kept alongside so nothing is
    hidden. ``vetoed_count`` stays 0 because no Sagnac veto stage exists in this
    runner; a VETOED status arriving from an adapter is disclosed in
    ``outcome_breakdown`` and counted under FAILED (it is not a pass).
    """
    breakdown: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        breakdown[status] = breakdown.get(status, 0) + 1
    execution_errors = breakdown.get("EXECUTION_ERROR", 0)
    passed = breakdown.get("PASSED", 0)
    attempted = len(rows) - execution_errors
    failed = attempted - passed
    return {"item_rows": len(rows), "attempted": attempted, "passed": passed,
            "failed": failed, "execution_errors": execution_errors,
            "outcome_breakdown": breakdown}


# --- the runner ---------------------------------------------------------------

def _build_item_row(item: dict[str, Any], outcome: AdapterOutcome,
                    item_index: int) -> dict[str, Any]:
    status = str(outcome.status)
    if status not in STATUS_VALUES:
        raise EvalRunnerError(
            f"adapter returned invalid status {status!r}; allowed={sorted(STATUS_VALUES)}")
    sagnac = outcome.sagnac_delta
    if sagnac is not None:
        sagnac = float(sagnac)
        if not (0.0 <= sagnac <= 2.0):
            raise EvalRunnerError(
                f"normalized sagnac delta {sagnac} outside [0,2] for item {item_index}")
    prompt = item.get("prompt")
    prompt = prompt if isinstance(prompt, str) else ""
    return {
        "item_id": str(item.get("item_id", f"item-{item_index:05d}")),
        "item_index": item_index,
        "prompt_sha256": sha256_text_lf(prompt),
        "raw_stdout_sha256": sha256_text_lf(outcome.raw_stdout or ""),
        "raw_stderr_sha256": sha256_text_lf(outcome.raw_stderr or ""),
        "status": status,
        "elapsed_ms": float(outcome.elapsed_ms),
        "tool_calls": int(outcome.tool_calls),
        "retries": int(outcome.retries),
        "candidate_scores": [float(s) for s in outcome.candidate_scores],
        "sagnac_delta": sagnac,
        "token_top1": outcome.token_top1,
        "token_entropy": (None if outcome.token_entropy is None
                          else float(outcome.token_entropy)),
    }


def run_evaluation(cfg: RunConfig, adapter: BenchmarkAdapter) -> RunResult:
    """Run one benchmark evaluation under the full fail-closed pilot chain."""
    if cfg.run_index == 0 and cfg.adaptation_enabled:
        raise FrozenFirstRunViolation(
            "run_index 0 is the frozen first run: adaptation must be off "
            "(measured defect: an adapting first run cannot serve as a baseline)")

    requires_exec = bool(getattr(adapter, "requires_code_execution", True))
    run_id = cfg.run_id or run_id_new()
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_output_dir(output_root, cfg.commit_sha, cfg.benchmark_id, run_id))

    dataset_sha256: str | None = None
    if cfg.dataset_path is not None:
        dp = Path(cfg.dataset_path)
        if not dp.is_file():
            raise EvalRunnerError(f"dataset path missing: {dp}")
        dataset_sha256 = sha256_bytes(dp.read_bytes())

    ctx = PilotContext(
        run_id=run_id,
        run_dir=run_dir,
        commit_sha=cfg.commit_sha,
        benchmark_id=cfg.benchmark_id,
        adapter_id=str(getattr(adapter, "adapter_id", cfg.adapter_id)),
        bundle_manifest=Path(cfg.bundle_manifest),
        dataset_path=cfg.dataset_path,
        dataset_sha256=dataset_sha256,
        store_paths=tuple(cfg.store_paths),
        checkpoint_policy=cfg.checkpoint_policy,
        checkpoint_path=cfg.checkpoint_path,
        checkpoint_sha256_expected=cfg.checkpoint_sha256_expected,
        checkpoint_provenance=cfg.checkpoint_provenance,
        checkpoint_skip_reason=cfg.checkpoint_skip_reason,
        decoder_source_paths=tuple(cfg.decoder_source_paths),
        sandbox_mode=cfg.sandbox_mode,
        adapter_requires_execution=requires_exec,
    )

    gate_results, failure = run_pilot_gate_chain(ctx, cfg.gate_chain)
    _write_json_atomic(run_dir / "gate_receipt.json", {
        "schema_id": SCHEMA_GATE_CHAIN,
        "run_id": run_id,
        "commit_sha256": cfg.commit_sha,
        "benchmark_id": cfg.benchmark_id,
        "adapter_id": ctx.adapter_id,
        "order": [r.gate for r in gate_results],
        "gates": [r.as_dict() for r in gate_results],
        "halted_at": failure.gate if failure else None,
        "passed": failure is None,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    env = torch_environment()
    if failure is not None:
        blocked = {
            "schema_id": RUN_EVIDENCE_SCHEMA,
            "status": "BLOCKED",
            "run_id": run_id,
            "commit_sha256": cfg.commit_sha,
            "command": cfg.command or " ".join(sys.argv),
            "benchmark_id": cfg.benchmark_id,
            "dataset_source": cfg.dataset_source,
            "dataset_sha256": dataset_sha256,
            "evaluator_id": cfg.evaluator_id,
            "evaluator_version": cfg.evaluator_version,
            "evaluator_sha256": _evaluator_sha256(cfg),
            "checkpoint_load_status": None,
            "trained_decoder_active": None,
            "device": env["device"],
            "torch_version": env["torch_version"],
            "cuda_version": env["cuda_version"],
            "item_count": 0, "attempted_count": 0, "passed_count": 0,
            "failed_count": 0, "execution_error_count": 0, "vetoed_count": 0,
            "raw_stdout_sha256": None, "raw_stderr_sha256": None,
            "item_results_sha256": None,
            "artifact_paths": [str(run_dir / "gate_receipt.json")],
            "limitations": f"blocked by pilot gate {failure.gate}: {failure.reason}",
            "grader_mode": cfg.grader_mode,
            "synthetic_source": bool(cfg.synthetic_source),
            "task_leakage_detected": False,
            "declared_split_count": cfg.declared_split_count,
            "blocked_by_gate": failure.gate,
            "blocked_reason": failure.reason,
            "gate_chain_sha256": sha256_text_lf(json.dumps(
                [r.as_dict() for r in gate_results], sort_keys=True, default=str)),
            "accounting": {"valid": True, "problems": [],
                           "note": "no item executed; accounting is vacuous by design"},
        }
        _write_json_atomic(run_dir / "run_receipt.json", blocked)
        raise GateFailure(failure)

    # ---- execution: per-item rows are fsync'd BEFORE any aggregation --------
    ledger_path = run_dir / "item_ledger.jsonl"
    ledger = ItemLedger(ledger_path)
    raw_out_path = run_dir / "raw_stdout.jsonl"
    raw_err_path = run_dir / "raw_stderr.jsonl"
    item_count = int(adapter.item_count())
    prompts_seen: list[str] = []
    run_error: str | None = None
    executed = 0
    t0 = time.time()
    with open(raw_out_path, "a", encoding="utf-8", newline="\n") as raw_out, \
            open(raw_err_path, "a", encoding="utf-8", newline="\n") as raw_err:
        try:
            for index, item in enumerate(adapter.items()):
                outcome = adapter.evaluate(item)
                prompt = item.get("prompt")
                if isinstance(prompt, str):
                    prompts_seen.append(prompt)
                row = _build_item_row(item, outcome, index)
                # append() validates, writes one line, flushes, and fsyncs.
                ledger.append(row)
                raw_out.write(json.dumps(
                    {"item_id": row["item_id"], "raw_stdout": outcome.raw_stdout},
                    sort_keys=True, default=str) + "\n")
                raw_err.write(json.dumps(
                    {"item_id": row["item_id"], "raw_stderr": outcome.raw_stderr},
                    sort_keys=True, default=str) + "\n")
                raw_out.flush()
                raw_err.flush()
                os.fsync(raw_out.fileno())
                os.fsync(raw_err.fileno())
                executed += 1
        except Exception as exc:
            run_error = f"{type(exc).__name__}: {exc}"

    # ---- aggregation reads the PERSISTED rows back from disk ----------------
    rows = list(ledger.rows())
    counts = classify_rows(rows)
    rec = reconcile(item_count, counts["attempted"], counts["passed"],
                    counts["failed"], counts["execution_errors"])
    problems = list(rec["problems"])
    if cfg.declared_split_count is not None and cfg.declared_split_count < item_count:
        problems.append(
            f"declared_split_count({cfg.declared_split_count}) < item_count({item_count})")
    if run_error:
        problems.append(f"RUN_ABORTED: {run_error}")
    valid = not problems
    status = "OBSERVED" if valid else "INVALID"

    persisted = run_dir / "item_ledger.jsonl"
    try:
        task_row_report = assert_no_task_rows_persisted(run_dir, prompts_seen)
    except TaskRowPersistenceViolation as exc:
        task_row_report = {"error": str(exc)}
        problems.append("TASK_ROW_PERSISTED")
        valid, status = False, "INVALID"

    artifact_paths = [str(p) for p in (run_dir / "gate_receipt.json", persisted,
                                       raw_out_path, raw_err_path)
                      if Path(p).exists()]
    limitations = ("; ".join(x for x in (
        cfg.extra_limitations,
        "first run frozen (adaptation off)" if cfg.run_index == 0 else
        f"run_index={cfg.run_index}",
        f"adapter={ctx.adapter_id}",
        f"sandbox_mode={cfg.sandbox_mode}",
    ) if x))

    receipt: dict[str, Any] = {
        "schema_id": RUN_EVIDENCE_SCHEMA,
        "status": status,
        "run_id": run_id,
        "commit_sha256": cfg.commit_sha,
        "command": cfg.command or " ".join(sys.argv),
        "benchmark_id": cfg.benchmark_id,
        "dataset_source": cfg.dataset_source,
        "dataset_sha256": dataset_sha256,
        "evaluator_id": cfg.evaluator_id,
        "evaluator_version": cfg.evaluator_version,
        "evaluator_sha256": _evaluator_sha256(cfg),
        "checkpoint_sha256": _checkpoint_digest(gate_results),
        "checkpoint_load_status": _checkpoint_status(gate_results),
        "trained_decoder_active": _checkpoint_trained(gate_results),
        "device": env["device"],
        "torch_version": env["torch_version"],
        "cuda_version": env["cuda_version"],
        "item_count": item_count,
        "attempted_count": counts["attempted"],
        "passed_count": counts["passed"],
        "failed_count": counts["failed"],
        "execution_error_count": counts["execution_errors"],
        "vetoed_count": 0,
        "raw_stdout_sha256": sha256_bytes(raw_out_path.read_bytes()),
        "raw_stderr_sha256": sha256_bytes(raw_err_path.read_bytes()),
        "item_results_sha256": ledger.sha256(),
        "artifact_paths": artifact_paths + [str(run_dir / "run_receipt.json")],
        "limitations": limitations,
        "grader_mode": cfg.grader_mode,
        "synthetic_source": bool(cfg.synthetic_source),
        "task_leakage_detected": False,
        "declared_split_count": cfg.declared_split_count,
        # --- extended evidence (henri_eval_infra extends, never replaces) ----
        "adapter_id": ctx.adapter_id,
        "sandbox_mode": cfg.sandbox_mode,
        "adaptation_enabled": bool(cfg.adaptation_enabled),
        "run_index": cfg.run_index,
        "elapsed_sec": round(time.time() - t0, 6),
        "items_executed": executed,
        "outcome_breakdown": counts["outcome_breakdown"],
        "task_row_persistence": task_row_report,
        "run_error": run_error,
        "gate_chain_sha256": sha256_text_lf(json.dumps(
            [r.as_dict() for r in gate_results], sort_keys=True, default=str)),
        "accounting": {"valid": valid, "problems": problems,
                       "item_rows_persisted": counts["item_rows"],
                       "ledger_path": str(persisted)},
    }
    _write_json_atomic(run_dir / "run_receipt.json", receipt)
    Path(run_dir / "run_receipt.json").exists()
    receipt["artifact_paths"] = artifact_paths + [str(run_dir / "run_receipt.json")]
    # rewrite so the receipt lists itself deterministically
    _write_json_atomic(run_dir / "run_receipt.json", receipt)

    # Final sweep over the COMPLETE artifact set, receipt included. Writing the
    # receipt is itself a chance to persist task text (an adapter could have
    # smuggled a prompt into a reason string), so the guard runs again and an
    # OBSERVED receipt is downgraded if it fails. The sweep's own report covers
    # run_receipt.json, so the receipt is written a final time: the returned
    # dict and the persisted file must be byte-identical, or the reported
    # evidence is not the evidence on disk.
    try:
        receipt["task_row_persistence"] = assert_no_task_rows_persisted(
            run_dir, prompts_seen)
    except TaskRowPersistenceViolation as exc:
        receipt["task_row_persistence"] = {"error": str(exc)}
        receipt["status"] = "INVALID"
        receipt["accounting"]["valid"] = False
        receipt["accounting"]["problems"].append("TASK_ROW_PERSISTED_AFTER_RECEIPT")
        status, valid = "INVALID", False
    _write_json_atomic(run_dir / "run_receipt.json", receipt)

    return RunResult(run_id=run_id, run_dir=run_dir, status=status,
                     receipt=receipt, gate_results=gate_results,
                     ledger_path=persisted, blocked_by_gate=None)


def _evaluator_sha256(cfg: RunConfig) -> str | None:
    if cfg.evaluator_source is None:
        return None
    p = Path(cfg.evaluator_source)
    return _lf_bytes_digest(p) if p.is_file() else None


def _gate(ctx_result: Sequence[GateResult], name: str) -> dict[str, Any]:
    for r in ctx_result:
        if r.gate == name:
            return r.evidence
    return {}


def _checkpoint_digest(results: Sequence[GateResult]) -> str | None:
    return _gate(results, "checkpoint_provenance").get("checkpoint_sha256")


def _checkpoint_status(results: Sequence[GateResult]) -> str | None:
    return _gate(results, "checkpoint_provenance").get("checkpoint_load_status")


def _checkpoint_trained(results: Sequence[GateResult]) -> bool | None:
    return _gate(results, "checkpoint_provenance").get("trained_decoder_active")


# --- plumbing mode -----------------------------------------------------------

class SyntheticPlumbingAdapter:
    """Deterministic adapter that exercises the harness without any benchmark.

    It executes NO candidate code (``requires_code_execution = False``), so it
    is the only adapter legal in plumbing sandbox mode. Outcomes are a fixed
    mix of PASSED / FAILED / EXECUTION_ERROR so the reconciliation arithmetic is
    exercised rather than trivially satisfied by all-passes.

    ``observe_path`` lets a test read the ledger file DURING execution, which is
    how "rows reach disk before aggregation" is proved rather than asserted.
    """

    adapter_id = "plumbing-synthetic-v1"
    requires_code_execution = False

    def __init__(self, n: int = 12, observe_path: Path | None = None):
        if n < 1 or n > PLUMBING_MAX_ITEMS:
            raise ValueError(
                f"plumbing mode runs 1..{PLUMBING_MAX_ITEMS} items, got {n}")
        self.n = int(n)
        self.observe_path = Path(observe_path) if observe_path else None
        self.observed_row_counts: list[int] = []
        self.evaluate_calls = 0

    def item_count(self) -> int:
        return self.n

    def items(self) -> Iterator[dict[str, Any]]:
        for i in range(self.n):
            yield {"item_id": f"synthetic-{i:03d}",
                   "prompt": f"synthetic plumbing item {i:03d}: no benchmark data"}

    def _observe(self) -> None:
        if self.observe_path is None:
            return
        if not self.observe_path.exists():
            self.observed_row_counts.append(0)
            return
        rows = [ln for ln in self.observe_path.read_text(
            encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        self.observed_row_counts.append(len(rows))

    def evaluate(self, item: dict[str, Any]) -> AdapterOutcome:
        self._observe()
        self.evaluate_calls += 1
        i = int(str(item["item_id"]).rsplit("-", 1)[-1])
        if i % 4 == 0:
            status = "EXECUTION_ERROR"
        elif i % 3 == 0:
            status = "FAILED"
        else:
            status = "PASSED"
        return AdapterOutcome(
            status=status,
            raw_stdout=f"{item['item_id']}:{status}",
            raw_stderr="" if status != "EXECUTION_ERROR" else "synthetic plumbing fault",
            candidate=None,
            elapsed_ms=0.001 * (i + 1),
            tool_calls=0,
            retries=0,
            candidate_scores=(1.0,) if status == "PASSED" else (0.0,),
            sagnac_delta=round(0.05 * (i % 20), 6),
            token_top1=f"tok{i}",
            token_entropy=0.5,
        )


def build_plumbing_config(root: Path, n: int = 12,
                          sandbox_mode: str = "plumbing") -> RunConfig:
    """Create a hermetic plumbing bundle (real files, real digests).

    Every artifact here is obviously synthetic and digests are computed from the
    bytes actually written, so the gate chain is exercised for real -- but the
    receipt is annotated ``synthetic_source=True`` and its dataset_source names
    the synthetic origin, so it can never be promoted to a score.
    """
    if n < 1 or n > PLUMBING_MAX_ITEMS:
        raise ValueError(f"plumbing mode runs 1..{PLUMBING_MAX_ITEMS} items, got {n}")
    import torch  # local import: only plumbing builds a probe checkpoint

    root = Path(root)
    bundle = root / "_plumbing_bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    (bundle / "decoder_clean.py").write_text(
        "def decode(x):\n"
        "    if x < 0:\n"
        "        raise ValueError('negative input')\n"
        "    try:\n"
        "        return x + 1\n"
        "    except Exception:\n"
        "        raise  # re-raise: never fabricate an answer\n",
        encoding="utf-8", newline="\n")
    (bundle / "eval_adapter.py").write_text(
        '"""synthetic plumbing adapter (no benchmark data)."""\n'
        "ADAPTER_ID = 'plumbing-synthetic-v1'\n",
        encoding="utf-8", newline="\n")
    dataset = bundle / "synthetic_items.jsonl"
    dataset.write_text("".join(
        json.dumps({"item_id": f"synthetic-{i:03d}",
                    "prompt": f"synthetic plumbing item {i:03d}: no benchmark data"}) + "\n"
        for i in range(n)), encoding="utf-8", newline="\n")

    store = bundle / "zonec_store.jsonl"
    store.write_text(json.dumps({"record_id": "zonec-synthetic-clean-0",
                                 "content": "synthetic clean store row"}) + "\n",
                     encoding="utf-8", newline="\n")

    ckpt_path = bundle / "synthetic_checkpoint.pt"
    torch.save({"w": torch.zeros(2), "b": torch.zeros(1)}, str(ckpt_path))
    ckpt_digest = sha256_bytes(ckpt_path.read_bytes())
    dataset_digest_raw = sha256_bytes(dataset.read_bytes())

    prov_path = bundle / "checkpoint_provenance.json"
    prov_path.write_text(json.dumps({
        "schema_id": SCHEMA_CHECKPOINT_PROVENANCE,
        "checkpoint_sha256": ckpt_digest,
        "dataset_sha256": dataset_digest_raw,
        "commit_sha256": "0" * 40,
        "step": 0,
        "trained_decoder_active": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    files = [
        {"path": "decoder_clean.py", "digest_mode": "text_lf",
         "sha256": _lf_bytes_digest(bundle / "decoder_clean.py")},
        {"path": "eval_adapter.py", "digest_mode": "text_lf",
         "sha256": _lf_bytes_digest(bundle / "eval_adapter.py")},
        {"path": "synthetic_items.jsonl", "digest_mode": "raw",
         "sha256": dataset_digest_raw},
    ]
    manifest = bundle / "static_bundle.json"
    manifest.write_text(json.dumps({"schema_id": SCHEMA_STATIC_BUNDLE, "files": files},
                                   indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")

    return RunConfig(
        benchmark_id="plumbing-synthetic-v1",
        commit_sha="0" * 40,
        output_root=root,
        bundle_manifest=manifest,
        adapter_id=SyntheticPlumbingAdapter.adapter_id,
        dataset_path=dataset,
        dataset_source="synthetic://plumbing-mode-no-benchmark-data",
        store_paths=(store,),
        checkpoint_policy="required",
        checkpoint_path=ckpt_path,
        checkpoint_sha256_expected=ckpt_digest,
        checkpoint_provenance=prov_path,
        decoder_source_paths=(bundle / "decoder_clean.py",),
        sandbox_mode=sandbox_mode,
        evaluator_id="henri_eval_runner.plumbing",
        evaluator_version="1",
        evaluator_source=bundle / "eval_adapter.py",
        grader_mode="plumbing_synthetic_no_grader",
        declared_split_count=n,
        run_index=0,
        adaptation_enabled=False,
        synthetic_source=True,
        command="henri_eval_runner.py --plumbing",
        extra_limitations=f"plumbing validation only; {n} synthetic items; "
                          "candidate code not executed",
    )


def run_plumbing_mode(root: Path | None = None, n: int = 12) -> RunResult:
    """Run the entire chain on a synthetic adapter. Zero infrastructure errors.

    Returns a RunResult. Raises if the chain does not pass cleanly -- a plumbing
    run that cannot pass its own gates is a harness defect, not a benchmark
    result.
    """
    if root is None:
        import tempfile
        root = Path(tempfile.mkdtemp(prefix="henri-plumbing-"))
    root = Path(root)
    cfg = build_plumbing_config(root, n=n)
    adapter = SyntheticPlumbingAdapter(n=n)
    result = run_evaluation(cfg, adapter)
    if result.status != "OBSERVED" or adapter.evaluate_calls != n:
        raise EvalRunnerError(
            f"plumbing run did not complete cleanly: status={result.status} "
            f"calls={adapter.evaluate_calls}/{n} problems={result.accounting.get('problems')}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--plumbing" not in argv:
        print("henri_eval_runner: pass --plumbing to validate the harness "
              "(real runs are driven through run_evaluation)", file=sys.stderr)
        return 2
    n = PLUMBING_MAX_ITEMS
    if "--items" in argv:
        n = int(argv[argv.index("--items") + 1])
    root = None
    if "--output-root" in argv:
        root = Path(argv[argv.index("--output-root") + 1])
    result = run_plumbing_mode(root=root, n=n)
    print(json.dumps(result.receipt, indent=2, sort_keys=True, default=str))
    return 0 if result.status == "OBSERVED" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
