#!/usr/bin/env python3
"""Compact holon envelope and result validator for the HENRI holonic loop.

A holon is a bounded execution unit that is both a complete local workflow and
part of a parent workflow. This module defines the two data contracts that
cross the boundary between a parent and a child holon, and it validates both.

    HolonContract  — what the parent SENDS to the child (the context firewall)
    HolonResult    — what the child RETURNS to the parent (the compact finding)

Why this exists
---------------
The parent must not receive raw tool traces, full PDFs, unbounded telemetry, or
the child's reasoning. It receives a bounded result and can retrieve the full
artifact by reference. That is the only mechanism here that reduces tokens: the
parent's context grows by a bounded envelope instead of by the child's whole
transcript.

Token accounting honesty
------------------------
``token_usage`` is OBSERVED only when an executor reports real counts. It is
never estimated, never inferred from character counts, and never defaulted to
zero. A result with no reported usage carries ``token_usage = None`` and
``token_usage_status = "BLOCKED"``. This keeps the step-8 local-versus-holonic
measurement honest: a missing measurement must not read as a cheap one.

What this module does NOT do
----------------------------
- It does not call an LLM, TrustGraph, or any network service.
- It does not approve anything. ``requires_approval`` records a requirement;
  only ``henri_governance.py`` seals a human decision.
- A validated result is evidence of a well-formed reply, NOT evidence that the
  finding is true or that a HENRI task improved.

CLI
---
    python scripts/henri_holon.py new-contract --help
    python scripts/henri_holon.py validate-result result.json
    python scripts/henri_holon.py seal-result result.json
    python scripts/henri_holon.py selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

CONTRACT_VERSION = 1

TASK_TYPES = {"research", "audit", "telemetry", "execution", "governance"}
# Mirrors agentic_event_store.CAUSAL_STATUSES so a result status can be sealed
# directly as an event causal_status without translation.
RESULT_STATUSES = {"observed", "derived", "inferred", "hypothesis", "falsified", "blocked"}
FAILURE_SCOPES = {"local", "parent", "global"}
ROUTE_PATTERNS = {"react", "plan-then-execute", "supervisor", "deterministic", "none"}

# Hard ceilings. A child that exceeds these has broken the context firewall and
# its result is rejected rather than truncated, because silently truncating a
# finding would hide the boundary violation from the parent.
MAX_QUESTION_CHARS = 2000
MAX_FINDING_CHARS = 4000
MAX_OUTPUT_CHARS_CEILING = 20000
MAX_ITERATIONS_CEILING = 25
MAX_EVIDENCE_REFS = 50


class HolonError(ValueError):
    """Raised when a contract or result violates the envelope schema."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _new_holon_id(prefix: str = "h") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Parent -> child: the context firewall contract.
# ---------------------------------------------------------------------------

@dataclass
class HolonContract:
    """The bounded instruction a parent hands to a child holon."""

    question: str
    task_type: str
    holon_id: str = field(default_factory=_new_holon_id)
    parent_holon_id: str | None = None
    contract_version: int = CONTRACT_VERSION
    created_at: str = field(default_factory=_utc_now)

    input_refs: list[str] = field(default_factory=list)
    source_hashes: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)

    route_pattern: str = "deterministic"
    max_iterations: int = 3
    max_output_chars: int = 4000

    acceptance: list[str] = field(default_factory=list)
    rejection: list[str] = field(default_factory=list)
    requires_approval: bool = False

    def validate(self) -> "HolonContract":
        if not self.question or not self.question.strip():
            raise HolonError("contract.question must be non-empty")
        if len(self.question) > MAX_QUESTION_CHARS:
            raise HolonError(
                f"contract.question exceeds {MAX_QUESTION_CHARS} chars "
                f"({len(self.question)}); a holon takes ONE bounded question"
            )
        if self.task_type not in TASK_TYPES:
            raise HolonError(f"contract.task_type invalid: {self.task_type!r}")
        if self.route_pattern not in ROUTE_PATTERNS:
            raise HolonError(f"contract.route_pattern invalid: {self.route_pattern!r}")
        if not (1 <= self.max_iterations <= MAX_ITERATIONS_CEILING):
            raise HolonError(
                f"contract.max_iterations must be 1..{MAX_ITERATIONS_CEILING}, "
                f"got {self.max_iterations}"
            )
        if not (1 <= self.max_output_chars <= MAX_OUTPUT_CHARS_CEILING):
            raise HolonError(
                f"contract.max_output_chars must be 1..{MAX_OUTPUT_CHARS_CEILING}, "
                f"got {self.max_output_chars}"
            )
        if self.holon_id == self.parent_holon_id:
            raise HolonError("contract.holon_id must differ from parent_holon_id")
        if not self.acceptance:
            raise HolonError(
                "contract.acceptance must pre-register at least one criterion; "
                "an unfalsifiable holon cannot be evaluated"
            )
        for name in ("input_refs", "source_hashes", "allowed_paths", "allowed_tools",
                     "acceptance", "rejection"):
            value = getattr(self, name)
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise HolonError(f"contract.{name} must be a list of strings")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def contract_hash(self) -> str:
        """Stable identity for duplicate-run detection (step 7 lease guard)."""
        payload = self.to_dict()
        payload.pop("created_at", None)
        payload.pop("holon_id", None)
        return _sha256(payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HolonContract":
        data = dict(data)
        # ``new-contract`` emits contract_hash as a convenience for the operator.
        # It is derived, not stored state, so drop it and recompute on demand.
        declared_hash = data.pop("contract_hash", None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = sorted(set(data) - known)
        if unknown:
            raise HolonError(f"contract has unknown fields: {', '.join(unknown)}")
        contract = cls(**data).validate()
        if declared_hash is not None and declared_hash != contract.contract_hash():
            raise HolonError(
                "contract_hash does not match the contract body; the file was edited "
                "after it was generated"
            )
        return contract


# ---------------------------------------------------------------------------
# Child -> parent: the compact result.
# ---------------------------------------------------------------------------

@dataclass
class HolonResult:
    """The bounded finding a child holon returns to its parent."""

    holon_id: str
    status: str
    finding: str
    next_action: str
    contract_version: int = CONTRACT_VERSION
    parent_holon_id: str | None = None
    completed_at: str = field(default_factory=_utc_now)

    evidence_refs: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)

    token_usage: dict[str, int] | None = None
    token_usage_status: str = "BLOCKED"
    route_pattern: str = "deterministic"
    iterations_used: int = 0
    failure_scope: str = "local"

    def validate(self, contract: HolonContract | None = None) -> "HolonResult":
        if self.status not in RESULT_STATUSES:
            raise HolonError(f"result.status invalid: {self.status!r}")
        if self.failure_scope not in FAILURE_SCOPES:
            raise HolonError(f"result.failure_scope invalid: {self.failure_scope!r}")
        if self.route_pattern not in ROUTE_PATTERNS:
            raise HolonError(f"result.route_pattern invalid: {self.route_pattern!r}")
        if not self.finding or not self.finding.strip():
            raise HolonError("result.finding must be non-empty")
        if len(self.finding) > MAX_FINDING_CHARS:
            raise HolonError(
                f"result.finding exceeds {MAX_FINDING_CHARS} chars ({len(self.finding)}); "
                "the parent takes a compact finding, not a transcript"
            )
        if not self.next_action or not self.next_action.strip():
            raise HolonError("result.next_action must state one bounded action")
        if len(self.evidence_refs) > MAX_EVIDENCE_REFS:
            raise HolonError(f"result.evidence_refs exceeds {MAX_EVIDENCE_REFS} entries")

        # A positive evidence class requires references. This is the tautology
        # guard: a child may not assert OBSERVED or DERIVED with no evidence.
        if self.status in {"observed", "derived"} and not self.evidence_refs:
            raise HolonError(
                f"result.status={self.status!r} requires at least one evidence_ref; "
                "an unreferenced positive finding is not admissible"
            )

        self._validate_token_usage()

        if contract is not None:
            if self.holon_id != contract.holon_id:
                raise HolonError(
                    f"result.holon_id {self.holon_id!r} does not match contract "
                    f"{contract.holon_id!r}"
                )
            if self.parent_holon_id != contract.parent_holon_id:
                raise HolonError("result.parent_holon_id does not match contract")
            if len(self.finding) > contract.max_output_chars:
                raise HolonError(
                    f"result.finding ({len(self.finding)} chars) exceeds the contract "
                    f"budget of {contract.max_output_chars}"
                )
            if self.iterations_used > contract.max_iterations:
                raise HolonError(
                    f"result.iterations_used {self.iterations_used} exceeds the contract "
                    f"budget of {contract.max_iterations}"
                )
        return self

    def _validate_token_usage(self) -> None:
        """Token usage is OBSERVED or BLOCKED. There is no estimated middle."""
        if self.token_usage is None:
            if self.token_usage_status != "BLOCKED":
                raise HolonError(
                    "result.token_usage is None so token_usage_status must be 'BLOCKED'"
                )
            return
        if self.token_usage_status != "OBSERVED":
            raise HolonError(
                "result.token_usage is present so token_usage_status must be 'OBSERVED'; "
                "estimated counts are not admissible"
            )
        if not isinstance(self.token_usage, dict):
            raise HolonError("result.token_usage must be an object")
        missing = {"input_tokens", "output_tokens"} - set(self.token_usage)
        if missing:
            raise HolonError(f"result.token_usage missing: {', '.join(sorted(missing))}")
        for key, value in self.token_usage.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise HolonError(f"result.token_usage.{key} must be a non-negative integer")

    def total_tokens(self) -> int | None:
        if self.token_usage is None:
            return None
        return int(self.token_usage.get("input_tokens", 0)) + int(
            self.token_usage.get("output_tokens", 0)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HolonResult":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = sorted(set(data) - known)
        if unknown:
            raise HolonError(f"result has unknown fields: {', '.join(unknown)}")
        return cls(**data)


# ---------------------------------------------------------------------------
# Admission decision.
# ---------------------------------------------------------------------------

def admit(result: HolonResult, contract: HolonContract) -> dict[str, Any]:
    """Decide whether a validated result may inform a parent decision.

    A well-formed result is not automatically admissible. A ``blocked`` or
    ``falsified`` child, or one whose contract requires approval, must not feed
    a parent synthesis as if it were a finished finding.
    """
    result.validate(contract)
    reasons: list[str] = []
    admitted = True

    if result.status in {"blocked", "falsified"}:
        admitted = False
        reasons.append(f"status={result.status} is not a usable finding")
    if result.status == "hypothesis":
        admitted = False
        reasons.append("status=hypothesis requires a test before it informs a decision")
    if contract.requires_approval:
        admitted = False
        reasons.append("contract requires a sealed human decision before use")
    if result.failure_scope in {"parent", "global"}:
        admitted = False
        reasons.append(f"failure_scope={result.failure_scope} must halt the parent route")

    return {
        "admitted": admitted,
        "reasons": reasons or ["result is well-formed and evidence-referenced"],
        "holon_id": result.holon_id,
        "status": result.status,
        "total_tokens": result.total_tokens(),
        "token_usage_status": result.token_usage_status,
        "contract_hash": contract.contract_hash(),
    }


def _vault_root() -> Path:
    configured = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Documents" / "HENRI_Research_Vault"


def seal_result(result: HolonResult, contract: HolonContract | None = None) -> dict[str, Any]:
    """Seal a validated result as a HOLON_RESULT event in the governance store.

    The event carries the compact envelope only. Full output stays at
    ``artifact_refs`` so the ledger does not become a transcript store.
    """
    from agentic_event_store import append_event

    result.validate(contract)
    payload = {
        "holon_id": result.holon_id,
        "parent_holon_id": result.parent_holon_id,
        "finding": result.finding,
        "evidence_refs": result.evidence_refs,
        "artifact_refs": result.artifact_refs,
        "route_pattern": result.route_pattern,
        "iterations_used": result.iterations_used,
        "failure_scope": result.failure_scope,
        "token_usage": result.token_usage,
        "token_usage_status": result.token_usage_status,
        "total_tokens": result.total_tokens(),
        "next_action": result.next_action,
    }
    if contract is not None:
        payload["contract_hash"] = contract.contract_hash()
        payload["question"] = contract.question
        payload["task_type"] = contract.task_type
    return append_event(
        "HOLON_RESULT",
        payload,
        stream="graph",
        actor="henri_holon",
        causal_status=result.status,
        parent_event_id=None,
        vault_path=str(_vault_root()),
    )


# ---------------------------------------------------------------------------
# Self-test. Exercises the real validators; no mocked logic.
# ---------------------------------------------------------------------------

def selftest() -> int:
    checks: list[tuple[str, bool, str]] = []

    def expect_error(name: str, fn) -> None:
        try:
            fn()
        except HolonError as exc:
            checks.append((name, True, f"rejected: {exc}"))
        except Exception as exc:  # wrong exception type is a failure
            checks.append((name, False, f"wrong exception {type(exc).__name__}: {exc}"))
        else:
            checks.append((name, False, "accepted an invalid input"))

    def expect_ok(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            checks.append((name, False, f"{type(exc).__name__}: {exc}"))
        else:
            checks.append((name, True, "accepted"))

    good_contract = HolonContract(
        question="Does the source support a candidate-specific constraint penalty?",
        task_type="research",
        input_refs=["event:538e4f51"],
        allowed_paths=["HENRI V2/efe_planner.py"],
        allowed_tools=["read_file", "search_files"],
        acceptance=["cites at least one source identifier"],
        rejection=["returns a finding with no provenance"],
    )
    expect_ok("contract: valid", good_contract.validate)
    expect_error(
        "contract: empty acceptance rejected",
        lambda: HolonContract(question="q", task_type="audit").validate(),
    )
    expect_error(
        "contract: bad task_type rejected",
        lambda: HolonContract(question="q", task_type="wave", acceptance=["a"]).validate(),
    )
    expect_error(
        "contract: oversize iterations rejected",
        lambda: HolonContract(
            question="q", task_type="audit", acceptance=["a"], max_iterations=999
        ).validate(),
    )

    good_result = HolonResult(
        holon_id=good_contract.holon_id,
        status="observed",
        finding="The source defines a per-candidate penalty term.",
        next_action="Audit efe_planner.py for the live consumer.",
        evidence_refs=["event:538e4f51", "doc:urn:henri:probe:abc"],
        artifact_refs=["vault:_agentic/events/538e4f51.json"],
        token_usage={"input_tokens": 1200, "output_tokens": 180},
        token_usage_status="OBSERVED",
        iterations_used=1,
    )
    expect_ok("result: valid", lambda: good_result.validate(good_contract))

    expect_error(
        "result: OBSERVED without evidence rejected",
        lambda: HolonResult(
            holon_id=good_contract.holon_id, status="observed",
            finding="f", next_action="n",
        ).validate(good_contract),
    )
    expect_error(
        "result: estimated token usage rejected",
        lambda: HolonResult(
            holon_id=good_contract.holon_id, status="derived", finding="f",
            next_action="n", evidence_refs=["e"],
            token_usage={"input_tokens": 10, "output_tokens": 2},
            token_usage_status="INFERRED",
        ).validate(good_contract),
    )
    expect_error(
        "result: token_usage None with OBSERVED status rejected",
        lambda: HolonResult(
            holon_id=good_contract.holon_id, status="derived", finding="f",
            next_action="n", evidence_refs=["e"], token_usage=None,
            token_usage_status="OBSERVED",
        ).validate(good_contract),
    )
    expect_error(
        "result: oversize finding rejected",
        lambda: HolonResult(
            holon_id=good_contract.holon_id, status="derived",
            finding="x" * (MAX_FINDING_CHARS + 1), next_action="n", evidence_refs=["e"],
        ).validate(good_contract),
    )
    expect_error(
        "result: holon_id mismatch rejected",
        lambda: HolonResult(
            holon_id="h_wrong", status="derived", finding="f",
            next_action="n", evidence_refs=["e"],
        ).validate(good_contract),
    )
    expect_error(
        "result: iterations over contract budget rejected",
        lambda: HolonResult(
            holon_id=good_contract.holon_id, status="derived", finding="f",
            next_action="n", evidence_refs=["e"], iterations_used=99,
        ).validate(good_contract),
    )

    verdict = admit(good_result, good_contract)
    checks.append((
        "admit: clean observed result admitted",
        verdict["admitted"] is True,
        json.dumps(verdict["reasons"]),
    ))

    blocked = HolonResult(
        holon_id=good_contract.holon_id, status="blocked",
        finding="TrustGraph flow unverified.", next_action="Verify flow id.",
        evidence_refs=[],
    )
    verdict_blocked = admit(blocked, good_contract)
    checks.append((
        "admit: blocked result refused",
        verdict_blocked["admitted"] is False,
        json.dumps(verdict_blocked["reasons"]),
    ))
    checks.append((
        "admit: missing usage reports BLOCKED not zero",
        verdict_blocked["total_tokens"] is None
        and verdict_blocked["token_usage_status"] == "BLOCKED",
        f"total_tokens={verdict_blocked['total_tokens']}",
    ))

    approval_contract = HolonContract(
        question="Apply the penalty change.", task_type="execution",
        acceptance=["scope matches approval"], requires_approval=True,
    )
    approval_result = HolonResult(
        holon_id=approval_contract.holon_id, status="derived",
        finding="Patch is ready.", next_action="Request human decision.",
        evidence_refs=["commit:26ef19e"],
    )
    verdict_approval = admit(approval_result, approval_contract)
    checks.append((
        "admit: approval-gated result refused",
        verdict_approval["admitted"] is False,
        json.dumps(verdict_approval["reasons"]),
    ))

    h1 = HolonContract(question="q", task_type="audit", acceptance=["a"]).contract_hash()
    h2 = HolonContract(question="q", task_type="audit", acceptance=["a"]).contract_hash()
    checks.append((
        "contract_hash: stable across instances (lease identity)",
        h1 == h2, f"{h1[:16]} == {h2[:16]}",
    ))
    h3 = HolonContract(question="q2", task_type="audit", acceptance=["a"]).contract_hash()
    checks.append((
        "contract_hash: differs on different question",
        h1 != h3, f"{h1[:16]} != {h3[:16]}",
    ))

    round_trip = HolonContract.from_dict(good_contract.to_dict())
    checks.append((
        "contract: dict round trip preserves identity",
        round_trip.holon_id == good_contract.holon_id, round_trip.holon_id,
    ))
    cli_shape = {**good_contract.to_dict(), "contract_hash": good_contract.contract_hash()}
    expect_ok(
        "contract: CLI round trip with contract_hash accepted",
        lambda: HolonContract.from_dict(cli_shape),
    )
    expect_error(
        "contract: tampered contract_hash rejected",
        lambda: HolonContract.from_dict({**cli_shape, "contract_hash": "0" * 64}),
    )
    expect_error(
        "contract: unknown field rejected",
        lambda: HolonContract.from_dict({**good_contract.to_dict(), "sneaky": 1}),
    )

    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    print(f"\n{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new-contract", help="emit a validated contract as JSON")
    new.add_argument("--question", required=True)
    new.add_argument("--task-type", required=True, choices=sorted(TASK_TYPES))
    new.add_argument("--parent-holon-id")
    new.add_argument("--input-ref", action="append", default=[])
    new.add_argument("--source-hash", action="append", default=[])
    new.add_argument("--allowed-path", action="append", default=[])
    new.add_argument("--allowed-tool", action="append", default=[])
    new.add_argument("--accept", action="append", default=[])
    new.add_argument("--reject", action="append", default=[])
    new.add_argument("--route-pattern", default="deterministic", choices=sorted(ROUTE_PATTERNS))
    new.add_argument("--max-iterations", type=int, default=3)
    new.add_argument("--max-output-chars", type=int, default=4000)
    new.add_argument("--requires-approval", action="store_true")

    val = sub.add_parser("validate-result", help="validate a result JSON file")
    val.add_argument("result_file")
    val.add_argument("--contract-file")

    seal = sub.add_parser("seal-result", help="validate and seal a result as an event")
    seal.add_argument("result_file")
    seal.add_argument("--contract-file")

    sub.add_parser("selftest", help="run the envelope validator self-test")
    args = parser.parse_args()

    try:
        if args.command == "selftest":
            return selftest()

        if args.command == "new-contract":
            contract = HolonContract(
                question=args.question,
                task_type=args.task_type,
                parent_holon_id=args.parent_holon_id,
                input_refs=args.input_ref,
                source_hashes=args.source_hash,
                allowed_paths=args.allowed_path,
                allowed_tools=args.allowed_tool,
                acceptance=args.accept,
                rejection=args.reject,
                route_pattern=args.route_pattern,
                max_iterations=args.max_iterations,
                max_output_chars=args.max_output_chars,
                requires_approval=args.requires_approval,
            ).validate()
            out = contract.to_dict()
            out["contract_hash"] = contract.contract_hash()
            print(json.dumps(out, indent=2, sort_keys=True))
            return 0

        contract = None
        if getattr(args, "contract_file", None):
            contract = HolonContract.from_dict(
                json.loads(Path(args.contract_file).read_text(encoding="utf-8"))
            )
        result = HolonResult.from_dict(
            json.loads(Path(args.result_file).read_text(encoding="utf-8"))
        )

        if args.command == "validate-result":
            if contract is None:
                result.validate()
                print(json.dumps({
                    "status": "VALID", "holon_id": result.holon_id,
                    "note": "validated without a contract; budgets unchecked",
                    "total_tokens": result.total_tokens(),
                    "token_usage_status": result.token_usage_status,
                }, indent=2, sort_keys=True))
            else:
                print(json.dumps(admit(result, contract), indent=2, sort_keys=True))
            return 0

        event = seal_result(result, contract)
        print(json.dumps({
            "status": "SEALED", "event_id": event["event_id"],
            "audit_hash": event["audit_hash"], "holon_id": result.holon_id,
        }, indent=2, sort_keys=True))
        return 0
    except HolonError as exc:
        print(json.dumps({"status": "REJECTED", "error": "HolonError", "message": str(exc)}))
        return 1
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__, "message": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
