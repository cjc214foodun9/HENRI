"""Multi-turn tool-using agent loop for AAII AutomationBench-AA style tasks.

WHY THIS MODULE EXISTS
----------------------

AAII's AutomationBench-AA grades "SaaS workflow automation with REST API
tools" by OBJECTIVE programmatic checks on the FINAL ENVIRONMENT STATE, and
gives ZERO CREDIT to any task that triggers a guardrail violation. Two
consequences shape every design decision below:

1. A guardrail violation is a first-class typed event. It does not warn. It
   ZEROES the task score and HALTS the loop. ``TaskOutcome.graded()`` refuses
   to let a later objective-check score overwrite a zeroed score, so an
   evaluator cannot accidentally "rescue" a violating run.

2. The environment state is what is graded, not the transcript. The loop
   therefore never interprets, retries, or repairs tool arguments. It parses
   the action, dispatches it, records the observation, and moves on.

HARD RULE - NO TOOL-CALL BODIES IN THE LOOP
-------------------------------------------

``ToolUsingAgentLoop`` contains no tool implementations. It:

  * asks an INJECTED policy callable for the next ``AgentAction``,
  * dispatches ``tool_call`` actions through an INJECTED callable,
  * records a typed ``TurnRecord`` per turn,
  * appends every turn to an append+fsync JSONL ledger.

Every test therefore runs fully offline with zero network. The built-in
toolbox (``BuiltinToolbox``) exists so the OFFLINE SCAFFOLD has something
real to drive; it is injected, never imported by the loop.

PRIOR MEASUREMENT (do not repeat)
---------------------------------

A previous "agent loop" iteration in this repository was a diagnostic-only
object with a constant objective and zero real iteration; it was rejected as
a mock loop. The scaffold here must therefore PROVE iteration: the turn index
strictly increases per task, and a later action's arguments are derived from
the previous turn's observation (``AgentAction.derived_from_turn_index``).
The scaffold summary reports ``reacted_turns``; a run with zero reacted turns
is a mock loop and the contract test fails it.

REQUIREMENT MAP
---------------

(1) ``TurnRecord``            typed per-turn record, exact field set.
(2) ``TurnBudget``            explicit exhaustion -> status BUDGET_EXHAUSTED.
(3) ``GuardrailViolation``    typed, zeroes the score, halts the task.
(4) ``dispatch`` injection    pluggable callable; built-ins: write_file,
                              rest_call (denied unless allowlisted).
(5) ``abstain`` action        status ABSTAINED, never FAILED.
(6) ``TurnLedger``            JSONL, flush+fsync per row -> crash keeps rows.
(7) ``run_offline_scaffold``  <=16 tasks, zero infrastructure errors.

Per-turn telemetry is persisted with a dedicated append+fsync JSONL writer
that mirrors the ``ItemLedger`` semantics from ``henri_eval_infra`` (row
validated, written, flushed, fsync'd before returning). ``ItemLedger``'s own
schema (``henri.run-item.v1``) is TASK-scoped with different required fields,
so the scaffold uses it for the task-level ledger and ``reconcile`` for the
accounting invariants - both imported, not reimplemented.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from functools import partial
from typing import Any, Callable, Iterator, Mapping, Sequence

from henri_eval_infra import (
    ItemLedger,
    build_run_receipt,
    reconcile,
    run_id_new,
    run_output_dir,
    sha256_bytes,
    sha256_text_lf,
)

# --- schema identity -------------------------------------------------------

TURN_SCHEMA = "henri.agent-turn.v1"
OUTCOME_SCHEMA = "henri.agent-task-outcome.v1"

#: The only legal ``TurnRecord.action_type`` values (requirement 1).
ACTION_TYPES: frozenset[str] = frozenset({"tool_call", "final_answer", "abstain"})

#: Terminal task statuses. BUDGET_EXHAUSTED is explicit, never silent (req 2).
TASK_STATUSES: frozenset[str] = frozenset(
    {
        "COMPLETED",
        "ABSTAINED",
        "BUDGET_EXHAUSTED",
        "GUARDRAIL_VIOLATION",
        "EXECUTION_ERROR",
    }
)

#: Statuses that end the task without a usable final answer.
HALTED_STATUSES: frozenset[str] = frozenset(
    {"BUDGET_EXHAUSTED", "GUARDRAIL_VIOLATION", "EXECUTION_ERROR"}
)

#: AAII AutomationBench-AA: a guardrail violation earns zero.
GUARDRAIL_ZERO_SCORE = 0.0

#: The complete guardrail rule space. A violation outside this set is a bug:
#: ``GuardrailViolation.__init__`` rejects unknown rule ids so a silent typo
#: cannot downgrade a typed violation into a generic exception.
GUARDRAIL_RULES: frozenset[str] = frozenset(
    {
        "REST_HOST_NOT_ALLOWLISTED",
        "REST_SCHEME_NOT_ALLOWED",
        "FS_PATH_ESCAPE",
        "FS_WRITE_TOO_LARGE",
    }
)

#: Exact field set of a per-turn JSONL row (requirement 1).
TURN_FIELDS: tuple[str, ...] = (
    "turn_index",
    "action_type",
    "tool_name",
    "tool_args",
    "observation",
    "elapsed_ms",
    "tokens_used",
)

#: Envelope fields every turn row carries so a multi-task run is attributable.
TURN_ENVELOPE: tuple[str, ...] = ("schema", "record_type", "run_id", "task_id")

#: Exact field set of a task-outcome JSONL row.
OUTCOME_FIELDS: tuple[str, ...] = (
    "schema",
    "record_type",
    "run_id",
    "task_id",
    "status",
    "score",
    "score_zeroed",
    "guardrail_rule",
    "turns_used",
    "max_turns",
    "policy_calls",
    "tool_calls",
    "tokens_used",
    "elapsed_ms",
    "final_answer",
    "halt_reason",
)


# --- typed errors ----------------------------------------------------------

class LoopContractError(RuntimeError):
    """The loop was handed an object that violates its own typed contract."""


class ToolDispatchError(RuntimeError):
    """A tool could not be dispatched: unknown name or transport failure.

    This is INFRASTRUCTURE, not a graded policy violation. It maps to status
    EXECUTION_ERROR. A guardrail violation, by contrast, is a scored zero.
    """

    def __init__(self, code: str, message: str, *, tool_name: str | None = None,
                 details: Mapping[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.tool_name = tool_name
        self.details = dict(details or {})

    def to_observation(self) -> dict[str, Any]:
        return {
            "dispatch_error": True,
            "code": self.code,
            "message": self.message,
            "tool_name": self.tool_name,
            "details": self.details,
        }


class GuardrailViolation(Exception):
    """Typed guardrail violation: ZEROES the score and HALTS the task.

    AAII AutomationBench-AA grants zero credit for a task that trips a
    guardrail, so this exception is not a warning channel. The loop catches
    it, records the violating turn, emits an outcome row with score 0.0 and
    ``score_zeroed=True``, and stops calling the policy.
    """

    zeroes_score: bool = True

    def __init__(self, rule_id: str, message: str, *,
                 tool_name: str | None = None,
                 tool_args: Mapping[str, Any] | None = None,
                 turn_index: int | None = None,
                 details: Mapping[str, Any] | None = None):
        if rule_id not in GUARDRAIL_RULES:
            raise ValueError(
                f"unknown guardrail rule_id {rule_id!r}; "
                f"allowed={sorted(GUARDRAIL_RULES)}")
        super().__init__(f"{rule_id}: {message}")
        self.rule_id = rule_id
        self.message = message
        self.tool_name = tool_name
        self.tool_args = dict(tool_args or {})
        self.turn_index = turn_index
        self.details = dict(details or {})

    def to_observation(self) -> dict[str, Any]:
        return {
            "guardrail_violation": True,
            "rule_id": self.rule_id,
            "message": self.message,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "details": self.details,
        }


class BudgetExhausted(RuntimeError):
    """Raised by ``TurnBudget.consume`` when no turn remains."""


# --- typed turn record (requirement 1) -------------------------------------

@dataclass(frozen=True)
class TurnRecord:
    """One turn of the conversation. Append-only, typed, JSON-serializable."""

    turn_index: int
    action_type: str
    tool_name: str | None
    tool_args: Mapping[str, Any]
    observation: Any
    elapsed_ms: float
    tokens_used: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.turn_index, int) or isinstance(self.turn_index, bool):
            raise LoopContractError(f"turn_index must be int, got {self.turn_index!r}")
        if self.turn_index < 0:
            raise LoopContractError(f"turn_index must be >= 0, got {self.turn_index}")
        if self.action_type not in ACTION_TYPES:
            raise LoopContractError(
                f"action_type {self.action_type!r} not in {sorted(ACTION_TYPES)}")
        if self.action_type == "tool_call":
            if not isinstance(self.tool_name, str) or not self.tool_name:
                raise LoopContractError("tool_call requires a non-empty tool_name")
        elif self.tool_name is not None:
            raise LoopContractError(
                f"{self.action_type} must not carry tool_name "
                f"(got {self.tool_name!r})")
        if not isinstance(self.tool_args, Mapping):
            raise LoopContractError("tool_args must be a mapping")
        if isinstance(self.elapsed_ms, bool) or not isinstance(self.elapsed_ms, (int, float)):
            raise LoopContractError(f"elapsed_ms must be numeric, got {self.elapsed_ms!r}")
        if not math.isfinite(float(self.elapsed_ms)) or float(self.elapsed_ms) < 0.0:
            raise LoopContractError(
                f"elapsed_ms must be finite and >= 0, got {self.elapsed_ms!r}")
        if isinstance(self.tokens_used, bool) or not isinstance(self.tokens_used, int):
            raise LoopContractError(
                f"tokens_used must be int, got {self.tokens_used!r}")
        if self.tokens_used < 0:
            raise LoopContractError(
                f"tokens_used must be >= 0, got {self.tokens_used}")
        try:
            json.dumps(self.observation, default=str)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise LoopContractError(f"observation is not JSON-serializable: {exc}")

    def to_row(self, *, run_id: str, task_id: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema": TURN_SCHEMA,
            "record_type": "turn",
            "run_id": run_id,
            "task_id": task_id,
            "turn_index": int(self.turn_index),
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "tool_args": dict(self.tool_args),
            "observation": self.observation,
            "elapsed_ms": round(float(self.elapsed_ms), 6),
            "tokens_used": int(self.tokens_used),
        }
        return row


def validate_turn_row(row: Mapping[str, Any]) -> None:
    """A persisted turn row must carry exactly the typed envelope + fields."""
    expected = set(TURN_ENVELOPE) | set(TURN_FIELDS)
    present = set(row)
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"{TURN_SCHEMA}: missing fields {missing}")
    extra = sorted(present - expected)
    if extra:
        raise ValueError(f"{TURN_SCHEMA}: unexpected fields {extra}")
    if row["record_type"] != "turn" or row["schema"] != TURN_SCHEMA:
        raise ValueError(f"{TURN_SCHEMA}: bad schema/record_type {row.get('schema')!r}")
    if row["action_type"] not in ACTION_TYPES:
        raise ValueError(f"{TURN_SCHEMA}: invalid action_type {row['action_type']!r}")


def validate_outcome_row(row: Mapping[str, Any]) -> None:
    expected = set(OUTCOME_FIELDS)
    present = set(row)
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"{OUTCOME_SCHEMA}: missing fields {missing}")
    if row["schema"] != OUTCOME_SCHEMA or row["record_type"] != "task_outcome":
        raise ValueError(
            f"{OUTCOME_SCHEMA}: bad schema/record_type {row.get('schema')!r}")
    if row["status"] not in TASK_STATUSES:
        raise ValueError(f"{OUTCOME_SCHEMA}: invalid status {row['status']!r}")
    if row["score_zeroed"] and float(row["score"] or 0.0) != GUARDRAIL_ZERO_SCORE:
        raise ValueError(
            f"{OUTCOME_SCHEMA}: score_zeroed row must score "
            f"{GUARDRAIL_ZERO_SCORE}, got {row['score']!r}")


# --- append+fsync per-turn telemetry (requirement 6) -----------------------

class TurnLedger:
    """Append-only per-turn JSONL ledger, flush AND fsync before returning.

    DEFECT IT PREVENTS: telemetry buffered inside the process disappears when
    the process dies. A task that crashes on turn 5 must still leave turns 0-4
    on disk. ``ItemLedger`` in ``henri_eval_infra`` solves this for task-level
    rows; this class applies the same discipline to per-turn rows, whose
    schema differs.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0
        self._turn_rows = 0
        self._outcome_rows = 0
        if self.path.exists():
            # A resumed run must count pre-existing rows honestly rather than
            # reporting only what this process wrote.
            for line in self.path.read_text(encoding="utf-8",
                                            errors="replace").splitlines():
                if not line.strip():
                    continue
                self._count += 1
                if '"record_type": "turn"' in line:
                    self._turn_rows += 1
                elif '"record_type": "task_outcome"' in line:
                    self._outcome_rows += 1

    def append(self, row: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(row)
        kind = row.get("record_type")
        if kind == "turn":
            validate_turn_row(row)
        elif kind == "task_outcome":
            validate_outcome_row(row)
        else:
            raise ValueError(
                f"unknown record_type {kind!r}; expected 'turn' or 'task_outcome'")
        try:
            line = json.dumps(row, sort_keys=True, default=str)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row is not JSON-serializable: {exc}")
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._count += 1
        if kind == "turn":
            self._turn_rows += 1
        else:
            self._outcome_rows += 1
        return row

    @property
    def count(self) -> int:
        return self._count

    @property
    def turn_rows(self) -> int:
        return self._turn_rows

    @property
    def outcome_rows(self) -> int:
        return self._outcome_rows

    def rows(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.path}: line {i + 1} is not valid JSON: {exc}") from exc

    def turns_for(self, task_id: str) -> list[dict[str, Any]]:
        return [r for r in self.rows()
                if r.get("record_type") == "turn" and r.get("task_id") == task_id]

    def sha256(self) -> str:
        if not self.path.exists():
            return sha256_bytes(b"")
        return sha256_bytes(self.path.read_bytes())


# --- budget (requirement 2) ------------------------------------------------

@dataclass
class TurnBudget:
    """Hard per-task turn budget.

    Exhaustion is never silent: the loop stops BEFORE calling the policy once
    the budget is spent and returns status ``BUDGET_EXHAUSTED`` with
    ``turns_used == max_turns`` and a human-readable ``halt_reason``.
    """

    max_turns: int
    turns_used: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int):
            raise LoopContractError(f"max_turns must be int, got {self.max_turns!r}")
        if self.max_turns < 1:
            raise LoopContractError(f"max_turns must be >= 1, got {self.max_turns}")
        self.turns_used = int(self.turns_used)

    @property
    def remaining(self) -> int:
        return max(0, self.max_turns - self.turns_used)

    def exhausted(self) -> bool:
        return self.turns_used >= self.max_turns

    def consume(self) -> int:
        if self.exhausted():
            raise BudgetExhausted(
                f"turn budget exhausted ({self.turns_used}/{self.max_turns})")
        self.turns_used += 1
        return self.turns_used


# --- policy side of the protocol ------------------------------------------

@dataclass(frozen=True)
class AgentAction:
    """What the policy decided for one turn. No interpretation, no retries."""

    action_type: str
    tool_name: str | None = None
    tool_args: Mapping[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    reason: str | None = None
    tokens_used: int = 0
    #: Index of the transcript turn whose observation this action consumed.
    #: Non-None is the EVIDENCE that the loop really iterated instead of
    #: replaying a constant objective (see the prior-measurement note above).
    derived_from_turn_index: int | None = None

    def validate(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise LoopContractError(
                f"action_type {self.action_type!r} not in {sorted(ACTION_TYPES)}")
        if self.action_type == "tool_call":
            if not isinstance(self.tool_name, str) or not self.tool_name:
                raise LoopContractError("tool_call requires a non-empty tool_name")
            if not isinstance(self.tool_args, Mapping):
                raise LoopContractError("tool_args must be a mapping")
        if self.action_type == "abstain" and not self.reason:
            raise LoopContractError("abstain requires a reason")
        if isinstance(self.tokens_used, bool) or not isinstance(self.tokens_used, int):
            raise LoopContractError(f"tokens_used must be int, got {self.tokens_used!r}")
        if self.tokens_used < 0:
            raise LoopContractError(f"tokens_used must be >= 0, got {self.tokens_used}")


@dataclass(frozen=True)
class AgentTask:
    """One graded task. ``objective`` is a callable over the FINAL env state."""

    task_id: str
    prompt: str
    objective: Callable[[Path], tuple[bool, str]] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionRequest:
    """Everything the policy is allowed to see. Offline by construction."""

    task: AgentTask
    turn_index: int
    transcript: tuple[TurnRecord, ...]
    turns_remaining: int
    tokens_used: int
    run_id: str

    @property
    def last_observation(self) -> Any:
        return self.transcript[-1].observation if self.transcript else None

    @property
    def last_action(self) -> TurnRecord | None:
        return self.transcript[-1] if self.transcript else None


Policy = Callable[[ActionRequest], AgentAction]
Dispatch = Callable[[str, Mapping[str, Any]], Any]


class ScriptedPolicy:
    """Deterministic offline policy for tests and the scaffold.

    Steps may be ``AgentAction`` objects or callables
    ``(ActionRequest) -> AgentAction`` so a step can react to the previous
    observation. NEVER calls a model.
    """

    def __init__(self, steps: Sequence[AgentAction | Policy]):
        self.steps = list(steps)
        self.requests: list[ActionRequest] = []
        self.emitted: list[AgentAction] = []

    def __call__(self, request: ActionRequest) -> AgentAction:
        self.requests.append(request)
        if request.turn_index >= len(self.steps):
            raise LoopContractError(
                f"ScriptedPolicy ran out of steps at turn {request.turn_index}; "
                f"{len(self.steps)} scripted")
        step = self.steps[request.turn_index]
        action = step(request) if callable(step) else step
        if not isinstance(action, AgentAction):
            raise LoopContractError(
                f"policy step {request.turn_index} returned {type(action).__name__}, "
                "expected AgentAction")
        self.emitted.append(action)
        return action


# --- task outcome -----------------------------------------------------------

@dataclass(frozen=True)
class TaskOutcome:
    """Terminal result for one task. ``score`` may be None until graded."""

    run_id: str
    task_id: str
    status: str
    turns_used: int
    max_turns: int
    policy_calls: int
    tool_calls: int
    tokens_used: int
    elapsed_ms: float
    final_answer: str | None = None
    halt_reason: str | None = None
    score: float | None = None
    score_zeroed: bool = False
    guardrail_rule: str | None = None

    def __post_init__(self) -> None:
        if self.status not in TASK_STATUSES:
            raise LoopContractError(
                f"status {self.status!r} not in {sorted(TASK_STATUSES)}")
        if self.status == "GUARDRAIL_VIOLATION":
            if not self.score_zeroed or self.guardrail_rule is None:
                raise LoopContractError(
                    "GUARDRAIL_VIOLATION requires score_zeroed=True and a rule id")
            if float(self.score or 0.0) != GUARDRAIL_ZERO_SCORE:
                raise LoopContractError(
                    f"GUARDRAIL_VIOLATION must score {GUARDRAIL_ZERO_SCORE}")
        if self.status == "BUDGET_EXHAUSTED" and self.turns_used != self.max_turns:
            raise LoopContractError(
                f"BUDGET_EXHAUSTED requires turns_used == max_turns "
                f"({self.turns_used} != {self.max_turns}); silent truncation "
                "is a contract violation")

    @property
    def halted(self) -> bool:
        return self.status in HALTED_STATUSES

    def graded(self, objective_score: float) -> "TaskOutcome":
        """Attach the objective-check score.

        AAII gives ZERO CREDIT for a guardrail violation, so a zeroed score is
        IMMUTABLE: whatever the environment looks like afterwards, the task
        keeps its zero. This is the enforcement point for requirement 3.
        """
        value = float(objective_score)
        if self.score_zeroed:
            return replace(self, score=GUARDRAIL_ZERO_SCORE)
        return replace(self, score=value)

    def to_row(self) -> dict[str, Any]:
        return {
            "schema": OUTCOME_SCHEMA,
            "record_type": "task_outcome",
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": self.status,
            "score": None if self.score is None else round(float(self.score), 6),
            "score_zeroed": bool(self.score_zeroed),
            "guardrail_rule": self.guardrail_rule,
            "turns_used": int(self.turns_used),
            "max_turns": int(self.max_turns),
            "policy_calls": int(self.policy_calls),
            "tool_calls": int(self.tool_calls),
            "tokens_used": int(self.tokens_used),
            "elapsed_ms": round(float(self.elapsed_ms), 6),
            "final_answer": self.final_answer,
            "halt_reason": self.halt_reason,
        }


# --- the loop (requirement 4: dispatch is injected) ------------------------

class ToolUsingAgentLoop:
    """Drive a task through multiple turns against an injected dispatcher.

    ``policy``   : (ActionRequest) -> AgentAction. Domain logic lives here.
    ``dispatch`` : (tool_name, tool_args) -> observation. ALL tool bodies live
                   here or behind it. The loop never touches the network, the
                   filesystem, or a model.
    ``ledger``   : append+fsync JSONL sink for per-turn telemetry.
    """

    def __init__(self, *, policy: Policy, dispatch: Dispatch, ledger: TurnLedger,
                 run_id: str | None = None, budget: TurnBudget | None = None,
                 max_turns: int = 8,
                 clock: Callable[[], float] = time.perf_counter):
        if not callable(policy):
            raise LoopContractError("policy must be callable")
        if not callable(dispatch):
            raise LoopContractError("dispatch must be callable")
        if not isinstance(ledger, TurnLedger):
            raise LoopContractError("ledger must be a TurnLedger")
        self.policy = policy
        self.dispatch = dispatch
        self.ledger = ledger
        self.run_id = run_id or run_id_new()
        self.budget = budget if budget is not None else TurnBudget(max_turns=max_turns)
        self.clock = clock
        self.transcript: list[TurnRecord] = []

    # -- internals ---------------------------------------------------------

    def _record(self, task: AgentTask, turn: TurnRecord) -> None:
        """Persist BEFORE the caller sees the turn: a crash keeps prior rows."""
        self.ledger.append(turn.to_row(run_id=self.run_id, task_id=task.task_id))
        self.transcript.append(turn)

    # -- the loop ----------------------------------------------------------

    def run(self, task: AgentTask) -> TaskOutcome:
        if not isinstance(task, AgentTask):
            raise LoopContractError("run() requires an AgentTask")
        budget = self.budget
        budget.turns_used = 0
        self.transcript = []

        started = self.clock()
        tool_calls = 0
        tokens_used = 0
        policy_calls = 0
        final_answer: str | None = None
        status: str | None = None
        halt_reason: str | None = None
        violation: GuardrailViolation | None = None

        while True:
            # (2) Explicit exhaustion: never silently truncate.
            if budget.exhausted():
                status = "BUDGET_EXHAUSTED"
                halt_reason = (
                    f"turn budget exhausted after {budget.turns_used} turns "
                    f"(max_turns={budget.max_turns}); no final answer or "
                    "abstention was produced")
                break

            budget.consume()
            request = ActionRequest(
                task=task,
                turn_index=len(self.transcript),
                transcript=tuple(self.transcript),
                turns_remaining=budget.remaining,
                tokens_used=tokens_used,
                run_id=self.run_id,
            )
            policy_calls += 1
            t0 = self.clock()
            action = self.policy(request)
            if not isinstance(action, AgentAction):
                raise LoopContractError(
                    f"policy returned {type(action).__name__}, expected AgentAction")
            action.validate()

            tool_name: str | None = None
            tool_args: Mapping[str, Any] = {}
            observation: Any = None
            status_after: str | None = None

            if action.action_type == "tool_call":
                tool_name = action.tool_name
                tool_args = dict(action.tool_args)
                try:
                    observation = self.dispatch(tool_name, tool_args)
                    tool_calls += 1
                except GuardrailViolation as gv:
                    # (3) Typed violation: record the turn, zero the score, halt.
                    gv.turn_index = len(self.transcript)
                    obs = gv.to_observation()
                    self._record(task, TurnRecord(
                        turn_index=len(self.transcript),
                        action_type="tool_call",
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=obs,
                        elapsed_ms=(self.clock() - t0) * 1000.0,
                        tokens_used=action.tokens_used,
                    ))
                    tokens_used += action.tokens_used
                    violation = gv
                    status = "GUARDRAIL_VIOLATION"
                    halt_reason = (
                        f"guardrail {gv.rule_id} tripped by tool "
                        f"{tool_name!r}; score zeroed and task halted")
                    break
                except ToolDispatchError as exc:
                    obs = exc.to_observation()
                    self._record(task, TurnRecord(
                        turn_index=len(self.transcript),
                        action_type="tool_call",
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=obs,
                        elapsed_ms=(self.clock() - t0) * 1000.0,
                        tokens_used=action.tokens_used,
                    ))
                    tokens_used += action.tokens_used
                    status = "EXECUTION_ERROR"
                    halt_reason = f"tool dispatch failed: {exc.code} ({exc.message})"
                    break
            elif action.action_type == "final_answer":
                final_answer = action.final_answer
                status_after = "COMPLETED"
            else:  # abstain
                # (5) Abstention is a legitimate terminal action: ABSTAINED,
                # deliberately NOT FAILED.
                final_answer = None
                status_after = "ABSTAINED"

            self._record(task, TurnRecord(
                turn_index=len(self.transcript),
                action_type=action.action_type,
                tool_name=tool_name,
                tool_args=tool_args,
                observation=observation,
                elapsed_ms=(self.clock() - t0) * 1000.0,
                tokens_used=action.tokens_used,
            ))
            tokens_used += action.tokens_used

            if status_after is not None:
                status = status_after
                if status_after == "ABSTAINED":
                    halt_reason = f"policy abstained: {action.reason}"
                break

        assert status is not None, "loop exited without a terminal status"
        score = GUARDRAIL_ZERO_SCORE if violation is not None else None
        outcome = TaskOutcome(
            run_id=self.run_id,
            task_id=task.task_id,
            status=status,
            turns_used=budget.turns_used,
            max_turns=budget.max_turns,
            policy_calls=policy_calls,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            elapsed_ms=(self.clock() - started) * 1000.0,
            final_answer=final_answer,
            halt_reason=halt_reason,
            score=score,
            score_zeroed=violation is not None,
            guardrail_rule=violation.rule_id if violation else None,
        )
        self.ledger.append(outcome.to_row())
        return outcome


# --- built-in toolbox (requirement 4) --------------------------------------

#: Default cap for a single file write. Exceeding it is a guardrail violation.
DEFAULT_MAX_WRITE_BYTES = 1 << 20

_REST_METHODS: frozenset[str] = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})


def normalize_allowlist_entry(entry: str) -> str:
    """Reduce an allowlist entry to a bare lowercase host (wildcards kept).

    Accepts 'api.example.com', 'https://api.example.com', 'api.example.com:443'
    and '*.example.com'; all normalize to a comparable host pattern.
    """
    e = str(entry).strip().lower()
    if "://" in e:
        e = e.split("://", 1)[1]
    e = e.split("/", 1)[0]          # drop path
    if e.startswith("*."):
        return "*." + e[2:].split(":", 1)[0]
    return e.split(":", 1)[0].split("@")[-1]


def host_is_allowlisted(host: str, allowlist: Sequence[str]) -> bool:
    """An ALLOWLIST, not a denylist: an empty list denies every host."""
    h = str(host).strip().lower()
    for entry in allowlist:
        pattern = normalize_allowlist_entry(entry)
        if not pattern:
            continue
        if pattern.startswith("*."):
            suffix = pattern[2:]
            if h.endswith("." + suffix) and h != suffix:
                return True
        elif h == pattern:
            return True
    return False


def _urllib_transport(url: str, method: str, body: Any, headers: Mapping[str, str],
                      timeout: float) -> dict[str, Any]:
    """Default REST transport. Reached ONLY for an allowlisted host."""
    payload = body.encode("utf-8") if isinstance(body, str) else body
    req = urllib.request.Request(url, data=payload, headers=dict(headers),
                                 method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(65536)
        return {"status": int(resp.status), "body": raw.decode("utf-8", "replace")}


@dataclass
class RestCallTool:
    """Built-in REST tool. DISABLED unless an allowlist entry matches.

    The allowlist check runs BEFORE any socket work: a disallowed host is
    refused without a single packet leaving the process. That ordering is what
    makes the offline contract test meaningful.
    """

    allowlist: Sequence[str] = ()
    timeout: float = 10.0
    transport: Callable[..., dict[str, Any]] | None = None

    def __call__(self, tool_args: Mapping[str, Any]) -> dict[str, Any]:
        url = tool_args.get("url")
        if not isinstance(url, str) or not url:
            raise ToolDispatchError("REST_BAD_REQUEST", "url must be a non-empty string",
                                    tool_name="rest_call")
        method = str(tool_args.get("method", "GET")).upper()
        if method not in _REST_METHODS:
            raise ToolDispatchError(
                "REST_BAD_METHOD", f"method {method!r} not allowed",
                tool_name="rest_call", details={"allowed": sorted(_REST_METHODS)})
        parsed = urllib.parse.urlsplit(url)
        scheme = (parsed.scheme or "").lower()
        if scheme not in {"http", "https"}:
            raise GuardrailViolation(
                "REST_SCHEME_NOT_ALLOWED",
                f"scheme {scheme!r} is not http/https",
                tool_name="rest_call", tool_args=dict(tool_args))
        host = (parsed.hostname or "").lower()
        if not host_is_allowlisted(host, self.allowlist):
            raise GuardrailViolation(
                "REST_HOST_NOT_ALLOWLISTED",
                f"host {host!r} is not in the rest allowlist "
                f"({sorted(normalize_allowlist_entry(e) for e in self.allowlist)})",
                tool_name="rest_call", tool_args=dict(tool_args),
                details={"host": host, "allowlist": list(self.allowlist)})
        transport = self.transport or _urllib_transport
        try:
            result = transport(url, method, tool_args.get("body"),
                               dict(tool_args.get("headers") or {}), self.timeout)
        except Exception as exc:
            raise ToolDispatchError(
                "REST_TRANSPORT_ERROR", f"{type(exc).__name__}: {exc}",
                tool_name="rest_call", details={"host": host}) from exc
        return {"host": host, "method": method, "url": url, **dict(result)}


@dataclass
class FileWriteTool:
    """Built-in file-write tool, sandboxed to ``root``.

    A path that resolves outside ``root`` is a guardrail violation, not a
    convenience error: writing outside the task environment would corrupt the
    final state that AutomationBench-AA grades.
    """

    root: Path
    max_bytes: int = DEFAULT_MAX_WRITE_BYTES

    def resolve(self, rel: str) -> Path:
        root_r = Path(self.root).resolve()
        target = (root_r / str(rel)).resolve()
        if target != root_r and root_r not in target.parents:
            raise GuardrailViolation(
                "FS_PATH_ESCAPE",
                f"path {rel!r} resolves outside the task environment",
                tool_name="write_file", tool_args={"path": str(rel)},
                details={"root": str(root_r), "resolved": str(target)})
        return target

    def __call__(self, tool_args: Mapping[str, Any]) -> dict[str, Any]:
        rel = tool_args.get("path")
        if not isinstance(rel, str) or not rel:
            raise ToolDispatchError("FS_BAD_REQUEST", "path must be a non-empty string",
                                    tool_name="write_file")
        content = tool_args.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, sort_keys=True)
        target = self.resolve(rel)
        raw = content.encode("utf-8")
        if len(raw) > self.max_bytes:
            raise GuardrailViolation(
                "FS_WRITE_TOO_LARGE",
                f"{len(raw)} bytes exceeds max_bytes={self.max_bytes}",
                tool_name="write_file", tool_args={"path": rel},
                details={"bytes": len(raw), "max_bytes": self.max_bytes})
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        return {
            "path": str(rel),
            "absolute_path": str(target),
            "bytes_written": len(raw),
            "sha256": sha256_bytes(raw),
        }


class BuiltinToolbox:
    """Registry of offline-safe built-ins, exposed as an injected dispatcher.

    Deliberately NOT imported by ``ToolUsingAgentLoop``: the loop receives
    ``toolbox.dispatch`` as a plain callable, which is what keeps the loop
    testable offline and prevents it from becoming a mock loop.
    """

    def __init__(self, root: str | Path, *, rest_allowlist: Sequence[str] = (),
                 rest_transport: Callable[..., dict[str, Any]] | None = None,
                 max_write_bytes: int = DEFAULT_MAX_WRITE_BYTES,
                 rest_timeout: float = 10.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._tools: dict[str, Callable[[Mapping[str, Any]], Any]] = {
            "write_file": FileWriteTool(self.root, max_bytes=max_write_bytes),
            "rest_call": RestCallTool(allowlist=tuple(rest_allowlist),
                                      transport=rest_transport,
                                      timeout=rest_timeout),
        }

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def dispatch(self, tool_name: str, tool_args: Mapping[str, Any]) -> Any:
        tool = self._tools.get(str(tool_name))
        if tool is None:
            # Infrastructure, not a graded policy violation: the environment
            # cannot distinguish a typo from an unauthorized tool.
            raise ToolDispatchError(
                "UNKNOWN_TOOL", f"no built-in tool named {tool_name!r}",
                tool_name=str(tool_name), details={"known": list(self.tool_names)})
        if not isinstance(tool_args, Mapping):
            raise ToolDispatchError("TOOL_BAD_ARGS", "tool_args must be a mapping",
                                    tool_name=str(tool_name))
        return tool(tool_args)


# --- offline scaffold (requirement 7) --------------------------------------

SCAFFOLD_MAX_TASKS = 16
SCAFFOLD_BENCHMARK_ID = "automationbench-aa-scaffold"


def _scaffold_records(task_index: int) -> list[dict[str, Any]]:
    """Deterministic synthetic 'SaaS records' - no dataset, no network."""
    n = 3 + (task_index % 4)
    return [
        {"id": f"REC-{task_index:03d}-{i}", "status": "open" if i % 2 else "closed",
         "amount": (i + 1) * 10}
        for i in range(n)
    ]


def _scaffold_objective(env_root: Path, *,
                        expected_rows: int) -> tuple[bool, str]:
    """OBJECTIVE check on the final environment state (AAII grading style)."""
    export = env_root / "export.json"
    summary = env_root / "summary.json"
    if not export.exists():
        return False, "missing export.json"
    if not summary.exists():
        return False, "missing summary.json"
    try:
        rows = json.loads(export.read_text(encoding="utf-8"))
        summ = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"artifact is not valid JSON: {exc}"
    if len(rows) != expected_rows:
        return False, f"export has {len(rows)} rows, expected {expected_rows}"
    raw = export.read_bytes()
    if summ.get("export_bytes") != len(raw):
        return False, (f"summary.export_bytes={summ.get('export_bytes')} != "
                       f"{len(raw)}")
    if summ.get("export_sha256") != sha256_bytes(raw):
        return False, "summary.export_sha256 does not match export.json"
    return True, f"export.json ({len(raw)} bytes) consistent with summary.json"


class ScaffoldPolicy:
    """Deterministic offline policy: 2 tool calls then a final answer.

    Turn 1's arguments are DERIVED from turn 0's observation, so a run that
    reports reacted_turns == 0 would prove this scaffold is not iterating.
    Never calls a model, never touches the network.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]]):
        self.records = [dict(r) for r in records]
        self.emitted: list[AgentAction] = []

    def __call__(self, request: ActionRequest) -> AgentAction:
        if request.turn_index == 0:
            action = AgentAction(
                action_type="tool_call", tool_name="write_file",
                tool_args={"path": "export.json",
                           "content": json.dumps(self.records, sort_keys=True)},
                tokens_used=12)
        elif request.turn_index == 1:
            obs = request.last_observation
            if not isinstance(obs, Mapping) or "sha256" not in obs:
                raise LoopContractError(
                    "turn 1 requires the write_file observation from turn 0")
            action = AgentAction(
                action_type="tool_call", tool_name="write_file",
                tool_args={
                    "path": "summary.json",
                    "content": json.dumps(
                        {"export_bytes": int(obs["bytes_written"]),
                         "export_sha256": str(obs["sha256"]),
                         "row_count": len(self.records)},
                        sort_keys=True),
                },
                tokens_used=8,
                derived_from_turn_index=0)
        else:
            obs = request.last_observation
            action = AgentAction(
                action_type="final_answer",
                final_answer=(
                    f"wrote {len(self.records)} records; summary sha256 "
                    f"{obs.get('sha256') if isinstance(obs, Mapping) else 'n/a'}"),
                tokens_used=5,
                derived_from_turn_index=request.turn_index - 1)
        self.emitted.append(action)
        return action


@dataclass(frozen=True)
class ScaffoldTask(AgentTask):
    """An AgentTask plus its objective-check expectation."""

    expected_rows: int = 0


def build_scaffold_tasks(n_tasks: int) -> list[ScaffoldTask]:
    if isinstance(n_tasks, bool) or not isinstance(n_tasks, int):
        raise ValueError(f"n_tasks must be int, got {n_tasks!r}")
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    if n_tasks > SCAFFOLD_MAX_TASKS:
        raise ValueError(
            f"the offline scaffold is capped at {SCAFFOLD_MAX_TASKS} tasks; "
            f"got {n_tasks}. A bigger run needs the real adapter, not the stub.")
    tasks: list[ScaffoldTask] = []
    for i in range(n_tasks):
        records = _scaffold_records(i)
        tasks.append(ScaffoldTask(
            task_id=f"scaffold-task-{i:02d}",
            prompt=(f"Export {len(records)} records to export.json, then write a "
                    "summary.json whose export_bytes/export_sha256 describe it."),
            objective=partial(_scaffold_objective, expected_rows=len(records)),
            metadata={"benchmark_id": SCAFFOLD_BENCHMARK_ID,
                      "suite": "automationbench-aa-scaffold",
                      "task_index": i,
                      "guardrail_policy": "zero_credit_on_violation"},
            expected_rows=len(records),
        ))
    return tasks


#: Map loop status -> ``henri.run-item.v1`` status (from henri_eval_infra).
_ITEM_STATUS = {
    "COMPLETED": None,   # decided by the objective check
    "ABSTAINED": "ABSTAINED",
    "BUDGET_EXHAUSTED": "TIMEOUT",
    "GUARDRAIL_VIOLATION": "FAILED",   # graded zero: attempted, not an infra error
    "EXECUTION_ERROR": "EXECUTION_ERROR",
}


def run_offline_scaffold(n_tasks: int = 12, *,
                         root: str | Path | None = None,
                         run_id: str | None = None,
                         commit_sha: str = "UNKNOWN",
                         max_turns: int = 6,
                         benchmark_id: str = SCAFFOLD_BENCHMARK_ID) -> dict[str, Any]:
    """Run <=16 tasks end-to-end with ZERO infrastructure errors.

    Everything here is offline: the stub policy, the built-in toolbox with an
    EMPTY rest allowlist, and the objective checks. Returns a summary plus a
    ``henri.run-evidence.v1`` receipt built by ``henri_eval_infra``.
    """
    tasks = build_scaffold_tasks(n_tasks)
    root_path = Path(root) if root is not None else (
        Path(tempfile.gettempdir()) / "henri_agent_loop_scaffold")
    run_id = run_id or run_id_new()
    run_dir = run_output_dir(root_path, commit_sha, benchmark_id, run_id)

    turn_ledger = TurnLedger(run_dir / "turns.jsonl")
    item_ledger = ItemLedger(run_dir / "items.jsonl")

    results: list[dict[str, Any]] = []
    passed = failed = execution_errors = abstained = 0
    tool_call_rows = 0
    reacted_turns = 0

    for task in tasks:
        env_root = run_dir / "tasks" / task.task_id
        env_root.mkdir(parents=True, exist_ok=True)
        policy = ScaffoldPolicy(_scaffold_records(int(task.metadata["task_index"])))
        toolbox = BuiltinToolbox(env_root, rest_allowlist=())
        loop = ToolUsingAgentLoop(policy=policy, dispatch=toolbox.dispatch,
                                  ledger=turn_ledger, run_id=run_id,
                                  budget=TurnBudget(max_turns=max_turns))
        try:
            outcome = loop.run(task)
        except Exception as exc:  # infra only: a stub policy that raises
            execution_errors += 1
            results.append({"task_id": task.task_id, "status": "EXECUTION_ERROR",
                            "error": f"{type(exc).__name__}: {exc}"})
            continue

        tool_call_rows += outcome.tool_calls
        reacted_turns += sum(1 for a in policy.emitted
                             if a.derived_from_turn_index is not None)

        ok, detail = (task.objective(env_root)
                      if outcome.status == "COMPLETED" else (False, "not completed"))
        graded = outcome.graded(1.0 if ok else 0.0)
        item_status = _ITEM_STATUS[outcome.status]
        if item_status is None:
            item_status = "PASSED" if ok else "FAILED"
        if item_status == "PASSED":
            passed += 1
        elif item_status == "FAILED":
            failed += 1
        elif item_status == "EXECUTION_ERROR":
            execution_errors += 1
        elif item_status == "ABSTAINED":
            abstained += 1

        item_ledger.append({
            "item_id": task.task_id,
            "prompt_sha256": sha256_text_lf(str(task.prompt)),
            "raw_stdout_sha256": sha256_text_lf(graded.final_answer or detail),
            "raw_stderr_sha256": sha256_text_lf(graded.halt_reason or ""),
            "status": item_status,
            "elapsed_ms": round(float(graded.elapsed_ms), 6),
            "tool_calls": int(graded.tool_calls),
            "retries": 0,
            "candidate_scores": [] if graded.score is None else [float(graded.score)],
            "sagnac_delta": None,
            "token_top1": None,
            "token_entropy": None,
        })
        results.append({
            "task_id": task.task_id,
            "status": graded.status,
            "item_status": item_status,
            "objective_ok": bool(ok),
            "objective_detail": detail,
            "score": graded.score,
            "score_zeroed": graded.score_zeroed,
            "turns_used": graded.turns_used,
            "tool_calls": graded.tool_calls,
            "reacted_turns": sum(1 for a in policy.emitted
                                 if a.derived_from_turn_index is not None),
        })

    attempted = passed + failed
    item_count = len(tasks)
    accounting = reconcile(item_count, attempted, passed, failed, execution_errors)
    if not accounting["valid"]:
        raise RuntimeError(f"INVALID_RUN: {accounting['problems']}")

    receipt = build_run_receipt(
        run_id=run_id, commit_sha=commit_sha, benchmark_id=benchmark_id,
        dataset_source="offline-scaffold:generated",
        dataset_sha256=sha256_text_lf(json.dumps(
            [t.task_id for t in tasks], sort_keys=True)),
        item_count=item_count, attempted=attempted, passed=passed, failed=failed,
        execution_errors=execution_errors, ledger=item_ledger,
        extra={
            "mode": "OFFLINE_SCAFFOLD",
            "stub_adapter": True,
            "guardrail_policy": "zero_credit_on_violation",
            "turn_rows": turn_ledger.turn_rows,
            "outcome_rows": turn_ledger.outcome_rows,
            "tool_call_rows": tool_call_rows,
            "reacted_turns": reacted_turns,
            "abstained_count": abstained,
            "turns_ledger_sha256": turn_ledger.sha256(),
        })

    return {
        "mode": "OFFLINE_SCAFFOLD",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "benchmark_id": benchmark_id,
        "item_count": item_count,
        "attempted": attempted,
        "passed": passed,
        "failed": failed,
        "abstained": abstained,
        "execution_errors": execution_errors,
        "tool_call_count": tool_call_rows,
        "turn_rows": turn_ledger.turn_rows,
        "outcome_rows": turn_ledger.outcome_rows,
        "reacted_turns": reacted_turns,
        "turns_ledger_sha256": turn_ledger.sha256(),
        "accounting": accounting,
        "results": results,
        "receipt": receipt,
    }


# --- CLI -------------------------------------------------------------------

def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-turn tool-using agent loop (protocol only; no model calls)")
    parser.add_argument("--scaffold", action="store_true",
                        help="run the offline <=16 task scaffold")
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--root", default=None,
                        help="output root (default: <tmp>/henri_agent_loop_scaffold)")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--commit-sha", default="UNKNOWN")
    args = parser.parse_args(argv)

    if not args.scaffold:
        parser.error("nothing to do: pass --scaffold")
    summary = run_offline_scaffold(args.tasks, root=args.root,
                                   commit_sha=args.commit_sha,
                                   max_turns=args.max_turns)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2, sort_keys=True))
    return 0 if summary["execution_errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
