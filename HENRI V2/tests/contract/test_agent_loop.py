"""Contract tests for the multi-turn tool-using agent loop.

Every test here runs OFFLINE. No model API, no network, no dataset. The one
subprocess test spawns a child that hard-kills itself with ``os._exit`` to
prove that per-turn telemetry survives a crash.

The tests are written against the AAII AutomationBench-AA grading contract:
objective checks on the FINAL environment state, and ZERO credit for a task
that trips a guardrail.
"""
import json
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from henri_agent_loop import (  # noqa: E402
    ACTION_TYPES,
    GUARDRAIL_ZERO_SCORE,
    SCAFFOLD_MAX_TASKS,
    TURN_ENVELOPE,
    TURN_FIELDS,
    AgentAction,
    AgentTask,
    BuiltinToolbox,
    FileWriteTool,
    GuardrailViolation,
    LoopContractError,
    RestCallTool,
    ScriptedPolicy,
    ToolDispatchError,
    ToolUsingAgentLoop,
    TurnBudget,
    TurnLedger,
    TurnRecord,
    build_scaffold_tasks,
    host_is_allowlisted,
    run_offline_scaffold,
    validate_outcome_row,
)

MODEL_FREE_TOOLS = ("write_file", "rest_call")


# --- offline guards --------------------------------------------------------

@pytest.fixture
def no_network(monkeypatch):
    """Any socket use fails the test: the loop and tools must be offline."""
    def boom(*args, **kwargs):  # pragma: no cover - only on violation
        raise AssertionError("network access attempted in an offline test")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    yield


def _loop(tmp_path, policy, *, tool_root=None, max_turns=8,
          rest_allowlist=(), run_id="test-run"):
    ledger = TurnLedger(tmp_path / "turns.jsonl")
    toolbox = BuiltinToolbox(tool_root or (tmp_path / "env"),
                             rest_allowlist=rest_allowlist)
    loop = ToolUsingAgentLoop(policy=policy, dispatch=toolbox.dispatch,
                             ledger=ledger, run_id=run_id,
                             budget=TurnBudget(max_turns=max_turns))
    return loop, ledger, toolbox


# --- (1) multi-turn completion, and PROOF that the loop iterates -----------

def test_multi_turn_completion_and_observation_dependent_next_action(
        tmp_path, no_network):
    """3 turns, and turn 1's ARGUMENTS come from turn 0's OBSERVATION.

    A diagnostic-only object with a constant objective would pass a naive
    "did it finish" test. This test pins the iteration: the transcript grows,
    the turn index strictly increases, and the second tool call's payload is
    a function of the first tool call's returned digest.
    """
    steps = [
        AgentAction(action_type="tool_call", tool_name="write_file",
                    tool_args={"path": "alpha.txt", "content": "alpha-payload"},
                    tokens_used=11),
        lambda req: AgentAction(
            action_type="tool_call", tool_name="write_file",
            tool_args={"path": "beta.txt",
                       "content": f"digest={req.last_observation['sha256']}",
                       "bytes": req.last_observation["bytes_written"]},
            tokens_used=7, derived_from_turn_index=0),
        lambda req: AgentAction(
            action_type="final_answer",
            final_answer=f"wrote alpha then {req.last_observation['path']}",
            tokens_used=5, derived_from_turn_index=1),
    ]
    policy = ScriptedPolicy(steps)
    loop, ledger, _ = _loop(tmp_path, policy, max_turns=8)
    outcome = loop.run(AgentTask(task_id="t-multi", prompt="write two files"))

    assert outcome.status == "COMPLETED"
    assert outcome.turns_used == 3 and outcome.policy_calls == 3
    assert outcome.tool_calls == 2
    assert outcome.tokens_used == 11 + 7 + 5
    assert outcome.final_answer == "wrote alpha then beta.txt"
    assert outcome.score is None, "grading belongs to the objective check"
    assert not outcome.score_zeroed and outcome.guardrail_rule is None

    rows = [r for r in ledger.rows() if r["record_type"] == "turn"]
    assert len(rows) == 3
    assert ledger.turn_rows == 3 and ledger.outcome_rows == 1
    # (2) strictly increasing turn index
    assert [r["turn_index"] for r in rows] == [0, 1, 2]
    assert [r["action_type"] for r in rows] == ["tool_call", "tool_call",
                                                "final_answer"]
    # (1) exact typed field set on disk
    expected_keys = set(TURN_ENVELOPE) | set(TURN_FIELDS)
    for row in rows:
        assert set(row) == expected_keys
        assert row["elapsed_ms"] >= 0.0 and row["tokens_used"] >= 0
        assert row["action_type"] in ACTION_TYPES
        if row["action_type"] == "tool_call":
            assert isinstance(row["observation"], dict)
        else:
            assert row["observation"] is None, "no tool call, no observation"
    # the multi-turn dependency is visible IN THE TELEMETRY
    assert rows[1]["tool_args"]["content"] == f"digest={rows[0]['observation']['sha256']}"
    assert rows[1]["tool_args"]["content"] != rows[0]["tool_args"]["content"]
    assert (rows[0]["tool_args"]["path"], rows[1]["tool_args"]["path"]) == \
        ("alpha.txt", "beta.txt")
    # transcript handed to the policy grows strictly: real iteration
    assert [len(r.transcript) for r in policy.requests] == [0, 1, 2]
    assert [r.turn_index for r in policy.requests] == [0, 1, 2]
    assert policy.requests[1].last_observation["path"] == "alpha.txt"


# --- (2) turn budget with an explicit exhaustion outcome -------------------

def test_turn_budget_exhaustion_is_explicit_never_silent(tmp_path, no_network):
    steps = [AgentAction(action_type="tool_call", tool_name="write_file",
                         tool_args={"path": f"f{i}.txt", "content": f"turn-{i}"},
                         tokens_used=3) for i in range(5)]
    policy = ScriptedPolicy(steps)
    loop, ledger, _ = _loop(tmp_path, policy, max_turns=3)
    outcome = loop.run(AgentTask(task_id="t-budget", prompt="never finish"))

    assert outcome.status == "BUDGET_EXHAUSTED"
    assert outcome.turns_used == 3 == outcome.max_turns
    assert outcome.policy_calls == 3, "the policy must not be called past budget"
    assert len(policy.requests) == 3
    assert outcome.final_answer is None
    assert outcome.halt_reason and "exhausted" in outcome.halt_reason
    assert outcome.score is None and not outcome.score_zeroed

    rows = list(ledger.rows())
    turns = [r for r in rows if r["record_type"] == "turn"]
    outcomes = [r for r in rows if r["record_type"] == "task_outcome"]
    assert len(turns) == 3 and len(outcomes) == 1
    assert [r["turn_index"] for r in turns] == [0, 1, 2]
    # the exhaustion is ON DISK, with an explicit reason and full budget use
    assert outcomes[0]["status"] == "BUDGET_EXHAUSTED"
    assert outcomes[0]["turns_used"] == outcomes[0]["max_turns"] == 3
    assert "exhausted" in outcomes[0]["halt_reason"]


def test_budget_exhaustion_outcome_rejects_silent_truncation():
    """A BUDGET_EXHAUSTED outcome that under-reports turns is a contract bug."""
    from henri_agent_loop import TaskOutcome

    with pytest.raises(LoopContractError):
        TaskOutcome(run_id="r", task_id="t", status="BUDGET_EXHAUSTED",
                    turns_used=2, max_turns=3, policy_calls=2, tool_calls=2,
                    tokens_used=0, elapsed_ms=1.0, halt_reason="truncated")


# --- (3) guardrail violation zeroes the score and halts --------------------

def test_guardrail_violation_zeroes_score_and_halts(tmp_path, no_network):
    steps = [
        AgentAction(action_type="tool_call", tool_name="rest_call",
                    tool_args={"url": "https://api.slack.com/api/chat.postMessage",
                               "method": "POST", "body": "{}"},
                    tokens_used=9),
        # This turn must NEVER execute: the violation halts the task.
        AgentAction(action_type="final_answer", final_answer="posted", tokens_used=4),
    ]
    policy = ScriptedPolicy(steps)
    loop, ledger, _ = _loop(tmp_path, policy, max_turns=8, rest_allowlist=())
    outcome = loop.run(AgentTask(task_id="t-guard", prompt="post to slack"))

    assert outcome.status == "GUARDRAIL_VIOLATION"
    assert outcome.score == GUARDRAIL_ZERO_SCORE == 0.0
    assert outcome.score_zeroed is True
    assert outcome.guardrail_rule == "REST_HOST_NOT_ALLOWLISTED"
    assert outcome.turns_used == 1 and outcome.policy_calls == 1
    assert outcome.tool_calls == 0, "a refused call is not a completed tool call"
    assert len(policy.requests) == 1, "the loop must halt on a violation"
    assert outcome.halted is True

    # AAII zero-credit is IMMUTABLE: an objective check cannot rescue it.
    assert outcome.graded(1.0).score == 0.0
    assert outcome.graded(1.0).score_zeroed is True

    rows = list(ledger.rows())
    violating = [r for r in rows if r["record_type"] == "turn"][0]
    assert violating["observation"]["guardrail_violation"] is True
    assert violating["observation"]["rule_id"] == "REST_HOST_NOT_ALLOWLISTED"
    assert violating["tool_name"] == "rest_call"
    final = rows[-1]
    assert final["record_type"] == "task_outcome"
    assert final["status"] == "GUARDRAIL_VIOLATION"
    assert final["score"] == 0.0 and final["score_zeroed"] is True
    assert final["guardrail_rule"] == "REST_HOST_NOT_ALLOWLISTED"


def test_filesystem_path_escape_is_a_typed_guardrail_violation(tmp_path):
    tool = FileWriteTool(tmp_path / "env")
    with pytest.raises(GuardrailViolation) as ei:
        tool({"path": "../outside.txt", "content": "escape"})
    assert ei.value.rule_id == "FS_PATH_ESCAPE"
    assert ei.value.zeroes_score is True
    assert not (tmp_path / "outside.txt").exists()


def test_unknown_guardrail_rule_id_is_rejected():
    with pytest.raises(ValueError):
        GuardrailViolation("MADE_UP_RULE", "silent typo")


# --- (5) abstention path ---------------------------------------------------

def test_abstention_is_abstained_not_failed(tmp_path, no_network):
    steps = [
        AgentAction(action_type="tool_call", tool_name="write_file",
                    tool_args={"path": "partial.txt", "content": "attempt"},
                    tokens_used=6),
        lambda req: AgentAction(
            action_type="abstain",
            reason=f"cannot verify record state after {req.last_observation['path']}",
            tokens_used=2, derived_from_turn_index=0),
    ]
    policy = ScriptedPolicy(steps)
    loop, ledger, _ = _loop(tmp_path, policy, max_turns=8)
    outcome = loop.run(AgentTask(task_id="t-abstain", prompt="maybe impossible"))

    assert outcome.status == "ABSTAINED"
    assert outcome.status != "FAILED"
    assert outcome.halted is False, "abstention is a terminal action, not a halt"
    assert outcome.score is None and outcome.score_zeroed is False
    assert outcome.guardrail_rule is None
    assert outcome.turns_used == 2 and outcome.tool_calls == 1
    assert "cannot verify" in outcome.halt_reason

    rows = [r for r in ledger.rows() if r["record_type"] == "turn"]
    assert [r["action_type"] for r in rows] == ["tool_call", "abstain"]
    assert rows[-1]["tool_name"] is None


def test_abstain_on_turn_zero_needs_no_tool_call(tmp_path, no_network):
    policy = ScriptedPolicy([AgentAction(action_type="abstain", reason="out of scope",
                                         tokens_used=3)])
    loop, ledger, _ = _loop(tmp_path, policy, max_turns=4)
    outcome = loop.run(AgentTask(task_id="t-abstain0", prompt="unclear"))
    assert outcome.status == "ABSTAINED"
    assert outcome.turns_used == 1 and outcome.tool_calls == 0
    turn = [r for r in ledger.rows() if r["record_type"] == "turn"][0]
    assert turn["turn_index"] == 0 and turn["action_type"] == "abstain"
    assert turn["observation"] is None


# --- (4) REST tool: disabled unless the allowlist matches ------------------

def test_disallowed_rest_host_refused_before_any_socket_work(tmp_path, no_network):
    tool = RestCallTool(allowlist=("api.allowed.example",))
    with pytest.raises(GuardrailViolation) as ei:
        tool({"url": "https://evil.example.com/steal", "method": "POST"})
    assert ei.value.rule_id == "REST_HOST_NOT_ALLOWLISTED"
    assert ei.value.tool_args["url"] == "https://evil.example.com/steal"

    # empty allowlist denies everything; non-http schemes are refused too
    with pytest.raises(GuardrailViolation) as ei2:
        RestCallTool(allowlist=())({"url": "https://api.allowed.example/x"})
    assert ei2.value.rule_id == "REST_HOST_NOT_ALLOWLISTED"
    with pytest.raises(GuardrailViolation) as ei3:
        RestCallTool(allowlist=("api.allowed.example",))({"url": "ftp://api.allowed.example/x"})
    assert ei3.value.rule_id == "REST_SCHEME_NOT_ALLOWED"

    # host matching is pure and offline; only allowlisted hosts get through
    assert host_is_allowlisted("api.allowed.example",
                               ["https://API.Allowed.Example:443/base"]) is True
    assert host_is_allowlisted("sub.allowed.example", ["*.allowed.example"]) is True
    assert host_is_allowlisted("allowed.example", ["*.allowed.example"]) is False
    assert host_is_allowlisted("api.allowed.example", []) is False


def test_allowlisted_host_uses_injected_transport_not_the_network(tmp_path,
                                                                  no_network):
    seen = {}

    def fake_transport(url, method, body, headers, timeout):
        seen.update(url=url, method=method, body=body, timeout=timeout)
        return {"status": 200, "body": '{"ok":true}'}

    tool = RestCallTool(allowlist=("api.allowed.example", ),
                        transport=fake_transport)
    out = tool({"url": "https://api.allowed.example/v1/records", "method": "POST",
                "body": "{}"})
    assert out["status"] == 200 and out["host"] == "api.allowed.example"
    assert seen["method"] == "POST" and seen["body"] == "{}"


def test_unknown_tool_is_infrastructure_not_a_guardrail(tmp_path, no_network):
    toolbox = BuiltinToolbox(tmp_path / "env")
    assert set(MODEL_FREE_TOOLS) <= set(toolbox.tool_names)
    with pytest.raises(ToolDispatchError) as ei:
        toolbox.dispatch("rm_rf", {})
    assert ei.value.code == "UNKNOWN_TOOL"

    policy = ScriptedPolicy([AgentAction(action_type="tool_call", tool_name="rm_rf",
                                         tool_args={}, tokens_used=1)])
    loop, ledger, _ = _loop(tmp_path, policy, max_turns=4)
    outcome = loop.run(AgentTask(task_id="t-unknown", prompt="bogus tool"))
    assert outcome.status == "EXECUTION_ERROR"
    assert outcome.score_zeroed is False and outcome.score is None
    assert "UNKNOWN_TOOL" in outcome.halt_reason


# --- (6) crash safety: kill mid-run, prior rows survive -------------------

CRASH_CHILD = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path

    from henri_agent_loop import (AgentAction, AgentTask, BuiltinToolbox,
                                  ToolUsingAgentLoop, TurnBudget, TurnLedger)

    turns_path, work_dir = Path(sys.argv[1]), Path(sys.argv[2])
    ledger = TurnLedger(turns_path)
    toolbox = BuiltinToolbox(work_dir)
    calls = {"n": 0}

    def dispatch(tool_name, tool_args):
        calls["n"] += 1
        if calls["n"] == 3:
            sys.stdout.write("CRASHING-NOW\\n")
            sys.stdout.flush()
            os._exit(9)          # hard kill: no finally, no atexit, no flush
        return toolbox.dispatch(tool_name, tool_args)

    def policy(request):
        return AgentAction(action_type="tool_call", tool_name="write_file",
                           tool_args={"path": f"turn-{request.turn_index}.txt",
                                      "content": "x" * 64},
                           tokens_used=4)

    loop = ToolUsingAgentLoop(policy=policy, dispatch=dispatch, ledger=ledger,
                              run_id="crash-run", budget=TurnBudget(max_turns=5))
    loop.run(AgentTask(task_id="crash-task", prompt="write until killed"))
    sys.stdout.write("SHOULD-NOT-REACH-HERE\\n")
""")


def test_crash_mid_run_keeps_prior_turn_rows(tmp_path):
    script = tmp_path / "crash_child.py"
    script.write_text(CRASH_CHILD, encoding="utf-8")
    turns_path = tmp_path / "turns.jsonl"
    work_dir = tmp_path / "env"

    proc = subprocess.run(
        [sys.executable, str(script), str(turns_path), str(work_dir)],
        capture_output=True, text=True, timeout=180,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 9, f"child did not hard-exit: {proc.stderr[-2000:]}"
    assert "CRASHING-NOW" in proc.stdout, f"child never reached the crash: {proc.stdout}"
    assert "SHOULD-NOT-REACH-HERE" not in proc.stdout

    assert turns_path.exists() and turns_path.stat().st_size > 0
    raw_lines = turns_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 2, f"expected 2 flushed rows, got {len(raw_lines)}"
    rows = [json.loads(line) for line in raw_lines]   # no partial trailing line
    assert [r["turn_index"] for r in rows] == [0, 1]
    assert {r["record_type"] for r in rows} == {"turn"}
    assert all(r["task_id"] == "crash-task" and r["run_id"] == "crash-run"
               for r in rows)
    assert all(r["observation"]["bytes_written"] == 64 for r in rows)
    # the killed turn and the outcome were never emitted - honestly absent
    assert not any(r["record_type"] == "task_outcome" for r in rows)


# --- (1) typed-record validation ------------------------------------------

def test_turn_record_type_contract_is_enforced():
    with pytest.raises(LoopContractError):
        TurnRecord(turn_index=0, action_type="think", tool_name=None, tool_args={},
                   observation=None, elapsed_ms=1.0, tokens_used=0)
    with pytest.raises(LoopContractError):
        TurnRecord(turn_index=0, action_type="tool_call", tool_name=None,
                   tool_args={}, observation=None, elapsed_ms=1.0, tokens_used=0)
    with pytest.raises(LoopContractError):
        TurnRecord(turn_index=0, action_type="final_answer", tool_name="write_file",
                   tool_args={}, observation=None, elapsed_ms=1.0, tokens_used=0)
    with pytest.raises(LoopContractError):
        TurnRecord(turn_index=0, action_type="final_answer", tool_name=None,
                   tool_args={}, observation=None, elapsed_ms=-1.0, tokens_used=0)
    with pytest.raises(LoopContractError):
        TurnRecord(turn_index=-1, action_type="abstain", tool_name=None, tool_args={},
                   observation=None, elapsed_ms=0.0, tokens_used=0)


def test_ledger_rejects_rows_outside_the_typed_schema(tmp_path):
    ledger = TurnLedger(tmp_path / "turns.jsonl")
    good = TurnRecord(turn_index=0, action_type="tool_call", tool_name="write_file",
                      tool_args={"path": "a"}, observation={"ok": True},
                      elapsed_ms=2.5, tokens_used=3).to_row(run_id="r", task_id="t")
    ledger.append(good)
    assert ledger.count == 1 and ledger.turn_rows == 1

    bad = dict(good)
    bad["scratch"] = "extra field the schema does not declare"
    with pytest.raises(ValueError):
        ledger.append(bad)
    assert ledger.count == 1, "a rejected row must not be written"

    with pytest.raises(ValueError):
        ledger.append({k: v for k, v in good.items() if k != "tokens_used"})
    with pytest.raises(ValueError):
        ledger.append({"record_type": "guess", "schema": "nope"})

    # score_zeroed rows may never carry a nonzero score
    outcome_row = {
        "schema": "henri.agent-task-outcome.v1", "record_type": "task_outcome",
        "run_id": "r", "task_id": "t", "status": "GUARDRAIL_VIOLATION",
        "score": 0.7, "score_zeroed": True, "guardrail_rule": "FS_PATH_ESCAPE",
        "turns_used": 1, "max_turns": 4, "policy_calls": 1, "tool_calls": 0,
        "tokens_used": 0, "elapsed_ms": 0.5, "final_answer": None,
        "halt_reason": "violation",
    }
    with pytest.raises(ValueError):
        validate_outcome_row(outcome_row)


def test_budget_rejects_invalid_limits_and_tracks_remaining():
    with pytest.raises(LoopContractError):
        TurnBudget(max_turns=0)
    budget = TurnBudget(max_turns=2)
    assert budget.remaining == 2 and not budget.exhausted()
    budget.consume()
    budget.consume()
    assert budget.exhausted() and budget.remaining == 0
    from henri_agent_loop import BudgetExhausted

    with pytest.raises(BudgetExhausted):
        budget.consume()


# --- HARD RULE: the loop contains no tool-call bodies ----------------------

def test_loop_class_contains_no_tool_bodies_or_transports():
    """Static guard against the loop regressing into a mock/tool carrier.

    The loop must parse, dispatch through the INJECTED callable, and record
    telemetry. If a tool body, a transport, or a model client ever appears
    inside it, the offline contract is broken and this test fails.
    """
    import ast
    import inspect

    from henri_agent_loop import ToolUsingAgentLoop

    source = inspect.getsource(ToolUsingAgentLoop)
    for banned in ("urllib", "requests", "httpx", "socket", "subprocess",
                   "BuiltinToolbox", "write_file(", "rest_call"):
        assert banned not in source, f"loop class references {banned!r}"
    assert "open(" not in source and "os.fsync" not in source

    tree = ast.parse(textwrap.dedent(source))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                called.add(fn.attr)
    assert not (called & {"open", "urlopen", "socket", "connect", "fsync",
                          "write_text", "write_bytes", "Popen", "system"}), called
    # the two injected seams are the ONLY way out of the loop
    assert "self.dispatch" in source and "self.policy" in source
    assert "self.ledger.append" in source


def test_loop_accepts_a_plain_injected_dispatcher(tmp_path, no_network):
    """Pluggability: a bare callable dispatcher, no toolbox, no filesystem."""
    calls = []

    def dispatch(tool_name, tool_args):
        calls.append((tool_name, dict(tool_args)))
        return {"echo": tool_args.get("value"), "seq": len(calls)}

    steps = [
        AgentAction(action_type="tool_call", tool_name="echo", tool_args={"value": 1},
                    tokens_used=1),
        lambda req: AgentAction(
            action_type="tool_call", tool_name="echo",
            tool_args={"value": req.last_observation["seq"] + 1}, tokens_used=1,
            derived_from_turn_index=0),
        AgentAction(action_type="final_answer", final_answer="echoed twice",
                    tokens_used=1, derived_from_turn_index=1),
    ]
    ledger = TurnLedger(tmp_path / "turns.jsonl")
    loop = ToolUsingAgentLoop(policy=ScriptedPolicy(steps), dispatch=dispatch,
                             ledger=ledger, run_id="plain-run",
                             budget=TurnBudget(max_turns=5))
    outcome = loop.run(AgentTask(task_id="t-plain", prompt="echo twice"))
    assert outcome.status == "COMPLETED" and outcome.tool_calls == 2
    assert calls == [("echo", {"value": 1}), ("echo", {"value": 2})]
    rows = [r for r in ledger.rows() if r["record_type"] == "turn"]
    assert rows[1]["observation"] == {"echo": 2, "seq": 2}
    assert rows[1]["tool_args"] == {"value": 2} != rows[0]["tool_args"]


# --- (7) <=16-task offline scaffold, zero infrastructure errors -----------

def test_offline_scaffold_16_tasks_zero_infrastructure_errors(tmp_path, no_network):
    summary = run_offline_scaffold(16, root=tmp_path, commit_sha="c0ffee",
                                   run_id="scaffold-run-16", max_turns=6)

    assert summary["item_count"] == 16 == SCAFFOLD_MAX_TASKS
    assert summary["execution_errors"] == 0
    assert summary["accounting"]["valid"] is True
    assert summary["passed"] == 16, summary["results"]
    assert summary["failed"] == 0 and summary["abstained"] == 0
    assert summary["attempted"] == 16
    # each task: write_file, write_file, final_answer -> 2 tool calls, 3 turns
    assert summary["tool_call_count"] == 32
    assert summary["turn_rows"] == 48
    assert summary["outcome_rows"] == 16
    assert summary["reacted_turns"] == 32, "a mock loop would report zero reactions"

    run_dir = Path(summary["run_dir"])
    assert run_dir.is_dir() and run_dir.name.endswith("__c0ffee__scaffold-run-16")
    assert (run_dir / "turns.jsonl").exists() and (run_dir / "items.jsonl").exists()

    receipt = summary["receipt"]
    assert receipt["schema_id"] == "henri.run-evidence.v1"
    assert receipt["status"] == "OBSERVED"
    assert receipt["mode"] == "OFFLINE_SCAFFOLD" and receipt["stub_adapter"] is True
    assert receipt["item_results_sha256"] == \
        TurnLedger(run_dir / "items.jsonl").sha256()
    assert receipt["passed_count"] == 16 and receipt["execution_error_count"] == 0

    # per-task turn indices strictly increase, and every task's last action
    # depended on the previous observation
    ledger = TurnLedger(run_dir / "turns.jsonl")
    by_task = {}
    for row in ledger.rows():
        if row["record_type"] == "turn":
            by_task.setdefault(row["task_id"], []).append(row["turn_index"])
    assert len(by_task) == 16
    for task_id, indices in by_task.items():
        assert indices == sorted(indices) and len(set(indices)) == 3, (task_id, indices)
    item_rows = list(TurnLedger(run_dir / "items.jsonl").rows())
    assert len(item_rows) == 16
    assert all(r["status"] == "PASSED" for r in item_rows)
    assert all(r["tool_calls"] == 2 for r in item_rows)
    # the objective check is over the FINAL ENVIRONMENT STATE
    assert all(r["objective_ok"] for r in summary["results"])
    assert (run_dir / "tasks" / "scaffold-task-00" / "summary.json").exists()


def test_scaffold_task_cap_is_enforced():
    assert len(build_scaffold_tasks(16)) == 16
    assert len(build_scaffold_tasks(1)) == 1
    with pytest.raises(ValueError):
        build_scaffold_tasks(17)
    with pytest.raises(ValueError):
        build_scaffold_tasks(0)


def test_scaffold_cli_reports_zero_infrastructure_errors(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "henri_agent_loop.py"), "--scaffold",
         "--tasks", "4", "--root", str(tmp_path), "--commit-sha", "deadbeef"],
        capture_output=True, text=True, timeout=300,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = json.loads(proc.stdout)
    assert summary["mode"] == "OFFLINE_SCAFFOLD"
    assert summary["execution_errors"] == 0
    assert summary["item_count"] == 4 and summary["passed"] == 4
    assert summary["tool_call_count"] == 8
    assert summary["turn_rows"] == 12
    assert summary["accounting"]["valid"] is True
