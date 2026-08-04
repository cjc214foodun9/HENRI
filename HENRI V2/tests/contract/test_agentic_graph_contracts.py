import hashlib
import json
from pathlib import Path

import pytest

from agentic_graph.budgets import BudgetExceeded, GraphBudget, RetryController
from agentic_graph.context_packer import ContextPackError, ContextPacker
from agentic_graph.evidence_receipts import file_hash_receipt, receipt_from_result, verify_artifact_hash
from agentic_graph.runtime import GraphRuntime, GraphRuntimeError
from agentic_graph.verifier import promotion_status

ROOT = Path(__file__).parents[2]
SCHEMAS = ROOT / "agentic_graph" / "schemas"

def test_schemas_are_valid_json_and_versioned():
    names = ["henri_task_packet.schema.json", "henri_holon_result.schema.json", "henri_evidence_receipt.schema.json", "henri_state_delta.schema.json"]
    for name in names:
        data=json.loads((SCHEMAS/name).read_text(encoding="utf-8"))
        assert data["$schema"].endswith("2020-12/schema")
        assert data["$id"].endswith(".v1")

def packer_args():
    return dict(policy={"version":"1"}, contracts={"schema":"1"}, skills=["henri-architecture"], tools=["file","terminal"])

def test_dynamic_task_does_not_change_static_prefix():
    p=ContextPacker()
    a=p.pack(**packer_args(), task={"task_id":"a","question":"one"})
    b=p.pack(**packer_args(), task={"task_id":"b","question":"two"}, state_delta={"run_id":"different"})
    assert a.static_prefix == b.static_prefix
    assert a.static_prefix_sha256 == b.static_prefix_sha256
    assert a.dynamic_tail != b.dynamic_tail

def test_static_prefix_is_key_order_stable_and_changes_on_policy_change():
    p=ContextPacker()
    a=p.pack(policy={"a":1,"b":2}, contracts={"x":1}, skills=[], tools=[], task={})
    b=p.pack(policy={"b":2,"a":1}, contracts={"x":1}, skills=[], tools=[], task={})
    c=p.pack(policy={"a":1,"b":3}, contracts={"x":1}, skills=[], tools=[], task={})
    assert a.static_prefix_sha256 == b.static_prefix_sha256
    assert a.static_prefix_sha256 != c.static_prefix_sha256

def test_oversized_raw_artifact_is_externalized():
    p=ContextPacker(raw_artifact_chars=10)
    result=p.pack(**packer_args(), task={}, excerpts=[{"ref":"logs/run.txt","text":"x"*11}])
    assert result.omitted_artifact_refs == ["logs/run.txt"]
    assert "raw artifact" in result.rejection_reasons[0]
    assert "logs/run.txt" not in result.dynamic_tail

def test_dynamic_budget_fails_closed():
    p=ContextPacker(max_dynamic_chars=100)
    with pytest.raises(ContextPackError):
        p.pack(**packer_args(), task={"question":"x"*500})

def test_model_call_and_depth_budgets_stop_excess():
    b=GraphBudget(max_model_calls=1,max_depth=2)
    b.consume_model_call()
    with pytest.raises(BudgetExceeded): b.consume_model_call()
    b.enter_depth(2)
    with pytest.raises(BudgetExceeded): b.enter_depth(3)

def test_leaf_retry_is_once_and_only_failed_leaf():
    r=RetryController(max_leaf_retries=1)
    assert r.allow("leaf-a",{"x":1},"change evidence")
    assert r.retry_targets("leaf-a") == ["leaf-a"]
    assert not r.allow("leaf-a",{"x":1},"change evidence")
    assert not r.allow("leaf-a",{"x":1},"")

def test_missing_receipt_blocks_verified_promotion():
    assert promotion_status("verified",[]) == "unverified"

def test_failed_receipt_blocks_verified_promotion():
    r=receipt_from_result(receipt_id="r",kind="test",status="fail",subject_ref="x",summary="failed",artifact_ref="x",tool="pytest",command_or_method="pytest x")
    assert promotion_status("verified",[r]) == "unverified"

def test_passing_receipt_allows_verified_promotion():
    r=receipt_from_result(receipt_id="r",kind="test",status="pass",subject_ref="x",summary="passed",artifact_ref="x",tool="pytest",command_or_method="pytest x")
    assert promotion_status("verified",[r]) == "verified"

def test_runtime_rejects_invalid_transition_and_requires_receipt():
    g=GraphRuntime()
    with pytest.raises(GraphRuntimeError): g.transition("complete")
    g.transition("classified")
    g.transition("collected")
    g.transition("dispatched")
    with pytest.raises(GraphRuntimeError): g.promote("verified",[])

def test_runtime_retry_does_not_restart_parent():
    g=GraphRuntime()
    g.transition("classified"); g.transition("collected"); g.transition("dispatched")
    result=g.retry_leaf("leaf-a",{"x":1},"repair missing receipt")
    assert result["failed_node"] == "leaf-a"
    assert result["retry_count"] == 1
    assert result["to_state"] == "repair"
    assert g.state == "repair"

def test_file_hash_receipt_is_observable(tmp_path):
    p=tmp_path/"artifact.txt"; p.write_text("abc",encoding="utf-8")
    r=file_hash_receipt(str(p),"hash-1")
    assert r.status == "pass"
    assert r.subject_sha256 == "sha256:" + hashlib.sha256(b"abc").hexdigest()

def test_legacy_simulated_builder_is_blocked():
    import importlib.util
    legacy = ROOT / "_archive" / "legacy_graph" / "henri_langgraph_executable_agent_engine.py"
    spec = importlib.util.spec_from_file_location("legacy_henri_graph", legacy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="not a production route"):
        module.build_henri_dev_graph()

def test_terminal_state_cannot_reactivate():
    g=GraphRuntime()
    g.transition("blocked")
    with pytest.raises(GraphRuntimeError): g.transition("dispatched")

def test_unavailable_receipt_cannot_promote():
    r=receipt_from_result(receipt_id="r",kind="schema",status="unavailable",subject_ref="x",summary="validator unavailable",artifact_ref="x",tool="jsonschema",command_or_method="validate x")
    assert promotion_status("verified",[r]) == "unverified"

def test_receipt_subject_hash_is_validated():
    with pytest.raises(ValueError):
        receipt_from_result(receipt_id="r",kind="hash",status="pass",subject_ref="x",subject_sha256="bad",summary="bad",artifact_ref="x",tool="sha256",command_or_method="hash x")

def test_duplicate_retry_fingerprint_is_rejected_even_with_budget():
    r=RetryController(max_leaf_retries=2)
    assert r.allow("leaf-a",{"x":1},"repair")
    assert not r.allow("leaf-a",{"x":1},"repair")
    assert r.allow("leaf-a",{"x":2},"repair")


def test_task_packet_schema_rejects_missing_required_fields():
    from jsonschema import Draft202012Validator
    schema=json.loads((SCHEMAS/"henri_task_packet.schema.json").read_text(encoding="utf-8"))
    errors=list(Draft202012Validator(schema).iter_errors({"schema_version":"1.0"}))
    assert errors


def test_result_schema_requires_claim_and_evidence_status():
    from jsonschema import Draft202012Validator
    schema=json.loads((SCHEMAS/"henri_holon_result.schema.json").read_text(encoding="utf-8"))
    errors=list(Draft202012Validator(schema).iter_errors({"schema_version":"1.0","task_id":"t","status":"complete"}))
    assert errors


def test_context_packer_rejects_out_of_scope_and_raw_ast():
    p=ContextPacker(raw_artifact_chars=100)
    result=p.pack(**packer_args(), task={"allowed_paths":["src"]}, excerpts=[
        {"ref":"outside/log.txt","kind":"raw_log","text":"x"},
        {"ref":"src/tree.ast","kind":"raw_ast","text":"x"},
    ])
    assert result.omitted_artifact_refs == ["outside/log.txt", "src/tree.ast"]
    assert any("outside allowed" in r for r in result.rejection_reasons)
    assert any("raw raw_ast" in r for r in result.rejection_reasons)


def test_context_packer_accepts_bounded_excerpt_inside_scope():
    p=ContextPacker(raw_artifact_chars=10)
    result=p.pack(**packer_args(), task={"scope":{"allowed_paths":["src"]}}, excerpts=[
        {"ref":"src/module.py","kind":"excerpt","bounded":True,"text":"x"*20},
    ])
    assert result.omitted_artifact_refs == []
    assert "src/module.py" in result.dynamic_tail


def test_graph_budget_worker_fanout_has_separate_routine_and_risk_limits():
    routine=GraphBudget(max_workers=1)
    routine.admit_workers(1)
    with pytest.raises(BudgetExceeded): routine.admit_workers(1)
    high=GraphBudget(max_workers=1,max_hard_workers=3)
    high.admit_workers(3,high_risk=True)
    with pytest.raises(BudgetExceeded): high.admit_workers(1,high_risk=True)


def test_receipt_hash_mismatch_is_not_silently_marked_fail():
    r=receipt_from_result(receipt_id="r",kind="hash",status="pass",subject_ref="x",summary="hash",artifact_ref="x",tool="sha256",command_or_method="hash x",artifact_sha256="sha256:"+"0"*64)
    # A receipt constructor validates syntax only. Content matching requires a
    # verifier with both artifacts; this test records that boundary explicitly.
    assert r.status == "pass"
    assert r.artifact_sha256 != r.subject_sha256


def test_task_packet_schema_requires_source_hash_prefix_and_scope():
    from jsonschema import Draft202012Validator
    schema=json.loads((SCHEMAS/"henri_task_packet.schema.json").read_text(encoding="utf-8"))
    bad={"schema_version":"1.0","task_id":"t","task_type":"audit","question":"q","requested_outcome":"o","risk_class":"routine","input_refs":[],"selected_skills":[],"budget":{"max_model_calls":1,"max_leaf_retries":1,"max_dynamic_context_tokens":10,"max_output_tokens":10},"source_hashes":["bad"],"scope":{"allowed_paths":["src"],"allowed_tools":["file"]},"acceptance":["a"],"rejection":["r"]}
    assert list(Draft202012Validator(schema).iter_errors(bad))


def test_state_delta_schema_requires_terminal_accounting_fields():
    from jsonschema import Draft202012Validator
    schema=json.loads((SCHEMAS/"henri_state_delta.schema.json").read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors({"schema_version":"1.0","task_id":"t","from_state":"dispatched","to_state":"blocked","reason":"missing receipt","changed_refs":[],"budget_remaining":{}}))


def test_runtime_does_not_promote_unavailable_verification():
    g=GraphRuntime(); g.transition("classified"); g.transition("collected"); g.transition("dispatched")
    r=receipt_from_result(receipt_id="r",kind="schema",status="unavailable",subject_ref="x",summary="not run",artifact_ref="x",tool="jsonschema",command_or_method="validate x")
    with pytest.raises(GraphRuntimeError): g.promote("verified",[r])


def test_legacy_module_contains_no_production_success_marker():
    legacy=(ROOT/"_archive"/"legacy_graph"/"henri_langgraph_executable_agent_engine.py").read_text(encoding="utf-8")
    assert "EXPERIMENTAL ONLY" in legacy
    assert "build_experimental_simulated_henri_dev_graph" in legacy


def test_runtime_receipt_payload_matches_schema():
    from jsonschema import Draft202012Validator
    schema=json.loads((SCHEMAS/"henri_evidence_receipt.schema.json").read_text(encoding="utf-8"))
    receipt=receipt_from_result(receipt_id="r",kind="test",status="pass",subject_ref="x",summary="passed",artifact_ref="x",tool="pytest",command_or_method="pytest x",subject_sha256="sha256:"+"1"*64,artifact_sha256="sha256:"+"2"*64)
    Draft202012Validator(schema).validate(receipt.to_dict())


def test_hash_mismatch_receipt_is_fail():
    r=verify_artifact_hash(receipt_id="h",subject_ref="x",expected_sha256="sha256:"+"0"*64,artifact_ref="x",actual_bytes=b"abc")
    assert r.status == "fail"

def test_runtime_state_delta_contains_required_failure_accounting():
    g=GraphRuntime(); g.transition("classified"); g.transition("collected"); g.transition("dispatched")
    d=g.retry_leaf("leaf-a",{"x":1},"missing receipt")
    for key in ["status","completed_checks","unmet_checks","failed_node","retry_count","next_safe_action"]:
        assert key in d
