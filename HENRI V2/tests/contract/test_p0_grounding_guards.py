from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_arc_score_path_has_no_identity_or_private_target_access():
    source = _read("production_arc_run.py")
    assert "target_grid=init_grid" not in source
    assert "_game._levels" not in source
    assert "OBSERVED_TEST_TARGET_UNAVAILABLE" in source


def test_arc_score_path_fails_closed_on_live_zone_c_errors():
    source = _read("production_arc_run.py")
    assert "refusing JSONL-only production evidence" in source
    assert "Falling back to in-memory SegmentCache surrogate" not in source


def test_governance_score_path_has_no_unsealed_success_state():
    source = _read("henri_agentic_graph_engine.py")
    assert '"UNSEALED"' not in source
    assert "audit sealing failed; no experiment scorecard may be persisted" in source


def test_invalid_live_inference_runner_is_quarantined():
    assert not (ROOT / "run_live_inference_eval.py").exists()
    assert (ROOT / "_archive/invalid_evaluators/run_live_inference_eval.py").exists()
