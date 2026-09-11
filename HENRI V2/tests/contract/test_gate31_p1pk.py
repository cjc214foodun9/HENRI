"""Contract tests — Gate 3.1 P@1/P@k diagnostic (carrier/p1-pk-diagnostics).

Prereg: experiments/verification/gate31_p1pk_prereg.md (sealed).
Static contracts for verify_egress_closed_loop.py:
  - teacher-anchored single-shot definitions (P@1, P@5, perplexity as NLL);
  - frozen artifact pins (E2 ckpt 08747c70..., teacher rev 060db6499f32...,
    corpus e83889ba..., split seed 20260908);
  - pre-registered bounds as module constants, NOT env-overridable;
  - fail-closed artifact codes; single-run/no-retry; required metrics emitted.
"""

import sys
from pathlib import Path

HENRI2 = Path(__file__).resolve().parents[2]  # <wt>/HENRI V2
sys.path.insert(0, str(HENRI2))

RUNNER = HENRI2 / "verify_egress_closed_loop.py"


def test_runner_exists():
    assert RUNNER.exists(), "verify_egress_closed_loop.py must exist in HENRI V2/"


def test_runner_defines_bounds_as_constants():
    src = RUNNER.read_text(encoding="utf-8")
    # RETIRED 2026-09-11: the legacy 0.285/0.640 bound sits below the
    # measured trivial baseline and must never be reinstated.
    assert "P_AT_1_BOUND = 0.285" not in src
    assert "P_AT_5_BOUND = 0.640" not in src
    assert "load_registered_bounds" in src
    assert "LEGACY_BOUNDS_RETIRED" in src


def test_runner_pins_frozen_artifacts():
    src = RUNNER.read_text(encoding="utf-8")
    assert "08747c70" in src       # E2 checkpoint sha prefix
    assert "060db6499f32" in src   # teacher rev
    assert "e83889ba" in src       # corpus sha prefix
    assert "20260908" in src       # split seed


def test_runner_emits_required_metrics_and_label():
    src = RUNNER.read_text(encoding="utf-8")
    for key in ("p_at_1", "p_at_5", "perplexity", "random_p_at_1",
                "GATE31_VERDICT", "CONDITIONAL_SAME_CORPUS_HELDOUT",
                "teacher-anchored"):
        assert key in src, key


def test_runner_fail_closed_codes():
    src = RUNNER.read_text(encoding="utf-8")
    for code in ("E2_CKPT_MISSING", "TEACHER_SHARD_MISSING", "CORPUS_MISSING",
                 "E2_CKPT_SHA_MISMATCH", "CORPUS_SHA_MISMATCH",
                 "TEACHER_EMBED_KEY_MISSING", "EVAL_SPLIT_MISMATCH",
                 "GOLD_TOKEN_PREFIX_MISMATCH"):
        assert code in src, code


def test_runner_bounds_not_env_overridable():
    src = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ.get("P_AT_1_BOUND"' not in src
    assert 'os.environ.get("P_AT_5_BOUND"' not in src


def test_runner_no_retry_path():
    src = RUNNER.read_text(encoding="utf-8")
    low = src.lower().replace("no retries", "").replace("no retry", "")
    assert "retry" not in low.replace("retrying", ""), "no retry logic allowed"


def test_runner_uses_real_e2_split_builder():
    src = RUNNER.read_text(encoding="utf-8")
    assert "build_pairs_ordered" in src  # real e2_calibrate split builder


def test_runner_writes_structured_receipt():
    src = RUNNER.read_text(encoding="utf-8")
    assert "gate31_receipt.json" in src
    assert "json.dumps" in src
