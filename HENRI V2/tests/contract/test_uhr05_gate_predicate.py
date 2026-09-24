"""UHR-05 contract tests — the M1 gate's predicate withdrawal and validity controls.

CONTEXT (all measured by own execution this session)
====================================================
The M1 open-answer gate is the instrument that asserted the 40%-weight egress
blocker. Two defects were found by RUNNING it:

  DEFECT A — relocated-relative-import. `sys.path.insert(0, parents[1])` was correct
    for the script's ORIGINAL location one level below the package root; after
    relocation into `experiments/verification/` it resolved to `experiments/`, so the
    gate died with `ModuleNotFoundError: No module named 'henri_vla_tokenizer'`,
    exit 1, and wrote NO receipt. The gate could not run at all.

  DEFECT B — unsatisfiable predicate.
        p2      = distinct_ratio >= 0.50
        vacuous = rand_distinct_ratio >= 0.50     # the TREATMENT's floor applied
                                                  # to the RANDOM-WAVE control
        p5      = not vacuous
        M1_GATE_PASS requires p1..p5
    Measured random-control distinct_ratio: 0.5917 / 0.7000 / 0.5917 / 0.7417 across
    every arm and both egress settings — ABOVE the 0.50 floor every time. Therefore
    `vacuous` was True for EVERY possible treatment, and M1_GATE_PASS was unreachable.
    A falsifier that cannot return its non-vacuous branch is an absolute barrier
    mislabelled as a test.

  ROOT CAUSE (why the axis is not repairable by moving the floor): directional SPREAD
    is not content. 120 random waves are maximally spread in direction space and hit
    many codebook entries; a content-bearing encoder is CONSTRAINED (similar inputs to
    similar outputs) and so has FEWER distinct top-1s. Measured: random 0.59-0.74 vs
    treatment 0.12-0.48. The random control WINS the axis, so `distinct_ratio` is
    confounded and is RETIRED as an operative criterion, retained as a reported
    diagnostic — the same post-hoc-withdrawal pattern already used in this project for
    the UHR-03 G6 `role_coherence` precondition (commits 510d796 / 3bad85e).

  THE REPLACEMENT IS STRICTER, NOT LOOSER:
    * No floor moved. P1=1.0, P3=0.50, P4=0.50 exactly as pre-registered.
    * P2 withdrawn from the gate, still reported.
    * NEW P5 = gate VALIDITY: both shipped degenerate encoders (`dead`, `hash`) must
      FAIL the (P3, P4) pair. A structureless encoder passing would make the pair a
      rubber stamp, and `test_degenerate_encoders_fail_the_pair` fails loudly if so.
      This is the dead-input negative control every gate in this project is required
      to ship.
"""
from __future__ import annotations

import importlib.util
import pathlib
import random
import sys

import pytest
import torch

SB = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SB))

GATE = SB / "experiments/verification/m1_open_answer_gate.py"
_spec = importlib.util.spec_from_file_location("m1gate", GATE)
m1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m1)


def _build(mode: str = "fractional_shift"):
    cfg = m1.build_config(len(m1.VOCAB), mode)
    tok = m1.vt.HoloVLATokenizer(cfg)
    code = m1.vt.HoloEgressCodebook(cfg, tok, list(m1.VOCAB))
    prompts = m1.build_prompts(m1.N_PROMPTS, random.Random(m1.SEED))
    return cfg, code, prompts


def _degenerate_scores(code, prompts, kind):
    cfg_dim = code.cfg.ambient_dim_D
    rng = random.Random(m1.SEED)
    with torch.no_grad():
        a = code.logits(m1.degenerate_wave(prompts, cfg_dim, kind)).argmax(-1).tolist()
        b = code.logits(m1.degenerate_wave([m1.shuffled(p, rng) for p in prompts],
                                           cfg_dim, kind)).argmax(-1).tolist()
        c = code.logits(m1.degenerate_wave([m1.near_view(p) for p in prompts],
                                           cfg_dim, kind)).argmax(-1).tolist()
    order = sum(x != y for x, y in zip(a, b)) / len(a)
    equiv = sum(x == y for x, y in zip(a, c)) / len(a)
    return order, equiv


# --------------------------------------------------------- instrument integrity
def test_no_floor_was_moved():
    """The withdrawal must not have relaxed a single threshold."""
    assert m1.P1_DETERMINISM == 1.0
    assert m1.P3_ORDER_FLOOR == 0.50
    assert m1.P4_EQUIV_FLOOR == 0.50


def test_old_unsatisfiable_predicate_is_gone():
    """The predicate that could never pass must not be reinstated."""
    src = GATE.read_text(encoding="utf-8")
    assert 'vacuous = a["rand_distinct_ratio"]' not in src, (
        "the unsatisfiable `vacuous = rand >= TREATMENT_FLOOR` predicate is back")


def test_degenerate_encoders_fail_the_pair():
    """P5 VALIDITY, the load-bearing control.

    Both structureless encoders must FAIL (order_sensitivity, equivalence). If either
    passes, the pair does not measure content and the replacement would be a rubber
    stamp — so this test failing is the signal to re-derive the replacement.
    """
    cfg, code, prompts = _build()
    for kind in ("dead", "hash"):
        order, equiv = _degenerate_scores(code, prompts, kind)
        passed = (order >= m1.P3_ORDER_FLOOR) and (equiv >= m1.P4_EQUIV_FLOOR)
        assert not passed, (
            f"degenerate encoder {kind!r} PASSED the (P3,P4) pair "
            f"(order={order:.4f}, equiv={equiv:.4f}) -> the pair is not discriminating")


def test_gate_validity_helper_agrees_with_manual_computation():
    cfg, code, prompts = _build()
    valid, detail = m1.gate_validity(code, prompts, random.Random(m1.SEED))
    manual = all(not ((o >= m1.P3_ORDER_FLOOR) and (e >= m1.P4_EQUIV_FLOOR))
                 for o, e in (_degenerate_scores(code, prompts, k) for k in ("dead", "hash")))
    assert valid == manual, "gate_validity disagrees with the manual control computation"
    assert set(detail) == {"dead", "hash"}


# --------------------------------------------------------- verdict behaviour
def test_dead_encoder_passes_equivalence_but_fails_order():
    """Documents WHY the pair is needed: a constant wave satisfies equivalence alone."""
    cfg, code, prompts = _build()
    order, equiv = _degenerate_scores(code, prompts, "dead")
    assert equiv >= m1.P4_EQUIV_FLOOR, "constant wave should trivially satisfy equivalence"
    assert order < m1.P3_ORDER_FLOOR, "constant wave must fail order sensitivity"


def test_retired_statistic_no_longer_gates():
    """A dict with a LOW distinct_ratio but good order/equivalence must PASS.

    That is the point of the withdrawal: `distinct_ratio` was confounded, so a low
    value is not evidence against the egress. This test pins that behaviour so a
    future session cannot silently re-gate on the retired statistic.
    """
    a = {"determinism": 1.0, "distinct_ratio": 0.10, "order_sensitivity": 0.60,
         "equivalence": 0.70, "rand_distinct_ratio": 0.70}
    v = m1.verdict(a, control_valid=True)
    assert v["verdict"] == "M1_GATE_PASS", v
    assert v["P2_distinct_RETIRED"] is False, "the retired statistic is still reported"
    assert v["P5_control_valid"] is True


def test_invalid_control_blocks_the_pass():
    """If the validity control fails, no treatment may pass, however good it looks."""
    a = {"determinism": 1.0, "distinct_ratio": 0.90, "order_sensitivity": 1.0,
         "equivalence": 1.0, "rand_distinct_ratio": 0.10}
    v = m1.verdict(a, control_valid=False)
    assert v["verdict"] == "GATE_INVALID_DEGENERATE_ENCODER_PASSED"


def test_order_blind_and_content_blind_encoders_fail():
    """P3 and P4 each independently block a degenerate treatment."""
    order_blind = {"determinism": 1.0, "distinct_ratio": 0.9, "order_sensitivity": 0.0,
                   "equivalence": 1.0, "rand_distinct_ratio": 0.1}
    assert m1.verdict(order_blind)["verdict"] == "M1_GATE_FAIL:P3"
    hash_like = {"determinism": 1.0, "distinct_ratio": 0.9, "order_sensitivity": 1.0,
                 "equivalence": 0.0, "rand_distinct_ratio": 0.1}
    assert m1.verdict(hash_like)["verdict"] == "M1_GATE_FAIL:P4"


def test_gate_now_runs_and_emits_a_receipt():
    """DEFECT A regression guard: the gate must execute and write a receipt."""
    import json
    import os
    import subprocess
    import tempfile

    # UHR-05: the gate's default receipt path is now the COMMITTED location, so this
    # test redirects it with HENRI_RECEIPT_DIR (the documented override) instead of
    # asserting the old %TEMP% path -- that old assertion encoded the defect where the
    # committed receipt had no reproduction path.
    td = pathlib.Path(tempfile.mkdtemp(prefix="uhr05_gate_"))
    env = dict(os.environ, HENRI_RECEIPT_DIR=str(td))
    receipt = td / "m1_gate_receipt.json"
    r = subprocess.run([sys.executable, str(GATE)], cwd=str(SB), env=env,
                       capture_output=True, text=True, errors="replace", timeout=900)
    assert "ModuleNotFoundError" not in (r.stderr or ""), (
        "relocated-relative-import defect returned: " + (r.stderr or "")[-300:])
    assert receipt.exists(), "gate did not write a receipt"
    rec = json.loads(receipt.read_text(encoding="utf-8"))
    assert rec["preregistration"]["N"] >= 100
