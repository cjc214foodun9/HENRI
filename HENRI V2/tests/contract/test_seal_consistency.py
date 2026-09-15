"""Contract test: the Phase 10.1 seal must be SELF-CONSISTENT.

A seal is not one artifact, it is a SET of artifacts that must agree: a
human-readable doc and the machine receipt it describes. Commit 99fb88a shipped a
doc that contradicted its own committed receipt (doc cited koopman_named6 /
koopman_no_refl / "480 solves" / 5.75 / 6.0 s; the receipt has koopman_diagonly
arms / n_solves 360 / mean_rank_G 5.667). The doc READS correctly, so the mismatch
survived careful visual review. This test makes the agreement mechanical.

The second test is the important one: it proves the gate can FAIL. A consistency
check that passes a doc containing no arms at all would be vacuous.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]                      # ...\HENRI V2
sys.path.insert(0, str(REPO / "experiments" / "verification"))


def test_phase10_1_seal_is_consistent():
    """The live doc and the live receipt must agree on arms and figures."""
    import validate_seal_consistency as vsc

    rep = vsc.check(vsc.DOC_DEFAULT, vsc.RECEIPT_DEFAULT)
    assert rep["CONSISTENT"], f"seal inconsistency: {rep['problems']}"
    # arm table must be non-empty, else the comparison is trivially satisfied
    assert rep["doc_arms"], "no measured arms found in doc -> gate would be vacuous"
    assert rep["receipt_arms"], "no arms in receipt"


def test_gate_rejects_a_doc_with_no_arms(tmp_path):
    """NEGATIVE CONTROL: the gate must FAIL on a doc that cites nothing."""
    import validate_seal_consistency as vsc

    bad = tmp_path / "doc.md"
    bad.write_text("This document intentionally cites no arms or figures.\n", encoding="utf-8")
    rep = vsc.check(bad, vsc.RECEIPT_DEFAULT)
    assert not rep["CONSISTENT"], "gate passed a doc with no arms -> vacuous check"
    assert rep["problems"], "inconsistent report carried no problems"


def test_gate_rejects_a_stale_arm_name(tmp_path):
    """NEGATIVE CONTROL: a doc citing a deleted arm must FAIL."""
    import validate_seal_consistency as vsc

    doc = vsc.DOC_DEFAULT.read_text(encoding="utf-8")
    stale = doc + "\n\n| `koopman_named6@0.99` | +0.0000 | +0.0000 | 0.0000 | 0.0 % |\n"
    p = tmp_path / "doc.md"
    p.write_text(stale, encoding="utf-8")
    rep = vsc.check(p, vsc.RECEIPT_DEFAULT)
    assert not rep["CONSISTENT"], "gate accepted an arm absent from the receipt"
    assert any("koopman_named6" in q for q in rep["problems"]), rep["problems"]
