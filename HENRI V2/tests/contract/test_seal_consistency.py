"""Contract test: EVERY sealed pair must be SELF-CONSISTENT.

A seal is not one artifact, it is a SET of artifacts that must agree: a
human-readable doc and the machine receipt it describes. Commit 99fb88a shipped a
doc that contradicted its own committed receipt (doc cited koopman_named6 /
koopman_no_refl / "480 solves" / 5.75 / 6.0 s; the receipt has koopman_diagonly
arms / n_solves 360 / mean_rank_G 5.667). The doc READS correctly, so the mismatch
survived careful visual review. This test makes the agreement mechanical.

WHY IT IS PARAMETRISED OVER vsc.PAIRS
    Phase 10.3 added a second sealed pair. A test that hard-coded the Phase 10.1
    pair would keep passing while the new seal went unchecked -- the check would
    silently stop covering part of what it claims to cover. Parametrising over
    PAIRS means a newly added seal is covered the moment it is registered.

    The negative controls are the load-bearing tests: they prove the gate can
    FAIL. A consistency check that passes a doc containing no arms, or a doc
    citing a deleted arm, or a doc that hides an unequal-task-set comparison,
    would be vacuous.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]                      # ...\HENRI V2
sys.path.insert(0, str(REPO / "experiments" / "verification"))


def test_pairs_registry_is_not_empty_and_covers_phase_10_3():
    """A registry that silently shrinks would disable coverage without failing."""
    import validate_seal_consistency as vsc

    assert len(vsc.PAIRS) >= 2, f"expected >=2 sealed pairs, got {len(vsc.PAIRS)}"
    names = " ".join(str(d) + " " + str(r) for d, r in vsc.PAIRS)
    assert "phase10_1" in names, "Phase 10.1 seal missing from PAIRS"
    assert "phase10_3" in names, "Phase 10.3 seal missing from PAIRS"


def test_every_sealed_pair_is_consistent():
    """The live doc and the live receipt must agree, for EVERY registered pair."""
    import validate_seal_consistency as vsc

    failures = []
    for doc_p, receipt_p in vsc.PAIRS:
        rep = vsc.check(doc_p, receipt_p)
        if not rep["CONSISTENT"]:
            failures.append(f"{doc_p.name}: {rep['problems']}")
            continue
        # arm table must be non-empty, else the comparison is trivially satisfied
        if not rep["doc_arms"]:
            failures.append(f"{doc_p.name}: no measured arms found -> gate vacuous")
        if not rep["receipt_arms"]:
            failures.append(f"{doc_p.name}: no arms in receipt")
    assert not failures, "seal inconsistency: " + "; ".join(failures)


def test_gate_rejects_a_doc_with_no_arms(tmp_path):
    """NEGATIVE CONTROL: the gate must FAIL on a doc that cites nothing."""
    import validate_seal_consistency as vsc

    bad = tmp_path / "doc.md"
    bad.write_text("This document intentionally cites no arms or figures.\n",
                   encoding="utf-8")
    rep = vsc.check(bad, vsc.PAIRS[0][1])
    assert not rep["CONSISTENT"], "gate passed a doc with no arms -> vacuous check"
    assert rep["problems"], "inconsistent report carried no problems"


def test_gate_rejects_a_stale_arm_name(tmp_path):
    """NEGATIVE CONTROL: a doc citing a deleted arm must FAIL."""
    import validate_seal_consistency as vsc

    doc_p, rec_p = vsc.PAIRS[0]
    doc = doc_p.read_text(encoding="utf-8")
    stale = doc + "\n\n| `koopman_named6@0.99` | +0.0000 | +0.0000 | 0.0000 | 0.0 % |\n"
    p = tmp_path / "doc.md"
    p.write_text(stale, encoding="utf-8")
    rep = vsc.check(p, rec_p)
    assert not rep["CONSISTENT"], "gate accepted an arm absent from the receipt"
    assert any("koopman_named6" in q for q in rep["problems"]), rep["problems"]


def test_gate_rejects_a_doc_that_hides_unequal_task_sets(tmp_path):
    """NEGATIVE CONTROL for the Phase 10.3 rule: when a receipt scored arms on
    DIFFERENT task sets, the doc must disclose the common-subset comparison.

    The whole Phase 10.3 defect chain (D1, D1-B) came from comparing arms scored
    on different task sets. If a doc could omit that disclosure silently, the gate
    would not have caught the false ACCEPT that this rule exists to prevent.
    """
    import validate_seal_consistency as vsc

    doc_p, rec_p = vsc.PAIRS[1]
    rep_good = vsc.check(doc_p, rec_p)
    if not any("common" in n.lower() or "common" in str(rep_good).lower()
               for n in rep_good["notes"]) and "common subset" not in doc_p.read_text(
                   encoding="utf-8").lower():
        pytest.skip("receipt does not scored unequal task sets; rule not exercised")

    stripped = doc_p.read_text(encoding="utf-8")
    # Remove every common-subset disclosure, then bypass the other checks by
    # keeping all arm figures intact: ONLY rule 8 should fire.
    import re
    stripped = re.sub(r"(?i)common[ _-]?subset", "grouping", stripped)
    p = tmp_path / "doc.md"
    p.write_text(stripped, encoding="utf-8")
    rep = vsc.check(p, rec_p)
    assert not rep["CONSISTENT"], "gate accepted a doc hiding an unequal-task-set run"
    assert any("unequal task sets" in q for q in rep["problems"]), rep["problems"]
