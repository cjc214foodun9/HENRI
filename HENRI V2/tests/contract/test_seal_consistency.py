"""Contract test: EVERY sealed pair must be SELF-CONSISTENT.

A seal is not one artifact, it is a SET of artifacts that must agree: a human-readable doc
and the machine receipt it describes. Commit 99fb88a shipped a doc that contradicted its own
committed receipt (doc cited koopman_named6 / koopman_no_refl / "480 solves" / 5.75 / 6.0 s;
the receipt has koopman_diagonly arms / n_solves 360 / mean_rank_G 5.667). The doc READS
correctly, so the mismatch survived careful visual review. This test makes it mechanical.

WHY IT IS PARAMETRISED OVER vsc.PAIRS
    Phase 10.3 added a second sealed pair; Phase 10.4 binds FOUR receipts to one doc. A test
    that hard-coded one pair would keep passing while the new seals went unchecked.

WHY ARMS ARE REQUIRED PER *DOC*, NOT PER *RECEIPT*   (Phase 10.4 fix, recorded)
    Phase 10.4 registers a doc against four receipts: one carries the arm table, three are
    MEASUREMENT receipts (an adjoint decode, a rank sweep, byte arithmetic) that have no arm
    table by construction. The original assertion "every receipt must expose arms" therefore
    failed on a CORRECT seal. But the anti-vacuity property it protected is still required:
    a doc must cite measured arms, and it must be attested either by an arm table or by
    explicitly declared sealed scalars. So the rule is now stated PER DOCUMENT, which keeps
    the property while allowing a multi-receipt seal.

    The negative controls are the load-bearing tests: they prove the gate can FAIL. A
    consistency check that passes a doc containing no arms, a doc citing a deleted arm, a doc
    hiding an unequal-task-set run, or a doc that omits a declared scalar, would be vacuous.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]                      # ...\HENRI V2
sys.path.insert(0, str(REPO / "experiments" / "verification"))


def test_pairs_registry_is_not_empty_and_covers_every_sealed_phase():
    """A registry that silently shrinks would disable coverage without failing."""
    import validate_seal_consistency as vsc

    assert len(vsc.PAIRS) >= 2, f"expected >=2 sealed pairs, got {len(vsc.PAIRS)}"
    names = " ".join(str(d) + " " + str(r) for d, r in vsc.PAIRS)
    for phase in ("phase10_1", "phase10_3", "phase10_4"):
        assert phase in names, f"{phase} seal missing from PAIRS"


def test_every_sealed_pair_is_consistent():
    """Every live doc must agree with EVERY receipt registered against it.

    Anti-vacuity, stated per document (see module docstring): each doc must cite measured
    arms, and must be attested by an arm table OR by declared sealed scalars.
    """
    import validate_seal_consistency as vsc

    failures = []
    by_doc: dict = {}
    for doc_p, receipt_p in vsc.PAIRS:
        rep = vsc.check(doc_p, receipt_p)
        if not rep["CONSISTENT"]:
            failures.append(f"{doc_p.name} <-> {receipt_p.name}: {rep['problems']}")
        by_doc.setdefault(doc_p, []).append((receipt_p, rep))

    for doc_p, entries in by_doc.items():
        if not any(e[1]["doc_arms"] for e in entries):
            failures.append(f"{doc_p.name}: no measured arms found -> gate vacuous")
        has_arm_table = any(e[1]["receipt_arms"] for e in entries)
        has_declared_scalars = any(vsc.SCALAR_SPECS.get(e[0].name) for e in entries)
        if not (has_arm_table or has_declared_scalars):
            failures.append(f"{doc_p.name}: no receipt attests arms and no declared "
                            f"scalars -> nothing is actually verified")
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

    The whole Phase 10.3 defect chain (D1, D1-B) came from comparing arms scored on
    different task sets. If a doc could omit that disclosure silently, the gate would not
    have caught the false ACCEPT that this rule exists to prevent.
    """
    import re

    import validate_seal_consistency as vsc

    doc_p, rec_p = vsc.PAIRS[1]
    stripped = doc_p.read_text(encoding="utf-8")
    stripped = re.sub(r"(?i)common[ _-]?subset", "grouping", stripped)
    p = tmp_path / "doc.md"
    p.write_text(stripped, encoding="utf-8")
    rep = vsc.check(p, rec_p)
    assert not rep["CONSISTENT"], "gate accepted a doc hiding an unequal-task-set run"
    assert any("unequal task sets" in q for q in rep["problems"]), rep["problems"]


def test_gate_rejects_a_doc_that_omits_a_declared_scalar(tmp_path):
    """NEGATIVE CONTROL for the Phase 10.4 scalar rule.

    Measurement receipts have no arm table, so without this rule their numbers would be
    bound to the doc only by the generator that wrote them -- the exact Phase 10.1 failure
    (a doc that reads correctly while contradicting its receipt). This proves the rule can
    actually fire.
    """
    import validate_seal_consistency as vsc

    # pick a declared scalar whose formatted value occurs EXACTLY ONCE in its doc, so the
    # redaction removes that scalar and nothing else.
    target = None
    for doc_p, rec_p in vsc.PAIRS:
        specs = vsc.SCALAR_SPECS.get(rec_p.name)
        if not specs:
            continue
        rc = json.loads(rec_p.read_text(encoding="utf-8"))
        doc = doc_p.read_text(encoding="utf-8")
        for label, path, fmt in specs:
            val = vsc.dig(rc, path)
            if val is None:
                continue
            s = format(val, fmt)
            if doc.count(s) == 1:
                target = (doc_p, rec_p, label, s, doc)
                break
        if target:
            break
    if target is None:
        pytest.skip("no declared scalar has a unique formatted occurrence to redact")

    doc_p, rec_p, label, s, doc = target
    p = tmp_path / "doc.md"
    p.write_text(doc.replace(s, "REDACTED", 1), encoding="utf-8")
    rep = vsc.check(p, rec_p)
    assert not rep["CONSISTENT"], f"gate accepted a doc missing declared scalar {label!r}"
    assert any(label in q for q in rep["problems"]), rep["problems"]
