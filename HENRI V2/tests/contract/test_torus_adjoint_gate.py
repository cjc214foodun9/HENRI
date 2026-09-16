"""Contract test: the torus-adjoint evaluator's section-0 gate is ENFORCED.

WHY THIS EXISTS -- defect V4-NOGATE, found by audit, not by a failure.
    `evaluate_torus_adjoint.py` printed and recorded `verification_max_err` from its first
    version, and the sealed doc calls section 0 verification-FIRST. An audit of the committed
    file found NO assert, NO raise and NO tolerance anywhere in it: the error was computed,
    printed, stored -- and never checked. The claim had no mechanism behind it.
        The consequence is visible in the pre-fix log (`adj1.log`): the first adjoint attempt
    died on a tensor-SHAPE RuntimeError rather than a verification failure, so the safeguard
    was never actually exercised. Nothing gated because nothing could gate.

WHAT THE GATE PROTECTS
    `verification_max_err` is a SEAL-BOUND scalar: validate_seal_consistency.SCALAR_SPECS
    binds it (format `.3e`) to a verbatim string in the sealed doc. If the explicit algebra
    stopped reproducing `enc.encode`, every downstream rank and decode number would be
    meaningless while still looking plausible -- and a receipt would still be written. The
    gate refuses to continue in that case, BEFORE the first json.dump.

WHY THERE ARE BOTH A NEGATIVE AND A POSITIVE CONTROL
    A gate that has never been seen to fire is decoration. `test_gate_fires_...` proves it
    can stop a bad operator; `test_gate_does_not_block_...` proves it does not stop a good one
    and that the good run reproduces the sealed numbers exactly. Neither alone is evidence.

HOW THE SEAL IS PROTECTED FROM THESE TESTS
    `OUT` honours the `ADJ_OUT` environment variable, and every run here redirects it into
    tmp_path, so no test can write, truncate or rewrite the sealed receipt. The final test
    re-hashes the sealed file against its committed git blob to prove that held.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]                 # ...\HENRI V2
EVAL = REPO / "experiments" / "verification" / "evaluate_torus_adjoint.py"
RECEIPT = REPO / "experiments" / "verification" / "torus_encoder_adjoint_observed.json"
RECEIPT_REL = "HENRI V2/experiments/verification/torus_encoder_adjoint_observed.json"
ARC = os.environ.get("ARC_CORPUS", "C:/Users/chan/henri_data/ARC-AGI/data")

pytestmark = pytest.mark.skipif(
    not (Path(ARC) / "training").is_dir(),
    reason="ARC corpus absent: the adjoint evaluator needs it for the decode section")


def _run(tmp_out: Path, tol: str | None) -> subprocess.CompletedProcess:
    """Run the evaluator with its receipt redirected into tmp_path."""
    env = dict(os.environ)
    env["ADJ_OUT"] = str(tmp_out)
    env["ARC_CORPUS"] = ARC
    if tol is None:
        env.pop("ADJ_TOL", None)          # sealed configuration: default 1e-3
    else:
        env["ADJ_TOL"] = tol
    return subprocess.run([sys.executable, str(EVAL)], cwd=str(REPO), env=env,
                          capture_output=True, text=True, errors="replace", timeout=900)


def test_gate_fires_when_tolerance_is_impossible(tmp_path):
    """NEGATIVE CONTROL: an operator that cannot pass verification must STOP the run.

    An impossible bound (1e-30) is used so the gate must fire against the REAL measured
    error. This also proves ordering: the gate precedes the write, because no receipt may
    exist afterwards.
    """
    out = tmp_path / "never_written.json"
    p = _run(out, tol="1e-30")
    blob = (p.stdout or "") + (p.stderr or "")

    assert p.returncode != 0, (
        "gate did NOT stop an operator that fails verification; the safety claim is "
        f"unsupported. Output:\n{blob[-1200:]}")
    assert "operator verification FAILED" in blob, blob[-1500:]
    assert not out.exists(), (
        "a receipt WAS written even though verification failed -> the gate is not "
        "upstream of the write, so a bad run could seal meaningless numbers")


def test_gate_does_not_block_and_reproduces_the_sealed_numbers(tmp_path):
    """POSITIVE CONTROL: a correct operator passes, and the run reproduces the seal.

    Compares the fields that must be invariant (`verification_max_err`, `decode_summary`)
    against the sealed receipt. `utc`/`elapsed_secs` are excluded because they always
    differ -- that is why the seal binds CONTENT, not file bytes, across a re-run.
    """
    out = tmp_path / "reproduced.json"
    p = _run(out, tol=None)
    blob = (p.stdout or "") + (p.stderr or "")

    assert p.returncode == 0, f"a correct operator was blocked. Output:\n{blob[-1500:]}"
    assert out.exists(), "no receipt written on a successful run"
    got = json.loads(out.read_text(encoding="utf-8"))
    sealed = json.loads(RECEIPT.read_text(encoding="utf-8"))

    assert got["verification_max_err"] < 1e-3, got["verification_max_err"]
    assert got["verification_max_err"] == sealed["verification_max_err"], (
        got["verification_max_err"], sealed["verification_max_err"])
    assert got["decode_summary"] == sealed["decode_summary"], (
        got["decode_summary"], sealed["decode_summary"])
    dec = got["decode_summary"]
    assert dec["n_exact"] == dec["n_cases"] > 0, dec


def test_sealed_receipt_was_not_touched_by_these_tests():
    """The redirect must hold: the sealed receipt still matches its committed blob.

    Runs last and executes nothing. If ADJ_OUT were ignored, the two tests above would have
    overwritten the sealed receipt, and this assertion is what would catch it.
    """
    assert RECEIPT.is_file(), f"sealed receipt missing: {RECEIPT}"
    on_disk = subprocess.run(["git", "-C", str(REPO.parent), "hash-object", str(RECEIPT)],
                             capture_output=True, text=True, errors="replace").stdout.strip()
    committed = subprocess.run(["git", "-C", str(REPO.parent), "rev-parse",
                                f"HEAD:{RECEIPT_REL}"],
                               capture_output=True, text=True, errors="replace").stdout.strip()
    assert on_disk and committed, (on_disk, committed)
    assert on_disk == committed, (
        "the SEALED adjoint receipt drifted from its committed blob -- a test run wrote it")

    status = subprocess.run(["git", "-C", str(REPO.parent), "status", "--porcelain", "--",
                             RECEIPT_REL],
                            capture_output=True, text=True, errors="replace").stdout.strip()
    assert status == "", f"sealed receipt is dirty: {status!r}"
    digest = hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
    assert digest.startswith("21862da894c78906"), digest[:16]
