"""Phase 1 harness-isolation gate tests (specification action item 4).

Asserts the three invariants from `henri/harness_isolation.py`. These are
AUDITS over the live tree, so they can fail if someone later routes an
`--out` default back into the repository.
"""

import json
import subprocess
from pathlib import Path

from henri.harness_isolation import (
    gate_report,
    h1_no_tracked_modifications,
    h2_defaults_stay_outside_repo,
    h3_explicit_out_under_tmp,
    scan_out_defaults,
)


def test_h2_no_out_default_lands_in_the_repository():
    """H2: every discovered --out default must stay outside the repo."""
    bad = h2_defaults_stay_outside_repo()
    assert bad == [], f"in-tree --out defaults:\n" + "\n".join(bad)


def test_h2_scanner_actually_finds_defaults():
    """TAUTOLOGY GUARD: an empty scan would make H2 pass vacuously."""
    defaults = scan_out_defaults()
    assert len(defaults) >= 10, (
        f"only {len(defaults)} --out defaults found; the H2 gate would be vacuous"
    )
    assert all(isinstance(d, str) and d for _, d in defaults)


def test_h3_helper_detects_containment(tmp_path):
    """H3: containment check works in BOTH directions."""
    inside = tmp_path / "sub" / "receipt.json"
    assert h3_explicit_out_under_tmp(str(tmp_path), str(inside))
    assert h3_explicit_out_under_tmp(str(tmp_path), str(tmp_path))
    # TAUTOLOGY GUARD: a path outside tmp must NOT be accepted.
    assert not h3_explicit_out_under_tmp(str(tmp_path), "/tmp/somewhere_else")


def test_h1_reports_no_tracked_receipt_modifications():
    """H1: the working tree must not carry modified tracked receipts.

    NOTE: this asserts only the *receipt-shaped* subset (json/md/yaml), because
    the pre-existing dirty worktrees on this host are out of scope and must be
    preserved. The gate is deliberately narrow so it cannot be confused with a
    clean-tree requirement.
    """
    bad = h1_no_tracked_modifications()
    assert bad == [], "modified tracked receipt-shaped files:\n" + "\n".join(bad)


def test_report_is_json_serialisable_and_honest():
    rep = gate_report()
    text = json.dumps(rep)
    assert "henri.phase1.harness-isolation.v1" in text
    assert rep["remediation_performed"] is False
    assert rep["out_defaults_scanned"] >= 10


def test_no_ci_workflow_writes_into_the_tree():
    """Documents the recon finding: there is no .github/workflows directory."""
    workflows = Path(__file__).resolve().parents[3] / ".github" / "workflows"
    if workflows.exists():
        # If CI is ever added, it must not stage receipt writes.
        hits = [
            p.name for p in workflows.rglob("*.yml")
            if "henri_audit_chain" in p.read_text(encoding="utf-8", errors="replace")
        ]
        assert hits == [], f"CI workflows reference governance ledgers: {hits}"
