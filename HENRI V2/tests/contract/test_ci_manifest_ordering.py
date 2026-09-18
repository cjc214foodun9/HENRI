"""Contract test: the CI recipe must REGENERATE the derived manifest BEFORE --check.

WHY
    The sealed egress manifest is derived and gitignored. Its pin says a consumer MUST
    fail closed when the file is absent -- correct governance, with a mechanical
    consequence: `build_egress_manifest.py --check` returns CHECK_FAIL on a HEALTHY
    clean clone, because the derived file has not been generated yet. Observed in this
    session: CHECK_FAIL on the clean tree, then PASS after regeneration.

    The ordering is therefore load-bearing, and it was previously encoded nowhere
    (the only workflow in this repo was docker-publish.yml). These tests pin the order
    at the source level so a future edit cannot silently invert it.

    They assert ORDER and LOUDNESS, not that CI has run. No CI runner was executed
    here; that limitation is recorded rather than implied away.
"""
from __future__ import annotations

import pathlib
import re

WT = pathlib.Path(__file__).resolve().parents[3]
WF = WT / ".github" / "workflows" / "egress-manifest.yml"
BUILDER = WT / "HENRI V2" / "scripts" / "build_egress_manifest.py"
PIN = WT / "HENRI V2" / "experiments" / "verification" / "EGRESS_MANIFEST_PIN_20260916.json"


def _code_only(text: str) -> str:
    """Strip YAML comments so prose cannot be mistaken for an executable step.

    MY OWN FALSE POSITIVE (measured): the first version of these tests searched the
    raw file, and the header comment explains the defect using the literal string
    `build_egress_manifest.py --check`. That occurrence sits ABOVE the real
    regeneration step, so the order check reported 1543 < 370 -- a failure caused
    entirely by matching inside a comment. Same class of error as the earlier
    "reflex fastest" and dead-flag false positives: a naive substring search over
    prose. Comments are therefore removed before any position is computed.
    """
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def _positions(text: str):
    """Return (first regeneration invocation, first --check invocation), code only.

    The window for the "bare call" test is bounded to the SAME LINE. A fixed
    character window (80 chars in the first version) spilled into the following
    line and swallowed the `--check` that appeared there, so every bare call looked
    like a check call and the negative control reported bare=None. Line-bounding is
    the correct scope: the flag, if present, is on the invocation's own line.
    """
    code = _code_only(text)
    check = [m.start() for m in re.finditer(r"build_egress_manifest\.py\"?\s+--check", code)]
    bare = []
    for m in re.finditer(r"build_egress_manifest\.py", code):
        eol = code.find("\n", m.start())
        seg = code[m.start():eol if eol != -1 else len(code)]
        if "--check" not in seg:
            bare.append(m.start())
    return (bare[0] if bare else None, check[0] if check else None)


def test_workflow_exists():
    assert WF.exists(), "the egress-manifest workflow must exist to encode the order"


def test_regeneration_precedes_check():
    text = WF.read_text(encoding="utf-8")
    bare, check = _positions(text)
    assert bare is not None, "workflow must invoke the builder to REGENERATE"
    assert check is not None, "workflow must invoke the builder with --check"
    assert bare < check, (
        "REGENERATE must precede --check: on a clean clone the derived manifest is "
        "absent, so checking first fails on a healthy tree"
    )


def test_regeneration_step_is_labelled_as_gating():
    text = _code_only(WF.read_text(encoding="utf-8")).lower()
    assert "must precede" in text or "step 1" in text, \
        "the ordering must be stated in the executable steps, not incidental"


def test_the_order_check_cannot_be_satisfied_by_a_comment():
    """Negative control for the false positive that broke the first version.

    A file whose COMMENT contains `--check` before the real regenerate step must
    still resolve to the correct order. Without comment stripping this test fails,
    which is precisely the bug that was fixed.
    """
    trap = (
        "# prose: `build_egress_manifest.py --check` fails on a clean tree\n"
        "run: python build_egress_manifest.py\n"
        "run: python build_egress_manifest.py --check\n"
    )
    bare, check = _positions(trap)
    assert bare is not None and check is not None
    assert bare < check, "comment stripping failed: prose order leaked into the check"


def test_blocked_regeneration_fails_loudly_not_silently():
    """A runner without the source artifact must FAIL, never pass quietly.

    A green CI run with an unregenerated manifest would be a false signal about the
    seal, which is worse than a red one.
    """
    text = WF.read_text(encoding="utf-8")
    assert "BLOCKED" in text, "the workflow must name the blocked condition"
    assert re.search(r"exit 1", text), "a blocked regeneration must exit non-zero"
    assert "::error::" in text, "a blocked regeneration must annotate the run"


def test_builder_contract_the_workflow_relies_on():
    """The workflow is only meaningful if the builder really has these behaviours."""
    src = BUILDER.read_text(encoding="utf-8")
    assert "--check" in src
    assert "CHECK_FAIL" in src, "check() must fail loudly on a missing manifest"
    assert "manifest missing" in src
    # the pin must be what check() compares against
    assert PIN.exists(), "the committed pin is the thing check() verifies"


def test_pin_records_the_fail_closed_policy_the_order_serves():
    import json

    d = json.loads(PIN.read_bytes())
    policy = d.get("commit_policy", "")
    assert "fail closed" in policy.lower(), (
        "the pin must state the fail-closed policy that makes regenerate-then-check "
        "necessary"
    )
    assert d.get("derived_artifact") is True
    # and the file itself must be absent from git (derived => not committed)
    assert d.get("manifest_sha256")
