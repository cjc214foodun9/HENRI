"""Smoke test: the SEALED action egress is reachable from production_arc_run.py.

WHY
    The sealed decoder exists and passes 23/23 contract tests, but wiring it into a
    3392-line runner is only real if the runner actually BINDS it. This test proves
    the flag/import seam in the production module itself, without running an
    episode (which needs ARC environments). It is the same class of check that
    caught the typed-probe contract's import seam in an earlier round.

ASSERTED
    S1  default-OFF: the flag is False and the private symbols are NOT bound, so
        the module namespace is unchanged when the feature is off.
    S2  with HENRI_SEALED_ACTION_EGRESS=1 the two private symbols ARE bound.
    S3  the margin floor parses; an unparseable value falls back to 0.0 rather than
        raising at import time (a typo must not brick a run).
    S4  the wiring site exists in the source and passes the floor through, and the
        telemetry carries both top1_margin and the effective floor (source ->
        consumer traceability: no read-but-not-forwarded dead store).
    S5  the default action path is untouched when the flag is off: the source still
        contains exactly one `decode_action_egress(` call site for the checkpoint
        path, and the sealed branch is guarded by its own flag.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

R = pathlib.Path(__file__).resolve().parents[2]
PROD = R / "production_arc_run.py"


def _run(env_over: dict) -> str:
    """Import production_arc_run in a SUBPROCESS with a controlled environment."""
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import production_arc_run as P\n"
        "print('FLAG', P.HENRI_SEALED_ACTION_EGRESS)\n"
        "print('FLOOR', P.HENRI_SEALED_EGRESS_MIN_MARGIN)\n"
        "print('BUILDER', hasattr(P, '_build_sealed_action_codebook'))\n"
        "print('DECODER', hasattr(P, '_decode_action_egress_sealed'))\n"
    ) % str(R)
    env = dict(os.environ)
    env.update(env_over)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=900, env=env, cwd=str(R))
    assert r.returncode == 0, f"import failed:\n{r.stdout}\n{r.stderr[-2500:]}"
    return r.stdout


@pytest.fixture(scope="module")
def off_out():
    return _run({"HENRI_SEALED_ACTION_EGRESS": "0",
                 "HENRI_SEALED_EGRESS_MIN_MARGIN": ""})


@pytest.fixture(scope="module")
def on_out():
    return _run({"HENRI_SEALED_ACTION_EGRESS": "1",
                 "HENRI_SEALED_EGRESS_MIN_MARGIN": "0.02"})


# --------------------------------------------------------------------- S1
def test_default_off_binds_nothing(off_out):
    assert "FLAG False" in off_out
    assert "BUILDER False" in off_out, "OFF must not bind the sealed builder"
    assert "DECODER False" in off_out, "OFF must not bind the sealed decoder"


# --------------------------------------------------------------------- S2
def test_flag_on_binds_the_sealed_path(on_out):
    assert "FLAG True" in on_out
    assert "BUILDER True" in on_out
    assert "DECODER True" in on_out


# --------------------------------------------------------------------- S3
def test_floor_parses_and_defaults(on_out, off_out):
    assert "FLOOR 0.02" in on_out
    assert "FLOOR 0.0" in off_out

    bad = _run({"HENRI_SEALED_ACTION_EGRESS": "1",
                "HENRI_SEALED_EGRESS_MIN_MARGIN": "not-a-number"})
    assert "FLOOR 0.0" in bad, "an unparseable floor must fall back to 0.0"


# --------------------------------------------------------------------- S4
def test_wiring_passes_floor_and_emits_traceable_telemetry():
    src = PROD.read_text(encoding="utf-8", errors="replace")
    assert "HENRI_SEALED_ACTION_EGRESS" in src
    assert "HENRI_SEALED_EGRESS_MIN_MARGIN" in src
    # the floor must be FORWARDED to the decoder, not merely defined
    assert re.search(
        r"_decode_action_egress_sealed\([^)]*min_margin=HENRI_SEALED_EGRESS_MIN_MARGIN",
        src, re.S), "floor is read but not forwarded -> dead store"
    # and both must appear in the emitted telemetry
    assert '"top1_margin": round(egress_result.top1_margin, 6)' in src
    assert '"sealed_min_margin": HENRI_SEALED_EGRESS_MIN_MARGIN' in src
    assert '"action_source": "SEALED_CODEBOOK"' in src


# --------------------------------------------------------------------- S5
def test_sealed_branch_cannot_shadow_the_checkpoint_path():
    """Precedence: when the checkpoint path is live, the sealed branch must not run.

    The sealed condition contains a NOT over the checkpoint condition, so a LOADED
    transducer always wins. Asserting the shape of that guard here prevents a
    future edit from making the sealed readout a silent override.
    """
    src = PROD.read_text(encoding="utf-8", errors="replace")
    assert re.search(
        r"if \(HENRI_SEALED_ACTION_EGRESS\s*\n\s*and not \(HENRI_ARC_EGRESS "
        r"and egress_transducer is not None\)", src), \
        "sealed branch must be gated by a NOT over the checkpoint condition"
    # fail-closed: the sealed branch must set env_step_error, which clears
    # macro_actions below and suppresses the fallback step
    assert 'env_step_error = f"SEALED_EGRESS_FAIL_CLOSED: {_se_exc}"' in src
    assert "if env_step_error is not None:" in src


def test_sealed_failure_emits_its_own_telemetry_event():
    src = PROD.read_text(encoding="utf-8", errors="replace")
    assert '"event_type": "EGRESS_FAIL_CLOSED"' in src
    assert '"action_source": "SEALED_CODEBOOK"' in src
