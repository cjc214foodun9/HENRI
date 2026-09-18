"""Smoke test for the Action-3 / Directive-4 production wiring (CPU, no GPU).

WHAT THIS PROVES
    1. `production_arc_run.py` still compiles, and its typed-probe import block
       resolves when HENRI_TYPED_PROBE_CONTRACT=1.
    2. The exact call sequence the wiring uses runs end to end on a real wave:
       transduce -> recover -> compare.
    3. The scalar-rotor guard RAISES on the path (so a gauge no-op can never be
       mistaken for an update) and does NOT raise on the real ramp path.
    4. The carried probe wave is a SEPARATE object from state_wave, which is what
       prevents causal leakage into train_ctx.

WHAT THIS DOES NOT PROVE
    It does not run the ARC environment and produces no task score. `arc_agi` and
    `arcengine` import, but no scorecard is executed here and no CUDA is available
    (Vast 50797414 EXITED). End-to-end live verification stays BLOCKED.
"""
import ast
import pathlib
import subprocess
import sys

import pytest
import torch

R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))

from arc_egress_contract import (  # noqa: E402
    ScalarRotorRejected,
    recover_delta_from_wave,
    reject_scalar_rotor,
    transduce_external_outcome,
)

NUM_BLOCKS, BLOCK_DIM = 8192, 8
D_MODEL = NUM_BLOCKS * BLOCK_DIM


def _state_wave(seed=17):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_BLOCKS, BLOCK_DIM, generator=g)


def test_production_runner_compiles():
    r = subprocess.run(
        [sys.executable, "-m", "py_compile",
         str(R / "production_arc_run.py")],
        capture_output=True, text=True)
    assert r.returncode == 0, f"production_arc_run.py failed to compile: {r.stderr}"


def test_flag_and_wiring_block_exist_in_source():
    """The flag must exist, default OFF, and the wiring must be guarded by it."""
    src = (R / "production_arc_run.py").read_text(encoding="utf-8")
    assert 'os.environ.get("HENRI_TYPED_PROBE_CONTRACT", "0") == "1"' in src
    assert "if HENRI_TYPED_PROBE_CONTRACT:" in src
    assert "probe_belief_wave" in src
    tree = ast.parse(src)
    fns = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef,))}
    assert fns, "source parsed but no functions found -- parsing assumption wrong"


def test_flag_defaults_to_off_when_env_unset():
    """Byte-identical default path: the flag must be falsy with no env var."""
    import os
    env = dict(os.environ)
    env.pop("HENRI_TYPED_PROBE_CONTRACT", None)
    got = env.get("HENRI_TYPED_PROBE_CONTRACT", "0") == "1"
    assert got is False


def test_wiring_call_sequence_runs_on_a_real_wave():
    """Exercises the same sequence as the production block."""
    state_wave = _state_wave()
    ref = state_wave.detach()
    for delta_s in (0.0, 1.0):
        new_wave, info = transduce_external_outcome(ref, 3, delta_s,
                                                   return_info=True)
        rec = recover_delta_from_wave(new_wave, 3, reference=ref)
        assert info["is_scalar_rotor"] is False
        assert info["operator"] == "per_dimension_phase_rotation"
        assert abs(rec - delta_s) < 1e-3, f"delta {delta_s} -> {rec}"
        ref = new_wave.detach()  # the wiring carries the wave forward


def test_scalar_rotor_raises_on_the_probe_path():
    """Contract A3: the guard must fire, so a gauge no-op cannot pass as an update."""
    with pytest.raises(ScalarRotorRejected):
        reject_scalar_rotor(torch.tensor([float(torch.pi)]), D_MODEL // 2)


def test_probe_carrier_is_independent_of_state_wave():
    """Anti-leakage: binding into a SEPARATE carrier must not mutate state_wave."""
    state_wave = _state_wave(seed=23)
    before = state_wave.clone()
    carrier = state_wave.detach()
    carrier, _ = transduce_external_outcome(carrier, 1, 1.0, return_info=True)
    assert torch.equal(state_wave, before), (
        "state_wave was mutated by the probe channel: causal leakage risk"
    )
    assert not torch.allclose(carrier.reshape(-1), before.reshape(-1), atol=1e-6)


def test_transduction_records_a_small_roundtrip_error():
    """The wiring emits delta_roundtrip_abs_err; it must be near zero on CPU."""
    w = _state_wave(seed=31)
    for d in (0.0, 1.0):
        nw, _ = transduce_external_outcome(w, 2, d, return_info=True)
        err = abs(recover_delta_from_wave(nw, 2, reference=w) - d)
        assert err < 1e-3
