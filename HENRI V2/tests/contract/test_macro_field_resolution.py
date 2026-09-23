"""Macro-field RESOLUTION must track the run's num_blocks.

WHY THIS FILE EXISTS
    The Sagnac veto raised on every live call at reduced scale (60/60 payloads
    recorded UNAVAILABLE_SHAPE_MISMATCH). ROOT CAUSE, measured
    (experiments/verification/root_cause_num_channels.json):

        opine_object_mcts.py:18   def __init__(self, num_channels: int = 8192, ...)
        production_arc_run.py:2232  _opine = OPINEObjectMCTS(      <-- no num_channels
        opine_object_mcts.construct_macro_option -> [num_channels, 3, 3]
        SU3FieldWaveTransducer.field_to_wave      -> [B, N*8]

    So the macro field carried 8192 channels REGARDLESS of the run's scale, emitting a
    65536-wide wave, while a num_blocks=64 run's references are 512-wide. MEASURED:

        num_channels default 8192 -> wave 65536  (mismatch: refs are 512)
        num_channels 64           -> wave 512    (MATCHES the num_blocks=64 refs)
        and at matched width the veto RUNS and reaches BOTH hard_vetoed values
        (pass delta 0.0 -> not vetoed; veto delta 1.0 -> vetoed)

    8192 is ALSO the GPU-scale `num_blocks` (SCALE = dict(num_blocks=8192) when
    DEVICE=="cuda"), so the hardcoded default coincided with the run's resolution at
    full scale and the defect was invisible there. That is why d_model=65536 appeared
    to be "the configuration where the mismatch disappears" -- the real variable is
    BLOCK COUNT.

INVARIANTS
    M1  the runner constructs OPINEObjectMCTS with num_channels = SCALE["num_blocks"]
    M2  the default remains 8192 so other construction sites are unaffected
    M3  construct_macro_option honours the instance's num_channels (not a constant)
    M4  the width chain is internally consistent: num_channels * 8 == the flattened
        encoder width for the same block count
    M5  at matched width the veto reaches BOTH hard_vetoed values WITHOUT the
        diagnostic bridge -- the gate is bidirectional for the right reason
"""
from __future__ import annotations

import ast
import math
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNNER = ROOT / "production_arc_run.py"
OPINE = ROOT / "opine_object_mcts.py"


def _runner_src() -> str:
    return RUNNER.read_text(encoding="utf-8")


# ------------------------------------------------------------------ M1
def test_macro_resolution_is_scale_bound_behind_a_flag():
    """M1. A hardcoded 8192 fixed the macro field's resolution; binding it is OPT-IN.

    The chain, read from source:
        _num_channels = int(SCALE["num_blocks"]) if HENRI_MACRO_NUM_CHANNELS==1 else 8192
        ActionOutcomeGeneratorStore(num_channels=_num_channels)
        su3_field  <- produced BY that store
        OPINEObjectMCTS(num_channels=su3_field.shape[0])
        construct_macro_option -> [num_channels, 3, 3]
        field_to_wave          -> [B, N*8]

    ROOT CAUSE (measured, root_cause_num_channels.json):
        num_channels 8192 -> 65536-wide wave vs 512-wide refs at num_blocks=64
                          -> the veto raises every step (UNAVAILABLE_SHAPE_MISMATCH)
        num_channels 64   -> 512-wide, MATCHES, veto runs, BOTH hard_vetoed values,
                             with NO bridge

    WHY OPT-IN. Binding it removes the width mismatch in dual_channel_sagnac_veto but a
    SECOND coupled 8192 remains downstream: an einsum `"nij,njk->nik"` then pairs the
    64-wide macro field with an 8192-wide operand and raises
        RuntimeError: einsum(): subscript n has size 8192 for operand 1 ... size 64
    measured on 64/64 steps. So the default path must stay unchanged until that link is
    fixed. The flag makes the change explicit and reversible, which is the project rule
    for altering a default path.
    """
    src = _runner_src()

    # (a) the flag must exist and be checked
    assert "HENRI_MACRO_NUM_CHANNELS" in src, (
        "the scale-bound macro resolution must be opt-in via HENRI_MACRO_NUM_CHANNELS")

    # (b) both arms must be present: SCALE-bound when on, the historical literal when off
    assert re.search(r"_num_channels\s*=\s*\(int\(SCALE\[\"num_blocks\"\]\)", src), (
        "the SCALE-bound arm is missing")
    assert re.search(r"else\s+8192\)", src), (
        "the default arm no longer preserves the historical 8192; the existing CPU "
        "path would change behaviour silently")

    # (c) the store must consume the variable, not a duplicated literal
    store_hits = [m for m in re.finditer(
        r"ActionOutcomeGeneratorStore\(([^)]*)\)", src, re.S)]
    assert store_hits, "ActionOutcomeGeneratorStore is never constructed"
    assert any("num_channels=_num_channels" in h.group(1) for h in store_hits), (
        "the store does not receive the resolution variable, so the flag would have no "
        "effect on the macro field")

    # (d) no bare call-site literal may remain
    bad = [h.group(0)[:80] for h in re.finditer(
        r"(?<![_\w])num_channels\s*=\s*8192", src)]
    assert not bad, f"a hardcoded num_channels=8192 remains at a call site: {bad}"


def test_opine_inherits_the_store_resolution():
    """M1b. OPINE must inherit the field's own channel count, not a constant."""
    src = _runner_src()
    hits = [m for m in re.finditer(r"OPINEObjectMCTS\(([^)]*)\)", src, re.S)]
    assert hits, "OPINEObjectMCTS is never constructed in the runner"
    assert any("su3_field.shape[0]" in h.group(1) for h in hits), (
        "OPINEObjectMCTS does not inherit su3_field.shape[0]; the macro option's "
        "resolution would then not track the field the store produced")


# ------------------------------------------------------------------ M2
def test_default_is_preserved_for_other_callers():
    """M2. Changing the default would move every other construction site."""
    src = OPINE.read_text(encoding="utf-8")
    assert re.search(r"def __init__\(\s*self,\s*num_channels:\s*int\s*=\s*8192", src), (
        "the constructor default changed; other callers rely on 8192")


# ------------------------------------------------------------------ M3
def test_construct_honours_instance_num_channels():
    """M3. The instance value must be USED, not shadowed by a constant."""
    tree = ast.parse(OPINE.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "construct_macro_option":
            body = ast.unparse(n)
            assert "self.num_channels" in body, (
                "construct_macro_option does not reference self.num_channels, so the "
                "field's resolution cannot follow the run's scale")
            return
    pytest.fail("construct_macro_option not found")


# ------------------------------------------------------------------ M4
def test_width_chain_is_consistent():
    """M4. num_channels*8 must equal the flattened encoder width for the same blocks."""
    from henri_vision_encoder import HENRIVisionEncoder
    import numpy as np

    grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for blocks in (64, 512):
        enc = HENRIVisionEncoder(d_model=blocks * 8, k_blocks=blocks, device="cpu")
        ref = int(enc.encode_grid(grid).numel())
        assert ref == blocks * 8, (
            f"encoder width {ref} != {blocks}*8 for num_blocks={blocks}")


# ------------------------------------------------------------------ M5
def test_veto_is_bidirectional_at_matched_width_without_bridge(monkeypatch):
    """M5. The gate must be bidirectional for the RIGHT reason, not via the bridge."""
    monkeypatch.delenv("HENRI_SAGNAC_WIDTH_BRIDGE", raising=False)
    from henri_vision_encoder import HENRIVisionEncoder
    from sagnac_mcts_planner import SagnacMCTSPlanner
    import numpy as np

    blocks = 64
    enc = HENRIVisionEncoder(d_model=blocks * 8, k_blocks=blocks, device="cpu")
    ref = enc.encode_grid(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    pl = SagnacMCTSPlanner(d_model=blocks * 8, k_blocks=blocks, tau_veto=0.35,
                           device="cpu")
    W = int(ref.numel())
    assert W == blocks * 8
    u = torch.zeros(W)
    u[0] = 1.0
    r_agree = pl.dual_channel_sagnac_veto(u, u, u, epsilon_hard=pl.tau_veto)
    r_oppose = pl.dual_channel_sagnac_veto(-u, u, u, epsilon_hard=pl.tau_veto)
    assert r_agree[2] is False, (
        f"perfect agreement was vetoed (delta {r_agree[0]}); the gate is not "
        f"bidirectional at the run's own resolution")
    assert r_oppose[2] is True, (
        f"perfect disagreement was NOT vetoed (delta {r_oppose[0]})")
    assert pl.last_sagnac_bridge is None, (
        "the bridge must NOT be needed at matched width; if it was applied, the "
        "resolution fix did not take effect")


def test_macro_field_width_follows_num_channels():
    """M5b. The field's emitted width is num_channels*8, measured end to end."""
    from opine_object_mcts import OPINEObjectMCTS
    from universal_data_transducer import SU3FieldWaveTransducer

    s3 = 1.0 / math.sqrt(3.0)
    lm = [[[0, 1, 0], [1, 0, 0], [0, 0, 0]],
          [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
          [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
          [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
          [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
          [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
          [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
          [[s3, 0, 0], [0, s3, 0], [0, 0, -2 * s3]]]
    basis = torch.tensor(lm, dtype=torch.complex64)
    trans = SU3FieldWaveTransducer(basis)

    def gens(seed: int = 9):
        g = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(4):
            a = torch.randn(3, 3, generator=g, dtype=torch.complex64)
            a = a - a.conj().transpose(-2, -1)
            out.append(torch.matrix_exp(0.05 * a))
        return out

    for nch in (64, 512):
        eng = OPINEObjectMCTS(num_channels=nch)
        u = eng.construct_macro_option(gens(), device="cpu")
        assert u.shape[0] == nch, f"field channels {u.shape[0]} != {nch}"
        w = trans.field_to_wave(u.unsqueeze(0)).squeeze(0)
        assert int(w.numel()) == nch * 8, (
            f"field_to_wave emitted {int(w.numel())}, expected {nch}*8")
