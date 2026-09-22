"""Sagnac scale contract: the scale defect must not return, and must stay A/B-able.

WHY THIS FILE EXISTS
    `SagnacMCTSPlanner.dual_channel_sagnac_veto` used `torch.mean(w.conj() * w_ref)`
    as an inner product. That is valid ONLY for qFHRR unit-modulus waves (|w_n| = 1,
    so ||w||_2 = sqrt(D)). For L2-normalized waves (||w||_2 = 1) it evaluates

        delta = 1 - |<a,b>| / D

    which is 1 - O(1/D) for EVERY input. At D = 1024 that is 0.999023.

    MEASURED consequence (experiments/verification/sagnac_scale_defect.py):
        identical pair   delta 0.999023   (must be 0.0)
        range across identical / orthogonal / random   9.59e-04   <- flat
        root children pruned by the hard veto          9/9  (rate 1.000)
    Meanwhile the ROOT is scored by HENRIVisionEncoder.compute_sagnac_similarity,
    which uses the CORRECT convention. So the root looked healthy (0.497) and every
    child looked catastrophic (~0.999), and search() always returned Identity.

    This defect was ALREADY DOCUMENTED before this patch (arc_sagnac_veto.py,
    "FALSIFIED ... OBSERVED 2026-08-12"; re-confirmed in henri_dual_speed_harness.py
    2026-08-18). It is pinned here so it cannot silently return, and so the
    production consumer keeps a working sidecar.

INVARIANTS ENFORCED
    S1  IDENTICAL waves give delta ~ 0, not ~1. This is the single most basic
        property of a similarity-derived distance and the one that was violated.
    S2  The stress SEPARATES: the range across identical / orthogonal / random pairs
        must be large (measured 0.9817 normalized vs 9.59e-04 legacy).
    S3  The hard veto must NOT fire on a matching candidate, so the MCTS can expand.
    S4  The legacy scale stays reproducible under HENRI_SAGNAC_LEGACY_SCALE=1. A fix
        that cannot be turned off cannot be A/B'd.
    S5  Real and complex waves are both handled to the SAME [0,1] range, so the
        two conventions cannot silently diverge again.
    S6  Zero-energy input fails closed (no NaN, no div-by-zero).
    S7  REGRESSION GUARD for the patch itself: importing the module must succeed and
        every name the patched method references must be bound. `py_compile` and
        `ast.parse` BOTH passed while `import os` was missing, because neither
        resolves names -- so an import check is the only thing that would have
        caught it. That is recorded here because it is the same class of
        un-failable check that let four other instrument bugs through.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

D = 1024
TAU_VETO = 0.35


@pytest.fixture(scope="module")
def planner():
    from sagnac_mcts_planner import SagnacMCTSPlanner
    return SagnacMCTSPlanner(d_model=D, k_blocks=128, tau_veto=TAU_VETO,
                             device="cpu")


def _unit(n: int, seed: int, dim: int = D) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, dim, generator=g)
    return v / v.norm(dim=-1, keepdim=True)


# ------------------------------------------------------------------ S7
def test_module_imports_and_patched_names_are_bound():
    """S7. The only check that would have caught the missing `import os`.

    `py_compile` and `ast.parse` both pass on a module whose methods reference an
    unbound global, because neither resolves names. Importing and then CALLING the
    patched method is the difference between a syntactic and a semantic check.
    """
    mod = importlib.import_module("sagnac_mcts_planner")
    assert hasattr(mod.SagnacMCTSPlanner, "_norm_consistent_similarity"), (
        "the norm-consistent similarity helper is missing; the scale fix was reverted")
    # `os` must be bound, because the legacy A/B branch reads os.environ.
    assert hasattr(mod, "os"), (
        "module-level `os` is not bound. The HENRI_SAGNAC_LEGACY_SCALE branch calls "
        "os.environ and would raise NameError at runtime, while py_compile passes.")


# ------------------------------------------------------------------ S1 / S2
def test_identical_waves_give_zero_stress(planner):
    """S1. THE core invariant that was violated."""
    a = _unit(1, 1)[0]
    delta, _, _ = planner.dual_channel_sagnac_veto(a, a, a)
    assert abs(delta) < 1e-6, (
        f"identical waves gave delta {delta:.6f}; expected ~0. A value near "
        f"1 - 1/D means the mean-vs-inner-product scale defect is back.")


def test_stress_separates_identical_from_orthogonal_and_random(planner):
    """S2. A distance that does not separate carries no information."""
    a = _unit(1, 1)[0]
    b = _unit(1, 2)[0]
    r1a, r1b = _unit(1, 11)[0], _unit(1, 12)[0]
    r2a, r2b = _unit(1, 21)[0], _unit(1, 22)[0]

    d_id = planner.dual_channel_sagnac_veto(a, a, a)[0]
    d_orth = planner.dual_channel_sagnac_veto(a, b, b)[0]
    d_r1 = planner.dual_channel_sagnac_veto(r1a, r1b, r1b)[0]
    d_r2 = planner.dual_channel_sagnac_veto(r2a, r2b, r2b)[0]

    spread = max(d_id, d_orth, d_r1, d_r2) - min(d_id, d_orth, d_r1, d_r2)
    assert spread > 0.5, (
        f"stress spread across identical/orthogonal/random is only {spread:.2e}. "
        f"Measured legacy spread was 9.59e-04 and normalized spread 0.9817; a tiny "
        f"spread means the channel is constant and cannot inform selection.")
    assert d_id < 0.05 and d_orth > 0.5, (
        f"identical={d_id:.4f} orthogonal={d_orth:.4f}: identical must be low and "
        f"orthogonal high.")


# ------------------------------------------------------------------ S3
def test_hard_veto_does_not_fire_on_a_matching_candidate(planner):
    """S3. 9/9 children were pruned before the fix, so the tree could not expand."""
    ref = _unit(1, 7)[0]
    _, _, veto = planner.dual_channel_sagnac_veto(ref, ref, ref)
    assert veto is False, (
        "the hard veto fired on a wavefront that MATCHES the reference. With "
        "epsilon_hard = 0.35 this prunes every child and search() can only ever "
        "return Identity.")


# ------------------------------------------------------------------ S4
def test_legacy_scale_is_reproducible_for_ab(planner):
    """S4. The defect must stay reproducible, and the fix must be reversible."""
    script = ROOT / "experiments" / "verification" / "sagnac_scale_defect.py"
    if not script.exists():
        pytest.skip("scale-defect probe not present")
    env = dict(os.environ)
    env["HENRI_SAGNAC_LEGACY_SCALE"] = "1"
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, env=env, timeout=600, cwd=str(ROOT))
    out = proc.stdout
    # The legacy arm must show the pinned-near-1 pathology.
    assert "EXACT-MATCH stress" in out and "PRUNED" in out, (
        f"legacy probe did not report its numbers; stderr tail: {proc.stderr[-400:]}")
    # Pull the exact-match stress from the legacy run and assert it is the defect.
    line = [l for l in out.splitlines() if "EXACT-MATCH stress" in l]
    assert line, "no EXACT-MATCH stress line in legacy output"
    stress = float(line[0].split("=")[-1].split()[0])
    assert stress > 0.9, (
        f"legacy arm gave EXACT-MATCH stress {stress:.6f}; the recorded defect "
        f"(~0.999) is not reproducible, so the A/B escape hatch is broken.")


# ------------------------------------------------------------------ S5
def test_real_and_complex_waves_share_the_same_range(planner):
    """S5. Two conventions in one method was the root cause; they must agree now."""
    g = torch.Generator().manual_seed(5)
    real = torch.randn(D, generator=g)
    real = real / real.norm()
    ph = torch.rand(D, generator=g) * 6.283185307
    comp = torch.complex(torch.cos(ph), torch.sin(ph))
    comp = comp / comp.norm()

    d_real, _, _ = planner.dual_channel_sagnac_veto(real, real, real)
    d_comp, _, _ = planner.dual_channel_sagnac_veto(comp, comp, comp)
    assert abs(d_real) < 1e-6, f"real identical delta {d_real:.3e}"
    assert abs(d_comp) < 1e-6, f"complex identical delta {d_comp:.3e}"

    # And both must be bounded in [0, 1].
    for w in (real, comp):
        o = _unit(1, 99)[0]
        if w.is_complex():
            o = torch.complex(torch.cos(ph), torch.sin(ph))
            o = o / o.norm()
        d, _, _ = planner.dual_channel_sagnac_veto(w, o, o)
        assert 0.0 <= d <= 1.0, f"delta {d} outside [0,1]"


# ------------------------------------------------------------------ S6
def test_zero_energy_input_fails_closed(planner):
    """S6. No NaN, no div-by-zero."""
    z = torch.zeros(D)
    ref = _unit(1, 3)[0]
    d, _, _ = planner.dual_channel_sagnac_veto(z, ref, ref)
    assert d == d, "delta is NaN on a zero-energy candidate"
    assert 0.0 <= d <= 1.0, f"delta {d} outside [0,1] on a zero-energy candidate"


def test_norm_consistent_similarity_is_bounded_and_unit_invariant(planner):
    """The helper must be scale-invariant: scaling an input must not change S."""
    a = _unit(1, 41)[0]
    b = _unit(1, 42)[0]
    s1 = planner._norm_consistent_similarity(a, b)
    s2 = planner._norm_consistent_similarity(a * 7.5, b * 0.02)
    assert abs(s1 - s2) < 1e-6, (
        f"similarity changed under rescaling ({s1:.6f} -> {s2:.6f}); it is not "
        f"norm-consistent, which is the whole point of the fix")
    assert 0.0 <= s1 <= 1.0
    assert abs(planner._norm_consistent_similarity(a, a) - 1.0) < 1e-6
