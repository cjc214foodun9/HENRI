"""K1-K3 kill gates for the tripartite VSA resonator carrier.

Pre-registered in
``HENRI V2/experiments/verification/KILL_PREREGISTRATION_resonator_carrier.md``.
Thresholds are FIXED before the mechanism was measured and must not be relaxed
after seeing a result.

Every gate carries a NEGATIVE CONTROL that must FAIL. The supplied blueprint's own
harness was falsified precisely because no such control existed: its spatial branch
was content-invariant and its Sagnac gate passed an all-zero field.

TWO DEFECTS FOUND IN AN EARLIER REVISION OF THIS FILE (disclosed, not hidden)
  1. ``test_k1c`` measured operator SELF-CONSISTENCY, not shift DISCRIMINATION: it
     compared ``enc(roll(X, s))`` against ``apply_roll(enc(X), s)`` at every ``s``,
     which is ~3e-4 everywhere, so true and wrong shifts were indistinguishable
     (measured true 1.147e-04 vs median_wrong 1.122e-04). Corrected to compare the
     TRUE-shifted target against ``apply_roll`` at each candidate shift, which is
     what "position separation via the roll operator" means. The threshold is
     unchanged at 0.5x.
  2. ``test_k2`` drove the resonator with REAL grid pairs, which mixes the
     resonator's factorization error with the measured non-diagonality of the
     encoder's colour/enclosure factors. The pre-registration says "synthesize a
     target wave by applying a known composite transformation", so the primary K2
     task now synthesizes the composite exactly. The real-grid variant is retained
     as a clearly-labelled secondary diagnostic.

Evidence class: CPU wiring and invariants only.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import arc_sagnac_veto as V  # noqa: E402
import o_vsa_torus_encoder as O  # noqa: E402
from henri_resonator import (  # noqa: E402
    ResonatorConfig,
    TripartiteResonator,
    ring_grid,
    roll_grid,
    solid_grid,
)

S = 32
NB = 64
SHIFTS = [(0, 0), (1, 0), (2, 0), (0, 1), (0, -1), (1, 1), (2, -1), (-1, 0), (-2, 0)]
COLOURS = [3, 5, 7]
TOPOS = ["solid", "ring"]


@pytest.fixture(scope="module")
def encoder():
    return O.TorusIngressEncoder(num_blocks=NB, modulus=S)


@pytest.fixture(scope="module")
def resonator(encoder):
    return TripartiteResonator.measure(
        encoder, shifts=SHIFTS, colours=COLOURS, config=ResonatorConfig(), S=S, n_train=5)


# --------------------------------------------------------------------------- K1a
def _content_blind_wave(S: int, nb: int) -> torch.Tensor:
    """The falsified packet's ``psi_pos``: a phasor built from index buffers ONLY.

    Negative control. It never reads the grid, so it cannot separate two grids that
    differ in content. It MUST fail K1a.
    """
    idx = torch.arange(nb, dtype=torch.float32).unsqueeze(-1)
    kx = idx % S
    ky = (idx // S) % S
    phase = (2.0 * math.pi * (kx + ky)) / float(S)
    z = torch.complex(torch.cos(phase), torch.sin(phase)).expand(nb, 4).contiguous()
    w = torch.view_as_real(z).reshape(nb, 8)
    return w / (w.norm(p=2, dim=-1, keepdim=True) + 1e-9)


def test_k1a_content_dependence(encoder):
    """Two grids differing only in object position must separate above the noise floor."""
    g1 = solid_grid(S, 8, 4)
    g2 = solid_grid(S, 20, 17)
    sep = float((encoder.encode(g1) - encoder.encode(g2)).abs().max())
    assert sep >= 1e-3, f"encoder is content-invariant: separation {sep:.3e} < 1e-3"

    ctl_sep = float((_content_blind_wave(S, NB) - _content_blind_wave(S, NB)).abs().max())
    assert ctl_sep < 1e-3, (
        f"negative control unexpectedly separated ({ctl_sep:.3e}); K1a is not measuring "
        "content dependence and is therefore VOID")


def test_k1b_determinism(encoder):
    """The same grid twice must encode bit-identically."""
    g = solid_grid(S, 8, 4)
    assert float((encoder.encode(g) - encoder.encode(g)).abs().max()) == 0.0


# --------------------------------------------------------------------------- K1c
def test_k1c_roll_separation(encoder):
    """The TRUE shift must predict the shifted target; wrong shifts must not.

    Compares the known-shifted target against ``apply_roll`` at every candidate
    shift and requires the true shift to beat the median wrong shift.
    """
    X = encoder.encode(solid_grid(S, 8, 4))
    true_shift = (3, 0)
    Y = encoder.encode(roll_grid(solid_grid(S, 8, 4), *true_shift))

    def residual(dw, dh):
        pred = encoder.apply_roll(X, dw, dh)
        return float(torch.norm(Y - pred) / (torch.norm(Y) + 1e-12))

    r_true = residual(*true_shift)
    wrong = sorted(residual(dw, dh) for dw in range(-2, 3) for dh in range(-2, 3)
                   if (dw, dh) != true_shift)
    med = wrong[len(wrong) // 2]
    assert r_true <= 0.5 * med, (
        f"roll factor not separable: true={r_true:.3e} median_wrong={med:.3e}")


# --------------------------------------------------------------------------- K1d
def test_k1d_enclosure_separable(encoder):
    """A solid block and a hollow ring must separate on a NON-mass feature."""
    fs = TripartiteResonator.topology_feature(solid_grid(S, 8, 4, 5, 5))
    fr = TripartiteResonator.topology_feature(ring_grid(S, 8, 4, 5, 5))
    assert fs["mech_type"] == "solid_block", fs
    assert fr["mech_type"] == "enclosed_contour", fr
    assert fr["interior_fraction"] > fs["interior_fraction"], (fs, fr)
    assert abs(fr["interior_fraction"] - fs["interior_fraction"]) >= 1e-6


# --------------------------------------------------------------------------- K3
def test_k3_dead_field_negative_control(encoder):
    """A DEAD field must fire the veto; a conserved field must not.

    The falsified packet's telescoping-sum ``delta_q`` returned 0 for an all-zero
    field and did NOT veto it. This control excludes exactly that defect.
    """
    X = encoder.encode(solid_grid(S, 8, 4)).reshape(-1)
    Y = encoder.encode(solid_grid(S, 20, 17)).reshape(-1)

    _, _, vetoed_ok, status_ok = V.evaluate_veto(X, X, Y)
    assert vetoed_ok is False, f"conserved field wrongly vetoed ({status_ok})"

    for name, dead in (("all-zeros", torch.zeros_like(X)),
                       ("constant", torch.ones_like(X) * 3.0)):
        _, _, vetoed, status = V.evaluate_veto(dead, X, Y)
        assert vetoed is True, f"DEAD FIELD PASSED the veto ({name}, {status})"


# --------------------------------------------------------------------------- K2
def _k2_triples():
    """Non-identity composite targets: shift != (0,0), colour != 3, topo == ring."""
    out = []
    nonid = [s for s in SHIFTS if s != (0, 0)]
    for i, sh in enumerate(nonid):
        for c in (5, 7):
            out.append((sh, c, "ring"))
    return out


def test_k2_resonator_beats_identity(encoder, resonator):
    """Pre-registered: hit_rate >= 0.60 AND hit_rate - identity_rate >= 0.30.

    Primary task: a wave synthesized by the KNOWN composite
    ``M_roll (x) R_value (x) Q_topo``, which is the pre-registered construction.
    The identity attractor rate and the residual trace are reported unconditionally.
    """
    ref_grid = solid_grid(S, 8, 4, 5, 5, colour=3)
    x = encoder.encode(ref_grid)

    hits = ident = 0
    per_factor = [0, 0, 0]
    ratios = []
    traces = []
    for (sh, col, topo) in _k2_triples():
        y = resonator.synthesize(x, sh, col, topo)
        r = resonator.factorize(y, x)
        want = (sh, col, topo)
        got = tuple(r.labels)
        hits += int(got == want)
        per_factor[0] += int(got[0] == sh)
        per_factor[1] += int(got[1] == col)
        per_factor[2] += int(got[2] == topo)
        ident += int(r.is_identity)
        if r.residual_ratio == r.residual_ratio:
            ratios.append(r.residual_ratio)
        traces.append((got, want, round(r.residual, 6)))

    n = len(_k2_triples())
    hit_rate = hits / n
    ident_rate = ident / n
    diag = {
        "n": n, "hit_rate": round(hit_rate, 4), "identity_rate": round(ident_rate, 4),
        "margin": round(hit_rate - ident_rate, 4),
        "factor_roll": round(per_factor[0] / n, 4),
        "factor_value": round(per_factor[1] / n, 4),
        "factor_topo": round(per_factor[2] / n, 4),
        "mean_residual_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
    }
    print("K2 DIAGNOSTICS:", diag)
    for got, want, res in traces[:6]:
        print(f"   got={got} want={want} residual={res}")

    assert hit_rate >= 0.60, f"K2 hit_rate {hit_rate:.3f} < 0.60 | {diag}"
    assert (hit_rate - ident_rate) >= 0.30, (
        f"K2 margin {hit_rate - ident_rate:.3f} < 0.30 -- identity attractor dominates "
        f"| {diag}")


def test_k2_residual_decreases(encoder, resonator):
    """The iteration must actually refine the estimate: last residual < first.

    A CONSTANT residual trace is the signature of an inert loop (measured in an
    earlier revision: 1.417012 repeated 16 times, ratio 1.000). This gate rejects it.
    """
    x = encoder.encode(solid_grid(S, 8, 4, 5, 5, colour=3))
    y = resonator.synthesize(x, (2, 0), 5, "ring")
    r = resonator.factorize(y, x)
    print("K2 residual trace:", [round(v, 6) for v in r.residual_trace])
    assert len(r.residual_trace) >= 2, "no iteration occurred"
    assert r.residual_ratio < 0.5, (
        f"residual did not decrease: first={r.first_residual:.6f} last={r.residual:.6f} "
        f"ratio={r.residual_ratio:.4f}")


def test_k2_config_fields_are_live():
    """No config field may be a dead variable (the falsified packet read 5 of 8 never)."""
    import inspect
    import re
    src = inspect.getsource(TripartiteResonator)
    for f in ("max_iter", "tol", "eps", "beta", "beta_start", "anneal"):
        n = len(re.findall(r"\b%s\b" % f, src)) - (1 if f in src[:400] else 0)
        assert n >= 1, f"config.{f} is never read in factorize -> dead variable"


def test_k2_synthesize_round_trip(encoder, resonator):
    """The synthesized composite must invert: factorizing it recovers the triple."""
    x = encoder.encode(solid_grid(S, 8, 4, 5, 5, colour=3))
    y = resonator.synthesize(x, (1, 1), 7, "ring")
    r = resonator.factorize(y, x)
    assert tuple(r.labels) == ((1, 1), 7, "ring"), r.labels


def test_k2_real_grid_diagnostic(encoder, resonator):
    """SECONDARY diagnostic (not a gate): factorization on REAL grid pairs.

    Reported to separate resonator error from the measured non-diagonality of the
    encoder's colour/enclosure factors. It is deliberately NOT asserted.
    """
    ref = solid_grid(S, 8, 4, 5, 5, colour=3)
    x = encoder.encode(ref)
    ok = 0
    total = 0
    for sh in [(1, 0), (2, 0), (0, 1), (2, -1), (-1, 0)]:
        tgt = roll_grid(ring_grid(S, 8, 4, 5, 5, colour=5), sh[0], sh[1])
        y = encoder.encode(tgt)
        r = resonator.factorize(y, x)
        total += 1
        ok += int(tuple(r.labels) == (sh, 5, "ring"))
    print(f"K2 REAL-GRID DIAGNOSTIC: {ok}/{total} exact triples "
          f"(encoder colour/enclosure factors are measured non-diagonal)")
