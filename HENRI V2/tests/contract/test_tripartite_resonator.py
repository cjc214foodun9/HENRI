"""Contract tests: ARC tripartite resonator (factorized task operator).

CPU-only, no network, deterministic. These tests are the GATE for
``arc_tripartite_resonator.py``: each one is written to FAIL under a specific
injected defect (see ``check_tripartite_mutation_gate.py``).

Nothing here is a capability claim. The scenes are solvable BY CONSTRUCTION.
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import arc_tripartite_resonator as R          # noqa: E402
from o_vsa_torus_encoder import TorusIngressEncoder  # noqa: E402
from henri_wave_kb import HARDCODED_EPSILON   # noqa: E402

BLOCKS = 1024          # 1 canvas tile at S=32 -> fast; the algebra is block-count free
MODULUS = 32


# ------------------------------------------------------------------ helpers
def _grid(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 4, (n, n), generator=g)).tolist()


def _canvas(blocks: int = BLOCKS) -> R.TorusCanvas:
    return R.TorusCanvas(num_blocks=blocks, modulus=MODULUS)


def _waves(canvas: R.TorusCanvas, n: int = 4, size: int = 32) -> torch.Tensor:
    grids = [canvas.pad_to_canvas(_grid(size, 100 + i), canvas.geo.modulus)
             for i in range(n)]
    return canvas.encode_batch(grids)


def _solvable(canvas: R.TorusCanvas, n: int = 4):
    X = _waves(canvas, n)
    return R.scene_solvable(canvas, X, dx=3, dy=5)


@pytest.fixture(scope="module")
def ctx():
    canvas = _canvas()
    sol = _solvable(canvas)
    gate, ref_est = R.calibrate_reference_gate(canvas, sol["X"], sol["Y"])
    return {"canvas": canvas, "sol": sol, "gate": gate, "ref": ref_est}


# ============================================================ 0. wave plumbing
def test_to_real_matches_encoder_own_map(ctx):
    """to_real is the batched restatement of TorusIngressEncoder._to_real.

    Differential check against the encoder's OWN 2-D implementation: if this
    diverges, every wave-domain number in the receipt is measuring a different
    representation than the codebase's.
    """
    canvas = ctx["canvas"]
    w = _waves(canvas, 1)[0]
    mine = R.to_real(R.to_complex(w))
    theirs = TorusIngressEncoder._to_real(R.to_complex(w))
    assert torch.allclose(mine, theirs, atol=1e-6), "batched to_real != encoder _to_real"
    assert mine.shape == (canvas.geo.num_blocks, 8)


def test_geometry_is_modulus_not_array_length(ctx):
    canvas = ctx["canvas"]
    notes = canvas.geo.notes()
    assert notes["position_modulus_S"] == 32
    assert notes["canvas_cells_S_squared"] == 1024
    assert notes["array_length_num_blocks"] == canvas.geo.num_blocks
    assert notes["spec_block_size_1024_is_S_squared"] is True
    assert canvas.geo.tiles == canvas.geo.num_blocks // 1024
    # The array length is a SEPARATE quantity from the modulus.
    assert canvas.geo.num_blocks != 32


# ==================================================== 1. translation recovery
def test_translation_recovery_on_known_roll(ctx):
    """A probe orbit of a pure roll must return the exact offset it was given."""
    canvas = ctx["canvas"]
    X = _waves(canvas, 3)
    for dx, dy in ((1, 0), (0, 1), (3, 5), (31, 31), (17, 2), (0, 0)):
        got = canvas.probe_translation(R.to_complex(X), R.to_complex(canvas.translate(X, dx, dy)))
        assert (got["dx"], got["dy"]) == (dx, dy), (
            f"translation not recovered: truth {(dx, dy)} got {(got['dx'], got['dy'])}")
        assert got["identifiable"] is True


def test_translation_recovery_is_sharp_and_exact_for_eight_offsets(ctx):
    """8/8 offsets recovered by the circular phase correlation probe."""
    canvas = ctx["canvas"]
    X = _waves(canvas, 3)
    exact = 0
    for i in range(8):
        dx, dy = (i * 5) % MODULUS, (i * 7 + 1) % MODULUS
        got = canvas.probe_translation(R.to_complex(X),
                                       R.to_complex(canvas.translate(X, dx, dy)))
        exact += 1 if (got["dx"], got["dy"]) == (dx, dy) else 0
        assert got["peak_ratio"] is None or got["peak_ratio"] > 10.0
    assert exact == 8, f"only {exact}/8 offsets recovered"


def test_translation_norm_wrap_and_adjoint_under_modulus_32(ctx):
    """Pure roll: norm-preserving, adjoint exact 8/8, modulus S=32 not the array length."""
    canvas = ctx["canvas"]
    inv = canvas.translation_invariants()
    assert inv["adjoint_exact"] == "8/8", inv
    assert inv["norm_preserving_exact"] == "8/8", inv
    assert inv["modulus_S_identity_exact"] == "8/8", inv
    assert inv["tile_purity_exact"] == "8/8", inv
    assert inv["index_semantics_exact"] is True
    assert inv["nonzero_shift_is_not_identity"] is True
    assert inv["max_abs_norm_drift"] < 1e-6

    # No wrap aliasing: a shift of exactly S on either axis is the identity, and
    # a translation never mixes tiles.
    X = _waves(canvas, 1)
    assert torch.allclose(canvas.translate(X, MODULUS, 0), X, atol=0)
    assert torch.allclose(canvas.translate(X, 0, MODULUS), X, atol=0)
    t = canvas.translate(X, 5, 7).reshape(canvas.geo.tiles, 1024, 8)
    src = X.reshape(canvas.geo.tiles, 1024, 8)
    for ti in range(canvas.geo.tiles):
        u1, c1 = torch.unique(src[ti], dim=0, return_counts=True)
        u2, c2 = torch.unique(t[ti], dim=0, return_counts=True)
        assert torch.equal(u1, u2) and torch.equal(c1, c2), "tile contents mixed"


# ================================================================= 2. rotor
def test_rotor_orthogonality_and_proper_determinant(ctx):
    canvas = ctx["canvas"]
    X = _waves(canvas, 3)
    Y = canvas.translate(R.apply_rotor(X, R._rotor_from_axis_angle(5, 0.4)), 2, 7)
    est = R.estimate_factors(canvas, X, Y)
    rotor = est["operator"].rotor
    err = float((rotor.t() @ rotor - torch.eye(8)).abs().max())
    assert err < 1e-5, f"rotor not orthogonal: {err}"
    assert abs(float(torch.det(rotor)) - 1.0) < 1e-5, "rotor is not a proper rotation"


def test_rotor_reapplication_reproduces_target(ctx):
    """>= || A R - B ||_F must vanish for full-rank data generated by a true rotor."""
    g = torch.Generator().manual_seed(5)
    R_true = R._rotor_from_axis_angle(9, 0.55)
    A = torch.randn(512, 8, generator=g)
    B = A @ R_true
    out = R.orthogonal_procrustes(A, B)
    assert out.reapplication_error < 1e-5, out.reapplication_error
    assert out.orthogonality_error < 1e-5
    assert float((out.R - R_true).abs().max()) < 1e-5
    assert out.det_correction_applied is False
    assert abs(out.det_applied - 1.0) < 1e-5


def test_rotor_is_not_unique_on_wave_data_and_that_is_reported(ctx):
    """MEASURED LIMIT: the wave slot space is rank-deficient, so the rotor is not unique.

    The solver contract (orthogonality, det=+1, bounded re-application) survives;
    the ROTOR ITSELF is only identifiable up to the null space, and the module
    must report that rather than implying an exact recovery.
    """
    canvas = ctx["canvas"]
    X = _waves(canvas, 3)
    R_true = R._rotor_from_axis_angle(9, 0.55)
    spec = R.slot_space_spectrum(X)
    assert spec["effective_rank_1e-2"] < 8, spec["singular_values"]
    assert spec["condition_ratio_last_over_first"] < 1e-2

    out = R.orthogonal_procrustes(X.reshape(-1, 8),
                                  R.apply_rotor(X, R_true).reshape(-1, 8))
    assert out.orthogonality_error < 1e-5
    assert abs(out.det_applied - 1.0) < 1e-5
    # residual is small RELATIVE to the wave scale, but the rotor is not R_true
    assert out.reapplication_error / float(X.abs().max()) < 1e-3
    assert float((out.R - R_true).abs().max()) > 1e-4, (
        "if the rotor were exactly identified this test's premise is wrong")
    rcv = R.evaluate_arms(canvas, X, canvas.translate(R.apply_rotor(X, R_true), 2, 3))
    assert rcv["arms"]["treatment"]["slot_space_spectrum"]["effective_rank_1e-2"] < 8


def test_rotor_det_correction_forces_proper_rotation():
    """A reflection optimum MUST be corrected to the nearest rotation.

    MUTATION GATE (b): dropping the det<0 correction leaves a reflection here.
    """
    g = torch.Generator().manual_seed(41)
    A = torch.randn(256, 8, generator=g)
    refl = torch.eye(8)
    refl[0, 0] = -1.0                                   # det = -1
    B = A @ refl
    bad = R.orthogonal_procrustes(A, B, enforce_rotation=False)
    assert bad.det_raw < 0.0, "the uncorrected optimum is expected to be a reflection"
    assert bad.det_applied < 0.0

    good = R.orthogonal_procrustes(A, B, enforce_rotation=True)
    assert good.det_raw < 0.0, "det_raw must report the UNCORRECTED determinant"
    assert good.det_correction_applied is True, "the det<0 correction did not fire"
    assert abs(good.det_applied - 1.0) < 1e-5, (
        f"rotor is a reflection, not a rotation: det={good.det_applied}")
    assert good.orthogonality_error < 1e-5
    assert abs(float(torch.det(good.R)) - 1.0) < 1e-5


# ================================================================== 3. mask
def test_mask_recovery_exact_on_known_support(ctx):
    """Mask recovery is EXACT on a known support (unit test of estimate_mask).

    Construction (deterministic, 2 demos): coherence is 1 on row-block set A, 0 on
    set B (both with full energy -- B is demo-1 sign-flipped), and a quarter of the
    rows are zeroed in BOTH views (dead -> never on the support). The recovered
    support must be exactly A minus the dead rows.
    """
    canvas = ctx["canvas"]
    nb = canvas.geo.num_blocks
    g = torch.Generator().manual_seed(17)
    base = torch.complex(torch.randn(nb, 4, generator=g), torch.randn(nb, 4, generator=g))
    base = base / (base.abs().pow(2).sum(-1, keepdim=True).sqrt() + 1e-9)
    u = torch.stack([base, base])
    a = u.clone()
    rows = lambda lo, hi: (torch.arange(nb).reshape(-1, 1).expand(nb, 4) >= lo) & \
                          (torch.arange(nb).reshape(-1, 1).expand(nb, 4) < hi)
    A_set = rows(0, nb // 2)                   # 50% rows: coherent
    dead = rows(nb // 2, 3 * nb // 4)          # 25% rows: zeroed in both views
    a[1][rows(3 * nb // 4, nb)] *= -1.0        # 25% rows: anti-phase in demo 1
    u = u.clone()
    u[:, dead] = 0.0
    a[:, dead] = 0.0

    mask = R.estimate_mask(u, a)
    expected = A_set & ~dead
    assert mask.flat_distribution is False
    assert mask.live_fraction == pytest.approx(0.75, abs=0.01)
    assert mask.support[dead].sum().item() == 0, "dead channels are on the support"
    assert mask.support[~A_set].sum().item() == 0, "incoherent channels are on the support"
    assert torch.equal(mask.support, expected), (
        f"mask not recovered exactly: {int((mask.support != expected).sum())} channels wrong")

    # and it actually gates: a wave living only on dead channels composes to zero
    op = R.TripartiteOperator(
        canvas=canvas, dx=1, dy=2, mask=mask,
        rotor=R._rotor_from_axis_angle(3, 0.3))
    only_dead = torch.zeros(1, nb, 8)
    out = op.compose_linear(only_dead)
    assert float(out.abs().max()) == 0.0, "mask did not remove the dead channels"


def test_mask_flat_distribution_selects_no_channel_by_noise(ctx):
    """A live coherence distribution inside float noise must not select by noise."""
    canvas = ctx["canvas"]
    X = _waves(canvas, 3)
    mask = R.estimate_mask(R.to_complex(X), R.to_complex(X))
    assert mask.flat_distribution is True
    # it degenerates to the LIVE set (drops what carries no energy) and never
    # removes a live channel on a numerical difference
    assert mask.mass == pytest.approx(mask.live_fraction)
    assert mask.live_fraction < 1.0
    assert int(mask.support.sum().item()) == int(round(mask.live_fraction * mask.support.numel()))
    # a genuinely spread coherence must NOT degenerate: quantile path is taken
    X2 = _waves(canvas, 2)
    u2 = R.to_complex(X2)
    a2 = u2.clone()
    nbb = canvas.geo.num_blocks
    a2[1, nbb // 2:] = -a2[1, nbb // 2:]     # demo 1 anti-phase on half the blocks
    spread = R.estimate_mask(u2, a2)
    assert spread.flat_distribution is False
    assert 0.0 < spread.threshold < 1.0
    assert int(spread.support.sum().item()) < int(spread.support.numel())


# =========================================================== 4. composition
def _hand_operator(canvas: R.TorusCanvas, dx=3, dy=2) -> R.TripartiteOperator:
    nb = canvas.geo.num_blocks
    idx = torch.arange(nb * 4).reshape(nb, 4)
    support = ((idx + idx % 3) % 2 == 0)
    return R.TripartiteOperator(
        canvas=canvas, dx=dx, dy=dy,
        mask=R.TopologicalMask(support=support, coherence=support.float(),
                               threshold=0.5, quantile=0.5),
        rotor=R._rotor_from_axis_angle(23, 0.7))


def _small_op_canvas() -> R.TorusCanvas:
    return R.TorusCanvas(num_blocks=64, modulus=8)


def test_composition_equals_product_of_factors(ctx):
    """compose_linear == T_dense @ Pi_dense @ R_dense, and compose == normalize(linear)."""
    canvas = _small_op_canvas()
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, canvas.geo.num_blocks, 8, generator=g)
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    op = _hand_operator(canvas)
    dense = op.dense_product()
    lin = op.compose_linear(x).reshape(-1)
    err = float((lin - dense @ x.reshape(-1)).abs().max())
    assert err < 1e-5, f"compose_linear != dense product ({err})"
    assert float((op.compose(x) - R.to_real(R.to_complex(op.compose_linear(x)))).abs().max()) == 0.0


def test_composition_order_is_pinned(ctx):
    """The documented order must DIFFER from the reversed order (with a real mask)."""
    canvas = _small_op_canvas()
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, canvas.geo.num_blocks, 8, generator=g)
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    op = _hand_operator(canvas)
    assert 0.0 < op.mask.mass < 1.0, "order is unobservable with a trivial mask"
    diff = float((op.compose_reversed_order(x) - op.compose(x)).abs().max())
    assert diff > 1e-6, "the reversed factor order produced the same operator"
    assert op.factors() == ("T_(dx,dy)", "Pi_mask", "R_Clifford")
    assert op.receipt()["order"].startswith("T_(dx,dy) * Pi_mask * R_Clifford")


# ======================================================== 5. gate reporting
def test_gate_reports_both_epsilons_and_used_is_calibrated(ctx):
    gate = ctx["gate"]
    rec = gate.receipt()
    assert rec["epsilon_hardcoded"] == HARDCODED_EPSILON == 0.0431
    assert "epsilon_calibrated" in rec
    assert rec["epsilon_used"] == pytest.approx(gate.epsilon_used)
    assert rec["epsilon_used"] != rec["epsilon_hardcoded"], (
        "the gate is running on the hardcoded epsilon")
    assert rec["epsilon_used_source"] in ("calibrated", "floor-clipped-calibrated")
    for key in ("accept_rate_same_origin_at_calibrated",
                "accept_rate_different_origin_at_calibrated",
                "accept_rate_same_origin_at_hardcoded",
                "accept_rate_different_origin_at_hardcoded"):
        assert key in rec, key
    assert rec["n_different_origin"] > 0, "no negative control population"
    assert rec["doc_reported_self_stress"] == pytest.approx(0.993424)
    assert rec["hardcoded_rejects_doc_reported_self_stress"] is True


def test_calibrated_and_hardcoded_are_functionally_distinguishable():
    """A population straddling 0.0431 must give the two thresholds DIFFERENT accept rates."""
    g = torch.Generator().manual_seed(3)
    v = torch.randn(256, generator=g).to(torch.complex64)
    v = v / v.abs().pow(2).sum().sqrt()
    orth = torch.randn(256, generator=g).to(torch.complex64)
    orth = orth - v * (torch.vdot(v, orth))
    orth = orth / orth.abs().pow(2).sum().sqrt()

    def pair(stress: float):
        cos = 1.0 - stress
        ang = math.acos(max(-1.0, min(1.0, cos)))
        p = math.cos(ang) * v + math.sin(ang) * orth
        return (p, v)

    same = [pair(0.02), pair(0.03)] + [pair(0.10), pair(0.11)]
    diff = [pair(0.30), pair(0.40)]
    gate = R.gate_from_pair_populations(same, diff, source="straddle")
    rec = gate.receipt()
    hard = rec["accept_rate_same_origin_at_hardcoded"]
    cal = rec["accept_rate_same_origin_at_calibrated"]
    assert hard == 0.5, f"hardcoded accept rate {hard} (expected the 2 sub-0.0431 pairs only)"
    assert cal > hard, (
        f"hardcoded and calibrated thresholds accept the same set ({hard} vs {cal})")
    assert cal >= 0.75, cal
    assert rec["epsilon_calibrated"] > HARDCODED_EPSILON
    assert rec["calibration_verdict"] == "NON_VACUOUS"


def test_calibration_vacuity_check_fires_on_non_separating_pairs():
    """If the negative population is not separated, the calibration must say VACUOUS.

    The negative population is constructed to be INDISTINGUISHABLE from the
    positive one (both consist of exact self-pairs, stress 0). A gate that accepts
    every negative pair is a gate that separates nothing, and the calibration must
    report VACUOUS rather than a pass.
    """
    canvas = _canvas()
    ws = [R.to_complex(_waves(canvas, 4)[i]).reshape(-1) for i in range(4)]
    same = [(w, w) for w in ws]
    diff = [(w, w) for w in ws]          # label differs, content does not
    gate = R.gate_from_pair_populations(same, diff, source="non-separating")
    rec = gate.receipt()
    assert rec["n_different_origin"] > 0
    assert rec["calibration_verdict"] == "VACUOUS", rec["calibration_verdict"]
    assert rec["non_vacuous"] is False
    assert rec["accept_rate_different_origin_at_calibrated"] > 0.0
    assert rec["separation_margin"] is not None and rec["separation_margin"] <= 0.0
    assert rec["epsilon_calibrated"] == pytest.approx(0.0, abs=1e-6)


def test_calibration_separates_on_the_fitted_operator_population(ctx):
    """The gate the relaxation actually uses IS non-vacuous on its own population."""
    rec = ctx["gate"].receipt()
    assert rec["calibration_verdict"] == "NON_VACUOUS"
    assert rec["non_vacuous"] is True
    assert rec["accept_rate_different_origin_at_calibrated"] == 0.0
    assert rec["separation_margin"] > 0.1


def test_raw_view_population_readout_reproduces_the_documented_trap(ctx):
    """The readout must report the hardcoded self-veto AND the calibrated vacuity."""
    canvas = ctx["canvas"]
    sol = ctx["sol"]
    out = R.raw_view_population_readout(canvas, sol["X"], sol["Y"], label="t")
    assert out["readout_only"] is True
    # R-1: the hardcoded epsilon vetoes genuine same-origin view pairs.
    assert out["accept_rate_same_origin_at_hardcoded"] < 0.5
    # R-2: the calibrated epsilon on the same population separates nothing.
    assert out["accept_rate_different_origin_at_calibrated"] == 1.0
    assert out["calibration_verdict"] == "VACUOUS"


# ======================================================== 6. relaxation
def test_convergence_on_solvable_instance(ctx):
    """Exactly-solvable instance: converges, cap NOT hit, and on the calibrated epsilon.

    MUTATION GATE (a): if the relaxation switches to the hardcoded epsilon the
    receipt reports a different epsilon_used than the gate's.
    """
    canvas, sol, gate = ctx["canvas"], ctx["sol"], ctx["gate"]
    rel = R.TripartiteResonator(canvas, gate, max_iters=8).relax(sol["X"], sol["Y"])
    rec = rel.receipt()
    assert rel.converged is True, rec["reasons"]
    assert rel.cap_hit is False
    assert rel.n_iterations <= 8
    assert rec["epsilon_used"] == pytest.approx(gate.epsilon_used), (
        "the relaxation is not iterating against the gate's calibrated epsilon")
    assert rec["epsilon_used"] != pytest.approx(HARDCODED_EPSILON)
    assert rec["epsilon_hardcoded"] == pytest.approx(HARDCODED_EPSILON)
    assert rec["epsilon_calibrated"] == pytest.approx(gate.calibrated_epsilon)
    assert rec["convergence_status"] == "CONVERGED"
    # trajectory must be reported and must show the stress falling
    traj = rec["trajectory"]
    assert len(traj) >= 2
    assert traj[0]["iteration"] == 0 and traj[-1]["iteration"] == rel.n_iterations
    assert traj[-1]["sagnac_stress_mean"] < traj[0]["sagnac_stress_mean"]
    for step in traj:
        for key in ("dx", "dy", "sagnac_stress_mean", "residual_fro_total",
                    "mask_mass", "rotor_orthogonality_error"):
            assert key in step, key
    assert (traj[-1]["dx"], traj[-1]["dy"]) == (sol["ground_truth"]["dx"],
                                                sol["ground_truth"]["dy"])
    assert traj[-1]["rotor_orthogonality_error"] < 1e-5
    assert abs(traj[-1]["rotor_det"] - 1.0) < 1e-5
    assert rec["verdict"] != R.STATUS_VOID_NONCONVERGENT


def test_nonconvergence_on_forced_impossible_instance_is_honest(ctx):
    """Forced-impossible instance: cap honored, verdict non-convergent, NO exception.

    MUTATION GATE (d): if the iteration cap is not enforced, n_iterations exceeds
    the requested cap.
    """
    canvas, gate = ctx["canvas"], ctx["gate"]
    base = _waves(canvas, 1)[0]
    scene = R.scene_impossible(canvas, base)
    rel = R.TripartiteResonator(canvas, gate, max_iters=5).relax(scene["X"], scene["Y"])
    rec = rel.receipt()
    assert rel.converged is False
    assert rel.cap_hit is True
    assert rel.max_iters == 5
    assert rel.n_iterations == 5, f"cap not honored: {rel.n_iterations} iterations"
    assert rel.n_iterations <= 5
    assert rec["convergence_status"] == "NON_CONVERGENT_ITERATION_CAP_HIT"
    assert rec["verdict"] == R.STATUS_VOID_NONCONVERGENT
    assert any("cap" in r for r in rec["reasons"])
    # the last trajectory point is a real measurement, not a fabricated success
    assert rec["trajectory"][-1]["sagnac_stress_mean"] > rec["epsilon_used"]


def test_degenerate_self_consistency_is_flagged_and_floored(ctx):
    """When the fit is exact the calibrated epsilon collapses; the floor binds and is reported."""
    gate = ctx["gate"]
    rec = gate.receipt()
    assert gate.calibrated_epsilon <= 1e-3
    assert gate.epsilon_used >= R.TRIPARTITE_EPS_FLOOR
    assert rec["epsilon_clipped_by_floor"] is True
    assert gate.epsilon_used == pytest.approx(R.TRIPARTITE_EPS_FLOOR)
    # `self_consistency_degenerate` is DEFINED as calibrated <= 0; assert the invariant
    assert rec["self_consistency_degenerate"] == (rec["epsilon_calibrated"] <= 0.0)
    assert rec["epsilon_used_source"] == "floor-clipped-calibrated"
    assert rec["epsilon_calibrated"] < rec["epsilon_hardcoded"]
    assert rec["epsilon_used"] < rec["epsilon_hardcoded"]
    # the FLOOR still rejects garbage: a different-origin pair is far above it
    assert min(rec["different_origin_stress"]) > R.TRIPARTITE_EPS_FLOOR


# ============================================================= 7. arms
def test_arms_are_all_scored_on_the_held_out_pair(ctx):
    canvas, sol = ctx["canvas"], ctx["sol"]
    arms = R.evaluate_arms(canvas, sol["X"], sol["Y"], hold_out_index=-1)
    got = arms["arms"]
    for name in ("treatment", "control_diag_ls", "control_identity",
                 "control_shuffled", "control_random_direction"):
        assert name in got, name
        assert isinstance(got[name]["held_out_cos"], float)
        assert got[name]["prediction_sha256"]
    assert arms["hold_out_index"] == sol["X"].shape[0] - 1
    assert got["treatment"]["role"] == "treatment"
    assert got["control_diag_ls"]["role"] == "control"
    assert "per_slot_diagonal_ridge_ls" in got["control_diag_ls"]["operator_family"]


def test_shuffled_control_does_not_beat_treatment(ctx):
    """The shuffled-demonstration arm must NOT beat the real arm."""
    canvas, sol = ctx["canvas"], ctx["sol"]
    arms = R.evaluate_arms(canvas, sol["X"], sol["Y"])
    treat = arms["treatment_held_out_cos"]
    shuf = arms["arms"]["control_shuffled"]["held_out_cos"]
    assert shuf < treat - 0.1, f"shuffled arm {shuf} vs treatment {treat}"
    assert arms["arms"]["control_shuffled"]["permutation"] is not None
    assert all(i != p for i, p in enumerate(arms["arms"]["control_shuffled"]["permutation"]))
    assert arms["verdict"] == R.STATUS_OK
    assert arms["arms_not_separated_by_treatment"] == []


def test_identity_control_behaves_as_identity(ctx):
    """The identity arm's prediction must be byte-identical to the held-out INPUT.

    MUTATION GATE (c): if the identity arm is silently swapped for the treatment,
    this digest comparison fails.
    """
    canvas, sol = ctx["canvas"], ctx["sol"]
    arms = R.evaluate_arms(canvas, sol["X"], sol["Y"])
    ida = arms["arms"]["control_identity"]
    assert ida["prediction_is_held_out_input_exact"] is True
    held_in = R.to_complex(sol["X"][-1]).reshape(-1)
    assert ida["prediction_sha256"] == R._sha256_tensor(held_in), (
        "identity arm is not predicting the held-out input")
    assert ida["held_out_cos"] != arms["arms"]["treatment"]["held_out_cos"]
    assert ida["prediction_sha256"] != arms["arms"]["treatment"]["prediction_sha256"]


def test_verdict_is_measurement_driven_not_unconditional(ctx):
    """Same machinery, two ground truths -> two DIFFERENT verdicts.

    On the solvable instance the treatment wins; on the identity-truth instance the
    identity control is at or above the treatment and the verdict must be VOID. If
    the verdict were printed unconditionally, one of these two fails.
    """
    canvas, sol = ctx["canvas"], ctx["sol"]
    good = R.evaluate_arms(canvas, sol["X"], sol["Y"])
    assert good["verdict"] == R.STATUS_OK
    assert good["closest_control_margin"] > 0.1

    ident = R.scene_identity_truth(canvas, sol["X"])
    bad = R.evaluate_arms(canvas, ident["X"], ident["Y"])
    assert bad["verdict"] == R.STATUS_VOID_CONTROL, bad["arms_not_separated_by_treatment"]
    assert "control_identity" in bad["arms_not_separated_by_treatment"]
    assert bad["controls_held_out_cos"]["control_identity"] >= \
        bad["treatment_held_out_cos"] - R.TIE_TOL


def test_identity_truth_scene_is_a_tie_at_production_geometry():
    """At the production geometry the identity-truth tie is measured, not decided by noise.

    The treatment must fit exactly the identity map, so on the identity scene it
    ties the identity control to within the score's numerical resolution.
    """
    prod = R.TorusCanvas(num_blocks=2048, modulus=32)
    X = _waves(prod, 4)
    ident = R.scene_identity_truth(prod, X)
    arms = R.evaluate_arms(prod, ident["X"], ident["Y"])
    assert arms["verdict"] == R.STATUS_VOID_CONTROL
    gap = arms["treatment_held_out_cos"] - arms["controls_held_out_cos"]["control_identity"]
    assert abs(gap) <= R.TIE_TOL, f"identity-truth gap {gap} is not a tie"
    assert arms["arms"]["treatment"]["operator"]["mask"]["flat_distribution"] is True


# ======================================================== 8. determinism
def _strip(obj):
    """Drop nothing but wall-clock fields if any ever appear."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items()
                if "timestamp" not in k.lower() and "elapsed" not in k.lower()
                and "runtime" not in k.lower()}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def test_determinism_same_inputs_identical_receipt(ctx):
    canvas, sol, gate = ctx["canvas"], ctx["sol"], ctx["gate"]
    a = R.TripartiteResonator(canvas, gate, max_iters=6).relax(sol["X"], sol["Y"]).receipt()
    b = R.TripartiteResonator(canvas, gate, max_iters=6).relax(sol["X"], sol["Y"]).receipt()
    assert _strip(a) == _strip(b)
    aa = R.evaluate_arms(canvas, sol["X"], sol["Y"])
    bb = R.evaluate_arms(canvas, sol["X"], sol["Y"])
    assert _strip(aa) == _strip(bb)


def test_selfcheck_receipt_shape_and_non_claims():
    """--selfcheck must emit the required machine-readable fields and the non-claims."""
    rec = R.run_selfcheck(num_blocks=1024, modulus=32, max_iters=4,
                          arc_root=os.environ.get("ARC_CORPUS"))
    for key in ("schema_id", "geometry", "translation_invariants",
                "composition_exactness", "reference_gate", "scenarios", "verdict",
                "epsilon_hardcoded_vs_calibrated", "non_claims", "config"):
        assert key in rec, key
    assert rec["schema_id"] == R.SCHEMA_ID
    assert rec["non_claims"] and any("INSTRUMENT" in s for s in rec["non_claims"])
    assert rec["verdict"]["instrument_only"] is True
    assert rec["verdict"]["capability_claim"] is False
    assert rec["verdict"]["branch"] in (
        "INSTRUMENT_VALID_NO_CAPABILITY_CLAIM", R.STATUS_VOID_CONTROL,
        R.STATUS_VOID_CALIBRATION, R.STATUS_VOID_NONCONVERGENT)
    assert "solvable" in rec["scenarios"]
    assert "identity_truth" in rec["scenarios"]
    assert "impossible" in rec["scenarios"]
    assert rec["translation_invariants"]["adjoint_exact"] == "8/8"
    # the JSON round-trips (machine-readable)
    import json
    assert json.loads(json.dumps(rec, default=str))["schema_id"] == R.SCHEMA_ID
