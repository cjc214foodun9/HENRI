"""Readout contract: the wave->observable direction must exist and be honest.

WHY THIS FILE EXISTS
    `evaluate_test_time_plan` derived its Sagnac stress by comparing a proposed
    wavefront against `zone_c_axioms` -- RANDOM unit phasors. Measured pass rate
    for that construction at tau=0.35 is 0.0000 with mean stress 0.9742
    (readout_probe_observed.json, gate G). A veto that passes 0 of 300 correct
    predictions is not a calibration error; the DECODE DIRECTION WAS MISSING, so
    the stress carried no information about the proposal.

    `henri_wave_readout.py` supplies that direction. These tests hold it in place.

WHAT IS ENFORCED
    R1  decode is exact on held-out observable draws, with positive margin
    R2  observation-space stress separates from the internal latent measure:
        a wavefront that the INTERNAL measure vetoes must PASS observationally
        when it decodes to the reference. Without R2 the readout could be a no-op.
    R3  a corrupted observable is rejected
    R4  zero-energy input fails closed PER SAMPLE with no NaN
    R5  a zero row does not corrupt its neighbours in the same batch
        (the reference design used ONE global energy test for the whole batch)
    R6  codebook buffers are never rebound during decode
    R7  the legacy random-axiom veto is reproducible and essentially never passes
        (kept as evidence that the old measure was un-passable by construction)
    R8  REGRESSION: decode is batch-size invariant
    R9  REGRESSION: encode binds each slot to ITS OWN role
    R10 continuous comparisons are dimension-normalized
    R11 the module exposes no random-axiom path under a production-looking name

    R8 and R9 are regression guards for two SILENT bugs found while writing this
    module, each of which produced plausible-looking wrong numbers rather than an
    error:
      * `matmul([B,S,D], [S,D,V])` broadcast at B=1 into [S,S,V] and returned a
        wrong argmax. Now `einsum("bsd,svd->bsv")`.
      * `value_codebook.gather(1, slot_values)` used the BATCH index on the ROLE
        axis, so slots were bound to role-0's values. Now advanced indexing with
        an explicit per-slot role index.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from henri_wave_readout import WaveObservationReadout  # noqa: E402

DIM, SLOTS, VALUES = 8192, 8, 16
TAU_VETO = 0.35
# Measured baselines (readout_probe_observed.json) recorded so a drift in these
# numbers is visible as a test failure, not re-discovered later.
MEASURED_SINGLE_RT_COS = 0.713772
MEASURED_EXACT_DECODE_RATE = 1.0
MEASURED_DECODE_MIN_MARGIN = 0.29


@pytest.fixture(scope="module")
def ro() -> WaveObservationReadout:
    return WaveObservationReadout(dim=DIM, n_slots=SLOTS, n_values=VALUES, seed=0)


# ------------------------------------------------------------------ R1
def test_decode_is_exact_on_held_out_draws(ro):
    g = torch.Generator().manual_seed(1234)
    exact = 0
    min_margin = float("inf")
    for _ in range(100):
        vals = torch.randint(0, VALUES, (SLOTS,), generator=g)
        idx, _, marg = ro.decode(ro.encode(vals), return_margins=True)
        if bool((idx[0] == vals).all()):
            exact += 1
        min_margin = min(min_margin, float(marg.min().item()))
    assert exact / 100 >= 0.99, f"decode exactness {exact}/100 below 0.99"
    assert min_margin > 0.0, (
        f"decode margin {min_margin:.3e} not positive; the snap is not decisive")


# ------------------------------------------------------------------ R2
def test_observable_stress_separates_from_the_internal_latent_measure(ro):
    """The load-bearing test: internal says mismatch, observable says match."""
    vals = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    clean = ro.encode(vals)
    alt = ro.encode_with_orthogonal_content(vals, alpha=3.0, seed=777)

    internal_cos = float(torch.abs((clean[0].conj() * alt[0]).sum()).item())
    internal_stress = 1.0 - internal_cos
    obs = ro.delta_sagnac_observational(alt, vals)
    obs_stress = float(obs["stress"][0].item())

    assert internal_stress > TAU_VETO, (
        f"internal stress {internal_stress:.4f} did not exceed tau; the control is "
        f"not far enough from the clean wavefront to demonstrate separation")
    assert obs_stress <= TAU_VETO, (
        f"observable stress {obs_stress:.4f} exceeded tau; the readout did not "
        f"recover the observable, so it cannot fix the un-passable veto")


# ------------------------------------------------------------------ R3
def test_corrupted_observable_is_rejected(ro):
    vals = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    corrupt = (vals + 1) % VALUES
    obs = ro.delta_sagnac_observational(ro.encode(vals), corrupt)
    assert float(obs["stress"][0].item()) > TAU_VETO, (
        "a corrupted observable passed the veto")


# ------------------------------------------------------------------ R4
def test_zero_energy_input_fails_closed_without_nan(ro):
    vals = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    zero = torch.zeros(1, DIM, dtype=torch.complex64)
    obs = ro.delta_sagnac_observational(zero, vals)
    assert float(obs["stress"][0].item()) == 1.0
    assert not bool(obs["valid"][0].item())
    assert not bool(torch.isnan(obs["stress"]).any().item())


# ------------------------------------------------------------------ R5
def test_zero_row_does_not_corrupt_neighbours_in_batch(ro):
    vals = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    psi = ro.encode(vals)
    mixed = torch.stack([psi[0], torch.zeros(DIM, dtype=torch.complex64), psi[0]])
    ref = torch.stack([vals[0], vals[0], vals[0]])
    obs = ro.delta_sagnac_observational(mixed, ref)
    s = [float(x) for x in obs["stress"]]
    assert s[0] <= TAU_VETO and s[2] <= TAU_VETO, (
        f"good rows were corrupted by a zero neighbour: stresses {s}")
    assert s[1] == 1.0, f"zero row did not fail closed: {s}"


# ------------------------------------------------------------------ R6
def test_codebook_buffers_are_not_rebound_during_decode(ro):
    vals = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    psi = ro.encode(vals)
    ids = (id(ro.role_codebook), id(ro.value_codebook))
    r_before = ro.role_codebook.clone()
    for _ in range(5):
        ro.decode(psi)
    assert (id(ro.role_codebook), id(ro.value_codebook)) == ids
    assert bool(torch.equal(r_before, ro.role_codebook)), "role codebook mutated"


# ------------------------------------------------------------------ R7
def test_legacy_random_axiom_veto_is_reproducible_and_unpassable(ro):
    g = torch.Generator().manual_seed(4242)
    n = 60
    passed = 0
    stresses = []
    for _ in range(n):
        v = torch.randint(0, VALUES, (SLOTS,), generator=g)
        st = float(ro.legacy_random_axiom_stress(ro.encode(v))[0].item())
        stresses.append(st)
        if st <= TAU_VETO:
            passed += 1
    mean_stress = sum(stresses) / n
    assert mean_stress > 0.90, (
        f"legacy random-axiom mean stress {mean_stress:.4f} is not near 1.0; the "
        f"'un-passable by construction' evidence no longer reproduces")
    assert passed <= 0.05 * n, (
        f"legacy random-axiom veto passed {passed}/{n}; it was supposed to be "
        f"un-passable. If this ever passes, the readout fix is unnecessary "
        f"and this whole contract must be re-derived.")


# ------------------------------------------------------------------ R8
def test_decode_is_batch_size_invariant(ro):
    """REGRESSION: matmul broadcast at B=1 and returned a wrong argmax."""
    vals = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                         [8, 7, 6, 5, 4, 3, 2, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0]])
    psi = ro.encode(vals)
    idx_batch, _ = ro.decode(psi)
    for i in range(vals.shape[0]):
        idx_single, _ = ro.decode(psi[i:i + 1])
        assert bool((idx_single[0] == idx_batch[i]).all()), (
            f"row {i}: single decode {idx_single[0].tolist()} != batched "
            f"{idx_batch[i].tolist()}. A broadcasting contraction is back.")


# ------------------------------------------------------------------ R9
def test_encode_binds_each_slot_to_its_own_role(ro):
    """REGRESSION: gather used the batch index on the role axis."""
    base = torch.zeros(1, SLOTS, dtype=torch.long)
    idx_base, _ = ro.decode(ro.encode(base))
    assert bool((idx_base[0] == base[0]).all()), (
        "all-equal-value wavefront did not decode; role indexing is wrong")

    for slot in range(SLOTS):
        v = base.clone()
        v[0, slot] = 5
        idx, _ = ro.decode(ro.encode(v))
        changed = [s for s in range(SLOTS) if int(idx[0, s]) != int(idx_base[0, s])]
        assert changed == [slot], (
            f"changing slot {slot} changed decoded slots {changed}; slots are not "
            f"bound to independent roles")


# ------------------------------------------------------------------ R10
def test_continuous_comparison_is_dimension_normalized(ro):
    a = torch.complex(torch.ones(1, 64), torch.zeros(1, 64))
    b = torch.complex(torch.zeros(1, 64), torch.zeros(1, 64))
    d = float(WaveObservationReadout.dimension_normalized_l2(a, b).item())
    assert d == pytest.approx(1.0, abs=1e-6), (
        f"normalized L2 is {d}, expected 1.0 (||a-b||_2/sqrt(d)). Reusing a raw L2 "
        f"threshold across dimensions is the dimension-blindness fallacy.")


# ------------------------------------------------------------------ R11
def test_module_has_no_production_named_random_axiom_path():
    """A random-axiom veto must not be reachable under a neutral name.

    Checked on the SYNTAX TREE, not by substring search. A substring search counts
    the module docstring's legitimate quotation of `zone_c_axioms` while explaining
    the defect -- which is the SAME measurement error made earlier in this project,
    when a string-count invariant counted a docstring as answer-coupling. Prose is
    not code, and only the parse tree can tell them apart.
    """
    import ast as _ast

    import henri_wave_readout as m
    tree = _ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    banned = {"zone_c_axioms", "axiom_veto"}
    hits = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name) and node.id in banned:
            hits.append((node.id, node.lineno))
        elif isinstance(node, _ast.Attribute) and node.attr in banned:
            hits.append((node.attr, node.lineno))
    assert not hits, (
        f"{hits} referenced as CODE; a random-axiom veto must stay quarantined as "
        f"reproducible evidence (legacy_random_axiom_stress), never wired as "
        f"production. Docstring mentions are correctly ignored.")
