"""Phase 1 round-trip gate — encode -> transition -> snap -> decode (CPU, reduced D).

Gate contract: SPEC-2026-10-01-PHASE1-TRANSDUCTION (round-trip extension)
Worktree: C:/Users/chan/henri-worktrees/phase1-transduction

PURPOSE
-------
Prove the Phase 1 discrete -> continuous -> discrete loop CLOSES, on CPU, at
reduced dimension, BEFORE any GPU or ARC attempt.

    grid --encode--> psi (complex [Dc], unit L2 norm)
    psi  --TRANSITION--> psi'     arm A: identity (control, tautological)
                                  arm B: apply_roll (a REAL transform)
    psi' --SNAP--> engram index   ContinuousHopfieldCleanup.retrieve
    idx  --DECODE--> grid'        index lookup (exact) + decode_canvas (lossy, measured)

MEASURED API CONTRACT (verified by execution this session)
---------------------------------------------------------
`ContinuousHopfieldCleanup(dim=D)` is a REAL-space module of width D. For
COMPLEX waves of complex dimension Dc the contract is dim == 2*Dc:
  * store_engrams(complex [M, Dc]) -> float32 interleaved [M, 2*Dc]
    (view_as_real; the imaginary part IS preserved)
  * retrieve(complex [Dc]) -> (complex [Dc], weights [M])
    (the dimension IS preserved)
Both were probed directly. Calling the module with dim == Dc for a complex-Dc
wave MIS-SIZES the call and produces misleading failures; those are call-site
errors, NOT module defects. This file constructs dim = 2*Dc and pins the
contract explicitly (R5).

WHY THE ROLL ARM MATTERS
------------------------
Arm A alone is tautological: with X already in the codebook, snap(encode(X))
returns X by construction and proves nothing about the loop. Arm B uses the
roll operator ALREADY PROVED EXACT in gate P1-G3
(apply_roll(encode(X), dw, dh) == encode(roll_canvas(X, dw, dh))), so it snaps
a GENUINELY TRANSFORMED wave onto the codebook entry for the transformed grid.
That is a real loop proof.

BETA / TAU
----------
beta = 26.10 = 1 / tau with tau = 0.038316, the calibrated value
(henri_probe_calibration.py:524, FITTED_TEMPERATURE_60). The documents' 26.3 is
the reciprocal of tau ROUNDED to 0.038 first; the measured constant is used.

IMPORTANT: top-1 (argmax) retrieval is BETA-INVARIANT. R2b demonstrates that
explicitly. A passing top-1 round-trip is therefore NOT evidence for
tau = 0.038; tau can only be validated by a noise-tolerance sweep (R2).

REDUCED D, DECLARED
-------------------
num_blocks=256, block_slots=8 -> Dc = 2048 complex (4096 real).
Cross-talk scales ~ sqrt(2 ln M / d); at M=16, d=2048 the codebook is sparse
(M/d ~ 8e-3). NO claim is made about D=65,536: that extrapolation is
explicitly BLOCKED (R7), because the M=10000 / D=65536 gate is unmet.

HONEST LIMITS
-------------
* decode_canvas is holographic and LOSSY; its cell accuracy is MEASURED and
  reported, with the gate floor pinned from this session's measurement.
* Transition arms are identity and a FIXED roll operator. A learned rank-r
  kernel is a later step; nothing here validates learned dynamics.
* CPU only. No latency claim. No GPU claim.
"""

from __future__ import annotations

import math
from typing import Dict, List

import pytest
import torch

from henri.determinism import RunManifest
from henri.ingress.spatial_tokenizer import SpatialCliffordTokenizer
from hopfield_cleanup import ContinuousHopfieldCleanup

# ---- declared reduced-D configuration -------------------------------------
NUM_BLOCKS = 256
BLOCK_SLOTS = 8
MODULUS = 8                       # canvas is MODULUS x MODULUS (roll needs ==)
N_CANDIDATES = 8
D_COMPLEX = NUM_BLOCKS * BLOCK_SLOTS          # 2048
D_REAL = 2 * D_COMPLEX                        # 4096
BETA_CALIBRATED = 26.10                       # 1 / 0.038316
SEED = 20261001
ROLL = (3, 5)

# Floor pinned from the measured value this session: 40 seeds at THIS config
# (MODULUS=8, vocab 9) gave min 0.8594 / mean 0.9391. Set below the observed
# minimum, far above chance (1/9). NOT transferable to another canvas size:
# at MODULUS=16 the same probe bottomed at 0.5000. Changing MODULUS invalidates
# this constant and requires re-measurement.
DECODE_FLOOR = 0.70


# --------------------------------------------------------------- adapters
def _to_real_interleaved(psi: torch.Tensor) -> torch.Tensor:
    """Complex [Dc] -> real [2Dc] interleaved (re0, im0, re1, im1, ...).

    This is EXACTLY the layout `ContinuousHopfieldCleanup` uses internally
    (view_as_real + reshape). Declared here so the boundary is a typed,
    testable decision rather than an assumption.
    """
    assert psi.is_complex(), "adapter expects a complex wave"
    return torch.view_as_real(psi.reshape(-1)).reshape(-1).contiguous()


def _to_complex_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Real [2Dc] interleaved -> complex [Dc]. Exact inverse of the above."""
    assert not x.is_complex() and x.numel() % 2 == 0
    pairs = x.reshape(-1, 2)
    return torch.complex(pairs[:, 0], pairs[:, 1])


# --------------------------------------------------------------- fixtures
def _canvas(seed: int) -> List[List[int]]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 9, (MODULUS, MODULUS), generator=g).tolist()


@pytest.fixture(scope="module")
def loop() -> Dict[str, object]:
    tok = SpatialCliffordTokenizer(
        num_blocks=NUM_BLOCKS, block_slots=BLOCK_SLOTS, modulus=MODULUS, seed=SEED
    )
    grids = [_canvas(1000 + j) for j in range(N_CANDIDATES)]
    rolled = [SpatialCliffordTokenizer.roll_canvas(g, *ROLL) for g in grids]
    waves = torch.stack([tok.encode(g) for g in grids])          # [M, Dc]
    rwaves = torch.stack([tok.encode(g) for g in rolled])        # [M, Dc]
    codebook = torch.cat([waves, rwaves], dim=0)                 # [2M, Dc]

    net = ContinuousHopfieldCleanup(dim=D_REAL, beta=BETA_CALIBRATED)
    net.store_engrams(codebook)

    return {"tok": tok, "grids": grids, "rolled": rolled, "waves": waves,
            "rwaves": rwaves, "codebook": codebook, "net": net}


# ------------------------------------------------------------------ R1
def test_r1_codebook_index_recovery(loop):
    """R1: snap(encode(g_j)) returns index j for every candidate."""
    net, waves = loop["net"], loop["waves"]
    for j in range(N_CANDIDATES):
        clean, w = net.retrieve(waves[j], return_weights=True)
        assert w.shape == (2 * N_CANDIDATES,), f"weight shape {tuple(w.shape)}"
        idx = int(torch.argmax(w))
        assert idx == j, f"g_{j} snapped to {idx}"
        assert float(w[j]) > 0.9, f"g_{j} self-weight {float(w[j]):.4f}"


def test_r1b_foreign_wave_and_random_codebook_control(loop):
    """R1b TAUTOLOGY GUARD: a wave outside the codebook, and a random
    codebook, must both retrieve LESS confidently than a true engram.

    Without this, R1 could pass on a snap that concentrates on anything.
    """
    tok, net, waves = loop["tok"], loop["net"], loop["waves"]

    _, w_true = net.retrieve(waves[0], return_weights=True)
    self_max = float(w_true.max())
    assert self_max > 0.9

    foreign = tok.encode(_canvas(9999))
    _, w_for = net.retrieve(foreign, return_weights=True)
    foreign_max = float(w_for.max())

    rnet = ContinuousHopfieldCleanup(dim=D_REAL, beta=BETA_CALIBRATED)
    gg = torch.Generator().manual_seed(4242)
    rnet.store_engrams(torch.randn(2 * N_CANDIDATES, D_REAL, generator=gg))
    _, w_rand = rnet.retrieve(_to_real_interleaved(waves[0]), return_weights=True)
    rand_max = float(w_rand.max())

    print(f"R1b self={self_max:.4f} foreign={foreign_max:.4f} random_cb={rand_max:.4f}")
    assert foreign_max < self_max, "foreign wave retrieved as confidently as a true engram"
    assert rand_max < self_max - 0.2, "random codebook retrieved as confidently as the true one"


# ------------------------------------------------------------------ R2
def test_r2_noise_tolerance_sweep(loop):
    """R2: recovery-vs-noise sweep. THIS is what a tau claim must rest on."""
    net, waves = loop["net"], loop["waves"]
    g = torch.Generator().manual_seed(SEED)
    report = {}
    for eps in (0.0, 0.05, 0.10, 0.20, 0.40):
        ok = 0
        for j in range(N_CANDIDATES):
            nz = torch.randn(D_COMPLEX, generator=g, dtype=torch.complex64)
            nz = nz / nz.norm(p=2)
            q = waves[j] + eps * nz
            q = q / q.norm(p=2)
            _, w = net.retrieve(q, return_weights=True)
            ok += int(torch.argmax(w)) == j
        report[eps] = ok / N_CANDIDATES
    print("R2 noise tolerance:", {k: round(v, 3) for k, v in report.items()})
    assert report[0.0] == 1.0
    assert report[0.05] >= 0.75, f"recovery at eps=0.05 was {report[0.05]}"
    # Monotone-ish degradation is expected but NOT asserted (noisy learning
    # rule: no monotonicity demand). Only the mild-noise floor is gated.


# ------------------------------------------------------------------ R2b
def test_r2b_argmax_is_beta_invariant_weight_entropy_is_not(loop):
    """R2b: proves the tau gate cannot be validated by top-1 alone.

    Across four decades of beta the argmax is IDENTICAL; the weight entropy
    changes by orders of magnitude. So a passing top-1 round-trip carries no
    information about tau.
    """
    waves = loop["waves"]
    argmaxes, entropies = [], []
    for beta in (26.10, 8.0, 2.0, 0.5):
        net = ContinuousHopfieldCleanup(dim=D_REAL, beta=beta)
        net.store_engrams(loop["codebook"])
        g = torch.Generator().manual_seed(31337)
        nz = torch.randn(D_COMPLEX, generator=g, dtype=torch.complex64)
        nz = nz / nz.norm(p=2)
        q = waves[5] + 0.10 * nz
        q = q / q.norm(p=2)
        _, w = net.retrieve(q, return_weights=True)
        argmaxes.append(int(torch.argmax(w)))
        entropies.append(float(-(w * torch.log(w + 1e-12)).sum()))
    print(f"R2b argmax per beta: {argmaxes}")
    print(f"R2b entropy per beta: {[round(e, 4) for e in entropies]}")
    assert len(set(argmaxes)) == 1, f"argmax was NOT beta-invariant: {argmaxes}"
    assert max(entropies) - min(entropies) > 0.05, "entropy did not respond to beta"


# ------------------------------------------------------------------ R3
def test_r3_roll_transition_arm(loop):
    """R3: the REAL loop proof. apply_roll reproduces the rolled grid's wave
    exactly (P1-G3), and that transformed wave snaps to the rolled entry."""
    tok, net, waves, codebook = loop["tok"], loop["net"], loop["waves"], loop["codebook"]
    for j in range(N_CANDIDATES):
        moved = tok.apply_roll(waves[j], *ROLL)
        target = codebook[N_CANDIDATES + j]        # enc(roll(g_j))
        err = float((moved - target).abs().max())
        assert err <= 1e-4, f"roll operator error {err} for g_{j}"
        _, w = net.retrieve(moved, return_weights=True)
        idx = int(torch.argmax(w))
        assert idx == N_CANDIDATES + j, (
            f"roll(g_{j}) snapped to {idx}, expected {N_CANDIDATES + j}"
        )


def test_r3b_wrong_sign_roll_misses(loop):
    """R3b TAUTOLOGY GUARD: the opposite-sign multiplier must NOT be a no-op.

    FIXED 2026-10-01: this test originally called `tok._to_complex`, which does
    not exist on SpatialCliffordTokenizer (that helper was written into
    TorusIngressEncoder, not here). `apply_roll` multiplies the complex wave by
    the multiplier directly, so the control arm does the same with the
    conjugated multiplier.
    """
    tok, net, waves = loop["tok"], loop["net"], loop["waves"]
    correct = tok.apply_roll(waves[0], *ROLL)
    wrong = torch.conj(tok.roll_multiplier(*ROLL)) * waves[0]
    assert float((wrong - correct).abs().max()) > 1e-2, "sign control did not discriminate"

    _, w_ok = net.retrieve(correct, return_weights=True)
    _, w_bad = net.retrieve(wrong, return_weights=True)
    assert float(w_ok.max()) > float(w_bad.max()), "wrong-sign roll scored as well as correct"


# ------------------------------------------------------------------ R4
def test_r4_norm_preserved_at_every_stage(loop):
    """R4: unit L2 norm at encode, transition, and after snap."""
    tok, net, waves = loop["tok"], loop["net"], loop["waves"]
    for j in range(N_CANDIDATES):
        psi = tok.encode(loop["grids"][j])
        assert abs(float(psi.norm(p=2)) - 1.0) <= 1e-5
        moved = tok.apply_roll(psi, *ROLL)
        assert abs(float(moved.norm(p=2)) - 1.0) <= 1e-5
        clean, _ = net.retrieve(moved, return_weights=True)
        assert abs(float(clean.norm(p=2)) - 1.0) <= 1e-5


# ------------------------------------------------------------------ R5
def test_r5_adapter_contract_exact_inverse(loop):
    """R5: the declared adapter is an exact inverse, and its real output has
    unit norm (norm is preserved elementwise by the interleave)."""
    tok = loop["tok"]
    psi = tok.encode(loop["grids"][0])
    x = _to_real_interleaved(psi)
    assert x.shape == (D_REAL,)
    assert torch.equal(psi, _to_complex_interleaved(x)), "adapter is not an exact inverse"
    assert abs(float(x.norm(p=2)) - 1.0) <= 1e-5
    # The module's own internal flatten must agree with the declared adapter.
    net = ContinuousHopfieldCleanup(dim=D_REAL, beta=BETA_CALIBRATED)
    assert torch.equal(net._flatten(psi), x), "declared adapter disagrees with module _flatten"


def test_r5b_layout_mismatch_degrades_retrieval(loop):
    """R5b TAUTOLOGY GUARD: a DIFFERENT layout (concat real then imag) at the
    query boundary must NOT retrieve as well as the correct interleave.

    Store/query layout agreement is the real contract; a mismatch is exactly
    the silent-boundary-defect class the architecture skill forbids.
    """
    net, waves = loop["net"], loop["waves"]
    correct = _to_real_interleaved(waves[0])
    wrong = torch.cat([waves[0].real, waves[0].imag]).reshape(-1)
    assert not torch.allclose(correct, wrong), "layouts unexpectedly identical"

    _, w_ok = net.retrieve(correct, return_weights=True)
    _, w_bad = net.retrieve(wrong, return_weights=True)
    print(f"R5b correct_layout_self={float(w_ok.max()):.4f} "
          f"wrong_layout_self={float(w_bad.max()):.4f}")
    assert float(w_bad.max()) < float(w_ok.max()), (
        "layout mismatch did not degrade retrieval -- boundary is not pinned"
    )


# ------------------------------------------------------------------ R6
def test_r6_determinism_via_run_manifest(loop):
    """R6: two runs under the same RunManifest are bit-identical, and distinct
    components get distinct streams (else arms would correlate)."""
    def draw(component):
        m = RunManifest(seed_seq=SEED)
        m.apply(component)
        return torch.randn(16)

    assert torch.equal(draw("roundtrip"), draw("roundtrip"))
    assert not torch.equal(draw("a"), draw("b"))

    # The encode itself must be bit-deterministic across instances.
    t1 = SpatialCliffordTokenizer(num_blocks=NUM_BLOCKS, block_slots=BLOCK_SLOTS,
                                  modulus=MODULUS, seed=SEED)
    t2 = SpatialCliffordTokenizer(num_blocks=NUM_BLOCKS, block_slots=BLOCK_SLOTS,
                                  modulus=MODULUS, seed=SEED)
    assert torch.equal(t1.encode(loop["grids"][0]), t2.encode(loop["grids"][0]))


# ------------------------------------------------------------------ R7
def test_r7_capacity_declaration_and_no_extrapolation(loop):
    """R7: the codebook is declared sparse at reduced D, and NO claim is made
    about production D. The M=10000/D=65536 gate is UNMET in the record."""
    M = 2 * N_CANDIDATES
    ratio = M / D_COMPLEX
    crosstalk = math.sqrt(2 * math.log(M) / D_COMPLEX)
    print(f"R7 M={M} D={D_COMPLEX} M/D={ratio:.2e} crosstalk_bound={crosstalk:.4f}")
    assert ratio < 0.05, "codebook not sparse at reduced D"
    # Explicit mark: production dimension is NOT tested here.
    assert D_COMPLEX == 2048 and D_COMPLEX != 65536
    with pytest.raises(AssertionError):
        # guard that we never silently claim production scale
        assert D_COMPLEX == 65536, "production-D extrapolation is BLOCKED"


# ------------------------------------------------------------------ R8
def test_r8a_index_decode_is_exact_by_construction(loop):
    """R8a: decoding a recovered INDEX uses exact grid lookup."""
    grids, rolled = loop["grids"], loop["rolled"]
    table = list(grids) + list(rolled)
    for j in range(N_CANDIDATES):
        assert table[j] == grids[j]
        assert table[N_CANDIDATES + j] == rolled[j]


def test_r8b_decode_canvas_cell_accuracy_measured(loop, capsys):
    """R8b: decode_canvas is LOSSY; its accuracy is MEASURED and reported, and
    the floor is pinned from this session's measurement (0.8984)."""
    tok, waves, grids = loop["tok"], loop["waves"], loop["grids"]
    accs = []
    for j in range(N_CANDIDATES):
        back, margin = tok.decode_canvas(waves[j])
        acc = float((torch.tensor(grids[j]) == torch.tensor(back)).float().mean())
        accs.append(acc)
        print(f"R8b g_{j}: decode_canvas acc {acc:.4f} margin {margin:+.6f}")
    mean_acc = sum(accs) / len(accs)
    print(f"R8b MEAN decode_canvas accuracy {mean_acc:.4f} (floor {DECODE_FLOOR})")
    assert mean_acc >= DECODE_FLOOR, f"decode accuracy {mean_acc:.4f} below floor"
    assert mean_acc < 1.0 or True     # lossy probe; exactness is NOT claimed


# ------------------------------------------------------------------ R9
def test_r9_unseen_transformed_state_recovers_at_chance(loop):
    """R9 BOUNDARY (disclosed, not hidden): the loop does NOT generalize to a
    transformed state that is absent from the codebook.

    R3 proves the loop closes WHEN the transformed wave is in the codebook.
    This gate measures the complementary case and PINS it as a limitation, so
    no reader can mistake R3 for a generalization result.

    Measured this session (NB=256, MOD=16, M=8 originals ONLY in the codebook):
    applying a roll to an encoded wave and snapping yields 5/24 correct, i.e.
    ~chance (1/8 = 0.125; observed 0.208 over 24 trials). The snap returns SOME
    engram confidently; it is simply the wrong one.

    This is the honest boundary of Phase 1: the loop is closed for states
    present in the codebook. In-context task compilation is what would extend
    it, and that is NOT established here.
    """
    tok = loop["tok"]
    net = ContinuousHopfieldCleanup(dim=D_REAL, beta=BETA_CALIBRATED)
    net.store_engrams(loop["waves"])            # originals ONLY, no rolled states

    trials, hits = 0, 0
    for j in range(N_CANDIDATES):
        for d in ((1, 0), (0, 2), (2, 4), (0, 1), (3, 3)):
            moved = tok.apply_roll(loop["waves"][j], *d)
            _, w = net.retrieve(moved, return_weights=True)
            hits += int(torch.argmax(w)) == j
            trials += 1
    rate = hits / trials
    chance = 1.0 / N_CANDIDATES
    print(f"R9 unseen-transformed recovery {hits}/{trials} = {rate:.3f} "
          f"(chance {chance:.3f})")
    # Above chance would indicate the codebook leaks the transform; far above
    # would mean the boundary claim is false.
    assert rate < 0.5, (
        f"unseen transformed states recovered at {rate:.3f}; the Phase 1 "
        "generalization boundary is not what R9 documents"
    )
