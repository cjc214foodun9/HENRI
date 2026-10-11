"""Discriminating tests for henri_attributable_veto.

TWO WAVE FAMILIES -- AND WHY THEY MUST NOT BE CONFUSED
    The shipped Sagnac statistic is  S = |mean(conj(a) * b)|  over VECTOR ENTRIES.
    Its measured chance level on unrelated waves is 0.011049 at D = 8192, which matches
    sqrt(pi/(4D)) for a family with UNIT-MODULUS ENTRIES (|a_i| = 1, so ||a|| = sqrt(D)).

    A vector whose SPECTRUM is unit modulus (|a_hat(k)| = 1) is a DIFFERENT object. Its
    entries have magnitude ~1/sqrt(D), so its vector-domain chance level is
    sqrt(pi/(4 D^3)) = 1.2e-6 at D = 8192 -- FOUR ORDERS OF MAGNITUDE smaller.

    This exact confusion is the hidden premise in the URC spec's Theorem 4.1. My first
    draft of these tests fed the spectrum-unimodular family to a statistic built for the
    entry-unimodular one. Measured effect: A_resonance went NEGATIVE (mean -0.024) and a
    working statistic looked broken. The fix was in the FIXTURE, not the module. One test
    below pins the distinction so the two families cannot be swapped again.

WHAT THE TESTS PIN
    - the shipped statistic SATURATES (fire-rate 1.0 on both classes), reproducing the
      recorded arm2_fixed_axiom.trigger_rate = 1.0
    - the bank-mean reference makes the defect worse, not better
    - the replacement separates on a separable fixture and does NOT on a null fixture
    - V_E catches a single-feature-in-costume score, and the near-collinear case that
      trips it is the AFFINE COPY (r = 1.0), not t vs t^2 (r = 0.967)
    - the frozen bars are the pre-registered values, unmodified
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", ".."))          # tests/unit -> HENRI V2
for p in (V, os.path.join(V, "henri_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from henri_core.henri_attributable_veto import (          # noqa: E402
    AUC_BAR, IDENTITY_R_BAR, REJECT_HI, REJECT_LO,
    AttributableVeto, auc, chance_coherence, chance_max_coherence,
    max_feature_identity_r, sagnac_coherence,
)

D = 2048
N_AX = 16


def _uent(n, gen):
    """ENTRY-UNIMODULAR: |z_i| = 1 for every i, so ||z|| = sqrt(n), NOT 1.

    MY DEFECT, self-caught a third time. The first draft divided by sqrt(n), producing the
    unit-L2-norm family whose chance level is sqrt(pi/(4 D^3)) = 1.2e-6. The shipped
    statistic's chance level is 1/sqrt(D) = 0.011 at D = 8192 and belongs to THIS family.
    Measured reconciliation:
        entry-unimodular  mean 1.0034e-02  rms 1.1266e-02
        receipt constant  0.011049 / sqrt(pi/(4D)) = 1.128426 = 1/sqrt(pi/4)
    The module divides by the NORMS, so it accepts either family; the FIXTURE must supply
    the one the chance constants describe.
    """
    th = torch.rand(n, generator=gen, dtype=torch.float32) * (2 * math.pi)
    return torch.exp(1j * th.to(torch.complex64))                 # |z_i| = 1


def _uspec(n, gen):
    """UNIT-MODULUS SPECTRUM: |z_hat(k)| = 1 for every k. The URC Stage-2 family."""
    phi = torch.rand(n, generator=gen, dtype=torch.float32) * (2 * math.pi)
    z = torch.fft.ifft(torch.exp(1j * phi.to(torch.complex64)))
    return z / z.norm().clamp_min(1e-30)


# ============================================ THE TWO FAMILIES, PINNED AS A FINDING
def test_the_two_wave_families_have_different_chance_levels():
    """PINNED FINDING, and the repair of my own fixture defect.

    `_uent` has |z_i| = 1/sqrt(D) after L2 normalisation, so |mean(conj(a)*b)| sits near
    sqrt(pi/(4D)). `_uspec` has |z_hat(k)| = 1, so its ENTRIES are ~1/sqrt(D) and the
    entry-domain statistic sits four orders of magnitude lower. Assert both, so the two
    cannot be silently interchanged.
    """
    g = torch.Generator().manual_seed(1)
    n = 400

    ent = [float(sagnac_coherence(_uent(8192, g), _uent(8192, g))) for _ in range(400)]
    spec = [float(sagnac_coherence(_uspec(8192, g), _uspec(8192, g))) for _ in range(400)]
    m_ent = sum(ent) / len(ent)
    m_spec = sum(spec) / len(spec)

    c = chance_coherence(8192)                        # sqrt(pi/(4D)) = 0.009792  (MEAN)
    c_rms = 1.0 / math.sqrt(8192)                     # 0.011049                    (RMS)
    c_spectrum = math.sqrt(math.pi / 4) / 8192 ** 1.5  # 1.195e-6

    assert abs(m_ent - c) < 0.3 * c, f"entry family {m_ent:.6f} vs closed mean {c:.6f}"
    assert abs(m_spec - c_spectrum) < 1.0 * c_spectrum, \
        f"spectrum family {m_spec:.2e} vs closed {c_spectrum:.2e}"
    assert m_spec < m_ent / 100.0, (
        "the spectrum-unimodular family must sit FAR below the entry-unimodular one")
    # THE RECONCILIATION: the receipt's 0.011049 is the RMS of the ENTRY family
    assert abs(0.011049 - c_rms) < 0.02 * c_rms, "receipt constant is the entry-family RMS"
    # and the mean/RMS ratio is the Rayleigh factor 1/sqrt(pi/4)
    ratio = c_rms / c
    assert abs(ratio - 1.0 / math.sqrt(math.pi / 4)) < 0.01, f"ratio {ratio:.6f}"


def test_chance_max_exceeds_chance_single():
    """Best-of-K exceeds single-trial chance, roughly by sqrt(2 ln K)."""
    s = chance_coherence(D)
    m = chance_max_coherence(D, N_AX)
    assert m > s
    ratio = m / s
    expected = math.sqrt(2 * math.log(N_AX))
    assert 0.6 * expected < ratio < 1.6 * expected, f"ratio {ratio:.2f} vs EVT {expected:.2f}"


# ============================================ THE SATURATION DEFECT, REPRODUCED
def test_shipped_statistic_saturates_on_unrelated_waves():
    """The shipped gate is `(1 - S) > 0.35`, i.e. keep when S > 0.65. On unrelated
    entry-unimodular waves E[S] = 0.0098, so the gate fires on EVERY candidate regardless
    of correctness. Assert the saturation, so a future change that lifts S is flagged."""
    g = torch.Generator().manual_seed(2)
    fires, n = 0, 80
    for _ in range(n):
        s = float(sagnac_coherence(_uent(2048, g), _uent(2048, g)))
        if (1.0 - s) > 0.35:
            fires += 1
    assert fires == n, f"shipped gate fired {fires}/{n}; saturation not reproduced"
    assert chance_coherence(2048) < 0.65


def test_bank_mean_reference_makes_the_defect_worse():
    """Receipt: arm2 (fixed bank-mean axiom) trigger_rate = 1.0 -- the most saturated arm.
    A bank mean is near-orthogonal to every member, so every candidate looks maximally
    dissimilar. Assert no genuine member survives the bank-mean gate."""
    g = torch.Generator().manual_seed(3)
    bank = torch.stack([_uent(2048, g) for _ in range(8)])
    mean = bank.mean(dim=0)
    kept = sum(1 for i in range(8) if float(sagnac_coherence(bank[i], mean)) > 0.65)
    assert kept == 0, f"{kept}/8 genuine bank members survived the bank-mean gate"


# ============================================ THE REPLACEMENT'S BEHAVIOUR
def test_exact_axiom_member_outscores_a_random_candidate():
    """THE DISCRIMINATING STATEMENT the veto needs, at the veto's own scale.

    `feature_vector` divides the inner product by sqrt(D), which is correct when the waves
    have unit-modulus entries. An EXACT axiom member then scores A_resonance near +0.97,
    while a random candidate scores near 0. My first draft fed unit-NORM vectors instead,
    which drove A_resonance to -0.024 and made this look broken.
    """
    veto = AttributableVeto(d=D, n_axioms=N_AX)
    g = torch.Generator().manual_seed(4)
    axioms = torch.stack([_uent(D, g) for _ in range(N_AX)])
    exact, rand = [], []
    for _ in range(40):
        gi = int(torch.randint(0, N_AX, (1,), generator=g))
        peers = torch.stack([_uent(D, g) for _ in range(4)])
        exact.append(veto.feature_vector(axioms[gi], axioms, peers)["A_resonance"])
        rand.append(veto.feature_vector(_uent(D, g), axioms, peers)["A_resonance"])
    me = float(torch.tensor(exact).mean())
    mr = float(torch.tensor(rand).mean())
    assert me > 0.5, f"exact axiom member should score high; got {me:.4f}"
    assert me - mr > 0.5, f"exact {me:.4f} must clear random {mr:.4f} by a wide margin"


def test_separable_fixture_does_separate():
    """Positive control: gold = an axiom member must be found above distractors."""
    veto = AttributableVeto(d=512, n_axioms=8)
    g = torch.Generator().manual_seed(5)
    axioms = torch.stack([_uent(512, g) for _ in range(8)])
    scores, labels = [], []
    for _ in range(120):
        gi = int(torch.randint(0, 8, (1,), generator=g))
        cands = [axioms[gi]] + [_uent(512, g) for _ in range(3)]
        for ci, c in enumerate(cands):
            others = torch.stack([x for cj, x in enumerate(cands) if cj != ci])
            scores.append(veto.score(c, axioms, others))
            labels.append(1 if ci == 0 else 0)
    a = auc(scores, labels)
    assert a > AUC_BAR, f"separable fixture must clear AUC {AUC_BAR}, got {a:.4f}"


def test_null_fixture_does_not_separate():
    """The null. Gold drawn EXACTLY like a distractor must give AUC ~0.5, and a random
    score must too. If either separates, the metric leaks."""
    veto = AttributableVeto(d=512, n_axioms=8)
    g = torch.Generator().manual_seed(6)
    axioms = torch.stack([_uent(512, g) for _ in range(8)])
    scores, labels = [], []
    for _ in range(120):
        cands = [_uent(512, g) for _ in range(4)]
        for ci, c in enumerate(cands):
            others = torch.stack([x for cj, x in enumerate(cands) if cj != ci])
            scores.append(veto.score(c, axioms, others))
            labels.append(1 if ci == 0 else 0)
    a = auc(scores, labels)
    assert 0.35 < a < 0.65, f"null fixture separated at AUC {a:.4f}; the metric leaks"
    r = torch.rand(len(labels), generator=g)
    assert abs(auc(r.tolist(), labels) - 0.5) < 0.10


# ============================================ THE IDENTITY CLAUSE (V_E)
def test_identity_clause_catches_a_single_feature_in_costume():
    """V_E. Measured sensitivities: an AFFINE COPY gives r = 1.000000 (must trip);
    independent features mixed give r < 0.995 (must pass).

    MY TEST DEFECT, self-caught twice: (a) I first used t and t^2 on [0,1] and assumed
    they were near-collinear -- measured r = 0.9668, BELOW the 0.995 bar, so they do not
    trip it; (b) mixing t with t^2 gives 0.9919 .. 0.9997 depending on weight, i.e. the
    pair is only borderline. The reliable near-collinear case is an affine copy.
    """
    t = torch.linspace(0, 1, 40)
    f = [float(x) for x in t]

    bad = [2.0 * x + 1.0 for x in f]                       # affine copy of one feature
    assert max_feature_identity_r({"f": f}, bad) > IDENTITY_R_BAR, "affine copy must trip V_E"

    g = torch.Generator().manual_seed(11)
    f1 = torch.randn(40, generator=g).tolist()
    f2 = torch.randn(40, generator=g).tolist()
    mixed = [a + 0.7 * b for a, b in zip(f1, f2)]
    assert max_feature_identity_r({"f1": f1, "f2": f2}, mixed) < IDENTITY_R_BAR, \
        "an independent mixture must pass V_E"

    # and record the measured t-vs-t^2 value so the old assumption cannot silently return
    t2 = [float(x * x) for x in t]
    r_t2 = max_feature_identity_r({"t2": t2}, f)
    assert 0.90 < r_t2 < IDENTITY_R_BAR, f"measured r(t,t^2) = {r_t2:.4f}; expected ~0.967"


def test_frozen_bars_are_the_preregistered_values():
    """Bar discipline. These come from henri_attributable_veto_preregistration.json and are
    NOT tunable in this module."""
    assert AUC_BAR == 0.55
    assert (REJECT_LO, REJECT_HI) == (0.02, 0.98)
    assert IDENTITY_R_BAR == 0.995


def test_rejection_band_flags_saturation():
    """V_D. A gate pinned at 0.0 or 1.0 is saturated and must be reported as such.
    Note torch.median on an EVEN count returns the lower middle, so a two-valued fixture
    genuinely saturates under a median split -- my first draft assumed otherwise."""
    veto = AttributableVeto(d=64, n_axioms=4)
    assert veto.rejection_band([1.0] * 50)["saturated"] is True
    assert veto.rejection_band([float(i) for i in range(50)])["in_band"] is True
    assert veto.rejection_band([1.0] * 25 + [2.0] * 25)["saturated"] is True
