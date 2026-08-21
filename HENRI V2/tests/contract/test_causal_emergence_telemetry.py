"""Contract tests for causal_emergence_telemetry (packet HENRI-CLASS47-CE-TELEMETRY-2026-08-21).

T1 calibration on analytic Markov chains (exact Hoel EI under uniform
intervention). Labels: OBSERVED (deterministic python/torch execution).
"""

import math

import pytest
import torch

from causal_emergence_telemetry import (
    CausalEmergenceTelemetry,
    causal_emergence,
    effective_information,
    estimate_ei_from_sequence,
)

N = 2000  # steps; 1000 transitions per row => smoothing bias ~0.0014 bits


def _run_chain(transitions, n=N, seed=7):
    """Sample a state sequence from a dict {state: [probs]} transition table."""
    g = torch.Generator().manual_seed(seed)
    states = [0]
    for _ in range(n - 1):
        cur = states[-1]
        probs = torch.tensor(transitions[cur], dtype=torch.float64)
        states.append(int(torch.multinomial(probs, 1, generator=g).item()))
    return torch.tensor(states, dtype=torch.int64)


def test_t1a_deterministic_identity_chain_ei_one_bit():
    """Deterministic 2-state identity chain: EI = 1.000 ± 0.02 (exact analytic 1.0)."""
    seq = _run_chain({0: [0.0, 1.0], 1: [1.0, 0.0]})
    rep = estimate_ei_from_sequence(seq, k=2)
    assert rep["status"] == "ok"
    assert rep["ei"] is not None
    assert 0.98 <= rep["ei"] <= 1.02, f"EI={rep['ei']}"


def test_t1b_random_chain_ei_near_zero():
    """Coin-flip 2-state chain: EI < 0.005 (analytic 0)."""
    seq = _run_chain({0: [0.5, 0.5], 1: [0.5, 0.5]})
    rep = estimate_ei_from_sequence(seq, k=2)
    assert rep["status"] == "ok"
    assert rep["ei"] < 0.005, f"EI={rep['ei']}"


def test_t1c_lumpable_3_to_2_positive_ce():
    """Lumpable 3->2 chain: EI_micro ~ 0.918, EI_macro ~ 1.000, CE ~ +0.082.

    Hoel EI is defined under do-intervention: EACH ROW of the TPM is sampled
    independently (row-stratified), because a single free-run trajectory
    starves rows that the chain visits rarely (absorbing states). We therefore
    build the sequence by interleaving per-row samples — exactly the empirical
    TPM the estimator computes. Chain: 0->{0,1}, 1->{0,1}, 2->2 (absorbing);
    macro lump A={0,1}, B={2} gives the deterministic identity TPM.
    """
    transitions = {
        0: [0.5, 0.5, 0.0],
        1: [0.5, 0.5, 0.0],
        2: [0.0, 0.0, 1.0],
    }
    g = torch.Generator().manual_seed(7)
    parts = []
    for s in range(3):
        probs = torch.tensor(transitions[s], dtype=torch.float64)
        for _ in range(1000):
            parts += [s, int(torch.multinomial(probs, 1, generator=g).item())]
    seq = torch.tensor(parts, dtype=torch.int64)  # row-stratified do-intervention
    micro = estimate_ei_from_sequence(seq, k=3)
    assert micro["status"] == "ok"
    assert 0.88 <= micro["ei"] <= 0.96, f"EI_micro={micro['ei']}"  # analytic 0.918
    macro_seq = torch.where(seq < 2, torch.tensor(0, dtype=torch.int64), torch.tensor(1, dtype=torch.int64))
    macro = estimate_ei_from_sequence(macro_seq, k=2)
    assert macro["status"] == "ok"
    assert 0.98 <= macro["ei"] <= 1.02, f"EI_macro={macro['ei']}"  # analytic 1.000
    ce = macro["ei"] - micro["ei"]
    assert 0.04 <= ce <= 0.12, f"CE={ce}"  # analytic +0.082


def test_effective_information_exact_identity_tpm():
    """Exact TPM [[0,1],[1,0]] gives exactly 1.000 bit."""
    tpm = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert effective_information(tpm) == pytest.approx(1.0, abs=1e-9)


def test_structured_waves_finite_ce_vs_noise_near_zero():
    """Amended T2 v3: noise (iid) |CE| < 0.02 @ T=256; coupled control CE > 0.01 > noise.

    A TEMPORALLY COUPLED control is required for discrimination: iid frame
    sequences have no causal structure for any estimator to find. The coupled
    control alternates two macro clusters deterministically (macro EI = 1)
    with within-cluster jitter (micro noisy) — the exact Hoel structure where
    CE > 0 exists after null-surrogate correction.
    """
    g = torch.Generator().manual_seed(11)
    # Noise: iid white noise, T=256 (amended gate).
    noise = torch.randn(256, 4096, generator=g).float()
    rep_n = causal_emergence(noise)
    assert rep_n["status"] == "ok"
    assert rep_n["ce_bits"] is not None
    assert abs(rep_n["ce_bits"]) < 0.02, f"noise corrected CE={rep_n['ce_bits']}"

    # Coupled control: deterministic macro alternation A,B,A,B,... with jitter.
    t = torch.linspace(0, 1, 4096)
    pa = torch.sin(2 * torch.pi * 3 * t)
    pb = torch.cos(2 * torch.pi * 3 * t)
    gj = torch.Generator().manual_seed(5)
    frames = []
    for i in range(256):
        base = pa if i % 2 == 0 else pb
        frames.append(base + 0.25 * torch.randn(4096, generator=gj))
    coupled = torch.stack(frames).float()
    rep_c = causal_emergence(coupled)
    assert rep_c["status"] == "ok"
    assert rep_c["ce_bits"] is not None
    assert rep_c["ce_bits"] > 0.01, f"coupled corrected CE={rep_c['ce_bits']} (must > 0.01)"
    assert rep_c["ce_bits"] > rep_n["ce_bits"], "coupled CE must exceed noise CE"


def test_short_window_and_nonfinite_do_not_crash():
    tele = CausalEmergenceTelemetry(window=16)
    assert tele.report() is None  # empty
    tele.push(torch.full((8, 8), float("nan")))  # non-finite dropped
    assert tele.report() is None  # still too short
    for _ in range(16):
        tele.push(torch.randn(8, 8))
    rep = tele.report()
    assert rep is not None and rep["status"] == "ok"
    assert "ce_after_erase" in rep
    assert 0 <= rep["macro_m"] <= 16


def test_ce_bounded_range():
    """CE stays in a sane range on structured data (packet gate S)."""
    t = torch.linspace(0, 40 * math.pi, 128)
    waves = torch.stack([torch.sin(t * (1 + 0.05 * i)) for i in range(8)], dim=1)
    waves = waves.repeat(1, 65536 // 8).float()
    rep = causal_emergence(waves, r=8, k=3, m=16)
    assert rep["status"] == "ok"
    assert rep["ce_bits"] is not None
    assert -0.5 <= rep["ce_bits"] <= 3.0
