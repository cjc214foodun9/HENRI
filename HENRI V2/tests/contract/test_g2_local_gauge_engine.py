"""Contract tests for Carrier G2 local block-gauge affordance engine.

Directive: user message (2026-09-01) + Example code.pdf
(HENRI-EVAL-2026-09-V3-G1-FALSIFICATION-AUDIT, 9d36971c..., 175,315 B).

C1  flag gate (default-OFF)
C2  stall-cosine label (cos < tau_stall) partitions the full-D synthetic bank
C3  full-D affordance AUC >= 0.8800 on separable synthetic bank (all 7 actions)
C4  PDF skew-Hermitian form degeneracy: Re(psi^T (iW) psi) == 0 -> AUC ~ 0.5
C5  label-shuffle control: AUC <= 0.60
C6  scattering identity: pi=0 -> psi; pi=1 -> T_free psi (D=64 kinematics)
C7  homotopy beam prunes blocked actions (J ~ 0)
C8  online affordance update moves prediction toward observed
C9  G4 single-pass scattered consistency ~ 0 in both regimes
C10 verdict precedence (PG1 kill / G1 -> G2 -> G3 -> G4)
C11 bridge-erasure reproduction: mean-pooled D=64 fit FAILS where full-D passes
    (the exact G1 kill class; representation change is the causal fix)
C12 fast-encoder equivalence vs HENRIVisionEncoder (cos >= 0.9999, max|d| <= 1e-3)
"""
from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "HENRI V2"))
sys.path.insert(0, str(ROOT / "experiments" / "verification"))

try:
    from arc_g2_local_gauge_engine import (
        G2Engine,
        FastFullDWaveEncoder,
        compute_auc,
        fit_local_gauge_classifiers,
        local_gauge_scores,
        predict_affordance_logits,
        require_flag,
        scatter_prediction,
        stall_cosine_labels,
        PG1_MIN_AUC,
        G1_LATENCY_MS,
        G2_MIN_SOLVED,
        G3_MIN_DELTA_NU,
        G4_MAX_AFFORDANCE,
        TAU_STALL,
        POOL_BETA,
        TAU_BASE,
        SIGMA_REF,
        MOVING_THRESH,
    )
except Exception as exc:  # pragma: no cover - import isolation
    raise AssertionError(f"G2 engine import failed: {exc!r}")

try:
    from arc_g1_topological_engine import (
        compile_free_generators_capped,
        fit_affordance_classifiers,
    )
except Exception:  # pragma: no cover
    compile_free_generators_capped = None
    fit_affordance_classifiers = None

try:
    from arc_f21_1_vectorized_engine import PatchIngress, _bridge_to_d64_batch
except Exception:  # pragma: no cover
    PatchIngress = None
    _bridge_to_d64_batch = None

try:
    from henri_vision_encoder import HENRIVisionEncoder
except Exception as exc:  # pragma: no cover
    raise AssertionError(f"HENRIVisionEncoder import failed: {exc!r}")

D = 64          # kinematics subspace
B_FULL = 8192   # blocks
BLK = 8         # block dim
SEED = 20260925
N_OPEN = 24
N_BLOCK = 24
N_ACT = 7
RHO_OPEN = 0.50   # |eps|/|m| for open (moving) rows -> per-block cos 0.6 < 0.90
RHO_BLOCK = 0.05  # |eta|/|m| for wall (blocked) rows -> cos ~ 1


def _unit(n, d, seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), p=2, dim=-1)


def _skew_rot(theta=0.25, seed=1, d=D, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    S = 0.5 * (A - A.T)
    n = torch.linalg.matrix_norm(S, ord=2)
    return torch.linalg.matrix_exp(S * (theta / n))


def _full_bank(seed=SEED, n_actions=N_ACT, n_open=N_OPEN, n_block=N_BLOCK,
               device="cpu", num_blocks=B_FULL):
    """Full-D synthetic bank with per-action distinct OPEN and WALL regions.

    Open rows:  psi_t = normalize(m_open + rho*u_b),
                psi_next = normalize(m_open - rho*u_b), u_b per-block unit,
                orthogonal to m_open, EXACTLY zero-mean over each 512-block
                bridge group (paired +/- within the group, |u_b| = 1) ->
                per-block cos (1-rho^2)/(1+rho^2) = 0.6 < tau_stall => MOVING,
                while the mean-pooled D=64 bridge sees EXACTLY zero
                displacement (erasure; |m + rho*u| = sqrt(1+rho^2) exactly).
    Wall rows:  psi_t = normalize(m_wall + rho*eta_b), psi_next = psi_t
                -> cos 1 (BLOCKED). m_wall orthogonal to m_open per block.
    Region directions m_open/m_wall are SHARED across all blocks per action
    (consistent structure the sum-over-blocks metric can learn).
    Returns (psi [N, M, 8], nxt [N, M, 8], onehot [N, A], y [N]).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    psi_list, nxt_list, oh_list, y_list = [], [], [], []
    groups = 16
    G = num_blocks // groups  # 512 blocks per bridge group
    assert num_blocks % groups == 0 and G % 2 == 0

    def _paired(noise, m_dir, rho):
        """unit, orthogonal-to-m, EXACT-zero-bridge-mean, magnitude rho.

        The G1 bridge reshapes the wave to [16, 4096] and averages the 16
        rows at fixed intra-chunk position k. To erase it exactly, pair the
        rows ACROSS groups: e[g+8, b, d] = -e[g, b, d] for g < 8, so the
        coordinate-wise mean over the 16 groups is 0 identically (per-block
        |e| stays 1 -> per-block cos 0.6 preserved).
        """
        e = noise - (noise * m_dir.unsqueeze(0)).sum(-1, keepdim=True) * m_dir.unsqueeze(0)
        e = F.normalize(e, p=2, dim=-1)
        e = e.view(noise.shape[0], groups, G, BLK)
        e[:, groups // 2:] = -e[:, :groups // 2]   # cross-group sign flip
        return e.view(noise.shape[0], num_blocks, BLK) * rho

    for a in range(n_actions):
        m_open = F.normalize(torch.randn(BLK, generator=g), p=2, dim=-1)   # [8]
        m_wall = F.normalize(torch.randn(BLK, generator=g), p=2, dim=-1)
        dots = float((m_wall * m_open).sum().item())
        m_wall = F.normalize(m_wall - dots * m_open, p=2, dim=-1)
        m_o = m_open.unsqueeze(0).expand(num_blocks, BLK)   # shared across blocks
        m_w = m_wall.unsqueeze(0).expand(num_blocks, BLK)

        # open rows: move (cos 0.6), bridge group-mean EXACTLY preserved
        eps = _paired(torch.randn(n_open, num_blocks, BLK, generator=g), m_o, RHO_OPEN)
        xo = F.normalize(m_o.unsqueeze(0) + eps, p=2, dim=-1)
        yo = F.normalize(m_o.unsqueeze(0) - eps, p=2, dim=-1)

        # wall rows: blocked (cos 1), distinct mean region
        eta = _paired(torch.randn(n_block, num_blocks, BLK, generator=g), m_w, RHO_BLOCK)
        xb = F.normalize(m_w.unsqueeze(0) + eta, p=2, dim=-1)
        yb = xb.clone()

        psi_list += [xo, xb]
        nxt_list += [yo, yb]
        oh = torch.zeros(n_open + n_block, n_actions)
        oh[:, a] = 1.0
        oh_list.append(oh)
        y_list += [torch.ones(n_open), torch.zeros(n_block)]
    psi = torch.cat(psi_list)
    nxt = torch.cat(nxt_list)
    onehot = torch.cat(oh_list)
    y = torch.cat(y_list)
    return psi, nxt, onehot, y


def _bridge(x):
    """Mean-pooled D=64 bridge (G1 path, no PatchIngress)."""
    flat = x.reshape(x.shape[0], -1).float()
    pooled = flat.reshape(x.shape[0], 16, -1).mean(dim=1)
    return F.normalize(pooled, p=2, dim=-1)


def _kin_bank(seed=7, n_actions=2, n_open=N_OPEN, n_block=N_BLOCK):
    """D=64 kinematics bank (G1-style): open clusters move, blocked stay."""
    g = torch.Generator().manual_seed(seed)
    centers, _ = torch.linalg.qr(torch.randn(D, n_actions * 2, generator=g))
    psi_list, nxt_list, onehot_list = [], [], []
    for a in range(n_actions):
        e_open = centers[:, a]
        e_block = centers[:, n_actions + a]
        xo = F.normalize(e_open.unsqueeze(0) + 0.2 * torch.randn(n_open, D, generator=g), dim=-1)
        xb = F.normalize(e_block.unsqueeze(0) + 0.2 * torch.randn(n_block, D, generator=g), dim=-1)
        T = _skew_rot(theta=0.30, seed=seed + a)
        yo = F.normalize(xo @ T.T, dim=-1)
        yb = xb.clone()
        psi_list += [xo, xb]
        nxt_list += [yo, yb]
        oh = torch.zeros(n_open + n_block, n_actions)
        oh[:, a] = 1.0
        onehot_list.append(oh)
    return torch.cat(psi_list), torch.cat(nxt_list), torch.cat(onehot_list), centers


def _engine(seed=SEED):
    """Kinematics from _kin_bank (D=64) + full-D affordance from _full_bank."""
    psi, nxt, onehot, centers = _kin_bank()
    comp = compile_free_generators_capped(psi, nxt, onehot, omega_bound=math.pi / 32.0)
    W, b = fit_affordance_classifiers(psi, onehot, comp["is_moving"])
    fpsi, fnxt, foh, fy = _full_bank(n_actions=2)
    Wf, bf, tauf = fit_local_gauge_classifiers(fpsi, foh, fy)
    eng = G2Engine(
        generators=comp["generators"], transitions=comp["transitions"],
        t_pow=comp["t_pow"], recon=comp["recon"], W_contact=Wf, b_contact=bf,
        tau_a=tauf, action_names=["0", "1"], n_actions=2, seed=seed,
        horizon=8, device="cpu", omega_bound=math.pi / 32.0,
        waypoints=[F.normalize(_unit(1, D, seed=3)[0], dim=-1)],
        waypoint_advance_thresh=0.60, langevin_temp=0.50,
        eta_affordance=0.10, moving_thresh=MOVING_THRESH,
    )
    return eng, centers, (fpsi, foh, fy)


def test_c1_flag_gate():
    os.environ.pop("HENRI_G2_LOCAL_GAUGE", None)
    try:
        require_flag()
        raise AssertionError("flag gate must refuse without env")
    except RuntimeError:
        pass
    os.environ["HENRI_G2_LOCAL_GAUGE"] = "1"
    require_flag()
    os.environ.pop("HENRI_G2_LOCAL_GAUGE", None)


def test_c2_stall_cosine_label():
    psi, nxt, onehot, y = _full_bank(n_actions=2, n_open=16, n_block=16)
    cos = (psi * nxt).sum(-1).mean(-1)  # per-block cos, mean over blocks
    y_hat = (cos < TAU_STALL).float()
    assert torch.equal(y_hat, y), "stall-cosine label must match construction"
    open_cos = cos[y == 1.0]
    block_cos = cos[y == 0.0]
    assert float(open_cos.max()) < TAU_STALL
    assert float(block_cos.min()) > 0.99
    # bridge erasure: bridged open rows look blocked (l2 ~ 0)
    psi64 = _bridge(psi)
    nxt64 = _bridge(nxt)
    l2 = torch.norm(nxt64 - psi64, p=2, dim=-1)
    assert float(l2[y == 1.0].max()) < 0.05, "bridge must erase open motion"


def test_c3_full_d_affordance_auc():
    psi, nxt, onehot, y = _full_bank(n_actions=N_ACT)
    W, b, tau = fit_local_gauge_classifiers(psi, onehot, y)
    pooled = local_gauge_scores(psi, W, POOL_BETA)
    logits = predict_affordance_logits(pooled, b, tau)
    pi = torch.sigmoid(logits)
    for a in range(N_ACT):
        mask = onehot[:, a].bool()
        auc = compute_auc(pi[mask, a], y[mask])
        assert auc >= PG1_MIN_AUC, f"action {a} AUC {auc:.4f} < {PG1_MIN_AUC}"


def test_c4_skew_form_degenerate():
    """PDF form Re(psi^T (i W_skew) psi) is identically zero for real psi."""
    psi, nxt, onehot, y = _full_bank(n_actions=2, n_open=16, n_block=16)
    g = torch.Generator().manual_seed(11)
    W_skew = torch.randn(2, 9, 9, generator=g)
    W_skew = 0.5 * (W_skew - W_skew.transpose(-1, -2))
    phi = torch.cat([psi, torch.ones(psi.shape[0], psi.shape[1], 1)], dim=-1)
    q = torch.einsum("bkp,apq,bkq->abk", phi, W_skew, phi)
    assert float(q.abs().max()) < 1e-5, "skew form must vanish for real states"
    pi = torch.sigmoid(torch.zeros(psi.shape[0], 2))
    for a in range(2):
        mask = onehot[:, a].bool()
        auc = compute_auc(pi[mask, a], y[mask])
        assert abs(auc - 0.5) < 0.02, f"skew-form AUC {auc:.4f} must be ~0.5"


def test_c5_label_shuffle():
    # More rows (800) at 512 blocks keep the fit capacity/sample ratio low
    # (p=45 quadratics vs N=800 => expected in-sample AUC ~ 0.53) so the
    # control is a real anti-memorization check, not an overfit artifact.
    psi, nxt, onehot, y = _full_bank(n_actions=2, n_open=200, n_block=200,
                                     num_blocks=512)
    g = torch.Generator().manual_seed(21)
    y_shuf = y[torch.randperm(y.shape[0], generator=g)]
    W, b, tau = fit_local_gauge_classifiers(psi, onehot, y_shuf)
    pooled = local_gauge_scores(psi, W, POOL_BETA)
    logits = predict_affordance_logits(pooled, b, tau)
    pi = torch.sigmoid(logits)
    for a in range(2):
        mask = onehot[:, a].bool()
        auc = compute_auc(pi[mask, a], y[mask])
        assert auc <= 0.60, f"shuffled-label AUC {auc:.4f} must be <= 0.60"


def test_c6_scattering_identity():
    eng, _, _ = _engine()
    psi = F.normalize(_unit(1, D, seed=5)[0], dim=-1)
    pred0 = scatter_prediction(psi, eng.transitions[0], 0.0)
    pred1 = scatter_prediction(psi, eng.transitions[0], 1.0)
    assert torch.allclose(pred0, psi, atol=1e-5)
    assert torch.allclose(pred1, psi @ eng.transitions[0].T, atol=1e-5)


def test_c7_beam_prunes_blocked():
    eng, centers, (fpsi, foh, fy) = _engine()
    e_open0 = centers[:, 0]
    psi64 = F.normalize(e_open0, dim=-1)
    wp = F.normalize(psi64 @ eng.transitions[0].T, dim=-1)
    # action-0 OPEN full state (moving region): pi_0 ~ 1, pi_1 ~ 0
    psi_full = fpsi[foh[:, 0].bool() & (fy == 1.0)][0]
    js = eng.score_all_actions(psi64, psi_full, wp)
    assert js["0"] > js["1"], f"moving action must outscore blocked: {js}"
    assert js["1"] < 1e-2, f"blocked action must be pruned: {js['1']:.4f}"


def test_c8_online_affordance_update():
    eng, _, (fpsi, foh, fy) = _engine()
    psi_full = fpsi[foh[:, 0].bool() & (fy == 1.0)][0]
    pi_before = eng.predict_affordance(psi_full)[0, 0].item()
    eng.update_online_affordance(psi_full, 0, psi_full, eta=0.10)  # prod eta
    pi_after = eng.predict_affordance(psi_full)[0, 0].item()
    assert pi_after < pi_before, f"collision must lower pi: {pi_before:.4f} -> {pi_after:.4f}"
    assert eng.affordance_updates == 1


def test_c9_g4_scattered_consistency():
    eng, centers, (fpsi, foh, fy) = _engine()
    e_open0 = centers[:, 0]
    psi_m = F.normalize(e_open0, dim=-1)
    psi_next_m = F.normalize(psi_m @ eng.transitions[0].T, dim=-1)
    open_full = fpsi[foh[:, 0].bool() & (fy == 1.0)][0]
    d_m = eng.g4_single_pass(psi_m, open_full, 0, psi_next_m)
    assert d_m < 1e-2, f"moving-regime Delta {d_m:.6f} must be small"
    e_block1 = centers[:, 3]
    psi_b = F.normalize(e_block1, dim=-1)
    wall_full = fpsi[foh[:, 1].bool() & (fy == 0.0)][0]
    d_b = eng.g4_single_pass(psi_b, wall_full, 1, psi_b)
    assert d_b < 1e-2, f"blocked-regime Delta {d_b:.6f} must be small"


def test_c10_verdict_precedence():
    eng, _, _ = _engine()
    v = eng._decide_verdict(
        mean_latency=3.0, solved=2, mean_delta_nu=0.02, g4_mean=0.01,
        steps_done=100, updates=100)
    assert v == "G2_GATE_G1_FAILED", v
    v = eng._decide_verdict(
        mean_latency=1.0, solved=0, mean_delta_nu=0.02, g4_mean=0.01,
        steps_done=100, updates=100)
    assert v == "G2_GATE_G2_FAILED", v
    v = eng._decide_verdict(
        mean_latency=1.0, solved=1, mean_delta_nu=0.001, g4_mean=0.01,
        steps_done=100, updates=100)
    assert v == "G2_GATE_G3_FAILED", v
    v = eng._decide_verdict(
        mean_latency=1.0, solved=1, mean_delta_nu=0.02, g4_mean=0.10,
        steps_done=100, updates=100)
    assert v == "G2_GATE_G4_FAILED", v
    v = eng._decide_verdict(
        mean_latency=1.0, solved=1, mean_delta_nu=0.02, g4_mean=0.01,
        steps_done=100, updates=100)
    assert v == "G2_PASS", v


def test_c11_bridge_erasure_reproduction():
    """The G1 kill class: mean-pooled fit fails where full-D passes."""
    psi, nxt, onehot, y = _full_bank(n_actions=2, n_open=16, n_block=16)
    # Full-D classifier
    W, b, tau = fit_local_gauge_classifiers(psi, onehot, y)
    pooled = local_gauge_scores(psi, W, POOL_BETA)
    pi_full = torch.sigmoid(predict_affordance_logits(pooled, b, tau))
    aucs_full = []
    for a in range(2):
        mask = onehot[:, a].bool()
        aucs_full.append(compute_auc(pi_full[mask, a], y[mask]))
    assert min(aucs_full) >= PG1_MIN_AUC, f"full-D must pass: {aucs_full}"

    # Bridged (G1 D=64) classifier: same fit rule on the REAL bridge
    # (block-mean 4096 -> PatchIngress -> 64), like the G1 pipeline.
    ingress64 = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32, seed=SEED)
    psi64 = _bridge_to_d64_batch(psi.reshape(psi.shape[0], -1), ingress=ingress64, seed=SEED)
    nxt64 = _bridge_to_d64_batch(nxt.reshape(nxt.shape[0], -1), ingress=ingress64, seed=SEED)
    y64 = (torch.norm(nxt64 - psi64, p=2, dim=-1) > MOVING_THRESH).float()
    W64, b64 = fit_affordance_classifiers(psi64, onehot, y64)
    pi64 = torch.sigmoid(torch.einsum("bd,ade,be->ba", psi64, W64, psi64) / 0.05 + b64.unsqueeze(0))
    aucs64 = []
    for a in range(2):
        mask = onehot[:, a].bool()
        aucs64.append(compute_auc(pi64[mask, a], y64[mask]))
    assert max(aucs64) < 0.6, f"bridged fit must fail the gate: {aucs64}"
    # and the bridged LABEL itself is wrong (all blocked)
    assert float(y64.sum()) == 0.0, "bridge must erase all motion in this bank"


def test_c12_fast_encoder_equivalence():
    enc_fast = FastFullDWaveEncoder(d_model=65536, device="cpu", seed=SEED)
    enc_prod = HENRIVisionEncoder(d_model=65536, device="cpu")
    for (h, w, seed) in [(10, 10, 1), (15, 15, 2), (20, 20, 3)]:
        g = torch.Generator().manual_seed(seed)
        grid = torch.randint(0, 10, (h, w), generator=g)
        wf = enc_fast.encode_grid(grid)
        wp = enc_prod.encode_grid(grid)
        cos = float(F.cosine_similarity(wf.unsqueeze(0), wp.unsqueeze(0)).item())
        max_d = float((wf - wp).abs().max().item())
        assert cos >= 0.9999, f"({h}x{w}) cos {cos:.6f} < 0.9999"
        assert max_d <= 1e-3, f"({h}x{w}) max|d| {max_d:.2e} > 1e-3"


def test_c13_stall_label_norm_invariance():
    """Regression guard: the stall-cosine label must be norm-invariant.

    The REAL bank's psi is NOT unit-norm (OBSERVED ||psi_t|| ~ 14-22 while
    next_wave ||.|| = 1.0); a raw-dot label gave 0.8% positives instead of
    the true 21% (G2 launch defect, 0 live steps, relaunched with identical
    bounds). stall_cosine_labels must recover the same labels from scaled
    and unscaled waves.
    """
    psi, nxt, onehot, y = _full_bank(n_actions=2, n_open=16, n_block=16)
    psi_flat = psi.reshape(psi.shape[0], -1)
    nxt_flat = nxt.reshape(nxt.shape[0], -1)
    y1, cos1 = stall_cosine_labels(psi_flat, nxt_flat, TAU_STALL)
    assert torch.equal(y1, y), "label must match construction on canonical waves"
    # scaled (unnormalized) bank: label must be IDENTICAL
    y2, cos2 = stall_cosine_labels(psi_flat * 17.3, nxt_flat, TAU_STALL)
    assert torch.equal(y1, y2), "label must be scale-invariant (norm-divided)"
    assert torch.allclose(cos1, cos2, atol=1e-4), "cosine must be scale-invariant"
    # raw-dot would FAIL: frac positives ~ 0 on scaled bank
    raw_frac = float((torch.abs((psi_flat * 17.3 * nxt_flat).sum(-1)) < TAU_STALL).float().mean())
    assert raw_frac < float(y.mean()), f"raw-dot label must be corrupted: {raw_frac:.3f}"
