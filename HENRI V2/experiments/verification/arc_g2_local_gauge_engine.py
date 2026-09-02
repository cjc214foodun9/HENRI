"""Carrier G2 — Local Block-Gauge Affordance Engine (HENRI V3).

Directive: user message (2026-09-01) + Example code.pdf
(HENRI-EVAL-2026-09-V3-G1-FALSIFICATION-AUDIT, 9d36971c..., 175,315 B).
Prereg: docs/spec/g2_local_block_gauge_preregistration.md (023ece11..., sealed
#671ae4ac @ ledger 1,161). Branch carrier/g2-local-block-gauge.

Mechanism (PDF "Local Block-Gauge Affordance", degeneracy fixed):
  Lever 1 (local blocks): affordance evaluated on the FULL [8192, 8] wave;
    per-block augmented vector phi_b = [psi_b; 1] in R^9; per-action metric
    W_a in Sym(9) fit as centered quadratic correlation (G1 fit rule at
    full-D, block-separable).
  Lever 2 (stall-cosine label): y_moving = [cos(psi_t, psi_{t+1}) < tau_stall],
    tau_stall = 0.90, measured on the FULL-D bank (OBSERVED 2026-09-01:
    per-action moving minority 0.19-0.234 at cos < 0.90 — balanced classes).
  Lever 3 (per-action temperature): logit_a = pooled_a / tau_a + b_a,
    tau_a = tau_base * exp(clamp(log(std_a(pooled)/sigma_ref), log .1, log 10)).
  Softmax pooling: pooled_a = sum_b softmax(beta * q_a(psi)_b) * q_a(psi)_b,
    beta = 10 — isolates the maximum local collision coordinate.
  PDF-form control: Re(psi^T (i W_skew) psi) == 0 for real psi (identically
    degenerate) — replaced by a real SYMMETRIC metric (a metric, not a
    generator). Same defect class as G1's rejected W_contact = I*(mean-0.5).

Kinematics (unchanged from G1, D=64 bridge): T_free,a = exp(D_free,a) in
SO(64) fit moving-only (||Psi_{t+1}-Psi_t||_2 > 0.05), omega-bound pi/32,
ridge 1e-4; scattering Psi_hat = pi*T*Psi + (1-pi)*Psi; homotopy beam K=8,
J = align * pi^K; dual-speed online affordance update eta=0.10 (label via
full-D stall-cosine).

Live representation: FastFullDWaveEncoder — vectorized chunked phase
accumulation + CC-OS parity, EQUIVALENT to HENRIVisionEncoder (cos >= 0.9999,
max|d| <= 1e-3 on 3 synthetic grids; C2d gate, verified on CUDA before the
gauntlet). Motivation (OBSERVED): production encoder = 738 ms/step at 30x30
(per-cell Python loop) — unusable live.

PG1 (binding): per-action in-sample AUC >= 0.8800 (>= 10 samples), all 7
actions. PRE-FLIGHT KILL -> no live run. C2b bridge reproduction (G1 D=64
fit on the REAL bank) reported as diagnostic (measured G1: 0.7768).

Verdicts G2_*. Flag HENRI_G2_LOCAL_GAUGE=1 (default-OFF fail-closed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from arc_g1_topological_engine import (
        _bridge_to_d64_single,
        advance_waypoint_index,
        compute_auc,
        compile_free_generators_capped,
        extract_waypoints,
        fit_affordance_classifiers,
        langevin_escape_tick,
        scatter_prediction,
        _safe_levels,
        DEFAULT_HORIZON,
        G1_LATENCY_MS,
        G2_MIN_SOLVED,
        G3_MIN_DELTA_NU,
        G4_MAX_AFFORDANCE,
        LANGEVIN_STEPS,
        LANGEVIN_TEMP,
        MIN_AUC_SAMPLES,
        MOVING_THRESH,
        OMEGA_BOUND,
        RIDGE,
        MIN_MOVING,
        WAYPOINT_ADVANCE_THRESH,
        WAYPOINT_DELTA_THETA,
        WAYPOINT_MAX,
        WAYPOINT_MIN,
        WAYPOINT_STRIDE,
    )
except Exception:  # pragma: no cover - test isolation
    _bridge_to_d64_single = advance_waypoint_index = compute_auc = None
    compile_free_generators_capped = fit_affordance_classifiers = None
    extract_waypoints = langevin_escape_tick = scatter_prediction = None
    _safe_levels = None
    DEFAULT_HORIZON = 8
    G1_LATENCY_MS = 2.0
    G2_MIN_SOLVED = 1
    G3_MIN_DELTA_NU = 0.0150
    G4_MAX_AFFORDANCE = 0.0500
    LANGEVIN_STEPS = 3
    LANGEVIN_TEMP = 0.50
    MIN_AUC_SAMPLES = 10
    MOVING_THRESH = 0.05
    OMEGA_BOUND = math.pi / 32.0
    RIDGE = 1e-4
    MIN_MOVING = 5
    WAYPOINT_ADVANCE_THRESH = 0.60
    WAYPOINT_DELTA_THETA = 0.35
    WAYPOINT_MAX = 6
    WAYPOINT_MIN = 2
    WAYPOINT_STRIDE = 15

try:
    from arc_f21_1_vectorized_engine import _bridge_to_d64_batch, PatchIngress
except Exception:  # pragma: no cover - test isolation
    _bridge_to_d64_batch = None
    PatchIngress = None

try:
    from arc_f15_trajectory_engine import DEFAULT_ENVS, resolve_trajectory_goal
except Exception:  # pragma: no cover - test isolation
    DEFAULT_ENVS = []
    resolve_trajectory_goal = None

try:
    from connected_component_segmenter import (
        ConnectedComponentSegmenter,
        ParityContourMask,
    )
except Exception:  # pragma: no cover - test isolation
    ConnectedComponentSegmenter = None
    ParityContourMask = None

FLAG = "HENRI_G2_LOCAL_GAUGE"
D_SUB = 64
B_FULL = 8192
BLK = 8
AUG = 9
PG1_MIN_AUC = 0.8800
TAU_STALL = 0.90
POOL_BETA = 10.0
TAU_BASE = 0.05
SIGMA_REF = 0.05
TAU_MIN_FACTOR = 1e-3
TAU_MAX_FACTOR = 1e3
ETA_AFFORDANCE = 0.10
SEED = 20260925
FAST_ENC_COS_MIN = 0.9999
FAST_ENC_MAX_D = 1e-3


def require_flag():
    if os.environ.get(FLAG) != "1":
        raise RuntimeError(f"{FLAG}=1 is required (carrier default-OFF flag).")


# ---------------------------------------------------------------------------
# FastFullDWaveEncoder: vectorized production-equivalent UWE.
# ---------------------------------------------------------------------------
class FastFullDWaveEncoder(nn.Module):
    """Vectorized chunked UWE equivalent to HENRIVisionEncoder (cos >= 0.9999).

    Replaces the production per-cell Python loop (738 ms/step at 30x30,
    OBSERVED 2026-09-01) with per-color separable einsum accumulation
    (O(H*W*D/2) complex MACs, ~2-6 ms at 30x30 on CUDA).
    """

    def __init__(self, d_model: int = 65536, max_grid_dim: int = 128,
                 device: str = "cpu", seed: int = 0):
        super().__init__()
        self.d_model = d_model
        self.max_grid_dim = max_grid_dim
        self.device = device
        self.seed = seed
        half = d_model // 2
        phases_x = torch.linspace(0, 2 * math.pi * 127, half, device=device)
        phases_y = torch.linspace(0, 2 * math.pi * 127, half, device=device)
        coords = torch.arange(max_grid_dim, device=device, dtype=torch.float32)
        # [max_grid_dim, half]
        self.register_buffer("basis_x", torch.exp(1j * (coords.unsqueeze(1) * phases_x.unsqueeze(0))))
        self.register_buffer("basis_y", torch.exp(1j * (coords.unsqueeze(1) * phases_y.unsqueeze(0))))
        angles = torch.linspace(0, 2 * math.pi * 15 / 16, 16, device=device).unsqueeze(1)
        freqs = torch.arange(1, half + 1, device=device, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("color_codebook", torch.exp(1j * (angles * freqs)))

    @torch.no_grad()
    def encode_grid(self, grid) -> torch.Tensor:
        if not isinstance(grid, torch.Tensor):
            grid = torch.as_tensor(np.ascontiguousarray(grid), dtype=torch.long, device=self.device)
        else:
            grid = grid.to(dtype=torch.long, device=self.device)
        if grid.dim() == 3 and grid.shape[0] == 1:
            grid = grid.squeeze(0)
        H, W = grid.shape
        assert H <= self.max_grid_dim and W <= self.max_grid_dim
        grid_clamped = torch.clamp(grid, 0, 15).cpu().numpy()

        parity = np.ones((H, W), dtype=np.float32)
        if ConnectedComponentSegmenter is not None:
            segmenter = ConnectedComponentSegmenter(background_color=0)
            components = segmenter.segment_grid(grid_clamped)
            for comp in components:
                interior_px, _ = ParityContourMask.compute_parity_contour(
                    (H, W), comp.pixels)
                for r_i, c_i in interior_px:
                    parity[r_i, c_i] = -1.0
        P = torch.as_tensor(parity, dtype=torch.float32, device=self.device)

        X = self.basis_x[:W]      # [W, half]
        Y = self.basis_y[:H]      # [H, half]
        cb = self.color_codebook  # [16, half]

        superposed = torch.zeros(self.d_model // 2, dtype=torch.complex64, device=self.device)
        for v in range(16):
            Mv = (grid_clamped == v).astype(np.float32)
            if not Mv.any():
                continue
            Wv_t = torch.as_tensor(Mv, dtype=torch.float32, device=self.device)
            Wv = (Wv_t * P).to(torch.complex64)   # [H, W] weights (complex for einsum)
            T = torch.einsum("hw,wd->hd", Wv, X)  # [H, half]
            S = torch.einsum("hd,hd->d", Y, T)    # [half]
            superposed.add_(cb[v] * S)
        real_wave = torch.cat([superposed.real, superposed.imag], dim=-1)
        return F.normalize(real_wave, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Local block-gauge affordance.
# ---------------------------------------------------------------------------
def _augment(psi: torch.Tensor) -> torch.Tensor:
    """[..., M, 8] -> [..., M, 9] with a constant bias coordinate."""
    ones = torch.ones(*psi.shape[:-1], 1, dtype=psi.dtype, device=psi.device)
    return torch.cat([psi, ones], dim=-1)


def local_gauge_scores(psi: torch.Tensor, W: torch.Tensor,
                       beta: float = POOL_BETA) -> torch.Tensor:
    """pooled_a = sum_b softmax(beta * q_a(psi)_b) * q_a(psi)_b.

    psi [B, 8192, 8]; W [A, 9, 9] (symmetric); returns pooled [B, A].
    """
    B = psi.shape[0]
    A = W.shape[0]
    phi = _augment(psi)  # [B, M, 9]
    pooled = torch.empty(B, A, dtype=psi.dtype, device=psi.device)
    for a in range(A):
        q = torch.einsum("bmp,pq,bmq->bm", phi, W[a], phi)  # [B, M]
        w = F.softmax(beta * q, dim=-1)
        pooled[:, a] = (w * q).sum(-1)
    return pooled


def predict_affordance_logits(pooled: torch.Tensor, b: torch.Tensor,
                              tau: torch.Tensor) -> torch.Tensor:
    return pooled / tau.unsqueeze(0) + b.unsqueeze(0)


def stall_cosine_labels(psi_flat, nxt_flat, tau_stall=TAU_STALL):
    """Norm-invariant stall-cosine labels on canonical per-block unit waves.

    psi_flat/nxt_flat [N, 65536] (or [N, M, 8]). Normalizes per block to the
    canonical unit-norm geometry (HENRI invariant ||w_k||_2 = 1.0), then
    divides the flat dot by the flat norms. Guard against the unnormalized-
    bank defect class (OBSERVED: bank psi ||.|| ~ 14-22 while next_wave
    ||.|| = 1.0; a raw-dot label corrupted the G2 launch to 0.8% positives
    instead of the true 21%).
    """
    psi_full = F.normalize(psi_flat.float().view(psi_flat.shape[0], -1, BLK), p=2, dim=-1)
    nxt_full = F.normalize(nxt_flat.float().view(nxt_flat.shape[0], -1, BLK), p=2, dim=-1)
    flat_p = psi_full.reshape(psi_full.shape[0], -1)
    flat_n = nxt_full.reshape(nxt_full.shape[0], -1)
    cos = (flat_p * flat_n).sum(-1) / (flat_p.norm(dim=-1) * flat_n.norm(dim=-1) + 1e-12)
    return (cos < tau_stall).float(), cos


def fit_local_gauge_classifiers(psi: torch.Tensor, onehot: torch.Tensor,
                                y: torch.Tensor,
                                min_samples: int = MIN_AUC_SAMPLES):
    """W_a = (1/N_a) sum_i (y_i - mean_a) sum_b phi_i,b phi_i,b^T (Sym(9)).

    psi [N, 8192, 8]; onehot [N, A] float; y [N] float labels.
    Returns (W [A,9,9] symmetric, b [A], tau [A]).
    """
    psi = psi.float()
    y = y.float().to(device=psi.device)
    onehot = onehot.to(device=psi.device, dtype=torch.float32)
    A = onehot.shape[1]
    phi = _augment(psi)                     # [N, M, 9]
    A_i = torch.einsum("nbp,nbq->npq", phi, phi)  # [N, 9, 9]
    W = torch.zeros(A, AUG, AUG, dtype=psi.dtype, device=psi.device)
    b = torch.zeros(A, dtype=psi.dtype, device=psi.device)
    tau = torch.full((A,), TAU_BASE, dtype=psi.dtype, device=psi.device)
    for a in range(A):
        mask = onehot[:, a].bool()
        if mask.sum() >= min_samples:
            y_a = y[mask]
            mean_a = float(y_a.mean().item())
            yc = y_a - mean_a
            W[a] = torch.einsum("i,ipq->pq", yc, A_i[mask]) / y_a.numel()
            W[a] = 0.5 * (W[a] + W[a].T)
            # Scale-free metric: normalize by Frobenius norm; the per-action
            # temperature tau_a then calibrates the sharpness (logit =
            # pooled/tau + b). A scale-invariant direction is the only
            # meaningful content of a quadratic correlation metric.
            W[a] = W[a] / (W[a].norm().clamp(min=1e-8))
            b[a] = torch.logit(torch.clamp(torch.tensor(mean_a, device=psi.device), 0.05, 0.95))
            # Per-action temperature from the in-sample pooled std (calibrated
            # scale; deterministic closed-form, no label-peeking beyond fit).
            pooled_a = local_gauge_scores(psi[mask], W[a:a + 1], POOL_BETA)[:, 0]
            std_a = float(pooled_a.std().clamp(min=1e-8).item())
            factor = math.exp(min(max(math.log(std_a / SIGMA_REF),
                                     math.log(TAU_MIN_FACTOR)), math.log(TAU_MAX_FACTOR)))
            tau[a] = TAU_BASE * factor
        else:
            prior = float(y[mask].mean().item()) if mask.sum() > 0 else 0.5
            b[a] = torch.logit(torch.clamp(torch.tensor(prior, device=psi.device), 0.05, 0.95))
    return W, b, tau


def fast_encoder_equivalence(enc_fast: FastFullDWaveEncoder,
                             enc_prod, grids) -> dict:
    """C2d: cos >= 0.9999 and max|d| <= 1e-3 vs HENRIVisionEncoder."""
    results = {}
    for (h, w, seed) in grids:
        g = torch.Generator().manual_seed(seed)
        grid = torch.randint(0, 10, (h, w), generator=g)
        wf = enc_fast.encode_grid(grid)
        wp = enc_prod.encode_grid(grid)
        cos = float(F.cosine_similarity(wf.unsqueeze(0), wp.unsqueeze(0)).item())
        max_d = float((wf - wp).abs().max().item())
        results[f"{h}x{w}"] = {"cos": cos, "max_d": max_d}
    ok = all(r["cos"] >= FAST_ENC_COS_MIN and r["max_d"] <= FAST_ENC_MAX_D
             for r in results.values())
    return {"pass": ok, "grids": results}


# ---------------------------------------------------------------------------
# G2 engine (kinematics D=64 unchanged; affordance full-D).
# ---------------------------------------------------------------------------
class G2Engine:
    def __init__(self, generators, transitions, t_pow, recon,
                 W_contact, b_contact, tau_a,
                 action_names=None, n_actions=7, seed=SEED,
                 horizon=DEFAULT_HORIZON, device="cuda",
                 omega_bound=OMEGA_BOUND, waypoints=None,
                 waypoint_advance_thresh=WAYPOINT_ADVANCE_THRESH,
                 langevin_temp=LANGEVIN_TEMP, eta_affordance=ETA_AFFORDANCE,
                 moving_thresh=MOVING_THRESH, tau_stall=TAU_STALL,
                 pool_beta=POOL_BETA, tau_base=TAU_BASE):
        self.generators = generators
        self.transitions = list(transitions)
        self.t_pow = t_pow
        self.recon = recon
        self.W_contact = W_contact
        self.b_contact = b_contact
        self.tau_a = tau_a
        self.action_names = list(action_names) if action_names else [str(i) for i in range(n_actions)]
        self.n_actions = n_actions
        self.seed = seed
        self.horizon = horizon
        self.device = device
        self.omega_bound = omega_bound
        self.waypoints = list(waypoints) if waypoints else None
        self.waypoint_advance_thresh = waypoint_advance_thresh
        self.langevin_temp = langevin_temp
        self.eta_affordance = eta_affordance
        self.moving_thresh = moving_thresh
        self.tau_stall = tau_stall
        self.pool_beta = pool_beta
        self.tau_base = tau_base
        self.creeps = 0
        self.waypoint_advances = 0
        self.langevin_escapes = 0
        self.affordance_updates = 0
        self.escape_state = {"steps": 0, "active": False}
        self._wp_idx = 0

    def _active_waypoint(self):
        if self.waypoints is None or len(self.waypoints) == 0:
            return F.normalize(
                torch.randn(D_SUB, generator=torch.Generator().manual_seed(self.seed)),
                dim=-1)
        k = min(self._wp_idx, len(self.waypoints) - 1)
        return self.waypoints[k]

    def predict_affordance(self, psi_full):
        """Pi [B, n] = sigmoid(pooled_a / tau_a + b_a) on the FULL wave.

        Normalizes to canonical per-block unit norm (identity on already-
        canonical input; guards the unnormalized-bank defect class).
        """
        psi_full = psi_full.float().to(self.device)
        if psi_full.dim() == 2:
            psi_full = psi_full.unsqueeze(0)
        psi_full = F.normalize(psi_full, p=2, dim=-1)
        pooled = local_gauge_scores(
            psi_full, self.W_contact.to(self.device), self.pool_beta)
        logits = predict_affordance_logits(
            pooled, self.b_contact.to(self.device), self.tau_a.to(self.device))
        return torch.sigmoid(logits)

    def score_all_actions(self, psi64, psi_full, waypoint=None):
        """J_a = |<Psi_hat_{t+K}(a), Psi_wp>| * Pi_pass,a^K (D=64 kinematics,
        full-D affordance)."""
        psi64 = F.normalize(psi64.float().to(self.device), dim=-1)
        wp = F.normalize((waypoint if waypoint is not None else self._active_waypoint())
                         .float().to(self.device), dim=-1)
        tpow = self.t_pow.to(self.device)
        rolled = tpow @ psi64  # [n, K, D]
        steps = F.normalize(rolled, dim=-1)
        aligns = (steps * wp).sum(-1).abs()  # [n, K]
        align = aligns[:, -1]
        pi = self.predict_affordance(psi_full)[0]  # [n]
        j = align * (pi ** self.horizon)
        if self.escape_state.get("active"):
            g = torch.Generator().manual_seed(self.seed + self.escape_state.get("steps", 0))
            noise = torch.sqrt(torch.tensor(2.0 * self.langevin_temp, device=self.device)) * \
                torch.randn(self.n_actions, generator=g).to(self.device)
            j = j + noise
        names = list(self.action_names)
        return {names[i]: float(j[i].item()) for i in range(self.n_actions)}

    def step_once(self, psi64, psi_full, waypoint=None):
        js = self.score_all_actions(psi64, psi_full, waypoint)
        if not js:
            return None, js
        return max(js, key=js.get), js

    def update_online_affordance(self, psi_full, action_idx, psi_full_next,
                                 eta=None):
        """Full-D stall-cosine plasticity on the executed action (norm-divided
        cosine on canonical per-block unit waves)."""
        eta = self.eta_affordance if eta is None else eta
        psi_full = F.normalize(psi_full.float().to(self.device), p=2, dim=-1)
        psi_full_next = F.normalize(psi_full_next.float().to(self.device), p=2, dim=-1)
        a_f = psi_full.reshape(-1)
        b_f = psi_full_next.reshape(-1)
        cos = min(1.0, float((a_f * b_f).sum().abs().item() /
                             ((a_f.norm() * b_f.norm()).clamp(min=1e-12).item())))
        # NOTE: clamp AFTER division. A clamp on the raw sum (magnitude
        # 8192 for per-block-unit waves) caps it at 1.0 -> cos ~ 1.2e-4
        # -> was_moving=1 even for identical pairs -> W increases on a
        # collision (C8 defect class, fixed 2026-09-01).
        was_moving = 1.0 if cos < self.tau_stall else 0.0
        pi = self.predict_affordance(psi_full)[0, action_idx].item()
        error = was_moving - pi
        phi = _augment(psi_full)  # [M, 9]
        A = torch.einsum("bp,bq->pq", phi, phi) / B_FULL  # mean over blocks
        # Scale-matched update: W is unit-Frobenius (fit normalization), so
        # the update direction must be unit-Frobenius too; eta then bounds
        # the relative perturbation to |eta*error| <= eta per step (C8 fix:
        # an unnormalized A with ||A||_F ~ 1.8 flipped the metric direction
        # at eta >= 1 and raised pi on a collision).
        A = A / (A.norm().clamp(min=1e-8))
        self.W_contact[action_idx] = self.W_contact[action_idx] + eta * error * A
        self.b_contact[action_idx] = self.b_contact[action_idx] + eta * error
        self.affordance_updates += 1
        return error

    def g4_single_pass(self, psi64, psi_full, action_idx, psi64_actual):
        """Scattered prediction vs ACTUAL post-action observation (D=64,
        pi from full-D)."""
        psi64 = F.normalize(psi64.float().to(self.device), dim=-1)
        psi64_actual = F.normalize(psi64_actual.float().to(self.device), dim=-1)
        pi = self.predict_affordance(psi_full)[0, action_idx].item()
        pred = scatter_prediction(psi64, self.transitions[action_idx], pi)
        sim = (pred * psi64_actual).sum(-1).abs().clamp(0.0, 1.0)
        return 1.0 - sim

    def advance_waypoint_index(self, psi, waypoint, k):
        return advance_waypoint_index(psi, waypoint, k, thresh=self.waypoint_advance_thresh)

    def _decide_verdict(self, mean_latency, solved, mean_delta_nu, g4_mean,
                        steps_done, updates):
        if steps_done > 0 and updates == 0:
            return "G2_NO_AFFORDANCE_ENGAGEMENT"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            return "G2_GATE_G1_FAILED"
        if solved < G2_MIN_SOLVED:
            return "G2_GATE_G2_FAILED"
        if mean_delta_nu is not None and mean_delta_nu < G3_MIN_DELTA_NU:
            return "G2_GATE_G3_FAILED"
        if g4_mean is not None and g4_mean > G4_MAX_AFFORDANCE:
            return "G2_GATE_G4_FAILED"
        return "G2_PASS"

    def run_gauntlet(self, env_names, fast_encoder, steps_per_env=150,
                     seed=SEED, trajectory_bank=None, trajectory_jsonl=None,
                     ingress=None, out_dir=None, receipt_out=None,
                     allow_kill=True, pg1_min_auc=None, env_goals=None):
        t0 = time.time()
        latencies, g4s, dnus = [], [], []
        waypoint_align_first, waypoint_align_last = None, None
        solved, steps_done, resets = 0, 0, 0
        env_levels = {}
        if pg1_min_auc is not None and allow_kill and pg1_min_auc < PG1_MIN_AUC:
            result = {"verdict": "G2_AFFORDANCE_FIT_COLLAPSE", "steps_done": 0,
                      "min_auc": float(pg1_min_auc), "n_actions": self.n_actions,
                      "seed": self.seed, "omega_bound": self.omega_bound,
                      "horizon": self.horizon, "tau_stall": self.tau_stall}
            if receipt_out:
                pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
            return result

        try:
            from arc_agi import Arcade
            arcade = Arcade()
        except Exception as exc:
            result = {"verdict": "G2_ARCADE_UNAVAILABLE", "steps_done": 0,
                      "reason": f"arcade_unavailable: {exc!r}"}
            if receipt_out:
                pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
            return result

        for env_name in env_names:
            self._wp_idx = 0
            goal = env_goals.get(env_name) if env_goals else None
            if goal is None and resolve_trajectory_goal is not None:
                try:
                    goal, _meta = resolve_trajectory_goal(
                        trajectory_bank, trajectory_jsonl, env_name,
                        device=self.device, ingress=ingress)
                except Exception:
                    goal = None
            if goal is None:
                goal = F.normalize(
                    torch.randn(D_SUB, generator=torch.Generator().manual_seed(seed)).to(self.device),
                    dim=-1)
            wps = [goal]
            if trajectory_bank:
                try:
                    data = np.load(trajectory_bank)
                    if "psi" in data.files:
                        waves = torch.from_numpy(np.asarray(data["psi"])).float()
                        env_indices = []
                        if trajectory_jsonl:
                            with open(trajectory_jsonl, "r", encoding="utf-8") as f:
                                for i, line in enumerate(f):
                                    try:
                                        rec = json.loads(line)
                                    except json.JSONDecodeError:
                                        continue
                                    if rec.get("env") == env_name:
                                        env_indices.append(i)
                        if len(env_indices) >= WAYPOINT_MIN:
                            curve_full = waves[env_indices]
                            curve = torch.stack([
                                _bridge_to_d64_single(w, ingress=ingress, seed=seed,
                                                      device=self.device)
                                for w in curve_full
                            ])
                            extracted = extract_waypoints(curve, goal)
                            if len(extracted) >= WAYPOINT_MIN:
                                wps = extracted
                except Exception:
                    pass
            self.waypoints = wps

            game = None
            try:
                game = arcade.make(env_name)
            except Exception as exc:
                result = {"verdict": "G2_ARCADE_MAKE_FAILED", "steps_done": steps_done,
                          "reason": f"arcade_make: {exc!r}"}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            if game is None:
                result = {"verdict": "G2_ARCADE_MAKE_NONE", "steps_done": steps_done}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            try:
                obs = game.reset()
            except Exception as exc:
                result = {"verdict": "G2_ARCADE_RESET_FAILED", "steps_done": steps_done,
                          "reason": f"arcade_reset: {exc!r}"}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            if obs is None or not getattr(obs, "frame", None):
                result = {"verdict": "G2_NULL_INITIAL_FRAME", "steps_done": steps_done,
                          "env": env_name}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            prev_levels = _safe_levels(obs)

            for _ in range(steps_per_env):
                t1 = time.time()
                frame = obs.frame[0]
                raw = torch.as_tensor(np.asarray(frame).reshape(-1).astype(np.float32),
                                      dtype=torch.float32, device=self.device)
                if raw.numel() < 4096:
                    raw = F.pad(raw, (0, 4096 - raw.numel()))
                else:
                    raw = raw[:4096]
                psi64 = ingress(raw.unsqueeze(0)).reshape(1, -1)[0].detach()
                if psi64.shape[-1] != D_SUB:
                    raise RuntimeError(f"PatchIngress boundary must flatten to [{D_SUB}], got {tuple(psi64.shape)}")
                psi_full = fast_encoder.encode_grid(
                    torch.as_tensor(np.asarray(frame), dtype=torch.long,
                                    device=self.device)).reshape(B_FULL, BLK).detach()
                wp = self._active_waypoint()
                if waypoint_align_first is None:
                    waypoint_align_first = float((psi64 * wp).sum(-1).abs().clamp(0.0, 1.0).item())
                best, js = self.step_once(psi64, psi_full, wp)
                if self.escape_state.get("active"):
                    self.escape_state = langevin_escape_tick(self.escape_state)
                if best is None:
                    best = self.action_names[0]
                env_actions = list(getattr(obs, "available_actions", None) or
                                   getattr(game, "action_space", None) or [])
                idx = self._action_index(best, env_actions)
                c_t = float((psi64 * wp).sum(-1).abs().clamp(0.0, 1.0).item())
                try:
                    if env_actions:
                        action = env_actions[idx % max(1, len(env_actions))]
                    else:
                        action = best
                    obs = game.step(action)
                except Exception:
                    obs = None
                latencies.append((time.time() - t1) * 1e3)
                steps_done += 1
                terminal = obs is None or (
                    getattr(obs, "state", None) and obs.state.name == "GAME_OVER")
                if terminal:
                    cur_levels = _safe_levels(obs) if obs is not None else prev_levels
                    if cur_levels > prev_levels:
                        solved += 1
                    prev_levels = cur_levels
                    env_levels[env_name] = prev_levels
                    self.escape_state = {"steps": 0, "active": True}
                    self.langevin_escapes += 1
                    resets += 1
                    try:
                        obs = game.reset()
                    except Exception:
                        obs = None
                    if obs is None or not getattr(obs, "frame", None):
                        break
                    prev_levels = _safe_levels(obs)
                    continue
                frame_next = obs.frame[0]
                raw_next = torch.as_tensor(np.asarray(frame_next).reshape(-1).astype(np.float32),
                                           dtype=torch.float32, device=self.device)
                if raw_next.numel() < 4096:
                    raw_next = F.pad(raw_next, (0, 4096 - raw_next.numel()))
                else:
                    raw_next = raw_next[:4096]
                psi64_next = ingress(raw_next.unsqueeze(0)).reshape(1, -1)[0].detach()
                psi_full_next = fast_encoder.encode_grid(
                    torch.as_tensor(np.asarray(frame_next), dtype=torch.long,
                                    device=self.device)).reshape(B_FULL, BLK).detach()
                g4s.append(float(self.g4_single_pass(psi64, psi_full, idx, psi64_next).item()))
                self.update_online_affordance(psi_full, idx, psi_full_next)
                c_next = float((psi64_next * wp).sum(-1).abs().clamp(0.0, 1.0).item())
                dnu = c_next - c_t
                dnus.append(dnu)
                if dnu > 0:
                    self.creeps += 1
                new_k = self.advance_waypoint_index(
                    psi64_next, self.waypoints[min(self._wp_idx, len(self.waypoints) - 1)],
                    self._wp_idx)
                if new_k > self._wp_idx:
                    self._wp_idx = new_k
                    self.waypoint_advances += 1
                waypoint_align_last = c_next
                cur_levels = _safe_levels(obs)
                if cur_levels > prev_levels:
                    solved += 1
                prev_levels = cur_levels
            env_levels[env_name] = prev_levels

        mean_latency = float(np.mean(latencies)) if latencies else None
        g4_affordance_mean = float(np.mean(g4s)) if g4s else None
        mean_delta_nu = float(np.mean(dnus)) if dnus else None
        result = {"verdict": None, "steps_done": steps_done, "resets": resets,
                  "mean_latency_ms": mean_latency, "g4_affordance_mean": g4_affordance_mean,
                  "mean_delta_nu_wp": mean_delta_nu,
                  "waypoint_align_first": waypoint_align_first,
                  "waypoint_align_last": waypoint_align_last,
                  "waypoint_advances": self.waypoint_advances,
                  "langevin_escapes": self.langevin_escapes,
                  "per_action_recon": {str(k): float(v) for k, v in self.recon.items()},
                  "affordance_updates": self.affordance_updates,
                  "creeps": self.creeps, "n_actions": self.n_actions, "seed": self.seed,
                  "envs_solved": solved, "env_levels": env_levels,
                  "wall_s": round(time.time() - t0, 3),
                  "omega_bound": self.omega_bound, "horizon": self.horizon,
                  "waypoint_advance_thresh": self.waypoint_advance_thresh,
                  "langevin_temp": self.langevin_temp,
                  "eta_affordance": self.eta_affordance,
                  "moving_thresh": self.moving_thresh, "tau_stall": self.tau_stall}
        verdict = self._decide_verdict(
            mean_latency=mean_latency, solved=solved, mean_delta_nu=mean_delta_nu,
            g4_mean=g4_affordance_mean, steps_done=steps_done,
            updates=self.affordance_updates)
        result["verdict"] = verdict
        if receipt_out:
            pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
        return result

    def _action_index(self, name, env_actions):
        if name in env_actions:
            return env_actions.index(name)
        try:
            return int(name) % max(1, len(env_actions))
        except (TypeError, ValueError):
            return 0


def build_parser():
    ap = argparse.ArgumentParser(description="Carrier G2 local block-gauge affordance gauntlet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--omega-bound", type=float, default=OMEGA_BOUND)
    ap.add_argument("--waypoint-advance-thresh", type=float, default=WAYPOINT_ADVANCE_THRESH)
    ap.add_argument("--langevin-temp", type=float, default=LANGEVIN_TEMP)
    ap.add_argument("--eta-affordance", type=float, default=ETA_AFFORDANCE)
    ap.add_argument("--moving-thresh", type=float, default=MOVING_THRESH)
    ap.add_argument("--tau-stall", type=float, default=TAU_STALL)
    ap.add_argument("--pool-beta", type=float, default=POOL_BETA)
    ap.add_argument("--tau-base", type=float, default=TAU_BASE)
    ap.add_argument("--trajectory-bank", required=True)
    ap.add_argument("--trajectory-jsonl", required=True)
    ap.add_argument("--envs", nargs="+", default=None,
                    help="12 named env ids (default: F15 DEFAULT_ENVS)")
    ap.add_argument("--out-dir", default="/tmp/henri_g2_local_gauge/")
    ap.add_argument("--receipt-out", default=None)
    return ap


def main():
    args = build_parser().parse_args()
    require_flag()
    env_names = list(args.envs) if args.envs else list(DEFAULT_ENVS)
    device = args.device

    data = np.load(args.trajectory_bank)
    psi_flat = torch.from_numpy(np.asarray(data["psi"])).float().to(device)
    nxt_flat = torch.from_numpy(np.asarray(data["next_wave"])).float().to(device)
    onehot = torch.from_numpy(np.asarray(data["actions_onehot"])).to(torch.uint8)

    # Canonical [8192, 8] geometry: per-block unit norm (HENRI invariant
    # ||w_k||_2 = 1.0). The bank's psi is NOT unit-norm (OBSERVED: ||psi_t||
    # ~ 14-22, next_wave ||.|| = 1.0) — a raw-dot label would be corrupted
    # (harness defect G2_HARNESS_DEFECT_LABEL_NORM, 0 live steps, relaunched
    # with identical bounds). Normalize per-block, then the stall-cosine
    # label is norm-invariant.
    psi_full = F.normalize(psi_flat.view(-1, B_FULL, BLK), p=2, dim=-1)
    nxt_full = F.normalize(nxt_flat.view(-1, B_FULL, BLK), p=2, dim=-1)

    # Full-D stall-cosine labels (PDF lever 2); norm-invariant helper.
    y, _cos = stall_cosine_labels(psi_flat, nxt_flat, args.tau_stall)

    W, b, tau = fit_local_gauge_classifiers(psi_full, onehot.float(), y)
    pooled = local_gauge_scores(psi_full, W, args.pool_beta)
    logits = predict_affordance_logits(pooled, b, tau)
    pi = torch.sigmoid(logits)

    per_action_auc = {}
    for a in range(pi.shape[1]):
        mask_a = onehot[:, a].bool()
        if mask_a.sum() >= MIN_AUC_SAMPLES:
            per_action_auc[str(a)] = compute_auc(pi[mask_a, a], y[mask_a])
        else:
            per_action_auc[str(a)] = None
    auc_vals = [v for v in per_action_auc.values() if v is not None]
    pg1_min_auc = min(auc_vals) if auc_vals else 0.0

    # C2b: G1 bridge reproduction (diagnostic — the representation change is
    # the causal fix; G1 measured min_auc 0.7768 on this exact pipeline).
    bridge_min_auc = None
    bridge_aucs = None
    if _bridge_to_d64_batch is not None and PatchIngress is not None:
        ingress64 = PatchIngress(in_dim=4096, d=D_SUB, num_blocks=8, p=32,
                                 seed=args.seed).to(device)
        psi64 = _bridge_to_d64_batch(psi_full.reshape(psi_full.shape[0], -1),
                                     ingress=ingress64, seed=args.seed)
        nxt64 = _bridge_to_d64_batch(nxt_full.reshape(nxt_full.shape[0], -1),
                                     ingress=ingress64, seed=args.seed)
        comp64 = compile_free_generators_capped(
            psi64, nxt64, onehot.float(), omega_bound=args.omega_bound,
            moving_thresh=args.moving_thresh, seed=args.seed)
        W64, b64 = fit_affordance_classifiers(psi64, onehot.float(), comp64["is_moving"])
        pi64 = torch.sigmoid(torch.einsum("bd,ade,be->ba", psi64, W64, psi64) / 0.05 + b64.unsqueeze(0))
        bridge_aucs = {}
        for a in range(pi64.shape[1]):
            mask_a = onehot[:, a].bool()
            if mask_a.sum() >= MIN_AUC_SAMPLES:
                bridge_aucs[str(a)] = compute_auc(pi64[mask_a, a], comp64["is_moving"][mask_a])
            else:
                bridge_aucs[str(a)] = None
        bridge_vals = [v for v in bridge_aucs.values() if v is not None]
        bridge_min_auc = min(bridge_vals) if bridge_vals else None

    receipt_out = args.receipt_out or str(pathlib.Path(args.out_dir) / "g2_gates_receipt.json")
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if pg1_min_auc < PG1_MIN_AUC:
        result = {"verdict": "G2_AFFORDANCE_FIT_COLLAPSE", "steps_done": 0,
                  "min_auc": pg1_min_auc, "per_action_auc": per_action_auc,
                  "bridge_min_auc": bridge_min_auc, "bridge_aucs": bridge_aucs,
                  "n_actions": len(W), "seed": args.seed,
                  "omega_bound": args.omega_bound, "horizon": args.horizon,
                  "tau_stall": args.tau_stall, "pool_beta": args.pool_beta,
                  "tau_base": args.tau_base,
                  "label_counts": {str(a): float(y[onehot[:, a].bool()].mean().item())
                                   for a in range(onehot.shape[1])}}
        pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2))
        return 1

    print(json.dumps({"pg1_min_auc": pg1_min_auc, "per_action_auc": per_action_auc,
                      "bridge_min_auc": bridge_min_auc, "bridge_aucs": bridge_aucs,
                      "label_counts": {str(a): float(y[onehot[:, a].bool()].mean().item())
                                       for a in range(onehot.shape[1])}}))

    # C2d: fast-encoder equivalence preflight (CUDA, 3 grids).
    from henri_vision_encoder import HENRIVisionEncoder
    enc_fast = FastFullDWaveEncoder(d_model=65536, device=device, seed=args.seed)
    enc_prod = HENRIVisionEncoder(d_model=65536, device=device)
    eq = fast_encoder_equivalence(enc_fast, enc_prod,
                                  [(10, 10, 1), (20, 20, 2), (30, 30, 3)])
    if not eq["pass"]:
        result = {"verdict": "G2_FAST_ENCODER_EQUIVALENCE_FAILED", "steps_done": 0,
                  "equivalence": eq}
        pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2))
        return 1

    ingress = PatchIngress(in_dim=4096, d=D_SUB, num_blocks=8, p=32,
                           seed=args.seed).to(device)
    psi64 = _bridge_to_d64_batch(psi_full.reshape(psi_full.shape[0], -1),
                                 ingress=ingress, seed=args.seed)
    nxt64 = _bridge_to_d64_batch(nxt_full.reshape(nxt_full.shape[0], -1),
                                 ingress=ingress, seed=args.seed)
    comp = compile_free_generators_capped(
        psi64, nxt64, onehot.float(), omega_bound=args.omega_bound,
        moving_thresh=args.moving_thresh, seed=args.seed)

    env_goals = {}
    if resolve_trajectory_goal is not None:
        for name in env_names:
            try:
                goal, meta = resolve_trajectory_goal(
                    args.trajectory_bank, args.trajectory_jsonl, name,
                    device=device, ingress=ingress)
                env_goals[name] = goal
            except Exception as exc:
                print(json.dumps({"goal_warning": str(exc), "env": name}))

    engine = G2Engine(
        generators=comp["generators"], transitions=comp["transitions"],
        t_pow=comp["t_pow"], recon=comp["recon"], W_contact=W, b_contact=b,
        tau_a=tau, action_names=comp.get("action_names"),
        n_actions=len(comp["generators"]), seed=args.seed, horizon=args.horizon,
        device=device, omega_bound=args.omega_bound,
        waypoint_advance_thresh=args.waypoint_advance_thresh,
        langevin_temp=args.langevin_temp, eta_affordance=args.eta_affordance,
        moving_thresh=args.moving_thresh, tau_stall=args.tau_stall,
        pool_beta=args.pool_beta, tau_base=args.tau_base)
    result = engine.run_gauntlet(
        env_names, fast_encoder=enc_fast, steps_per_env=args.steps_per_env,
        seed=args.seed, trajectory_bank=args.trajectory_bank,
        trajectory_jsonl=args.trajectory_jsonl, ingress=ingress,
        out_dir=args.out_dir, receipt_out=receipt_out,
        pg1_min_auc=pg1_min_auc, env_goals=env_goals)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
