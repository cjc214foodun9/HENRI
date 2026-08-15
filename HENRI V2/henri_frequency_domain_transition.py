"""
HENRI V2 Phase 8.9 — Frequency-Domain Diagonal Phase Rotators (qFHRR).

Action-conditioned diagonal phase transition (8.9-A) + closed-form Wirtinger
phase residual update (8.9-B), per the Phase 8.8 Postmortem / Phase 8.9
Blueprint (raw SHA ccacd145...).

Mechanism (Fourier Convolution Theorem; Plate 1995):
  spatial translation (dx, dy)  <->  diagonal phase multiplier
  Phi_a = exp(j * Theta_a), Theta_a = dx*Omega + dy*Theta  in [-pi, pi]^D
  Psi_{t+1} = Normalize(Psi_t (.) Phi_a)        [O(D) Hadamard]

This module is DIAGNOSTIC ONLY. Production LowRankCoupledTransition
(efe_planner.py:70) is untouched; this module is never imported by production.

Pre-registration: experiments/sweeps/phase89_diagonal_transition_design.md
Deviations from the blueprint sketch (pre-registered):
  1. update lr=1.0 (exact residual); sketch's 1e-2 cannot meet gate G3.
  2. Analytic complex CC-OS carriers (exp(j(r*Omega + c*Theta))), the canonical
     FHRR frequency-domain state; blueprint's own eq defines Psi = F{s_t} complex.
"""

import math

import torch
import torch.nn as nn

TAU = (math.sqrt(5.0) - 1.0) / 2.0  # golden-ratio conjugate: incommensurate spacing


def _band_limited_irrational(seed_base: float, dimension: int, carrier_scale: float) -> torch.Tensor:
    """Deterministic band-limited incommensurate frequency vector in [-2*pi*s, 2*pi*s].

    Omega_d = 2*pi*s * (2*frac(seed_base*(d+1)) - 1). Irrational seed_base makes
    frequencies incommensurate; the [0,1) fractional span is band-limited.
    """
    idx = torch.arange(1, dimension + 1, dtype=torch.float64)
    frac = (seed_base * idx) % 1.0
    return (2.0 * frac - 1.0) * (2.0 * math.pi * carrier_scale)


class FrequencyDomainDiagonalTransition(nn.Module):
    """Action-conditioned diagonal phase rotator (8.9-A) + Wirtinger update (8.9-B)."""

    def __init__(self, dimension: int = 65536, num_actions: int = 16, device: str = "cuda"):
        super().__init__()
        self.dimension = dimension
        self.num_actions = num_actions
        self.device = device
        # Action phase parameters Theta_a in [-pi, pi]; initialized to zero phase (identity).
        self.action_phases = nn.Parameter(
            torch.zeros(num_actions, dimension, dtype=torch.float32, device=device)
        )

    def phasor(self, action_idx: torch.Tensor) -> torch.Tensor:
        """Unit-modulus complex rotator exp(j*Theta_a); [B, D] complex64 or [D]."""
        theta_a = self.action_phases[action_idx]  # [B, D] or [D]
        return torch.polar(torch.ones_like(theta_a), theta_a)

    def forward(self, state_wave: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """Psi_{t+1} = Psi_t * exp(j*Theta_a). state_wave complex [B, D] or [D].

        NOTE (pre-registered deviation #3): the blueprint sketch's
        `Normalize(next_wave)` (vector L2) contradicts its own Sagnac formula
        1 - |<pred, actual>|/D — unit-L2 vectors would floor Sagnac at
        1 - 1/D ~= 0.99998 even for perfect prediction. FHRR phasor convention
        (per-element unit modulus, |z_d| = 1) keeps <a,a>/D = 1 -> Sagnac = 0 at
        perfect. Pure Hadamard phase rotation preserves per-element modulus, so
        no vector-level normalization is applied.
        """
        phi_a = self.phasor(action_idx)  # [B, D] or [D]
        next_wave = state_wave * phi_a  # Hadamard complex phase rotation
        return next_wave

    def update_online_wirtinger(
        self,
        state_t: torch.Tensor,
        state_tp1_actual: torch.Tensor,
        action_idx: int,
        lr: float = 1.0,
    ) -> float:
        """Closed-form frequency-domain phase learning (8.9-B).

        Theta_a <- Theta_a + lr * arg(Psi_{t+1} * conj(Psi_t) * exp(-j*Theta_a)).
        Returns Sagnac phase loss 1 - |<pred, actual>|/D after the update.
        """
        with torch.no_grad():
            theta_a = self.action_phases[action_idx]  # [D]
            phi_a = torch.polar(torch.ones_like(theta_a), theta_a)  # [D] complex
            # Target phase difference
            target_phase_diff = state_tp1_actual * torch.conj(state_t)  # [D] complex
            # Phase error residual
            phase_error = target_phase_diff * torch.conj(phi_a)
            angle_residual = torch.angle(phase_error)  # [-pi, pi]
            # Gradient update on phase parameters
            self.action_phases[action_idx] += lr * angle_residual.squeeze(0)
            # Measure Sagnac phase loss
            predicted_wave = self.forward(state_t, torch.tensor([action_idx], device=self.device))
            sagnac_loss = 1.0 - torch.abs(torch.sum(predicted_wave * torch.conj(state_tp1_actual))) / self.dimension
            return sagnac_loss.item()


class AnalyticSpatialCarriers:
    """Analytic complex CC-OS spatial carriers (8.9-C data source; deviation #2).

    Psi(r, c) = exp(j*(r*Omega + c*Theta)) in S^{D-1}, with band-limited
    incommensurate Omega/Theta. Translation is EXACT:
        Psi(r+dx, c+dy) = Psi(r, c) * exp(j*(dx*Omega + dy*Theta)).
    """

    def __init__(self, dimension: int = 65536, carrier_scale: float = 0.10, device: str = "cuda"):
        self.dimension = dimension
        self.carrier_scale = carrier_scale
        self.device = device
        self.omega = _band_limited_irrational(TAU, dimension, carrier_scale).to(
            device=device, dtype=torch.float32)
        self.theta = _band_limited_irrational(math.sqrt(3.0), dimension, carrier_scale).to(
            device=device, dtype=torch.float32)

    def encode(self, r: float, c: float) -> torch.Tensor:
        """Analytic complex carrier at integer/float position (r, c); [D] complex64."""
        phase = r * self.omega + c * self.theta  # [D] float32
        return torch.polar(torch.ones_like(phase), phase)

    def rotator(self, dx: float, dy: float) -> torch.Tensor:
        """Exact diagonal phase rotator for translation (dx, dy); [D] complex64."""
        return self.encode(dx, dy)

    def expected_sagnac(self, predicted: torch.Tensor, actual: torch.Tensor) -> float:
        return 1.0 - float(torch.abs(torch.sum(predicted * torch.conj(actual))) / self.dimension)


class FrequencyDomainDiagonalAdapter(nn.Module):
    """Phase 8.10 — production-interface adapter over the sealed 8.9 rotator.

    The production transition interface (EFEPlanner / LowRankCoupledTransition)
    is REAL [num_blocks, 8] per-block unit waves. This adapter keeps that exact
    interface while executing the frequency-domain diagonal phase rotation:

        z_d     = exp(j * arccos(clamp(w_d)))     # real wave -> unit phasor
        theta_d = pi * clamp(a_d)                 # action wave -> phase offset
        pred_d  = cos(arccos(s_d) + theta_d + phi_a_d)
        out     = per-block renormalize(pred)

    Identity at zero phase: cos(arccos(s)) == s exactly, so
    forward(s, zero-wave) == s (pre-registered gate G1).

    Per-action phase correction phi_a (nn.Parameter [num_actions, d], zero init
    = identity) is learned by the closed-form elementwise Wirtinger update
    (8.9-B port): phi_a[d] += lr * angle(z_tp1 * conj(z_t) * conj(phi_a))[d],
    exact in 1 step for any diagonal phase rotation (Fourier Convolution
    Theorem; Plate 1995). Lazy action indexing by cosine fingerprint of the
    action wave (deterministic decoder engrams -> stable fingerprints).

    Pre-registration: experiments/sweeps/phase810_diagonal_production_wiring_design.md
    """

    def __init__(
        self,
        num_blocks: int = 8192,
        block_dim: int = 8,
        num_actions: int = 16,
        device: str = "cpu",
        d_model: int = 65536,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.d = num_blocks * block_dim
        self.d_model = d_model
        self.num_actions = num_actions
        # Phase corrections per action; zero init = identity rotation.
        self.phase_correction = nn.Parameter(
            torch.zeros(num_actions, self.d, dtype=torch.float32, device=device)
        )
        # Lazy action-wave fingerprint prototypes (cosine identity).
        self.register_buffer("_fp_buf", torch.zeros(num_actions, self.d, dtype=torch.float32))
        self._fp_count = 0

    # -- production-compat surface -------------------------------------------
    @property
    def rank(self) -> int:
        return 0  # diagonal path has no low-rank field channel

    @property
    def requested_rank(self) -> int:
        return 0

    def _retract(self, residual_only: bool = False):
        """No-op for the diagonal path (phase params are unconstrained)."""
        return None

    # -- action indexing ------------------------------------------------------
    @torch.no_grad()
    def _action_index(self, action_wave: torch.Tensor) -> int:
        w = action_wave.detach().reshape(1, -1).to(self._fp_buf.device)
        w = w / (torch.norm(w, p=2, dim=-1, keepdim=True) + 1e-9)
        if self._fp_count == 0:
            self._fp_buf[0].copy_(w.reshape(-1))
            self._fp_count = 1
            return 0
        sims = torch.nn.functional.cosine_similarity(w, self._fp_buf[: self._fp_count], dim=-1)
        best = int(sims.argmax())
        if sims[best] > 0.999:
            return best  # same deterministic decoder engram
        if self._fp_count < self.num_actions:
            self._fp_buf[self._fp_count].copy_(w.reshape(-1))
            self._fp_count += 1
            return self._fp_count - 1
        return best  # capacity full -> nearest (bounded)

    # -- core ops -------------------------------------------------------------
    def _phasor(self, real_wave: torch.Tensor) -> torch.Tensor:
        """Real wave -> unit-modulus complex phasor via arccos bridge."""
        w = real_wave.reshape(-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.polar(torch.ones_like(w), torch.acos(w))  # [d] complex

    def forward(self, state_wave: torch.Tensor, action_wave: torch.Tensor) -> torch.Tensor:
        """Production interface: real [B, 8] -> real [B, 8] per-block unit."""
        s = state_wave.reshape(-1)
        a = action_wave.reshape(-1)
        idx = self._action_index(action_wave)
        alpha = torch.acos(s.clamp(-1.0 + 1e-6, 1.0 - 1e-6))        # [d]
        theta = math.pi * a.clamp(-1.0, 1.0)                        # [d]
        phi = self.phase_correction[idx].to(s.device)               # [d]
        pred = torch.cos(alpha + theta + phi)                       # [d] real
        out = pred.reshape(self.num_blocks, self.block_dim)
        return out / (torch.norm(out, p=2, dim=-1, keepdim=True) + 1e-9)

    def _real_sagnac(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """Production real-metric Sagnac: 1 - dot(p, o)/(|p||o|)."""
        p = pred.reshape(-1)
        o = actual.reshape(-1)
        return 1.0 - torch.dot(p, o) / (torch.norm(p) * torch.norm(o)).clamp(min=1e-12)

    # -- closed-form learning (8.9-B port, Phase 8.10 remedy) ------------------
    @torch.no_grad()
    def update_wirtinger(
        self,
        state_wave: torch.Tensor,
        action_wave: torch.Tensor,
        observed_next_wave: torch.Tensor,
        lr: float = 1.0,
        return_loss: bool = True,
    ) -> float:
        """Online phase update on the PRODUCTION real-Sagnac loss.

        Phase 8.10 remedy (pre-registered): the closed-form elementwise angle
        residual is BIASED in the production real regime — per-block L2
        normalization scales cos values, so acos(cos(phi+delta)/c) != phi+delta
        (OBSERVED contract-G3 post-fit Sagnac 0.2912 with the angle estimator).
        Gradient descent on the exact direction loss
        1 - dot(pred, obs)/(|pred||obs|) reaches Sagnac ~ 0: at phi = delta the
        prediction direction equals the target exactly. Step count scales with
        lr (runner semantics: lr=0.05 -> 1 damped step; lr=1.0 -> 20 steps).
        Returns the pre-update real-metric Sagnac loss.
        """
        with torch.no_grad():
            pred0 = self.forward(state_wave, action_wave)
            pre = float(self._real_sagnac(pred0, observed_next_wave.detach()))
        steps = max(1, int(round(lr * 20.0)))
        self._sgd_fit(
            [state_wave], [action_wave], [observed_next_wave],
            steps=steps, step_lr=0.05)
        return pre

    @torch.no_grad()
    def fit_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        observed_nexts: torch.Tensor,
        iters: int = 1,
        lr: float = 1.0,
    ) -> float:
        """Batch gradient fit on the production real-Sagnac loss.

        steps = max(1, iters * 25) at step_lr = lr * 0.04 (default: 25 steps
        @ 0.04). Returns the pre-fit mean real-metric Sagnac. The same
        phase-parameter mechanism as update_wirtinger; grouped only by the
        action fingerprint for multi-action batches.
        """
        with torch.no_grad():
            pre_losses = []
            for i in range(len(states)):
                pred = self.forward(states[i], actions[i])
                pre_losses.append(float(self._real_sagnac(pred, observed_nexts[i].detach())))
            pre = float(torch.tensor(pre_losses).mean()) if pre_losses else 0.0
        steps = max(1, int(iters) * 25)
        step_lr = float(lr) * 0.04
        self._sgd_fit(states, actions, observed_nexts, steps=steps, step_lr=step_lr)
        return pre

    def _sgd_fit(
        self,
        states,
        actions,
        observed_nexts,
        steps: int,
        step_lr: float,
    ):
        """SGD on the phase parameters minimizing the real-Sagnac direction loss."""
        phase = self.phase_correction
        opt = torch.optim.SGD([phase], lr=step_lr)
        with torch.enable_grad():
            for _ in range(steps):
                opt.zero_grad()
                total = torch.zeros((), device=phase.device, dtype=torch.float32)
                for i in range(len(states)):
                    pred = self.forward(states[i], actions[i])
                    total = total + self._real_sagnac(pred, observed_nexts[i].detach())
                total.backward()
                opt.step()

    # -- checkpoint persistence -------------------------------------------------
    def field_channel_wave(self) -> torch.Tensor:
        """Pack the phase corrections into a wave-shaped tensor (diagonal branch)."""
        return self.phase_correction.detach().reshape(-1).cpu()

    @torch.no_grad()
    def load_field_channel_wave(self, wave: torch.Tensor):
        """Inverse of field_channel_wave (diagonal branch)."""
        expected = self.num_actions * self.d
        wave = wave.detach().cpu().float()
        assert wave.numel() >= expected, (
            f"diagonal field wave too short: {wave.numel()} < {expected}")
        self.phase_correction.copy_(wave[:expected].reshape(self.num_actions, self.d))
