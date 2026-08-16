"""
Project HENRI V2: Adaptive Viscoelastic Thermostat & Koopman Relaxation Engine
Subsystem: Inner-Loop Parameter Adaptation / Anisotropic Langevin SDE Solver
Hardware Target: CUDA / PyTorch Tensor Substrate
Verified from: HENRI V8 Telemetry Evaluation.pdf (Run 1785290013 Audit)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List


def _haar_forward(x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Orthonormal 1D Haar transform. Returns (coarse, detail_bands) where
    detail_bands[0] is the finest level. Input length must be a power of two."""
    a = x.clone()
    detail_bands = []
    while a.shape[0] > 1:
        half = a.shape[0] // 2
        a_ = (a[0::2] + a[1::2]) / math.sqrt(2.0)
        d_ = (a[0::2] - a[1::2]) / math.sqrt(2.0)
        detail_bands.append(d_)
        a = a_
    return a, detail_bands


def _haar_inverse(coarse: torch.Tensor, detail_bands: List[torch.Tensor]) -> torch.Tensor:
    """Inverse of _haar_forward. detail_bands[0] is the finest level.

    Reconstruction iterates the bands COARSEST-first (reversed list): each
    detail band has the same length as the current reconstruction, doubling
    it each step back toward the original length.
    """
    rec = coarse.clone()
    for d_band in reversed(detail_bands):
        new = torch.empty(rec.shape[0] * 2, device=rec.device, dtype=rec.dtype)
        new[0::2] = (rec + d_band) / math.sqrt(2.0)
        new[1::2] = (rec - d_band) / math.sqrt(2.0)
        rec = new
    return rec


class AdaptiveViscoelasticThermostat(nn.Module):
    """
    Executes real-time parameter adaptation on stiff variational loss landscapes.
    Dynamically modulates Langevin friction and applies Stiefel manifold projection.
    """
    def __init__(
        self,
        d_model: int = 65536,
        base_learning_rate: float = 1e-3,
        lambda_threshold: float = 0.10,
        max_lambda: float = 5.0,
        stiefel_iters: int = 3,
        device: Optional[str] = None,
        signal_lock_steps: int = 12,
        use_wavelet_gating: bool = False,
        signal_mask_kappa: float = 4.0,
        signal_dominance_threshold: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.base_lr = base_learning_rate
        self.lambda_threshold = lambda_threshold
        self.max_lambda = max_lambda
        self.stiefel_iters = stiefel_iters
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        # Phase 5 Task 2.2 (WaiT wait-for-signal spectral masking): steps of
        # sustained signal dominance after which thermal noise is fully
        # silenced. Kept as a parameter so the preregistered lock criterion
        # (lock <= 12 steps) is testable at other budgets.
        self.signal_lock_steps = signal_lock_steps
        self._signal_lock_counter = 0
        # Named default-OFF flag: when False, step_viscoelastic_creep is
        # byte-identical to the pre-Phase-5 path (isotropic Langevin noise).
        # When True, thermal noise is band-gated by the Haar-decomposed
        # gradient (wait-for-signal) and fully silenced after the lock.
        self.use_wavelet_gating = use_wavelet_gating
        self.signal_mask_kappa = signal_mask_kappa
        self.signal_dominance_threshold = signal_dominance_threshold

    def compute_anisotropic_friction(
        self,
        lambda_active: float,
        sagnac_delta: float
    ) -> float:
        """
        Computes Langevin friction coefficient gamma inversely proportional to constraint stiffness.
        """
        if lambda_active <= self.lambda_threshold:
            return 1.0
        
        stiffness_ratio = lambda_active / self.max_lambda
        friction = 1.0 / (1.0 + math.log1p(stiffness_ratio * 10.0) * sagnac_delta)
        return max(0.05, friction)

    def project_stiefel_manifold(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
        Enforces orthogonality constraint W^T W = I using Newton-Schulz iterations.
        """
        W = weight_matrix
        if W.dim() != 2:
            return W
            
        rows, cols = W.shape
        if rows < cols:
            W = W.T
            
        identity = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
        for _ in range(self.stiefel_iters):
            W = 0.5 * W @ (3.0 * identity - W.T @ W)
            
        return W.T if rows < cols else W

    def compute_wavelet_gated_noise(
        self,
        weight_matrix: torch.Tensor,
        grad_loss: torch.Tensor,
        temperature: float,
        effective_lr: float,
        base_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, bool]:
        """Wait-for-signal spectral noise gating (Phase 5 Task 2.2 / WaiT).

        Haar-decompose the gradient; per-band gates g_b = 1/(1 + kappa*E_b/E_max)
        suppress thermal noise on strong-signal bands while keeping it on null
        bands (exploration where there is no signal). After
        signal_lock_steps of sustained dominance (E_max/E_total >= threshold),
        noise is fully silenced (lock). Returns (gated_noise, dominance, locked).

        The gradient length is padded to the next power of two for the Haar
        transform and truncated back; the returned noise matches
        weight_matrix.shape.
        """
        flat = grad_loss.reshape(-1).float()
        n = flat.shape[0]
        n_pad = 1 << (n - 1).bit_length() if n > 1 else 1
        x = flat
        if n_pad != n:
            x = torch.cat([flat, torch.zeros(n_pad - n, device=flat.device, dtype=flat.dtype)])
        coarse, detail = _haar_forward(x)
        energies = torch.stack(
            [d.pow(2).mean() for d in detail] + [coarse.pow(2).mean()])
        e_max = energies.max().clamp(min=1e-12)
        gates = 1.0 / (1.0 + self.signal_mask_kappa * energies / e_max)
        e_total = energies.sum().clamp(min=1e-12)
        dominance = float((e_max / e_total).item())
        if dominance >= self.signal_dominance_threshold:
            self._signal_lock_counter += 1
        else:
            self._signal_lock_counter = 0
        locked = self._signal_lock_counter >= self.signal_lock_steps
        if locked:
            gates = torch.zeros_like(gates)
        # Phase 5 P2 fix (gate lesson): the gradient measures band energies
        # and dominance only. The stochastic noise is a FRESH white draw
        # (base_noise when provided for paired A/B), Haar-transformed and
        # gated per coefficient INCLUDING the coarse coefficient. Reusing
        # gradient coefficients as noise (previous coarse*gates[-1]) leaked
        # deterministic gradient content into the stochastic term.
        if base_noise is not None:
            if base_noise.numel() != n:
                raise ValueError(
                    f"base_noise numel {base_noise.numel()} != gradient numel {n}")
            noise_flat = base_noise.reshape(-1).float()
        else:
            noise_flat = torch.randn(n_pad, device=flat.device, dtype=flat.dtype)
        if n_pad != noise_flat.shape[0]:
            noise_flat = torch.cat(
                [noise_flat,
                 torch.zeros(n_pad - noise_flat.shape[0],
                             device=noise_flat.device, dtype=noise_flat.dtype)])
        coarse_n, detail_n = _haar_forward(noise_flat)
        gated = _haar_inverse(coarse_n * gates[-1],
                              [d * gates[i] for i, d in enumerate(detail_n)])
        gated = gated[:n].reshape(weight_matrix.shape)
        scale = math.sqrt(2.0 * temperature * effective_lr)
        return gated * scale, dominance, locked

    def step_viscoelastic_creep(
        self,
        weight_matrix: torch.Tensor,
        grad_loss: torch.Tensor,
        lambda_active: float,
        sagnac_delta: float,
        temperature: float = 1e-4,
        base_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes adaptive Langevin SDE step with anisotropic damping and manifold projection.
        """
        friction = self.compute_anisotropic_friction(lambda_active, sagnac_delta)
        
        # Adaptive step size scaling based on stiffness
        effective_lr = (self.base_lr / friction) * (1.0 + sagnac_delta)
        
        # Phase 5 Task 2.2 (WaiT wait-for-signal): flag-gated spectral noise.
        # Default path (use_wavelet_gating=False) is byte-identical to the
        # legacy isotropic Langevin injection. Friction/LR/veto math is
        # untouched in both paths.
        if self.use_wavelet_gating:
            gated_noise, dominance, locked = self.compute_wavelet_gated_noise(
                weight_matrix, grad_loss, temperature, effective_lr,
                base_noise=base_noise)
            noise = gated_noise
        else:
            # Anisotropic noise injection (legacy isotropic path)
            # With base_noise given, use the paired draw (A/B harness only).
            if base_noise is not None:
                noise = base_noise * math.sqrt(2.0 * temperature * effective_lr)
            else:
                noise = torch.randn_like(weight_matrix) * math.sqrt(2.0 * temperature * effective_lr)
        
        # SDE update: dW = - (eta / gamma) * grad + noise
        updated_weight = weight_matrix - effective_lr * grad_loss + noise
        
        # Retract onto Stiefel Manifold if weight is 2D
        if weight_matrix.dim() == 2 and weight_matrix.shape[0] == weight_matrix.shape[1]:
            updated_weight = self.project_stiefel_manifold(updated_weight)
            
        telemetry = {
            "effective_lr": effective_lr,
            "langevin_friction": friction,
            "sagnac_delta": sagnac_delta,
            "lambda_active": lambda_active,
            "weight_norm": float(torch.norm(updated_weight).item())
        }
        if self.use_wavelet_gating:
            telemetry["wavelet_dominance"] = dominance
            telemetry["wavelet_locked"] = locked
        
        return updated_weight, telemetry


    def apply_anisotropic_langevin_creep(
        self,
        task_operator: torch.Tensor,
        grad_operator: torch.Tensor,
        sagnac_delta: torch.Tensor,
        t_base: float = 1e-4,
        alpha: float = 5.0,
        threshold: float = 0.1000,
        step_lr: float = 1e-3,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Phase 8.17 C3: channel-wise viscoelastic Langevin creep on W_task.

        Spec: HENRI-SPEC-2026-08-PHASE8.17-ALIGNMENT (SHA 1342944c...).
        T_d = t_base * exp(alpha * sagnac_delta_d); thermal variance is
        injected into failing channels (delta_d > threshold) only, preserving
        Lie invariants on stable channels. SU(3) polar retraction per block
        (SVD -> U V^dag -> det-correction) after the SDE step.
        """
        delta = sagnac_delta.reshape(-1)
        t_d = t_base * torch.exp(alpha * delta)  # [NB] channel temperatures
        failing = delta > threshold  # [NB] bool mask
        if noise is None:
            n = torch.randn(
                task_operator.shape,
                device=task_operator.device,
                dtype=task_operator.dtype,
            )
        else:
            n = noise
        # complex noise: std per real component = sqrt(T_d), E|n|^2 = 2 T_d
        scale = torch.sqrt(t_d).to(task_operator.dtype)[:, None, None]
        injected = n * scale
        updated = task_operator - step_lr * grad_operator + injected
        # polar retraction per block onto SU(3)
        uu, _s, vhv = torch.linalg.svd(updated)
        proj = torch.einsum("bij,bjk->bik", uu, vhv)
        det = torch.linalg.det(proj)
        phase = torch.pow(det.conj(), 1.0 / 3.0).unsqueeze(-1).unsqueeze(-1)
        proj = proj * phase
        t_f = float(t_d[failing].max().item()) if bool(failing.any()) else 0.0
        t_s = float(t_d[~failing].min().item()) if bool((~failing).any()) else t_base
        ratio = (t_f / t_s) if t_s > 0 else float("inf")
        telemetry = {
            "n_failing": int(failing.sum().item()),
            "t_failing": t_f,
            "t_stable": t_s,
            "thermal_ratio": ratio,
            "su3_det_min": float(torch.linalg.det(proj).abs().min().item()),
        }
        return proj, telemetry


def verify_thermostat_adaptation() -> bool:
    """Verification routine for the Adaptive Viscoelastic Thermostat."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thermostat = AdaptiveViscoelasticThermostat(d_model=4096, device=device)
    
    # Simulate a 2D Koopman weight matrix
    W = torch.eye(256, device=device) + torch.randn(256, 256, device=device) * 0.01
    grad = torch.randn(256, 256, device=device) * 0.5
    
    # Low constraint stiffness pass
    W_low, telem_low = thermostat.step_viscoelastic_creep(W, grad, lambda_active=0.005, sagnac_delta=0.07)
    
    # High constraint stiffness pass (Simulating Step 20 in telemetry)
    W_high, telem_high = thermostat.step_viscoelastic_creep(W, grad, lambda_active=0.377, sagnac_delta=0.424)
    
    print(f"Substrate Hardware: {device.upper()}")
    print(f"[Low Stiffness]   Effective LR: {telem_low['effective_lr']:.6f} | Friction: {telem_low['langevin_friction']:.4f}")
    print(f"[High Stiffness]  Effective LR: {telem_high['effective_lr']:.6f} | Friction: {telem_high['langevin_friction']:.4f}")
    
    assert telem_high['effective_lr'] > telem_low['effective_lr'], "Adaptive LR scaling failed to increase under stiffness."
    print("Thermostat adaptation verification passed successfully.")
    return True


if __name__ == "__main__":
    verify_thermostat_adaptation()
