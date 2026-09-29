"""HENRI continuous action head — flow-matching 7-DoF VLA action decoder.

WHY THIS EXISTS
===============
A vision-language-ACTION model must emit CONTINUOUS actions. HENRI's existing
heads emit DISCRETE tokens:
  * `arc_action_head.ActionHead`      -> logits over ARC actions (discrete)
  * `henri_decoder`                   -> argmax over a token vocabulary
  * `algebraic_action_head`           -> SU(3) lexical snap (discrete)
None of them can represent a 7-DoF manipulation command (dx,dy,dz,droll,dpitch,
dyaw,gripper), which is continuous.

METHOD (flow matching / rectified flow; Lipman et al. arXiv:2210.02747)
======================================================================
Learn a velocity field v_theta(x, t, cond) that transports a Gaussian source
x_0 ~ N(0, I) to the action distribution x_1 at t=1 along a straight path:

    x_t   = (1 - t) x_0 + t x_1
    u_t   = x_1 - x_0                 (target velocity, CONSTANT along t)
    L     = E || v_theta(x_t, t, cond) - u_t ||^2

Sampling integrates the ODE with Euler steps from t=0 to t=1. This is preferred
over DDPM here because the target velocity is analytic (no score estimation) and
a 4-8 step sampler is enough in practice for smooth action manifolds.

CONDITIONING
============
`cond` is the HENRI phase-vector state [B, D] (D=65,536 at production scale).
A fixed random projection D -> d_cond keeps the head small; the projection is
NOT trained (it is a deterministic sketch), so the head trains while the wave
core stays untouched. That keeps this head compatible with Contract A: no
[D, D] object is ever formed; the projection is [D, d_cond].

ACTION NORMALIZATION
====================
Actions are per-dimension standardized (mean 0 / std 1) before training and
de-standardized at inference. Without this the position dims (metres) and
rotation dims (radians) have incompatible scales and the flow target is
ill-conditioned.

HONEST LIMITS (stated, not implied away)
========================================
* No real robot data is attached yet. `synthetic_actions()` exists ONLY to make
  the head mechanically trainable and its loss curve measurable. A loss curve on
  synthetic data proves the head learns a FUNCTION, not that it drives a robot.
* 7-DoF is a convention (OSC/EEF delta commands); it is configurable and the
  numbers here are not a claim about any specific robot.
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

N_DOF_DEFAULT = 7


class ActionHeadError(RuntimeError):
    """Typed failure for the continuous action head."""


class FlowMatchingActionHead(nn.Module):
    """Velocity field v_theta(x_t, t, cond) for 7-DoF continuous actions."""

    def __init__(self, n_dof: int = N_DOF_DEFAULT, d_cond: int = 512,
                 d_hidden: int = 1024, n_layers: int = 4,
                 phase_dim: int = 65536, seed: int = 0):
        super().__init__()
        if n_dof < 1:
            raise ActionHeadError("n_dof must be >= 1")
        if phase_dim < 1 or d_cond < 1:
            raise ActionHeadError("phase_dim and d_cond must be >= 1")
        self.n_dof = int(n_dof)
        self.d_cond = int(d_cond)
        self.phase_dim = int(phase_dim)

        # Deterministic, NON-TRAINED sketch of the phase vector: [phase_dim, d_cond].
        # Registered as a buffer so it moves with .to(device) and is saved with
        # the state dict, but it never receives a gradient.
        g = torch.Generator().manual_seed(seed)
        proj = torch.randn(self.phase_dim, self.d_cond, generator=g) / math.sqrt(self.phase_dim)
        self.register_buffer("phase_sketch", proj, persistent=True)

        # Sinusoidal time embedding (standard in flow/diffusion heads).
        self.time_dim = 128
        self.register_buffer("time_freqs",
                             torch.exp(torch.linspace(0, math.log(1000.0), self.time_dim // 2)),
                             persistent=True)

        d_in = self.n_dof + self.d_cond + self.time_dim
        layers: list[nn.Module] = []
        prev = d_in
        for _ in range(max(1, n_layers)):
            layers += [nn.Linear(prev, d_hidden), nn.SiLU()]
            prev = d_hidden
        layers += [nn.Linear(prev, self.n_dof)]
        self.net = nn.Sequential(*layers)
        # Zero-init the last layer: the head starts by predicting v=0, so the
        # initial ODE is the identity path. This is the flow-matching analogue of
        # a zero-init residual head and makes early training stable.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    # ------------------------------------------------------------------ utils
    def encode_cond(self, phase: torch.Tensor) -> torch.Tensor:
        """[B, phase_dim] -> [B, d_cond]. Deterministic sketch, no gradient path."""
        if phase.shape[-1] != self.phase_dim:
            raise ActionHeadError(
                f"phase dim {phase.shape[-1]} != head phase_dim {self.phase_dim}")
        return phase.to(torch.float32) @ self.phase_sketch

    def time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] in [0,1] -> [B, time_dim]."""
        ang = t[:, None] * self.time_freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor,
                 cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, cond, self.time_embed(t)], dim=-1))

    # ------------------------------------------------------------------ train
    def loss(self, action: torch.Tensor, phase: torch.Tensor,
             t: torch.Tensor | None = None) -> torch.Tensor:
        """Conditional flow-matching loss. action: [B, n_dof] in NORMALIZED units."""
        B = action.shape[0]
        if t is None:
            t = torch.rand(B, device=action.device, dtype=action.dtype)
        x0 = torch.randn_like(action)
        x_t = (1.0 - t[:, None]) * x0 + t[:, None] * action
        u_t = action - x0
        v = self.velocity(x_t, t, self.encode_cond(phase))
        return F.mse_loss(v, u_t)

    # ------------------------------------------------------------------ infer
    @torch.no_grad()
    def sample(self, phase: torch.Tensor, steps: int = 8,
               x0: torch.Tensor | None = None) -> torch.Tensor:
        """Integrate the flow ODE x' = v(x, t, cond) with `steps` Euler steps."""
        if steps < 1:
            raise ActionHeadError("steps must be >= 1")
        cond = self.encode_cond(phase)
        B = cond.shape[0]
        if x0 is None:
            x0 = torch.randn(B, self.n_dof, device=cond.device, dtype=torch.float32)
        x = x0
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=cond.device, dtype=torch.float32)
            x = x + dt * self.velocity(x, t, cond)
        return x


# ------------------------------------------------------------------ normalizer
class ActionNormalizer:
    """Per-dimension standardisation. Position (m) and rotation (rad) differ in
    scale; without this the flow target is ill-conditioned."""

    def __init__(self, n_dof: int = N_DOF_DEFAULT):
        self.n_dof = int(n_dof)
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, actions: torch.Tensor) -> "ActionNormalizer":
        if actions.shape[-1] != self.n_dof:
            raise ActionHeadError(f"expected {self.n_dof} dims, got {actions.shape[-1]}")
        self.mean = actions.mean(dim=0).to(torch.float32)
        self.std = actions.std(dim=0).clamp(min=1e-6).to(torch.float32)
        return self

    def normalize(self, a: torch.Tensor) -> torch.Tensor:
        self._require()
        # DEVICE-AWARE (defect fixed 2026-09-28): mean/std are fitted on CPU but
        # the policy runs on CUDA; a bare multiply raised
        # "Expected all tensors to be on the same device". Move the stats to the
        # input's device/dtype instead of assuming either side.
        m = self.mean.to(device=a.device, dtype=torch.float32)
        s = self.std.to(device=a.device, dtype=torch.float32)
        return (a.to(torch.float32) - m) / s

    def denormalize(self, a: torch.Tensor) -> torch.Tensor:
        self._require()
        m = self.mean.to(device=a.device, dtype=torch.float32)
        s = self.std.to(device=a.device, dtype=torch.float32)
        return a.to(torch.float32) * s + m

    def _require(self) -> None:
        if self.mean is None or self.std is None:
            raise ActionHeadError("call fit() before normalize/denormalize")

    def state_dict(self) -> dict:
        self._require()
        return {"n_dof": self.n_dof,
                "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "ActionNormalizer":
        n = cls(int(d["n_dof"]))
        n.mean = torch.tensor(d["mean"], dtype=torch.float32)
        n.std = torch.tensor(d["std"], dtype=torch.float32)
        return n


# ------------------------------------------------------------------ synthetic
def synthetic_actions(n: int, n_dof: int = N_DOF_DEFAULT, phase_dim: int = 65536,
                     seed: int = 0, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Mechanically-trainable fixture. NOT robot data.

    The action is a smooth function of a few phase coordinates, so a correct head
    can drive the loss toward the noise floor. Its purpose is to prove the head
    learns a function and to give a measurable curve; it is NOT evidence about
    any robot or benchmark.
    """
    g = torch.Generator().manual_seed(seed)
    phase = torch.randn(n, phase_dim, generator=g)
    phase = phase / phase.norm(dim=-1, keepdim=True)
    # a low-rank teacher: 7 DoF driven by 5 random phase directions
    W = torch.randn(5, n_dof, generator=g)
    key = phase[:, : 5 * 32].reshape(n, 5, 32).mean(dim=-1)     # [n, 5]
    act = torch.tanh(key @ W) * 0.5
    return act.to(device), phase.to(device)


def _self_test() -> int:
    out = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            out[device] = "SKIP (no cuda)"
            continue
        n, pd = 256, 4096
        head = FlowMatchingActionHead(n_dof=7, d_cond=256, d_hidden=512,
                                      n_layers=3, phase_dim=pd).to(device)
        norm = ActionNormalizer(7)
        act, ph = synthetic_actions(n, 7, pd, seed=1, device=device)
        act = norm.fit(act).normalize(act)
        opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
        first = last = None
        for i in range(400):
            l = head.loss(act, ph)
            opt.zero_grad(); l.backward(); opt.step()
            if i == 0: first = float(l.item())
            last = float(l.item())
        # baseline: predicting the mean velocity (zero) gives E||u||^2
        base = float((act - torch.randn_like(act)).pow(2).mean().item())
        out[device] = {"loss_first": round(first, 5), "loss_last": round(last, 5),
                       "baseline_zero_vel": round(base, 5),
                       "learned": bool(last < 0.6 * base),
                       "n_params": sum(p.numel() for p in head.parameters()
                                       if p.requires_grad),
                       "sketch_frozen": not head.phase_sketch.requires_grad}
        # sampling sanity: finite, right shape, bounded
        s = head.sample(ph[:8], steps=8)
        out[device]["sample_shape"] = list(s.shape)
        out[device]["sample_finite"] = bool(torch.isfinite(s).all().item())
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
