"""F4 non-linear context-conditioned egress head (carrier F4).

Spec: HENRI-SPEC-2026-08-F4-NONLINEAR-EGRESS (sealed; branch carrier/f4).
Parent verdict: F3_GATES_VERDICT=K1_KILLED (event 8c47bf5c).

Three tiers:
  Tier 1: task-functor phase pre-inversion (qFHRR Z_256 unbinding) via the
          LIVE HolographicTaskFunctorCompiler (zone_c_epistemic_axiom_harness).
  Tier 2: 3-layer MLP 65536 -> 2048 -> 512 -> 7 (GELU + LayerNorm).
  Tier 3: in-situ SGLD on the final layer only (W3, b3), scheduled thermal
          noise T(t) = T0*(1+0.05t)^-0.55, unit-normalized increments.

DEFAULT-OFF: this module is never imported by production_arc_run.py or
henri_egress.py (contract-tested). Consumers must gate on HENRI_F4_EGRESS=1.
No dense [D,D] tensors: W1 is [hidden1, d_model] (134.2M params, 537 MB fp32).

AMENDMENT (disclosed, ratification pending): spec Tier-3 box states T0=0.5
with unit-normalized noise. At ||h2|| ~ sqrt(hidden2), one noise step shifts
logits by ~sqrt(hidden2) while a gradient step shifts by eta*||grad|| — CE
descent in 3 steps is unsatisfiable at T0=0.5. The same spec section names
adapt_in_context_sgld_wave (henri_decoder.py:209, default T0=1e-6) as the
reference protocol; Tier-3 default t0 := 1e-6.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Spec 4.4 hyperparameters (frozen; no CLI tuning in the gates harness).
FROZEN_LR = 1e-3
FROZEN_WD = 1e-4
FROZEN_BATCH = 64
FROZEN_EPOCHS = 20
FROZEN_RIDGE = 1e-3  # Tier-1 compile / linear-control ridge


def ring_to_real(q: torch.Tensor) -> torch.Tensor:
    """Z_256 uint8 ring -> real wave in [-1, 1] (sanctioned mapping)."""
    return q.to(torch.float32) / 255.0 * 2.0 - 1.0


def real_to_ring(w: torch.Tensor) -> torch.Tensor:
    """Real wave [-1, 1] -> Z_256 uint8 ring (sanctioned mapping)."""
    return ((w.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).to(torch.uint8)


def unbind_w_task(
    psi: torch.Tensor,
    w_task: torch.Tensor,
    codec,
    D: int = 65536,
) -> torch.Tensor:
    """Tier 1: circular-convolution unbinding in the qFHRR phase domain.

    psi may be a real wave (auto ring conversion) or an already-uint8 ring.
    Returns the unbound wave normalized to S^{D-1}.
    """
    ring = psi if psi.dtype == torch.uint8 else real_to_ring(psi)
    unbound_ring = codec.unbind_hadamard(ring, w_task.to(ring.device))
    real = ring_to_real(unbound_ring)
    return F.normalize(real, p=2.0, dim=-1)


def _sgld_thermal_schedule(t: int, t0: float) -> float:
    return t0 * (1.0 + 0.05 * float(t)) ** (-0.55)


class F4NonLinearEgressHead(nn.Module):
    """Tier-2 deep non-linear compression head (65536 -> 2048 -> 512 -> 7).

    W1/W2 Kaiming init, W3/b3 zero init (spec 4.4). Seed-deterministic via an
    explicit torch.Generator (cross-process reproducible state dict).
    """

    def __init__(
        self,
        d_model: int = 65536,
        hidden1: int = 2048,
        hidden2: int = 512,
        n_actions: int = 7,
        seed: int = 20260830,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.n_actions = int(n_actions)
        self.seed = int(seed)

        g = torch.Generator(device="cpu").manual_seed(self.seed)
        self.W1 = nn.Parameter(torch.empty(self.hidden1, self.d_model))
        self.b1 = nn.Parameter(torch.zeros(self.hidden1))
        self.W2 = nn.Parameter(torch.empty(self.hidden2, self.hidden1))
        self.b2 = nn.Parameter(torch.zeros(self.hidden2))
        self.W3 = nn.Parameter(torch.zeros(self.n_actions, self.hidden2))
        self.b3 = nn.Parameter(torch.zeros(self.n_actions))

        nn.init.kaiming_normal_(self.W1, a=math.sqrt(5), mode="fan_in",
                                nonlinearity="leaky_relu", generator=g)
        nn.init.kaiming_normal_(self.W2, a=math.sqrt(5), mode="fan_in",
                                nonlinearity="leaky_relu", generator=g)
        # W3/b3 stay zero (spec 4.4)

        self.ln1 = nn.LayerNorm(self.hidden1, eps=1e-5)
        self.ln2 = nn.LayerNorm(self.hidden2, eps=1e-5)
        # Spec 2: "Trainable: W1,W2,W3 at train time". LayerNorm scale/bias are
        # frozen so the trainable set is EXACTLY the three weight layers.
        self.ln1.weight.requires_grad_(False)
        self.ln1.bias.requires_grad_(False)
        self.ln2.weight.requires_grad_(False)
        self.ln2.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, d_model] fp32 -> logits [N, n_actions]."""
        h1 = F.gelu(self.ln1(F.linear(x, self.W1, self.b1)))
        h2 = F.gelu(self.ln2(F.linear(h1, self.W2, self.b2)))
        logits = F.linear(h2, self.W3, self.b3)
        return logits

    # ---- Tier-2 training (fold train envs) --------------------------------
    def train_head(
        self,
        x: torch.Tensor,
        onehot: torch.Tensor,
        lr: float = FROZEN_LR,
        wd: float = FROZEN_WD,
        batch: int = FROZEN_BATCH,
        epochs: int = FROZEN_EPOCHS,
        seed: int = 0,
    ) -> Dict[str, float]:
        """AdamW training over the full head (W1,W2,W3) on train-fold rows."""
        self.train()
        x = x.to(self.W1.device).to(torch.float32)
        onehot = onehot.to(self.W1.device).to(torch.float32)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        n = x.shape[0]
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        losses: List[float] = []
        for _ in range(int(epochs)):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, int(batch)):
                idx = perm[i:i + int(batch)]
                logits = self.forward(x[idx])
                loss = F.cross_entropy(logits, onehot[idx].argmax(dim=-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
        self.eval()
        return {
            "epochs": int(epochs),
            "n_train": int(n),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_mean": float(sum(losses) / len(losses)) if losses else None,
        }

    # ---- Tier 3: in-situ SGLD, W3/b3 only --------------------------------
    def adapt_w3_sgld(
        self,
        psi: torch.Tensor,
        onehot: torch.Tensor,
        steps: int = 3,
        eta: float = 1e-3,
        t0: float = 1e-6,
        dt: float = 1.0,
        seed: int = 0,
        weight_decay: float = 1e-4,
    ) -> Dict[str, float]:
        """Scheduled SGLD on the final layer only (spec Tier 3).

        L = CE(z, y_demo) + weight_decay*||W3||^2.
        Noise: unit-normalized, amplitude sqrt(2*T(t)*dt).
        W1/W2 are frozen (never updated).
        """
        self.train()
        psi = psi.to(self.W1.device).to(torch.float32)
        onehot = onehot.to(self.W1.device).to(torch.float32)
        if psi.dim() == 1:
            psi = psi.unsqueeze(0)
            onehot = onehot.unsqueeze(0)
        w3_before = self.W3.detach().clone()
        loss_first = None
        loss_last = None
        for t in range(int(steps)):
            temp_t = _sgld_thermal_schedule(t, t0=t0)
            logits = self.forward(psi)
            ce = F.cross_entropy(logits, onehot.argmax(dim=-1))
            reg = weight_decay * torch.sum(self.W3 ** 2)
            loss = ce + reg
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                g3 = self.W3.grad
                gb3 = self.b3.grad
                if g3 is not None:
                    g3 = torch.nan_to_num(g3, nan=0.0, posinf=0.0, neginf=0.0)
                    rng = torch.Generator(device=self.W3.device).manual_seed(
                        int(seed) + t)
                    xi = torch.randn_like(self.W3, generator=rng)
                    xi = F.normalize(xi, p=2.0, dim=-1)
                    noise = math.sqrt(2.0 * temp_t * dt) * xi
                    self.W3.data.sub_(eta * g3).add_(noise)
                if gb3 is not None:
                    gb3 = torch.nan_to_num(gb3, nan=0.0, posinf=0.0, neginf=0.0)
                    self.b3.data.sub_(eta * gb3)
            if loss_first is None:
                loss_first = float(loss.item())
            loss_last = float(loss.item())
        self.eval()
        delta_w3 = float((self.W3.detach() - w3_before).norm().item())
        return {
            "adapt_protocol": "w3_scheduled_sgld",
            "steps": int(steps),
            "delta_w3_fro": delta_w3,
            "loss_first": loss_first,
            "loss_last": loss_last,
            "temperature_schedule": "T0*(1+0.05t)^-0.55",
            "t0": float(t0),
            "finite": bool(torch.isfinite(self.W3).all().item()),
        }


def compile_env_w_task(
    codec,
    compiler,
    demo_psi: torch.Tensor,
    demo_actions: Sequence[str],
) -> torch.Tensor:
    """Compile a per-env W_task from the env's demo prefix.

    Pair convention (matches contract test + spec Tier 1): X = psi row ring,
    Y = qFHRR ring of the action name (encode_text(f"ACTION{k}")).
    Returns W_task in Z_256^D.
    """
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i, psi_i in enumerate(demo_psi):
        x_ring = real_to_ring(psi_i)
        y_ring = codec.encode_text(str(demo_actions[i]))
        pairs.append((x_ring, y_ring))
    return compiler.compile_functor(pairs)
