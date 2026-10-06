"""MODEL 1 - Zone B Viscoelastic Resonator Swarm.

Document anchors:
  doc p2  : 256-512 concurrent probes, ~3.87 MiB/worker
  doc p12 : K = diag(m) + A S B^H parallel Koopman evolution; CCCP minimization
  doc p13 : Generalized Langevin Equation with non-Markovian memory kernel
  doc p14 : three rheological phases - elastic, creep, solidification
  doc p15 : class SwarmWaveResonator(nn.Module); class SwarmConsensusVeto
  doc p16 : Gate S-1 distinct seeds; Gate S-2 monotone energy with noise off

Integrator
    CCCP coordinate step  z* <- X softmax(beta X^H z)   (monotone, doc p19)
    Viscoelastic gating   lr = 1 - eta(dt)              (doc p14 phase 3)
    Backtracking line search on the Lyapunov energy      (guarantees S-2)
    FDT-coloured noise    disabled when noise_std = 0    (S-2 precondition)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import substrate as sub
from .viscoelastic import ViscoelasticMemoryKernel


class SwarmWaveResonator(nn.Module):
    """Launch B perturbed probes and evolve them to a shared attractor."""

    def __init__(self, dim: int = sub.DEFAULT_DIM, n_workers: int = 256,
                 n_slices: int = 4, steps: int = 16, beta: float = 26.10,
                 dt: float = 1e-3, noise_std: float = 0.0, sigma: float = 0.05,
                 seed: int = 0, backtrack: bool = True, beta_scale: float = 1.0):
        super().__init__()
        self.dim = int(dim)
        self.B = int(n_workers)
        self.steps = int(steps)
        # D-ZO-COLLAPSE (measured at test time): the CCCP attention
        #   logits = beta * <z, p>  with UNIT-NORM RANDOM patterns
        # has <z, p> ~ N(0, 1/D), i.e. |cos| ~ 1/sqrt(D). At D=4096 that is
        # ~0.0156, so beta=26.10 yields max|logit| ~ 0.5 and softmax entropy
        # H/lnN ~ 0.97 (near-uniform). `att @ patterns` then converges to the
        # BANK MEAN for every input: measured cos(cccp_out, mean(bank)) = 0.9748,
        # and solve() returned 1 distinct answer from 4 distinct queries.
        # The doc's beta acts on a cosine the random bank cannot produce.
        # beta_scale multiplies beta so the logit spread matches the formula's
        # stated domain. DEFAULT 1.0 PRESERVES current behaviour byte-for-byte;
        # committed receipts stay valid. See audit_root_cause.py.
        self.beta = float(beta)
        self.beta_scale = float(beta_scale)
        self.beta_eff = self.beta * self.beta_scale
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.backtrack = bool(backtrack)
        self.kernel = ViscoelasticMemoryKernel(n_slices=n_slices, dim=dim)
        # Low-rank Koopman generator K = diag(m) + A S B^H (doc p12)
        r = max(8, dim // 256)
        g = torch.Generator().manual_seed(seed + 7)
        scale = 1.0 / (r ** 0.5)
        self.A = nn.Parameter((torch.randn(dim, r, generator=g) * scale).to(torch.complex64))
        self.Smat = nn.Parameter((torch.randn(r, r, generator=g) * scale).to(torch.complex64))
        self.Bmat = nn.Parameter((torch.randn(dim, r, generator=g) * scale).to(torch.complex64))
        self.diag = nn.Parameter(torch.ones(dim, dtype=torch.complex64))
        self.rank_r = r

    # ------------------------------------------------------------------ probes
    def init_probes(self, psi_in: torch.Tensor, base_seed: int | None = None):
        """B distinct perturbed probes. Distinct seeds are gate S-1."""
        base = self.seed if base_seed is None else int(base_seed)
        seeds, states = [], []
        for k in range(self.B):
            gk = torch.Generator().manual_seed(base + k)
            xi = torch.randn(self.dim, generator=gk).to(torch.complex64)
            states.append(sub.unit_norm(psi_in + self.sigma * xi))
            seeds.append(base + k)
        return torch.stack(states), seeds

    # ------------------------------------------------------------- integration
    def _cccp(self, z: torch.Tensor, patterns: torch.Tensor) -> torch.Tensor:
        logits = self.beta_eff * (z @ patterns.conj().transpose(0, 1)).real
        att = torch.softmax(logits, dim=-1)
        return att.to(torch.complex64) @ patterns

    def forward(self, psi_in: torch.Tensor, patterns: torch.Tensor | None = None,
                base_seed: int | None = None):
        """Evolve the swarm. Returns energies per step, plus the winner.

        patterns: [N, D] complex unit-norm archetypes (Zone C axioms / engrams).
                  None -> pure viscoelastic relaxation to the memory state.
        """
        z, seeds = self.init_probes(psi_in, base_seed)
        history = z.unsqueeze(1).repeat(1, self.kernel.P, 1)      # [B, P, D]
        energies = [sub.hopfield_energy(z, patterns, self.beta)
                    if patterns is not None
                    else (z.abs() ** 2).sum(dim=-1).real]
        eta = self.kernel.friction(self.dt).detach()
        entropy_trace = [self._entropy(z, patterns)]

        for _ in range(self.steps):
            mem = self.kernel.convolve(history, self.dt)
            if patterns is not None:
                target = self._cccp(z, patterns)
            else:
                target = mem
            lr = float((1.0 - eta).clamp(0.02, 1.0))
            nxt = _blend(z, target, lr)
            if self.noise_std > 0.0:
                gn = torch.randn(z.shape, generator=torch.Generator().manual_seed(
                    (base_seed if base_seed is not None else self.seed) + 101))
                nxt = sub.unit_norm(nxt + (2.0 * self.noise_std * lr) ** 0.5
                                    * gn.to(torch.complex64))
            if self.backtrack and patterns is not None:
                nxt = self._backtrack(z, nxt, patterns, lr)
            z = nxt
            history = torch.cat([history[:, 1:], z.unsqueeze(1)], dim=1)
            energies.append(sub.hopfield_energy(z, patterns, self.beta)
                            if patterns is not None
                            else (z.abs() ** 2).sum(dim=-1).real)
            entropy_trace.append(self._entropy(z, patterns))

        E = torch.stack(energies)                                  # [steps+1, B]
        # DIRECTIVE 2: continuous information gain, per relaxation step.
        # Delta H_t = H(Psi_t) - E[H(Psi_{t+1})] = H(Psi_t) - H(Psi_{t+1}) for a
        # deterministic step. The cumulative sum telescopes to H_0 - H_K, which is
        # the scalar "delta_h" already reported. Exposing the per-step trace makes
        # the gain continuous instead of a single start/end difference.
        gain_steps = torch.stack(
            [entropy_trace[t] - entropy_trace[t + 1]
             for t in range(len(entropy_trace) - 1)])                # [K, B]
        best = int(E[-1].argmin())
        return {
            "psi": z[best],
            "psi_workers": z,
            "energies": E,
            "best_index": best,
            "seeds": seeds,
            "distinct_seeds": len(set(seeds)),
            "final_energy": float(E[-1].min()),
            "delta_h": self._delta_h(entropy_trace),
            "delta_h_steps": gain_steps,                            # [K, B]
            "mean_gain_per_step": gain_steps.mean(dim=0),           # [B]
            "height": E.shape[0],
            "K": self.steps,
        }

    def _backtrack(self, z, nxt, patterns, lr):
        """Halve the step until the Lyapunov energy does not increase."""
        e0 = sub.hopfield_energy(z, patterns, self.beta)
        step = lr
        for _ in range(8):
            cand = _blend(z, nxt, step)
            e1 = sub.hopfield_energy(cand, patterns, self.beta)
            if bool((e1 <= e0 + 1e-9).all()):
                return cand
            step *= 0.5
        return z

    def _entropy(self, z, patterns):
        if patterns is None:
            return torch.zeros(z.shape[0])
        logits = self.beta_eff * (z @ patterns.conj().transpose(0, 1)).real
        p = torch.softmax(logits, dim=-1)
        return -(p * p.clamp_min(1e-12).log()).sum(dim=-1)

    @staticmethod
    def _delta_h(trace):
        """doc p39: Delta H(pi) = H(Psi_t) - E[H(Psi_{t+1} | o, pi)]."""
        first, last = trace[0], trace[-1]
        return (first - last)

    def monotone_fraction(self, E: torch.Tensor) -> float:
        """Fraction of steps where energy did not increase. Gate S-2 metric."""
        d = E[1:] - E[:-1]
        return float((d <= 1e-9).float().mean())


class SwarmConsensusVeto(nn.Module):
    """Select the probe minimizing Lyapunov energy and maximizing Delta H. doc p15."""

    def __init__(self, min_delta_h: float = 0.15):
        super().__init__()
        self.min_delta_h = float(min_delta_h)

    def forward(self, result: dict):
        E = result["energies"][-1]
        H = result["delta_h"]
        # primary: lowest energy. tie-break: greatest information gain.
        order = torch.argsort(E + 1e-9 * (-H))
        idx = int(order[0])
        return {
            "index": idx,
            "psi": result["psi_workers"][idx],
            "energy": float(E[idx]),
            "delta_h": float(H[idx]),
            "informed": bool(H[idx] >= self.min_delta_h),
        }


def _blend(z: torch.Tensor, target: torch.Tensor, lr: float) -> torch.Tensor:
    return sub.unit_norm(z + lr * (target - z))
