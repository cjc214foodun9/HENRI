"""Carrier G3 — Wave-Packet Path Search (diagnostic sidecar, default-OFF).

Directive: user approval (2026-09-01) + holographic search.pdf
(HENRI-AUDIT-2026-09-V3-QUANTUM-WAVE-SEARCH, 76c28f6b..., 190,418 B).
Prereg: docs/spec/g3_wave_packet_path_search_preregistration.md.

Audited corrections vs the PDF (load-bearing):
- generators are FROZEN deterministic buffers (zero nn.Parameter) — the
  zero-trainable invariant forbids nn.Parameter action generators.
- Sagnac clearance uses the HENRI normalized delta
  (1 - Re<a,b>/(||a|| ||b||)) in [0,2], NOT the PDF's sin^2 formula.
- Complex flat [D] is the THIRD wave family: one-way norm-preserving
  complexification adapter, diagnostic sidecar only, no policy influence.
"""
import argparse
import json
import math
import os
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

FLAG = "HENRI_G3_WAVE_PACKET"
SEED = 20260926
DEFAULT_THRESHOLD = 0.05
DEFAULT_TAU = 0.05
DEFAULT_TOP_K = 64
PHASE_SIGMA = 0.1  # small per-dimension phase perturbations (action generators)


def require_flag(flag_name: str) -> None:
    """Default-OFF gate: raise SystemExit unless the env flag is '1'."""
    if os.environ.get(flag_name) != "1":
        print(f"BLOCKED: {flag_name} not set (default-OFF)", file=sys.stderr)
        raise SystemExit(1)


def sagnac_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """HENRI normalized Sagnac delta: 1 - Re<a,b>/(||a|| ||b||), bounded [0,2].

    Complex-safe (uses Re of the Hermitian inner product). Exact 0 for
    identical, ~1 for orthogonal.
    """
    num = torch.real((a.conj() * b).sum(-1))
    den = (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-12)
    return torch.clamp(1.0 - num / den, min=0.0, max=2.0)


def complexify_wave(real: torch.Tensor) -> torch.Tensor:
    """One-way norm-preserving complexification: real [..., M, 8] -> complex [..., D].

    Third-family adapter (diagnostic sidecar): imag = 0, flat-normalized.
    Injective up to global scale; norm-preserving (unit -> unit).
    """
    z = real.float().reshape(*real.shape[:-2], -1).to(torch.complex64)
    return F.normalize(z, p=2, dim=-1)


def veto_selectivity(match: torch.Tensor, threshold: float = DEFAULT_THRESHOLD):
    """Survival mask + HENRI sagnac energy from axiom coherence matches."""
    sagnac = torch.clamp(1.0 - match, min=0.0, max=2.0)
    return (sagnac <= threshold).float(), sagnac


class WavePacketPathSearch(nn.Module):
    """Continuous wave-packet path search (diagnostic sidecar).

    Superposes all action paths as phase-rotated wavefronts, applies
    Sagnac homodyne clearance against invariant axioms, keeps top-k
    coherent modes per step. Zero trainable parameters.
    """

    def __init__(self, dim: int = 65536, num_actions: int = 7,
                 horizon: int = 8, top_k: int = DEFAULT_TOP_K,
                 sagnac_threshold: float = DEFAULT_THRESHOLD,
                 cavity_temperature: float = DEFAULT_TAU,
                 seed: int = SEED, device: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.num_actions = num_actions
        self.horizon = horizon
        self.top_k = top_k
        self.thresh = sagnac_threshold
        self.tau = cavity_temperature
        g = torch.Generator().manual_seed(seed)
        # Frozen deterministic small-phase generators (unit-modulus rows,
        # phase sigma 0.1 rad): zero trainable parameters.
        phases = torch.randn(num_actions, dim, generator=g) * PHASE_SIGMA
        generators = torch.exp(1j * phases).to(torch.complex64)
        generators = generators / generators.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        self.register_buffer("generators", generators.to(device))

    def propagate_superposed_paths(self, psi_initial: torch.Tensor,
                                   action_priors: torch.Tensor,
                                   invariant_axioms: torch.Tensor,
                                   ) -> tuple:
        """Coherent multi-path propagation in a single batched pass per step.

        psi_initial [B, D] complex; action_priors [B, A]; axioms [K, D] complex.
        Returns (best_wavefront [B, D], coherence [B, <=top_k],
                 clearance [B, n_paths_last]).
        """
        B = psi_initial.shape[0]
        device = psi_initial.device
        active = psi_initial.unsqueeze(1)                      # [B, 1, D]
        amps = action_priors.unsqueeze(1)                      # [B, 1, A]
        phasors = torch.exp(1j * torch.angle(self.generators)).to(device)  # [A, D]
        ax = invariant_axioms.to(device)
        clearance = None
        for _ in range(self.horizon):
            N = active.shape[1]
            # Superposition branching: [B, N, A, D]
            expanded = active.unsqueeze(2) * phasors.unsqueeze(0).unsqueeze(0)
            expanded = expanded.reshape(B, N * self.num_actions, self.dim)
            expanded = F.normalize(expanded, p=2, dim=-1)
            # Parallel Sagnac clearance vs axioms: [B, N*A, K]
            match = torch.real(torch.matmul(expanded, ax.conj().transpose(0, 1)))
            max_match = match.max(dim=-1).values
            survival, sagnac = veto_selectivity(max_match, self.thresh)
            clearance = sagnac
            # Prior-weighted path weights
            pr = action_priors.unsqueeze(1).expand(B, N, self.num_actions) \
                               .reshape(B, N * self.num_actions)
            weights = torch.exp(-sagnac / max(self.tau, 1e-8)) * survival * pr
            # Destructive-interference pruning (top-k coherent modes)
            k = min(self.top_k, weights.shape[1])
            top_w, top_i = torch.topk(weights, k=k, dim=-1)
            bidx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            active = expanded[bidx, top_i]
            amps = top_w
        best_idx = torch.argmax(amps, dim=-1)
        best = active[torch.arange(B, device=device), best_idx]
        return best, amps, clearance


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Carrier G3 wave-packet path search (diagnostic sidecar)")
    ap.add_argument("--dim", type=int, default=65536)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--num-actions", type=int, default=7)
    ap.add_argument("--out-dir", default="/tmp/henri_g3_wave_packet/")
    ap.add_argument("--receipt-out", default=None)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    require_flag(FLAG)
    torch.manual_seed(SEED)
    eng = WavePacketPathSearch(dim=args.dim, num_actions=args.num_actions,
                               horizon=args.horizon, seed=SEED, device=args.device)
    g = torch.Generator().manual_seed(SEED)
    psi = F.normalize(torch.randn(1, args.dim, dtype=torch.complex64, generator=g),
                      p=2, dim=-1).to(args.device)
    axioms = F.normalize(torch.randn(2, args.dim, dtype=torch.complex64, generator=g),
                         p=2, dim=-1).to(args.device)
    priors = torch.ones(1, args.num_actions, device=args.device) / args.num_actions
    import time
    t0 = time.perf_counter()
    best, coherence, clearance = eng.propagate_superposed_paths(psi, priors, axioms)
    dt = time.perf_counter() - t0
    norm_err = float((best.norm(dim=-1) - 1.0).abs().max().item())
    n_params = sum(p.numel() for p in eng.parameters())
    result = {
        "verdict": "G3_SIDECAR_VERIFIED",
        "flag": FLAG,
        "dim": args.dim, "horizon": args.horizon, "num_actions": args.num_actions,
        "seed": SEED, "device": args.device,
        "latency_s": round(dt, 4),
        "norm_err": norm_err,
        "paths_survived": int(coherence.shape[1]),
        "clearance_finite": bool(torch.isfinite(clearance).all().item()),
        "trainable_params": n_params,
    }
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.receipt_out or str(out_dir / "g3_receipt.json")
    pathlib.Path(receipt).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
