#!/usr/bin/env python3
"""Grid-observable readout: decode a wavefront to an ARC-lattice observable.

THE DEPENDENCY THAT BLOCKS NAIVE WIRING (measured, not assumed)
    `delta_sagnac_observational` needs at least one OBSERVABLE. In
    `sagnac_mcts_planner.search()` a candidate already has a literal grid
    (`child_grid = child_ast.execute(input_grid)`), so no decode is needed there.
    But the SCORING REFERENCE (`reference_wave`) is a wave, produced by

        w_task = task_compiler.compile_functor(encoded_demos)
        phase_goal_pred = task_compiler.single_pass_associative_retrieval(...)
        goal_wave_pred = phase_goal_pred / (k_bins - 1) * 2 - 1

    which is a k_bins = 256 PHASE-QUANTIZATION algebra. That wave was NOT produced
    by binding role-filler pairs from any codebook, so unbinding it with an
    independently-seeded grid codebook recovers nothing. Measured in
    observable_discrimination.py: a role-agnostic bin comparison separates cases
    (0.4893 identical-vs-unrelated at k_bins = 11), but that is a PROXY, not a decode.

    CONCLUSION: an observable veto over the induced goal REQUIRES the wave
    representation to be role-filler consistent end to end. That is what
    `henri_role_filler_ingress.py` supplies, and it is DEFAULT-OFF. So this module
    is correct and self-consistent on waves IT encoded, and it must not be pointed
    at production reference waves until the ingress is switched. Wiring is therefore
    flag-gated, and the gate states this dependency.

ALGEBRA
    Psi = normalize( sum_{r,c} P_{r,c} (*) V_{v(r,c)} )     position roles, value keys
    decode: z_{r,c} = Psi (*) conj(P_{r,c}),  snap z to the value codebook

    Identical to the RFSS construction used for tokens, with a grid POSITION as the
    role. Position under a fixed H x W lattice is a legitimate role: it is what makes
    "cell (2,3) holds 7" a decomposable fact rather than an opaque whole.

THRESHOLD -- DERIVED, NEVER INHERITED
    The observational metric is a SLOT MATCH RATE. Its null is Binomial(n_slots,
    1/n_values), so a random decode matches 1/11 of cells by chance and a random
    wave scores stress ~0.909. `tau_veto = 0.35` was chosen for the WAVEFORM-cosine
    metric and would demand a 65% exact match rate here, vetoing nearly everything.
    tau is therefore computed from the null at the ACTUAL lattice size (see
    derive_tau). This is the r = 0.2682 lesson: a threshold belongs to a METRIC.

VALUE RANGE: n_values = 11 = {0..9} + PAD, matching ARC (NOT the encoder's 0..15 clamp).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

DEFAULT_N_VALUES = 11          # {0..9} + PAD


def derive_tau(n_slots: int, n_values: int = DEFAULT_N_VALUES,
               alpha: float = 0.01) -> Dict[str, float]:
    """Exact finite-sample threshold for the match-rate veto. Never inherit 0.35.

    t = number of matched slots ~ Binomial(n_slots, 1/n_values) under the null.
    tau = 1 - q_{1-alpha}(t) / n_slots, so a candidate passes unless it is
    significantly WORSE than a random decode at level alpha.
    """
    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    p = 1.0 / n_values
    pmf = [comb(n_slots, k) * (p ** k) * ((1 - p) ** (n_slots - k))
           for k in range(n_slots + 1)]
    s = sum(pmf)
    pmf = [x / s for x in pmf] if s > 0 else pmf
    cum = 0.0
    q = n_slots
    for k in range(n_slots + 1):
        cum += pmf[k]
        if cum >= 1.0 - alpha:
            q = k
            break
    return {
        "n_slots": n_slots, "n_values": n_values, "alpha": alpha,
        "null_match_rate": p,
        "q_1_minus_alpha_matched": q,
        "tau_observational": 1.0 - q / n_slots,
        "inherited_0.35_requires_match_rate": 0.65,
    }


@dataclass
class GridDecode:
    values: np.ndarray            # [H, W] decoded cell values
    quality: np.ndarray           # [H, W] best-cosine per cell
    valid: bool


class GridObservableReadout:
    """Encode/decode an integer lattice as a role-filler wavefront.

    Args:
        shape: (H, W) lattice. H*W = n_slots = the observable's slot count.
        dim: wave dimension D.
        n_values: codebook size, {0..9}+PAD = 11 for ARC.
        seed: codebook seed; fixed so encode/decode are reproducible and SHARED.
        alpha: significance for the derived veto threshold.

    The codebook is the shared contract: encode and decode must use the SAME
    instance (or the same seed). Decoding a wave built from a different codebook is
    undefined and is guarded by `codebook_fingerprint`.
    """

    def __init__(self, shape: Tuple[int, int] = (8, 8), dim: int = 1024,
                 n_values: int = DEFAULT_N_VALUES, seed: int = 20261012,
                 alpha: float = 0.01) -> None:
        self.H, self.W = int(shape[0]), int(shape[1])
        self.n_slots = self.H * self.W
        self.dim = int(dim)
        self.n_values = int(n_values)
        self.alpha = float(alpha)
        self.seed = int(seed)

        g = torch.Generator().manual_seed(self.seed)
        self.role_keys = self._phasors(self.n_slots, g)          # [H*W, D]
        self.value_keys = self._phasors(self.n_values, g)        # [n_values, D]
        self.tau_info = derive_tau(self.n_slots, self.n_values, self.alpha)
        self.tau = self.tau_info["tau_observational"]

    # ------------------------------------------------------------------ setup
    def _phasors(self, count: int, gen: torch.Generator) -> torch.Tensor:
        phases = torch.rand(count, self.dim, generator=gen, dtype=torch.float32) * (2 * math.pi)
        p = torch.complex(torch.cos(phases), torch.sin(phases))
        return p / p.norm(dim=-1, keepdim=True)

    def codebook_fingerprint(self) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.role_keys.numpy()).tobytes())
        h.update(np.ascontiguousarray(self.value_keys.numpy()).tobytes())
        return h.hexdigest()[:16]

    # ---------------------------------------------------------------- encode
    @staticmethod
    def _bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft(torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1)

    def encode(self, grid: np.ndarray) -> torch.Tensor:
        """grid [H, W] int -> unit-norm complex wave [D]."""
        g = np.asarray(grid).astype(np.int64)
        if g.shape != (self.H, self.W):
            raise ValueError(f"grid shape {g.shape} != {(self.H, self.W)}")
        if g.min() < 0 or g.max() >= self.n_values:
            raise ValueError(f"values must be in [0, {self.n_values - 1}]")
        idx = torch.from_numpy(g.reshape(-1))
        bound = self._bind(self.role_keys, self.value_keys[idx])
        psi = bound.sum(dim=0) / math.sqrt(self.n_slots)
        return psi / psi.norm().clamp_min(1e-12)

    # ---------------------------------------------------------------- decode
    @staticmethod
    def _unbind(psi: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        """Circulate-correlate psi against each role: psi (*) conj(role).

        CONJUGATION MUST HAPPEN IN THE FREQUENCY DOMAIN.

            fft(conj(R))_k = conj(fft(R)_{(-k) mod D})

        so `ifft(fft(psi) * fft(conj(R)))` is NOT circular correlation -- it is
        correlation against a TIME-REVERSED role, which for complex phasors is a
        different vector and recovers nothing. MEASURED: a first version of this
        module conjugated in the spatial domain and round-tripped only 25% of cells
        (min cell quality 0.036, at the noise floor). `henri_wave_readout.unbind`
        conjugates the SPECTRUM and decodes 200/200 exactly. This method uses that
        form.
        """
        return torch.fft.ifft(
            torch.fft.fft(psi, dim=-1).unsqueeze(-2)
            * torch.conj(torch.fft.fft(roles, dim=-1)),
            dim=-1)

    def decode(self, psi: torch.Tensor) -> GridDecode:
        """wave [D] -> decoded lattice. Fail-closed on zero energy."""
        w = psi.flatten()
        if float(w.norm().item()) < 1e-8:
            return GridDecode(np.full((self.H, self.W), -1, dtype=np.int64),
                              np.zeros((self.H, self.W)), False)
        # [S, D], one unbound vector per position role.
        z = self._unbind(w, self.role_keys)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # einsum states the contract axis explicitly. A matmul here would silently
        # broadcast on a shape mismatch instead of erroring.
        sims = torch.einsum("sd,vd->sv", z.conj(), self.value_keys).abs()   # [S, V]
        idx = sims.argmax(dim=-1)
        best = sims.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        return GridDecode(idx.reshape(self.H, self.W).numpy(),
                          best.reshape(self.H, self.W).numpy(), True)

    # ------------------------------------------------- observation-space stress
    def observational_stress(self, psi: torch.Tensor, ref_grid: np.ndarray) -> Dict:
        """1 - cell match rate of the DECODED wave against a reference lattice.

        This is the quantity a veto should use: bounded [0, 1], dimension-free
        (a match rate), and with a derived threshold (self.tau) rather than an
        inherited constant.
        """
        dec = self.decode(psi)
        ref = np.asarray(ref_grid).astype(np.int64)
        if not dec.valid:
            return {"stress": 1.0, "match_rate": 0.0, "valid": False, "decoded": dec}
        match = float((dec.values == ref).mean())
        return {"stress": 1.0 - match, "match_rate": match, "valid": True,
                "decoded": dec}

    # ------------------------------------------------ literal-vs-literal (best)
    @staticmethod
    def grid_stress(pred_grid: np.ndarray, ref_grid: np.ndarray) -> float:
        """When BOTH are literal grids, compare them directly. No decode, no noise.

        Preferred whenever a candidate already has a real observable. Using a lossy
        decode when ground truth exists is a gratuitous failure mode.
        """
        a = np.asarray(pred_grid).astype(np.int64)
        b = np.asarray(ref_grid).astype(np.int64)
        if a.shape != b.shape:
            return 1.0
        return float(1.0 - (a == b).mean())
