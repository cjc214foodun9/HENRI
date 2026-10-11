"""henri_resonator_flow.py -- Tripartite Resonator Network (five-action directive 5).

PURPOSE
    Break the 0.1367 deduplicated-bank containment ceiling that holds Gate G-SJ3b
    BLOCKED. A tie group cannot be split by score ranking, because a strictly monotone
    transform preserves the maximizer set. Separation must come from ALGEBRA.

ALGEBRA -- the directive's "Torus (x) Mask (x) Rotor"
    Three factors act on DISJOINT coordinate blocks of one complex state vector.
    Disjoint support makes the factors commute EXACTLY, so the composed operator is
    associative and factorizes in closed form.
        Torus  block A : elementwise U(1) phase multiply, exp(i*phi)     (abelian)
        Mask   block B : elementwise sign flip, s in {-1,+1}             (order 2)
        Rotor  block C : SO(2) plane rotation by theta in fixed planes   (abelian)
    Inverse: phi -> -phi, s -> s, theta -> -theta.
    Compose: phi1+phi2, s1*s2, theta1+theta2.

WHY DISJOINT BLOCKS. A rotor does not commute with elementwise complex codes. If all
three shared one block the composed operator would not be associative and factorization
would be ill-posed. Disjoint support buys exact associativity at a real cost: each block
sees only d/3 coordinates, so per-block capacity is reduced. That cost is reported.

NON-CLAIMS
    This module does NOT clear Gate G-SJ3b. G-SJ3b needs a MEASURED top-1 >= 0.1600 on
    the FULL candidate bank. No bank data file exists in the tree, so the gate stays
    BLOCKED_ON_SUBSYSTEM. This module supplies the algebra and its discriminating test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

DEFAULT_D = 8192


@dataclass(frozen=True)
class TripartiteKey:
    """One relational operator: (phi, s, theta) over the three disjoint blocks."""
    phi: torch.Tensor      # [a] real phases
    s: torch.Tensor        # [b] real, each entry exactly +1.0 or -1.0
    theta: torch.Tensor    # scalar tensor, real radians


class TripartiteResonator:
    """Commuting Torus (x) Mask (x) Rotor factorization over disjoint blocks."""

    def __init__(self, d: int = DEFAULT_D, seed: int = 0):
        self.d = int(d)
        a = self.d // 3
        b = (self.d - a) // 2
        c = self.d - a - b
        self.a, self.b, self.c = a, b, c
        self.iA = slice(0, a)
        self.iB = slice(a, a + b)
        self.iC = slice(a + b, a + b + c)
        c2 = self.c - (self.c % 2)
        self.n_planes = c2 // 2
        g = torch.Generator().manual_seed(seed)
        # FIXED plane pairing. Part of the code, never resampled after construction.
        self.pair = torch.randperm(self.n_planes, generator=g) if self.n_planes > 1 \
            else torch.zeros(1, dtype=torch.long)

    # ---------- key construction ----------
    def make_key(self, src: torch.Tensor) -> TripartiteKey:
        """Derive a key from a real source vector of length d (deterministic)."""
        src = src.detach().to(torch.float32).flatten()
        phi = 2.0 * math.pi * torch.frac(src[self.iA])
        s = torch.where(src[self.iB] >= 0, 1.0, -1.0)
        theta = (2.0 * math.pi * torch.frac(src[self.iC].sum().unsqueeze(0))).flatten()[0]
        return TripartiteKey(phi=phi, s=s, theta=theta)

    # ---------- primitive actions ----------
    def _torus(self, x: torch.Tensor, phi: torch.Tensor, inv: bool) -> torch.Tensor:
        p = -phi if inv else phi
        out = x.clone()
        out[self.iA] = out[self.iA] * torch.exp(1j * p.to(torch.float32))
        return out

    def _mask(self, x: torch.Tensor, s: torch.Tensor, inv: bool) -> torch.Tensor:
        # s is its own inverse: s*s = 1 exactly.
        out = x.clone()
        out[self.iB] = out[self.iB] * s.to(out.dtype)
        return out

    def _rotor(self, x: torch.Tensor, theta: torch.Tensor, inv: bool) -> torch.Tensor:
        th = (-theta if inv else theta).to(torch.float32)
        co, si = torch.cos(th), torch.sin(th)
        out = x.clone()
        c2 = 2 * self.n_planes
        blk = out[self.iC][:c2]              # in-place view: writes land in `out`
        u, v = blk[0::2].clone(), blk[1::2].clone()   # clone: reads must not alias
        blk[0::2] = u * co - v * si
        blk[1::2] = u * si + v * co
        return out

    # ---------- public API ----------
    def bind(self, x: torch.Tensor, k: TripartiteKey) -> torch.Tensor:
        y = self._torus(x, k.phi, inv=False)
        y = self._mask(y, k.s, inv=False)
        y = self._rotor(y, k.theta, inv=False)
        return y

    def unbind(self, y: torch.Tensor, k: TripartiteKey) -> torch.Tensor:
        z = self._rotor(y, k.theta, inv=True)
        z = self._mask(z, k.s, inv=True)
        z = self._torus(z, k.phi, inv=True)
        return z

    @staticmethod
    def compose(k1: TripartiteKey, k2: TripartiteKey) -> TripartiteKey:
        """k1 then k2. Exact, because the three blocks act on disjoint support."""
        return TripartiteKey(phi=k1.phi + k2.phi, s=k1.s * k2.s, theta=k1.theta + k2.theta)

    # ---------- capacity reporting ----------
    def block_capacity(self) -> dict:
        return {"d": self.d, "torus_dims": self.a, "mask_dims": self.b,
                "rotor_dims": self.c, "rotor_planes": self.n_planes,
                "fraction_per_block": round(1.0 / 3.0, 4)}


if __name__ == "__main__":
    torch.manual_seed(0)
    R = TripartiteResonator(d=96, seed=0)
    print("capacity:", R.block_capacity())
    x = torch.randn(R.d, dtype=torch.complex64)
    k1, k2 = R.make_key(torch.randn(R.d)), R.make_key(torch.randn(R.d))

    rt = (R.unbind(R.bind(x, k1), k1) - x).abs().max().item()
    asso = (R.bind(R.bind(x, k1), k2)
            - R.bind(x, R.compose(k1, k2))).abs().max().item()
    print(f"round_trip_max_err      = {rt:.3e}")
    print(f"associativity_max_err   = {asso:.3e}")
    print("PASS" if rt < 1e-5 and asso < 1e-5 else "FAIL")
