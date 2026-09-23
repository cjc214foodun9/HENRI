#!/usr/bin/env python3
"""UHR-01 regime report: deterministic reproduction of the numbers quoted in
`uhr_rfss.py`'s "VERIFIED PROPERTIES" block.

WHY THIS FILE EXISTS
    Every number in that module docstring must be re-derivable by a command, not
    only assertable inside a pytest run. A docstring that names an artifact which
    does not exist is a dangling pointer (a defect class this repository has
    corrected before). This script IS that artifact.

WHAT IT PRINTS
    1. Sof(8) structural laws for Ad(U): orthogonality, det, homomorphism order.
    2. The character identity Tr(Ad U) = |Tr U|^2 - 1 and the closed form
       delta = 0.5 * (1 - Tr(Ad U)/8).
    3. The three populations that set tau_veto: identity, compliant, invalid.
    4. The axiom-sensitivity split: near-isotropic (the REAL baseplate) vs
       genuinely anisotropic roles.

RUN
    python experiments/verification/uhr01_regime.py
Local CPU (torch>=2.4) is sufficient: this is a scaffold-cost reproduction of
algebra, not a CUDA verification (which stays on the Vast target).
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from uhr_rfss import (  # noqa: E402
    adjoint_matrix,
    axiom_sensitivity,
    block_norm_deviation,
    character_delta,
    project_option_to_boundary_family,
)

TAU_VETO = 0.35
N_BLOCKS = 8192


def _basis() -> torch.Tensor:
    s3 = math.sqrt(3.0)
    return torch.tensor([
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[1 / s3, 0, 0], [0, 1 / s3, 0], [0, 0, -2 / s3]],
    ], dtype=torch.complex64)


def _gen(action: int, basis: torch.Tensor, scale: float, salt: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + action + salt)
    theta = torch.randn(8, generator=g) * scale
    return (1j * torch.einsum("a,aij->ij", theta.to(basis.dtype), basis)).unsqueeze(0)


def _seeder_like(n: int, seed: int, edit: bool = False) -> torch.Tensor:
    """Mirrors `zone_c_axiom_seeder.generate_seed_crystal_axioms`:
    normalize(randn(n, 8)) with a small structural edit on some axioms."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, 8, generator=g)
    if edit:
        w[:, 1:4] = -w[:, 1:4]
    return w / w.norm(dim=-1, keepdim=True)


def _axis_roles(n: int, axis: int) -> torch.Tensor:
    r = torch.zeros(n, 8)
    r[:, axis] = 1.0
    return r


def _delta(cand: torch.Tensor, ref: torch.Tensor) -> float:
    a = cand.flatten().to(torch.float32)
    b = ref.flatten().to(torch.float32)
    denom = float(a.norm()) * float(b.norm())
    if denom < 1e-12:
        return 0.0
    return 1.0 - 0.5 * (1.0 + float((a * b).sum()) / denom)


def main() -> int:
    basis = _basis()
    roles_iso = _seeder_like(N_BLOCKS, 4242)

    print("=" * 78)
    print("1. SO(8) STRUCTURAL LAWS for Ad(U)")
    print("=" * 78)
    g1, g2 = _gen(2, basis, 0.4), _gen(3, basis, 0.5)
    A1, A2 = adjoint_matrix(g1, basis), adjoint_matrix(g2, basis)
    U1 = torch.matrix_exp(g1[0].to(torch.complex128))
    U2 = torch.matrix_exp(g2[0].to(torch.complex128))
    lam = basis.to(torch.complex128)
    Ud = (U1 @ U2).conj().transpose(-2, -1)
    direct = torch.tensor(
        [[(torch.trace((U1 @ U2) @ lam[a] @ Ud @ lam[c]) / 2.0).real for a in range(8)]
         for c in range(8)], dtype=torch.float32)
    print(f"  ||A A^T - I||_F                  = {float((A1 @ A1.T - torch.eye(8)).norm()):.3e}")
    print(f"  det(A)                           = {float(torch.linalg.det(A1.double())):.6f}")
    print(f"  ||Ad(U1U2) - Ad(U1)Ad(U2)||_max  = {float((direct - A1 @ A2).abs().max()):.3e}")
    print(f"  ||Ad(U1U2) - Ad(U2)Ad(U1)||_max  = {float((direct - A2 @ A1).abs().max()):.3e}"
          "   <- must be LARGE (homomorphism, not anti-)")

    print()
    print("=" * 78)
    print("2. CHARACTER LAW   Tr(Ad U) == |Tr U|^2 - 1   and   delta = 0.5*(1 - Tr(A)/8)")
    print("=" * 78)
    print(f"  {'scale':>7} {'Tr(A)':>11} {'|TrU|^2-1':>11} {'closed form':>12} {'measured':>10}")
    for scale in (0.0, 0.2, 0.6, 1.2):
        gen = _gen(9, basis, scale)
        A = adjoint_matrix(gen, basis)
        U = torch.matrix_exp(gen[0].to(torch.complex128))
        tr_a = float(torch.trace(A))
        tr_u = abs(complex(torch.trace(U).item()))
        meas = _delta(project_option_to_boundary_family([gen], basis, roles_iso), roles_iso)
        print(f"  {scale:7.2f} {tr_a:11.6f} {tr_u ** 2 - 1:11.6f} "
              f"{character_delta(tr_a):12.6f} {meas:10.6f}")

    print()
    print("=" * 78)
    print(f"3. POPULATIONS at tau_veto={TAU_VETO}  (8 draws each, 4-step option, {N_BLOCKS} blocks)")
    print("=" * 78)
    populations = {
        "identity": lambda s: [],
        "compliant (0.10)": lambda s: [_gen(s * 4 + i, basis, 0.10) for i in range(4)],
        "mid (0.30)": lambda s: [_gen(s * 4 + i, basis, 0.30) for i in range(4)],
        "invalid (1.20)": lambda s: [_gen(s * 4 + i, basis, 1.20) for i in range(4)],
    }
    for name, mk in populations.items():
        ds = []
        for s in range(8):
            roles = _seeder_like(N_BLOCKS, 900 + s)
            ds.append(_delta(
                project_option_to_boundary_family(mk(s), basis, roles), roles))
        rate = sum(d > TAU_VETO for d in ds) / len(ds)
        print(f"  {name:17s} min={min(ds):.6f} max={max(ds):.6f} "
              f"mean={sum(ds) / len(ds):.6f} veto_rate={rate:.3f}")

    # Structural violation: a candidate in the family but NOT homologous.
    ds = []
    for s in range(8):
        roles = _seeder_like(N_BLOCKS, 900 + s)
        perm = torch.randperm(N_BLOCKS, generator=torch.Generator().manual_seed(50 + s))
        gens = [_gen(s * 4 + i, basis, 0.10) for i in range(4)]
        ds.append(_delta(
            project_option_to_boundary_family(gens, basis, roles,
                                              role_permutation=perm), roles))
    rate = sum(d > TAU_VETO for d in ds) / len(ds)
    print(f"  {'scrambled(0.10)':17s} min={min(ds):.6f} max={max(ds):.6f} "
          f"mean={sum(ds) / len(ds):.6f} veto_rate={rate:.3f}")

    print()
    print("=" * 78)
    print("4. AXIOM SENSITIVITY   isotropic (REAL baseplate) vs anisotropic")
    print("=" * 78)
    gens = [_gen(i, basis, 0.25) for i in range(4)]
    s_iso = max(
        axiom_sensitivity(project_option_to_boundary_family, gens, basis,
                          _seeder_like(4096, s), _seeder_like(4096, s + 100, True))
        for s in (1, 2, 3))
    obs = [_delta(project_option_to_boundary_family(gens, basis, _axis_roles(4096, a)),
                  _axis_roles(4096, a)) for a in range(4)]
    s_aniso = max(obs) - min(obs)
    print(f"  near-isotropic (real seeder) sensitivity = {s_iso:.6f}")
    print(f"  anisotropic axis deltas = {[round(d, 6) for d in obs]}")
    print(f"  anisotropic spread      = {s_aniso:.6f}   ratio = {s_aniso / max(s_iso, 1e-12):.1f}x")
    print()
    print("  LIMITATION (stated, not hidden): the REAL Zone C baseplate is")
    print("  near-isotropic, so UHR-01 restores OPTION-level discrimination and")
    print("  does NOT establish axiom-level correspondence.")

    print()
    print("=" * 78)
    print("5. BLOCK-NORM CONTRACT (loader AXIOM_NORM_TOL = 1e-4)")
    print("=" * 78)
    w = project_option_to_boundary_family(gens, basis, roles_iso)
    print(f"  shape={tuple(w.shape)} dtype={w.dtype} complex={w.is_complex()}")
    print(f"  block_norm_deviation = {block_norm_deviation(w):.3e}")
    print(f"  PASS = {block_norm_deviation(w) < 1e-4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
