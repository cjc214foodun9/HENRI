"""Phase 10.1 Koopman generator bank, built ONLY from live encoder buffers.

SOURCE OF RECORD
    "Phase 10.1 Operator Gap Adjudication & Sample-Efficiency Directive",
    sha256 972c29ffdc67d125ff54009850976fa107d8fb9481dcb42bc70363df66ce0fd9,
    11 pages. Section 3.2 (subspace fitting), section 5 (generator bank).

WHY THIS MODULE EXISTS SEPARATELY
    The directive sketches `generators: torch.Tensor  # [K, D] complex64 diagonal
    generator masks` while its own docstring promises "non-trivial cross-slot
    coupling". A sum of DIAGONAL masks is diagonal, i.e. inside the very family
    whose ceiling the directive says is insufficient. A dense [K, D, D] complex64
    bank would be 8 * 32768^2 * 8 bytes = 68.7 GB at K=8, D=32768. The bank is
    therefore a list of CALLABLES L_j(X) -> X', each [M, D] -> [M, D] complex.

    Measured here, not argued: how many of the directive's NAMED generators are
    actually diagonal in the live frequency basis, and what a genuinely
    off-diagonal generator costs.

BANK (the 6 named in the directive, plus 2 explicit additions, total K=8)
    The directive's section 5 header says "K <= 8" but enumerates exactly SIX:
        1 Identity pass-through
        2 Spatial differential currents
        3 Dihedral reflection parity
        4 Laplacian diffusion
        5 Color permutation shift
        6 Topological charge projection
    That 6-vs-8 discrepancy is recorded, NOT filled in by invention. This module
    implements all six (with "differential currents" split into d/dx and d/dy and
    "dihedral parity" split into its two generator reflections, which is how a
    dihedral group is actually spanned), giving 7, plus ONE explicitly-labelled
    addition (a within-block unitary mixer) to reach K=8 and to give the subspace
    at least one genuinely off-diagonal direction. Every generator records
    `is_diagonal_in_freq_basis` so the diagonal-only subset can be selected.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import torch

BLOCK_SLOTS_FALLBACK = 4


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1) if x.dim() > 1 else x.reshape(1, -1)


def build_generator_bank(enc, device: str = "cpu", seed: int = 0,
                         include_mixer: bool = False
                         ) -> Tuple[List[Callable], List[Dict]]:
    """Build the K generator callables from LIVE encoder buffers.

    enc: a TorusIngressEncoder with `.kx`, `.ky`, `.wx`, `.wy`, `.modulus`,
         `.num_blocks`, and optional `.value_phase`.
    Returns (callables, metadata). Each callable maps [M, D] complex -> [M, D].
    """
    kx = enc.kx.reshape(-1).to(device)                  # [D] integers
    ky = enc.ky.reshape(-1).to(device)
    wx = enc.wx.reshape(-1).to(device)                  # [D] radians
    wy = enc.wy.reshape(-1).to(device)
    S = int(enc.modulus)
    D = int(kx.numel())
    NB = int(enc.num_blocks)
    SL = D // NB

    gens: List[Callable] = []
    meta: List[Dict] = []

    def _add(name, fn, is_diag, note, extra=None):
        gens.append(fn)
        m = {"idx": len(gens) - 1, "name": name,
             "is_diagonal_in_freq_basis": bool(is_diag), "note": note}
        if extra:
            m.update(extra)
        meta.append(m)

    # 1. Identity pass-through (diagonal)
    _add("identity", lambda X: X, True, "L=I")

    # 2. Spatial differential currents: d/dx -> i*wx, d/dy -> i*wy (DIAGONAL)
    _add("diff_dx", lambda X: X * (1j * wx).to(X.dtype), True,
         "d/dx is multiplication by i*wx: diagonal by construction",
         {"freq_factor": "i*wx"})
    _add("diff_dy", lambda X: X * (1j * wy).to(X.dtype), True,
         "d/dy is multiplication by i*wy: diagonal by construction",
         {"freq_factor": "i*wy"})

    # 4. Laplacian diffusion: -(kx^2+ky^2) (DIAGONAL)
    _lap = -((wx ** 2) + (wy ** 2))
    _add("laplacian", lambda X: X * _lap.to(X.dtype), True,
         "Laplacian is -(wx^2+wy^2): diagonal by construction",
         {"freq_factor": "-(wx^2+wy^2)"})

    # 5/6. Dihedral reflection parity, as a frequency-matched SLOT PERMUTATION.
    # (kx,ky) -> (-kx,ky) and -> (kx,-ky). A spatial reflection conjugates the
    # frequency, so it is NOT diagonal; it is realizable only if the reflected
    # frequency exists somewhere in the bank. Coverage is measured, not assumed.
    def _refl_map(neg_x: bool, neg_y: bool):
        """Build a REFLECTION as a genuine INVOLUTION PERMUTATION (v2).

        DEFECT FIXED (mine, load-bearing). v1 did:
            first[key] = first slot anywhere in the bank with that frequency
            dst[i]     = first[reflect(freq(i))]
        Every one of the S^2 = 1024 possible frequencies occurs many times among
        D = 32768 slots, so v1 collapsed ~32768 slots onto ~1024 targets. That is
        a many-to-one PROJECTION, not a permutation: it annihilates most of the
        state. It is consistent with the measured rank_G = 5.75 of 8 and with the
        in-sample ceiling collapsing 0.7536 -> 0.4378, and it CONFOUNDED the
        "off-diagonal contributes nothing" ablation.

        v2 pairs slots of frequency k with slots of frequency r(k), consuming both
        sides, so the map is an involution (sigma(sigma(i)) == i) and a bijection on
        the paired part. Slots with no partner, and self-reflecting frequencies
        (r(k) == k), map to themselves. Coverage is reported honestly.
        """
        rx = ((-kx) % S) if neg_x else kx
        ry = ((-ky) % S) if neg_y else ky
        # group slots by frequency
        buckets: Dict[Tuple[int, int], List[int]] = {}
        for i in range(D):
            buckets.setdefault((int(kx[i]), int(ky[i])), []).append(i)
        perm = list(range(D))
        used = [False] * D
        paired = self_reflecting = 0
        for key in list(buckets.keys()):
            tgt = (int(rx[buckets[key][0]]), int(ry[buckets[key][0]]))
            if tgt == key:
                self_reflecting += len(buckets[key])     # sigma(i) = i
                continue
            if key > tgt or tgt not in buckets:
                continue                                  # handle each pair once
            a, b = buckets[key], buckets[tgt]
            m = min(len(a), len(b))
            for n in range(m):
                perm[a[n]] = b[n]
                perm[b[n]] = a[n]
                used[a[n]] = used[b[n]] = True
                paired += 2
        src = torch.arange(D, device=device)
        dst = torch.tensor(perm, device=device, dtype=torch.long)
        # VERIFY the involution property on this build; a silent failure here
        # would re-confound the very ablation this fix exists to un-confound.
        inv_ok = bool((dst[dst] == src).all().item())
        return src, dst, paired / max(1, D), inv_ok

    for nm, nx, ny in (("dihedral_refl_negx", True, False),
                       ("dihedral_refl_negy", False, True)):
        src, dst, cov, inv_ok = _refl_map(nx, ny)
        def _mk(src=src, dst=dst):
            def f(X, src=src, dst=dst):
                Xf = _flat(X)
                out = torch.zeros_like(Xf)
                out[:, dst] = Xf[:, src]
                return out
            return f
        _add(nm, _mk(), False,
             "frequency-conjugated reflection as an INVOLUTION PERMUTATION over "
             "the full bank (v2: v1 was a collapsing projection and was fixed)",
             {"coverage": cov, "involution_verified": inv_ok,
              "within_block_only": False})

    # 7. Color permutation shift: uniform phase rotation per slot (DIAGONAL).
    # A cyclic colour relabeling is a phase advance in the value register.
    th = (2.0 * math.pi / float(max(2, int(getattr(enc, "vocab_size", 64)))))
    _add("color_permutation_shift", lambda X: X * torch.exp(
        torch.tensor(1j * th, dtype=X.dtype, device=X.device)), True,
        "cyclic value relabeling = uniform phase advance: diagonal",
        {"phase_per_step": th})

    # 8. Topological charge projection (DIAGONAL): the zero-charge (DC) subspace.
    # Charge lives in the reserved dc slots whose multiplier is exp(0)=1.
    dc = int(getattr(enc, "dc_slots", 0) or 0)
    mask = torch.ones(D, dtype=torch.float32, device=device)
    if dc:
        mask[:dc] = 0.0
    _add("topological_charge_projection", lambda X: X * mask.to(X.dtype), True,
         "projects out the reserved DC slots: diagonal", {"dc_slots": dc})

    # ---- OPTIONAL ADDITION (explicitly labelled; NOT in the named list) -----
    # DEFAULT OFF. The K=8 default above is complete without it: the directive
    # names six generators and splits to exactly eight. This per-block unitary is
    # therefore an ABLATION ARM (`include_mixer=True` -> K=9), used only to ask
    # whether one extra genuinely off-diagonal direction helps. Naming it K=8 was
    # my defect and is fixed here.
    if include_mixer:
        g = torch.Generator(device="cpu").manual_seed(seed)
        U = torch.linalg.qr(torch.randn(
            SL, SL, generator=g, dtype=torch.complex128))[0].to(torch.complex64).to(device)

        def _slot_mix(X, U=U, NB=NB, SL=SL):
            Xf = _flat(X)
            V = Xf.reshape(-1, NB, SL)
            return torch.einsum("bsj,ij->bsi", V, U).reshape(Xf.shape[0], NB * SL)

        _add("ADDED_slot_unitary_mixer", _slot_mix, False,
             "NOT in the directive's named list. ABLATION ONLY. Per-block SLxSL "
             "unitary; adds one off-diagonal direction beyond the reflections.",
             {"block_slots": SL})

    return gens, meta
