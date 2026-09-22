#!/usr/bin/env python3
"""Role-Filler Sum Superposition ingress codebook for byte/character tokens.

THE DEFECT THIS RESOLVES
    `o_vsa_ingress_tokenizer.py` initialises every token as an INDEPENDENT random
    phasor:

        raw_basis = torch.randn(vocab_size, num_blocks, 8)
        canonical_basis = raw_basis / ||raw_basis||

    For independent random vectors the expected inner product between any two tokens
    is 0 with fluctuation O(1/sqrt(D)). So the metric is FLAT: no two tokens are
    "closer" than any other two. Semantic structure cannot exist in the codebook,
    which is the "ingress metric trap" -- and it is why downstream composition has
    nothing to compose.

THE VERIFIED CONSTRUCTION
    Role-Filler Sum Superposition (RFSS), the arm measured at +0.385 semantic gap
    with 36/36 exact unbinding in henri_grounded_lexical_codec.py, versus -0.004194
    for multiplicative convolution chains:

        Psi_token = normalize( sum_r R_r (*) V_{r, value(r)} )

    Adding SHARED terms across tokens is what creates overlap. Two tokens that
    agree on k of m roles overlap by ~k/m; tokens that agree on none are
    quasi-orthogonal. The metric becomes informative.

WHAT ROLES A BYTE TOKEN HAS
    A character genuinely has decomposable structure, so the roles are not
    decorative:
        class         the character class (lowercase, uppercase, digit, space,
                      punctuation, control)
        bucket        coarse numeric magnitude, idx // 16
        residue       fine numeric position, idx % 16
        parity        even/odd
    Consequence, and the reason this is testable: 'a' and 'b' share class, bucket,
    residue-range and parity, differing only in `residue`, so they must be
    measurably CLOSER than 'a' and 'Z', which share almost nothing.

DESIGN
    * DEFAULT-OFF. The tokenizer keeps its existing path unless
      HENRI_INGRESS_ROLE_FILLER=1. A codebook change alters every downstream
      encoding, so it must be an explicit A/B, not a silent swap.
    * ADDITIVE superposition. Never a convolution CHAIN: a chain rolls all phases
      onto one unit-modulus torus and destroys the metric (measured gap -0.004194).
    * Deterministic. Seeded phasors, so the same token always maps to the same
      wavefront and a checkpoint stays valid across processes.
    * Fail-closed. vocab_size <= 0 raises rather than returning an empty basis.

HONEST LIMITS
    * The roles above encode character STRUCTURE, not meaning. 'a' being near 'b'
      reflects numeric adjacency and class, not that 'a' and 'b' are semantically
      related. Real semantics require the frozen foundation adapter. This fixes the
      METRIC, not the knowledge.
    * Adding a role increases the number of superposed terms, so crosstalk grows as
      ~1/sqrt(n_roles). Measured in the companion probe rather than assumed.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch


def _phasors(count: int, dim: int, seed: int) -> torch.Tensor:
    """Deterministic unit-modulus phasors, L2-normalized so ||p||_2 = 1."""
    g = torch.Generator().manual_seed(int(seed))
    phases = torch.rand(count, dim, generator=g, dtype=torch.float32) * (2.0 * math.pi)
    p = torch.complex(torch.cos(phases), torch.sin(phases))
    return p / p.norm(dim=-1, keepdim=True)


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular convolution via the FFT Hadamard product.

    Norm-preserving only up to O(1/sqrt(D)) for L2-normalized phasors; measured
    dev_std*sqrt(D) = 0.50 (binding_norm_algebra.py). It is NOT exactly unitary, so
    no invariant of the form ||a(*)b|| == 1 may be asserted at a 1e-6 tolerance.
    """
    return torch.fft.ifft(torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1)


def char_roles(idx: int) -> Dict[str, str]:
    """Decompose a character index into (role -> value) assignments.

    Derived from the character itself, so the assignment is interpretable and
    reproducible rather than a hash.
    """
    c = chr(idx) if 0 <= idx < 0x110000 else ""
    if "a" <= c <= "z":
        klass = "lower"
    elif "A" <= c <= "Z":
        klass = "upper"
    elif "0" <= c <= "9":
        klass = "digit"
    elif c == " " or c in "\t\n\r":
        klass = "space"
    elif c and c.isprintable():
        klass = "punct"
    else:
        klass = "control"
    return {
        "class": klass,
        "bucket": str(idx // 16),
        "residue": str(idx % 16),
        "parity": "even" if idx % 2 == 0 else "odd",
    }


class RoleFillerIngressCodebook:
    """Builds a `canonical_basis` replacement of shape [vocab, num_blocks, 8].

    The output layout matches the live tokenizer exactly: a real tensor whose last
    axis packs complex pairs, i.e. complex64 of width D = num_blocks * 4 reshaped
    as (num_blocks, 4, 2) -> (num_blocks, 8).
    """

    ROLES: Tuple[str, ...] = ("class", "bucket", "residue", "parity")

    def __init__(self, vocab_size: int = 256, num_blocks: int = 8192,
                 seed: int = 20261012) -> None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.vocab_size = int(vocab_size)
        self.num_blocks = int(num_blocks)
        self.dim = self.num_blocks * 4          # complex width
        self.seed = int(seed)

        # Enumerate role values in a STABLE sorted order so index->key mapping is
        # reproducible across runs and processes.
        self.role_values: Dict[str, List[str]] = {r: [] for r in self.ROLES}
        assignments: List[Dict[str, str]] = []
        for idx in range(self.vocab_size):
            a = char_roles(idx)
            assignments.append(a)
            for r in self.ROLES:
                if a[r] not in self.role_values[r]:
                    self.role_values[r].append(a[r])
        for r in self.ROLES:
            self.role_values[r] = sorted(self.role_values[r])

        # Role keys and per-role value keys (frozen).
        self.role_keys = _phasors(len(self.ROLES), self.dim, self.seed + 1)
        self.value_keys: Dict[str, torch.Tensor] = {}
        for i, r in enumerate(self.ROLES):
            self.value_keys[r] = _phasors(len(self.role_values[r]), self.dim,
                                          self.seed + 31 + 17 * i)
        self._rindex = {r: i for i, r in enumerate(self.ROLES)}
        self._vindex = {r: {v: j for j, v in enumerate(self.role_values[r])}
                        for r in self.ROLES}
        self.assignments = assignments

    # ------------------------------------------------------------------ encode
    def encode_token(self, idx: int) -> torch.Tensor:
        """RFSS: normalize( sum_r R_r (*) V_{r, value(r)} ). Returns complex [D]."""
        a = self.assignments[idx] if 0 <= idx < len(self.assignments) else char_roles(idx)
        acc = torch.zeros(self.dim, dtype=torch.complex64)
        for r in self.ROLES:
            ri = self._rindex[r]
            vi = self._vindex[r][a[r]]
            acc = acc + bind(self.role_keys[ri], self.value_keys[r][vi])
        return acc / acc.norm().clamp_min(1e-12)

    def shared_roles(self, i: int, j: int) -> int:
        """How many role-value pairs two tokens agree on. Max = len(ROLES)."""
        ai, aj = self.assignments[i], self.assignments[j]
        return sum(1 for r in self.ROLES if ai[r] == aj[r])

    # ------------------------------------------------------------------- basis
    def build_basis(self, dtype: torch.dtype = torch.float32,
                    normalize: str = "row", chunk: int = 4096) -> torch.Tensor:
        """[vocab, num_blocks, 8] real tensor in the tokenizer's layout.

        normalize="row"   : each token's WHOLE row is unit L2 norm (||row||=1).
                            This is the geometrically meaningful normalization and
                            is what the RFSS construction produces naturally.
        normalize="block" : each 8-wide Clifford block is unit norm, so the row norm
                            is sqrt(num_blocks). This reproduces the LIVE tokenizer's
                            convention EXACTLY:
                                raw_basis / torch.norm(raw_basis, p=2, dim=-1)
                            with raw_basis shaped [vocab, num_blocks, 8] -- i.e. the
                            live code normalizes over the LAST axis only. Use this
                            mode for a drop-in replacement so no downstream consumer
                            sees a changed scale.

        The distinction is load-bearing, not cosmetic. A probe that compared an
        RFSS basis at ||row||=1 against a random basis at ||row||=sqrt(num_blocks)
        raised a scale-artifact gate. Cosine similarity is scale-invariant so the
        metric conclusions were unaffected, but ANY consumer that thresholds a raw
        inner product or L2 distance would see the wrong magnitude. A scale
        conflation of exactly this kind once fabricated a fake +18 gap in
        henri_grounded_lexical_codec.encode -- see the comment there.
        """
        if normalize not in ("row", "block"):
            raise ValueError(f"normalize must be 'row' or 'block', got {normalize!r}")
        rows: List[torch.Tensor] = []
        for start in range(0, self.vocab_size, chunk):
            stop = min(start + chunk, self.vocab_size)
            block = torch.stack([self.encode_token(i) for i in range(start, stop)])
            real = torch.view_as_real(block)                    # [B, D, 2]
            real = real.reshape(stop - start, self.num_blocks, 4, 2)
            out = real.reshape(stop - start, self.num_blocks, 8)
            if normalize == "block":
                out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rows.append(out.to(dtype))
        return torch.cat(rows, dim=0)


def build_random_basis_control(vocab_size: int, num_blocks: int,
                               seed: int = 0,
                               normalize: str = "block") -> torch.Tensor:
    """The LIVE construction, reproduced for a matched control.

    Indices independent random vectors and normalizes them the SAME way the live
    tokenizer does: over the LAST axis only (`dim=-1`), which normalizes each
    8-wide block and therefore leaves the row norm at sqrt(num_blocks). That is
    literal fidelity to o_vsa_ingress_tokenizer.py:

        raw_basis = torch.randn(vocab_size, num_blocks, 8)
        canonical_basis = raw_basis / torch.norm(raw_basis, p=2, dim=-1, keepdim=True)

    normalize="row" is offered so a probe can compare two bases at IDENTICAL row
    norms when it intends to threshold raw inner products. Choose the mode that
    matches the consumer under test; do not mix them in one comparison.
    """
    if normalize not in ("row", "block"):
        raise ValueError(f"normalize must be 'row' or 'block', got {normalize!r}")
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(vocab_size, num_blocks, 8, generator=g)
    if normalize == "block":
        return raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    flat = raw.reshape(vocab_size, -1)
    flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return flat.reshape(vocab_size, num_blocks, 8)
