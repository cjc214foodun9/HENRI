#!/usr/bin/env python3
"""HENRI wave -> observable READOUT. Closes the missing decode direction.

THE DEFECT THIS RESOLVES
    The engine encodes observables into phase space (Zone A) but never decodes
    back. `L_JEPA = 1 - cos(pred, target)` is computed entirely in latent phase
    space, and `evaluate_test_time_plan` derives its Sagnac stress by comparing a
    proposed wavefront against `zone_c_axioms` -- RANDOM unit phasors. For random
    axioms the best achievable agreement is O(sqrt(2 ln K / D)), so the stress is
    ~1.0 for every proposal and the veto can never pass. That is "un-passable by
    construction": not a calibration error, a MISSING DIRECTION.

    Consequence: every mechanism that reads Delta_Sagnac -- the grounding ratchet,
    the memory write policy, the tau_0-VLA compute-depth controller -- is fed a
    near-constant. The readout is upstream of all of them.

ALGEBRA (why the decode is cheap)
    Psi = (1/sqrt(M)) sum_m R_m (*) F_m            (RFSS superposition, additive)

    Unbinding with a role conjugate recovers that role's filler:

        Psi (*) conj(R_s) = (1/sqrt(M)) [ F_s + crosstalk ]

    because FFT(R (*) conj(R))_k = |FFT(R)_k|^2, which has mean D for a random
    unit-modulus phasor and relative fluctuation O(1/sqrt(D)). So the operation
    is near-identity on average with a small, D-suppressed deviation.

    Two measured facts that set the gate calibration (see the companion probe):
      * Binding is norm-preserving only up to O(1/sqrt(D)), NOT exactly. Measured
        dev_std * sqrt(D) = 0.50, constant from D=1024 to D=65536
        (experiments/verification/binding_norm_algebra.py). An invariant of the
        form | ||Psi (*) a|| - 1 | <= 1e-6 CANNOT hold at D=65536; ~3.4e-3 is the
        achievable bound. Do not write 1e-6 upper bounds anywhere.
      * A single binding/unbinding round trip does NOT return cos ~ 1. The
        |FFT(R)_k|^2 fluctuation is itself O(1) per frequency, so the recovered
        vector is signal + comparable-norm noise and cos lands near 1/sqrt(2).
        The probe measures this rather than assuming either value.

DESIGN DECISIONS, EACH TRACED TO A MEASURED FAILURE
    1. Delta_Sagnac is defined in OBSERVABLE space: 1 - (matched slots / slots).
       Two wavefronts that decode to the SAME observable can have near-zero
       internal cosine; the internal measure vetoes them, the observable measure
       correctly passes them. Probe gate B measures that separation.
    2. Fail CLOSED on zero-energy input, PER SAMPLE. The reference design used one
       global energy test for a whole batch, so a zero row could be normalized
       against another sample's norm. That branch is deleted, not disabled.
    3. NO buffer rebinding. Codebooks are created once and never reassigned, so
       decode cannot silently change what it decodes against. Asserted by test.
    4. Indexing is explicit. Advanced indexing for the value lookup, einsum for
       the similarity contraction. An earlier draft used
       `value_codebook.gather(1, ...)` (batch index leaking into the role axis)
       and `matmul([B,S,D],[S,D,V])` (which broadcasts to [S,S,V] at B=1 instead
       of aligning slots). Both returned plausible-looking wrong numbers. Neither
       pattern is used here.
    5. Continuous comparisons are dimension-normalized (||x-y||_2 / sqrt(d)).
       Reusing a raw L2 threshold across dimensions is the dimension-blindness
       fallacy.
    6. Random-axiom stress is retained ONLY as `legacy_random_axiom_stress`, so
       the un-passable baseline stays reproducible as evidence.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


class WaveObservationReadout:
    """Decode a complex wavefront back into discrete observable slots.

    Args:
        dim: wavefront dimension D.
        n_slots: number of observable slots to decode.
        n_values: codebook size per slot.
        seed: codebook seed, fixed so a decode is reproducible.
        dtype: complex dtype for codebooks.

    The instance is stateful in exactly one way: it holds codebooks. It mutates
    no buffer during decode.
    """

    def __init__(self, dim: int = 8192, n_slots: int = 8, n_values: int = 16,
                 seed: int = 0, dtype: torch.dtype = torch.complex64) -> None:
        self.dim = int(dim)
        self.n_slots = int(n_slots)
        self.n_values = int(n_values)
        self.dtype = dtype

        g = torch.Generator().manual_seed(seed)
        self.role_codebook = self._phasors(self.n_slots, g)                  # [S, D]
        vcb = self._phasors(self.n_slots * self.n_values, g)
        self.value_codebook = vcb.view(self.n_slots, self.n_values, self.dim)

    # ------------------------------------------------------------------ setup
    def _phasors(self, count: int, gen: torch.Generator) -> torch.Tensor:
        """Unit-modulus phasors with ||p||_2 = 1. This is the RFSS basis."""
        phases = (torch.rand(count, self.dim, generator=gen, dtype=torch.float32)
                  * (2.0 * math.pi))
        p = torch.complex(torch.cos(phases), torch.sin(phases)).to(self.dtype)
        return _normalize(p)

    def random_wave(self, seed: Optional[int] = None) -> torch.Tensor:
        """A unit-norm random wavefront, for constructing controls."""
        g = torch.Generator().manual_seed(seed if seed is not None else 12345)
        return self._phasors(1, g)[0]

    # ------------------------------------------------------------------ algebra
    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Circular convolution a (*) b via the FFT Hadamard product.

        Norm-preserving only up to O(1/sqrt(D)) for L2-normalized phasors; it is
        NOT exactly unitary. See the module docstring.
        """
        return torch.fft.ifft(torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1)

    @staticmethod
    def unbind(psi: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
        """Correlation with the role conjugate: psi (*) conj(role)."""
        return torch.fft.ifft(torch.fft.fft(psi, dim=-1)
                              * torch.conj(torch.fft.fft(role, dim=-1)), dim=-1)

    # ------------------------------------------------------------------ encode
    def encode(self, slot_values: torch.Tensor) -> torch.Tensor:
        """RFSS superposition: Psi = normalize( sum_s R_s (*) V_{s, val(s)} ).

        ADDITIVE, never a multiplicative convolution chain. A chain rolls every
        phase onto one unit-modulus torus and destroys metric structure (measured
        semantic gap -0.004194 at a039095); sum superposition preserves it
        (measured +0.385018).

        Args:
            slot_values: [B, S] long (or [S] for a single sample).
        Returns:
            [B, D] unit-norm complex wavefront.
        """
        if slot_values.dim() == 1:
            slot_values = slot_values.unsqueeze(0)
        b = slot_values.shape[0]
        dev = slot_values.device

        slots = torch.arange(self.n_slots, device=dev).unsqueeze(0).expand(b, -1)
        # ADVANCED INDEXING, not gather. [S,V,D] indexed by (slot, value) -> [B,S,D].
        vals = self.value_codebook[slots, slot_values]
        roles = self.role_codebook.unsqueeze(0).expand(b, -1, -1)
        bound = self.bind(roles, vals)                       # [B,S,D]
        acc = bound.sum(dim=1)                               # [B,D]
        return _normalize(acc / math.sqrt(self.n_slots))

    def encode_with_orthogonal_content(self, slot_values: torch.Tensor,
                                       alpha: float,
                                       seed: int = 777) -> torch.Tensor:
        """A wavefront that decodes to the SAME observable but is far from the
        clean one in wave space.

        psi_alt = normalize(psi_clean + alpha * xi), xi an independent unit
        phasor. The added content is spread across all frequencies, so it aligns
        with no single role-filler and does not move the decoded slots, while the
        internal cosine falls to 1/sqrt(1+alpha^2). This is the control that
        separates the internal measure from the observable one.
        """
        psi = self.encode(slot_values)
        xi = self.random_wave(seed).to(psi.dtype).unsqueeze(0)
        return _normalize(psi + alpha * xi)

    # ------------------------------------------------------------------ decode
    def decode(self, psi: torch.Tensor, return_margins: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a wavefront to slot value indices.

        For each slot s: z_s = psi (*) conj(R_s), then snap z_s to slot s's
        codebook by maximum Hermitian inner product. This is the egress snap.

        Returns:
            idx:     [B, S] long
            quality: [B, S] float, best cosine per slot in [0, 1]
            margins: [B, S] float, best minus runner-up cosine (if requested)
        """
        if psi.dim() == 1:
            psi = psi.unsqueeze(0)

        z = self.unbind(psi.unsqueeze(1), self.role_codebook.unsqueeze(0))   # [B,S,D]
        z = _normalize(z)
        # einsum states the contraction axes explicitly. A matmul here is what
        # silently broadcast at B=1 and produced a wrong argmax.
        sims = torch.einsum("bsd,svd->bsv", z.conj(), self.value_codebook).abs()  # [B,S,V]
        idx = sims.argmax(dim=-1)                                             # [B,S]
        best = sims.gather(-1, idx.unsqueeze(-1)).squeeze(-1)                 # [B,S]
        if not return_margins:
            return idx, best
        masked = sims.scatter(-1, idx.unsqueeze(-1), float("-inf"))
        return idx, best, best - masked.max(dim=-1).values

    # ------------------------------------------------ observation-space stress
    def delta_sagnac_observational(self, psi_pred: torch.Tensor,
                                   ref_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Delta_Sagnac in OBSERVABLE space: 1 - (matched slots / total slots).

        This is the quantity the veto SHOULD use. Bounded in [0,1], dimension-free
        by construction (a slot match rate), and unlike the random-axiom measure it
        is not ~1.0 for every proposal.

        Fail-closed PER SAMPLE: a sample whose wavefront has ~zero energy gets
        stress 1.0 and valid=False, and is never normalized against another
        sample's norm.
        """
        if psi_pred.dim() == 1:
            psi_pred = psi_pred.unsqueeze(0)
        if ref_values.dim() == 1:
            ref_values = ref_values.unsqueeze(0)

        energy = psi_pred.norm(dim=-1)
        valid = energy > 1e-8

        idx, quality = self.decode(psi_pred)
        matched = (idx == ref_values).sum(dim=-1).float() / float(self.n_slots)
        stress = 1.0 - matched
        stress = torch.where(valid, stress, torch.ones_like(stress))
        return {"stress": stress, "valid": valid, "match_rate": 1.0 - stress,
                "quality": quality}

    # ---------------------------------------------------- legacy, for evidence
    def legacy_random_axiom_stress(self, psi: torch.Tensor, n_axioms: int = 128,
                                   seed: int = 99) -> torch.Tensor:
        """THE UN-PASSABLE BASELINE, kept reproducible on purpose.

        stress = 1 - max_k |<psi, axiom_k>| over RANDOM unit phasors. For random
        axioms the expected best agreement is O(sqrt(2 ln K / D)), so the stress
        concentrates near 1.0 and a veto at tau=0.35 essentially never fires.
        Do not use this in new code; it exists so the claim is re-measurable.
        """
        if psi.dim() == 1:
            psi = psi.unsqueeze(0)
        g = torch.Generator().manual_seed(seed)
        ax = self._phasors(n_axioms, g)
        sim = torch.einsum("bd,kd->bk", _normalize(psi).conj(), ax).abs()
        return 1.0 - sim.max(dim=-1).values

    # ------------------------------------------------------- continuous variant
    @staticmethod
    def dimension_normalized_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """||a - b||_2 / sqrt(d). Never compare a raw L2 across dimensions."""
        return (a - b).norm(dim=-1) / math.sqrt(a.shape[-1])
