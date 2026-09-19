#!/usr/bin/env python3
"""Role-filler compositional lexical codec -- amendment reconciliation carrier.

SOURCE
    HENRI-SPEC-2026-09-ONTOLOGICAL-GROUNDING-VLA (amendment document 1),
    "The Orthogonal Isolation Trap" and its proposed remedy.

CLAIM UNDER TEST
    The O-VSA ingress initializes token keys as pseudo-orthogonal random phase
    vectors, so the hypersphere distance between "lion" and "feline" equals the
    distance between "lion" and "teaspoon". Remedy proposed by the document:
        K_lion = K_taxa:mammal (x) K_morph:quadruped (x) K_scale:large (x) ...
    i.e. build each concept key by CIRCULAR CONVOLUTION of its attributes.

LIVE-CODE CHECK (OBSERVED, this session)
    o_vsa_ingress_tokenizer.py:34  raw_basis = torch.randn(vocab_size, num_blocks, 8)
    o_vsa_ingress_tokenizer.py:39  spatial_theta_x = torch.rand(...) * 2*pi
    o_vsa_ingress_tokenizer.py:40  spatial_theta_y = torch.rand(...) * 2*pi
    The "trap" description IS accurate about the live initializer.

    BUT the amendment's own remedy code re-creates the trap:
        def _init_grounded_codebook(self, vocab_size, dim):
            phases = torch.rand(vocab_size, dim) * 2.0 * math.pi
            base_keys = torch.complex(cos(phases), sin(phases))
            return F.normalize(base_keys, p=2, dim=-1)
    That is the SAME independent-random-phasor construction the document calls
    the flaw. So the document describes a remedy it does not implement. This
    module tests the remedy properly, in three arms, and lets the measurement
    decide which reading is correct.

THE MATH -- AND WHERE THE LITERAL REMEDY FAILS
    Circular convolution is elementwise multiplication in the frequency domain.
    For the literal arm, K_c has frequency-domain form  PROD_a  A^_a,v :
        K_lion = A_taxa:mammal * A_morph:quadruped * ... * A_habitat:savanna
    Two concepts that share four of six attribute factors differ by the product
    of the two unshared factors -- still a unit-modulus vector. The time-domain
    correlation is the MEAN of that ratio vector, which for unit-modulus entries
    is ~0 with standard deviation 1/sqrt(D). So the literal arm should show NO
    semantic proximity. It is a testable prediction, not an opinion.

    Structure requires SHARING SUM TERMS, which is the standard VSA role-filler
    construction assembled by bundling (superposition):
        K_c = normalize( SUM_a  R_a  (x)  V_{a,v_a(c)} )
    Two concepts sharing an (attribute, value) pair share that term EXACTLY, so
    their bundles correlate by roughly (shared / total). Concepts sharing
    nothing stay near-orthogonal. The role key R_a makes the attribute
    order-sensitive and unbindable.

ARMS
    random_control         independent random phasors (the live trap)
    literal_convolution    the document's remedy, as literally described
    role_filler_bundle     the corrected construction (role-filler + bundling)

SCOPE
    Reduced dimension by default (D=4096) so the measurement is CPU-cheap; the
    construction is dimension-independent, and the null 1/sqrt(D) is reported so
    the reader can rescale. No production path is modified. No capability claim:
    this measures whether an ENCODER creates semantic neighbourhood structure,
    which is a necessary-not-sufficient property.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DIM = 4096

# A deliberately small, hand-authored ontology. Six attribute dimensions; the
# point is that two concepts SHARE attribute values, which is what a
# compositional codec must be able to exploit.
ONTOLOGY: Dict[str, Dict[str, str]] = {
    "lion":     dict(taxa="mammal",  morph="quadruped", scale="large",  trophic="carnivore", valence="apex_threat", habitat="savanna"),
    "feline":   dict(taxa="mammal",  morph="quadruped", scale="medium", trophic="carnivore", valence="threat",      habitat="jungle"),
    "cat":      dict(taxa="mammal",  morph="quadruped", scale="small",  trophic="carnivore", valence="neutral",     habitat="domestic"),
    "dog":      dict(taxa="mammal",  morph="quadruped", scale="medium", trophic="omnivore",  valence="neutral",     habitat="domestic"),
    "teaspoon": dict(taxa="artifact",morph="rod",       scale="small",  trophic="none",      valence="neutral",     habitat="kitchen"),
    "quantum":  dict(taxa="abstract",morph="field",     scale="tiny",   trophic="none",      valence="neutral",     habitat="vacuum"),
}


def _phasors(shape: Tuple[int, ...], seed: int, device: str = "cpu") -> torch.Tensor:
    """Deterministic unit-modulus complex phasors on the torus (|z_i| = 1)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    ph = torch.rand(*shape, generator=g, device="cpu") * (2.0 * math.pi)
    return torch.polar(torch.ones_like(ph), ph).to(device)


class GroundedLexicalCodec(nn.Module):
    """Three construction arms over one shared ontology."""

    ARMS = ("random_control", "literal_convolution", "role_filler_bundle")

    def __init__(self, dim: int = DEFAULT_DIM, seed: int = 20260918,
                 device: str = "cpu") -> None:
        super().__init__()
        self.dim = int(dim)
        self.device = device
        self.roles: List[str] = sorted({a for v in ONTOLOGY.values() for a in v})
        self.values: Dict[str, List[str]] = {
            a: sorted({v[a] for v in ONTOLOGY.values()}) for a in self.roles}
        self.concepts: List[str] = sorted(ONTOLOGY)

        self._ridx = {a: i for i, a in enumerate(self.roles)}
        self._vidx = {a: {v: j for j, v in enumerate(self.values[a])}
                      for a in self.roles}
        self._cidx = {c: i for i, c in enumerate(self.concepts)}

        # Role keys R_a and value keys V_{a,v}. Both are frozen buffers: this
        # carrier is an ENCODER, not a learned module.
        self.register_buffer(
            "role_keys",
            _phasors((len(self.roles), self.dim), seed + 1, device))
        vk = []
        for i, a in enumerate(self.roles):
            vk.append(_phasors((len(self.values[a]), self.dim),
                               seed + 7 + i, device))
        self.value_keys: List[torch.Tensor] = vk

    # ------------------------------------------------------------------ terms
    def attr_terms(self, concept: str) -> List[Tuple[int, int]]:
        return [(self._ridx[a], self._vidx[a][ONTOLOGY[concept][a]])
                for a in self.roles]

    def shared_attribute_values(self, c1: str, c2: str) -> int:
        """How many (role,value) pairs the two concepts share. Max = len(roles)."""
        return sum(1 for a in self.roles if ONTOLOGY[c1][a] == ONTOLOGY[c2][a])

    # --------------------------------------------------------------- encoders
    def encode(self, arm: str) -> torch.Tensor:
        """Return [n_concepts, dim] unit-norm complex keys for one arm."""
        if arm not in self.ARMS:
            raise ValueError(f"unknown arm: {arm!r}")

        if arm == "random_control":
            # The live trap: independent random phasors, no shared structure.
            # NORMALIZED like every other arm. An earlier version returned raw
            # unit-modulus phasors (norm sqrt(D)=64) while the other arms were
            # L2-normalized to 1, so this arm's inner products were ~D times
            # larger -- a SCALE CONFLATION that fabricated a fake "+18 semantic
            # gap" for the arm that should have shown ~0. Caught by the C1 gate.
            return F.normalize(
                _phasors((len(self.concepts), self.dim), 999, self.device),
                p=2, dim=-1)

        if arm == "literal_convolution":
            # The document's remedy, taken literally: convolve attribute keys.
            keys = []
            for c in self.concepts:
                acc = torch.ones(self.dim, dtype=torch.complex64,
                                 device=self.device)
                for ri, vi in self.attr_terms(c):
                    acc = acc * self.value_keys[ri][vi]
                keys.append(F.normalize(acc, p=2, dim=-1))
            return torch.stack(keys)

        # Corrected: role-filler binding assembled by bundling.
        keys = []
        for c in self.concepts:
            acc = torch.zeros(self.dim, dtype=torch.complex64,
                              device=self.device)
            for ri, vi in self.attr_terms(c):
                rk = self.role_keys[ri]
                vk = self.value_keys[ri][vi]
                bound = torch.fft.ifft(torch.fft.fft(rk) * torch.fft.fft(vk))
                acc = acc + bound
            keys.append(F.normalize(acc, p=2, dim=-1))
        return torch.stack(keys)

    # ------------------------------------------------------------- retrieval
    def similarity(self, arm: str) -> torch.Tensor:
        """[n_concepts, n_concepts] cosine similarity matrix (real part)."""
        K = self.encode(arm).to(torch.complex64)
        return torch.real(K @ K.conj().T)

    def recover_attribute(self, concept: str, role: str) -> Tuple[int, torch.Tensor]:
        """Unbind one attribute from a role_filler_bundle concept key.

        Unbinding = circular correlation = frequency-domain multiplication by the
        conjugate of the role key, then match against that role's value keys.
        Returns (argmax index, similarity vector).
        """
        if role not in self._ridx:
            raise ValueError(f"unknown role: {role!r}")
        K = self.encode("role_filler_bundle")
        ci, ri = self._cidx[concept], self._ridx[role]
        unbound = torch.fft.ifft(
            torch.fft.fft(K[ci]) * torch.conj(torch.fft.fft(self.role_keys[ri])))
        sims = torch.real(unbound @ self.value_keys[ri].conj().T)
        return int(torch.argmax(sims).item()), sims

    # ------------------------------------------------------------------ stats
    def gap_report(self, arm: str) -> Dict[str, float]:
        """Shared-attribute vs disjoint-pair cosine, plus the null scale.

        NULL SCALE: for independent unit-modulus phasors on C^D, the expected
        |<u,v>| is about sqrt(pi)/(2*sqrt(D)). Reported so a reader can compare
        the gap against the noise floor of the SAME dimension.
        """
        S = self.similarity(arm)
        shared, disjoint = [], []
        n = len(self.concepts)
        for i in range(n):
            for j in range(i + 1, n):
                k = self.shared_attribute_values(self.concepts[i],
                                                 self.concepts[j])
                if k == 0:
                    disjoint.append(float(S[i, j]))
                else:
                    shared.append(float(S[i, j]))
        null = math.sqrt(math.pi) / (2.0 * math.sqrt(self.dim))
        out = {
            "arm": arm,
            "n_shared_pairs": len(shared),
            "n_disjoint_pairs": len(disjoint),
            "mean_shared": (sum(shared) / len(shared)) if shared else None,
            "mean_disjoint": (sum(disjoint) / len(disjoint)) if disjoint else None,
            "null_abs_expected": null,
        }
        if shared and disjoint:
            out["semantic_gap"] = out["mean_shared"] - out["mean_disjoint"]
        else:
            out["semantic_gap"] = None
        return out


# ---------------------------------------------------------------- verification
# Pre-registered gates. Fixed BEFORE running, so the arms cannot be tuned.
GATE_GAP = 0.05          # a real codec must separate shared from disjoint pairs
GATE_RECOVERY = 0.90     # role-filler unbinding must recover the attribute value


def pre_registered_verdict(reports: Dict[str, Dict[str, float]],
                           recovery_rate: float) -> Tuple[str, Dict[str, bool]]:
    """Compare arms against the gates and emit a bounded verdict."""
    rnd = reports.get("random_control", {})
    lit = reports.get("literal_convolution", {})
    rf = reports.get("role_filler_bundle", {})

    checks = {
        "C1_random_has_no_gap": bool(
            abs(rnd.get("semantic_gap") or 0.0) < GATE_GAP),
        "C2_literal_convolve_has_no_gap": bool(
            abs(lit.get("semantic_gap") or 0.0) < GATE_GAP),
        "C3_role_filler_has_gap": bool(
            (rf.get("semantic_gap") or 0.0) >= GATE_GAP),
        "C4_role_filler_beats_literal": bool(
            (rf.get("semantic_gap") or 0.0) - (lit.get("semantic_gap") or 0.0)
            >= GATE_GAP),
        "C5_unbinding_recovers": bool(recovery_rate >= GATE_RECOVERY),
    }

    if not checks["C3_role_filler_has_gap"]:
        verdict = ("FALSIFIED_COMPOSITION_FIXES_NOTHING -- the corrected "
                   "construction does not create a measurable semantic gap "
                   "either, so the 'Orthogonal Isolation Trap' has no "
                   "compositional remedy at this scale.")
    elif not checks["C2_literal_convolve_has_no_gap"]:
        verdict = ("LITERAL_REMEDY_ALSO_WORKS -- the document's convolution "
                   "chain does create proximity; the corrected construction is "
                   "not uniquely responsible. Re-derive before claiming.") 
    elif checks["C4_role_filler_beats_literal"] and checks["C5_unbinding_recovers"]:
        verdict = ("CORRECTED_CONSTRUCTION_REQUIRED -- role-filler bundling "
                   "creates the semantic gap and supports exact unbinding, while "
                   "the document's literal convolution chain (and the live "
                   "random init) do not. The remedy is real; the mechanism in "
                   "the document is not the one that provides it.")
    else:
        verdict = ("PARTIAL -- a gap exists but either the literal arm also "
                   "succeeds or unbinding fails. Not promotable.")
    return verdict, checks
