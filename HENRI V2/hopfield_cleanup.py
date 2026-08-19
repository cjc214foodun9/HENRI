"""
Project HENRI: Continuous Modern Hopfield Cleanup Layer.

Boundary interface between the continuous wave core (Zone B) and the discrete
symbolic engram store (Zone C). A noisy wavefront exiting the diffractive core
is snapped onto the nearest canonical engram via a single softmax-weighted
superposition step.

Theory (Dense Associative Memory / Modern Hopfield Network):
    Energy:   E(psi) = -tau * log sum_k exp( Re(psi^dag M_k) / tau )
    Retrieval: s = sum_k softmax(beta * <r, v^k>) * v^k
    Capacity: M < exp(alpha * d)  -- exponential in dimension, with
              max cross-talk ~ sqrt(2 ln M / d), giving single-iteration
              convergence with probability 1 - o(1).

References: Ramsauer et al. 2020; notebook source synthesis (nlm_hopfield.md).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousHopfieldCleanup(nn.Module):
    """
    Continuous Modern Hopfield Network over complex or real wavefronts.

    Stores canonical engrams as rows of a memory matrix M of shape [M, D].
    Retrieval is a single softmax-weighted superposition over engrams; the
    inverse temperature beta = 1/tau controls selection sharpness.
    """

    def __init__(self, dim: int, beta: float = None):
        super().__init__()
        self.dim = dim
        # Default inverse temperature scales as sqrt(d) -- the proven regime
        # for clean separation when memories are ~orthogonal on the sphere.
        self.beta = beta if beta is not None else math.sqrt(dim)
        # Engram memory: registered as a buffer so it moves with .to(device)
        # but is not a trained parameter (Zone C owns long-term persistence).
        self.register_buffer("engrams", torch.zeros(0, dim))

    @torch.no_grad()
    def store_engrams(self, waves: torch.Tensor) -> int:
        """
        Crystallize canonical engrams into the cleanup matrix.
        waves: [M, D] (real or complex64). Rows are L2-normalized.
        Returns the number of stored engrams.
        """
        if waves.is_complex():
            waves = torch.view_as_real(waves).reshape(waves.shape[0], -1)
        waves = waves.to(self.engrams.device, torch.float32)
        assert waves.shape[-1] == self.dim, (
            f"Engram dim {waves.shape[-1]} != cleanup dim {self.dim}; "
            "store complex [M, dim/2] or real [M, dim] waves."
        )
        waves = F.normalize(waves, p=2, dim=-1)
        if self.engrams.numel() == 0:
            self.engrams = waves
        else:
            self.engrams = torch.cat([self.engrams, waves], dim=0)
        return self.engrams.shape[0]

    @torch.no_grad()
    def clear(self):
        self.engrams = torch.zeros(0, self.dim, device=self.engrams.device)

    def num_engrams(self) -> int:
        return self.engrams.shape[0]

    def energy(self, wave: torch.Tensor) -> torch.Tensor:
        """
        E(psi) = -tau * logsumexp(beta * Re(psi^dag M_k)), tau = 1/beta.
        wave: [..., D]. Returns scalar energy per leading element.
        """
        assert self.engrams.numel() > 0, "No engrams stored; call store_engrams first."
        if self.engrams.device != wave.device:
            self.engrams = self.engrams.to(wave.device)
        r = self._flatten(wave)
        sim = r @ self.engrams.T  # [..., M]
        tau = 1.0 / self.beta
        return -tau * torch.logsumexp(self.beta * sim, dim=-1)

    def retrieve(self, wave: torch.Tensor, return_weights: bool = False):
        """
        Single-step attractor cleanup.
        wave: [..., D] noisy wavefront (real or complex).
        Returns (clean_wave, weights) where clean_wave has the same shape/dtype
        family as the input (complex in -> complex out), snapped onto the
        engram attractor. weights is the softmax distribution over engrams.
        """
        assert self.engrams.numel() > 0, "No engrams stored; call store_engrams first."
        if self.engrams.device != wave.device:
            self.engrams = self.engrams.to(wave.device)
        was_complex = wave.is_complex()
        r = self._flatten(wave)
        r = F.normalize(r, p=2, dim=-1)
        sim = r @ self.engrams.T  # [..., M]
        weights = torch.softmax(self.beta * sim, dim=-1)
        clean = weights @ self.engrams  # [..., D]
        clean = F.normalize(clean, p=2, dim=-1)
        if was_complex:
            clean = torch.view_as_complex(
                clean.reshape(*clean.shape[:-1], -1, 2).contiguous()
            )
        if return_weights:
            return clean, weights
        return clean

    def hard_retrieve(self, wave: torch.Tensor):
        """
        Zero-entropy crystallization: snap to the single nearest engram.
        Returns (clean_wave, index, similarity).
        """
        _, weights = self.retrieve(wave, return_weights=True)
        idx = torch.argmax(weights, dim=-1)
        clean = self.engrams[idx]
        if wave.is_complex():
            clean = torch.view_as_complex(
                clean.reshape(*clean.shape[:-1], -1, 2).contiguous()
            )
        return clean, idx, weights.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    def lexical_snap(self, wave: torch.Tensor, top_k: int = 1):
        """
        Phase 8.34 Evolution I: multi-vector zero-entropy Lexical Snap.

        Batched hard retrieval over the engram codebook: each input wave
        [..., D] collapses to the nearest discrete engram index. Unlike
        hard_retrieve (single wave, returns the cleaned wave), this is the
        codebook-level egress primitive: continuous waves -> discrete
        symbolic indices, no BPTT.

        Returns (indices, confidences): indices [..., top_k] (argmax order),
        confidences [..., top_k] (raw engram similarities, not softmax).
        """
        assert self.engrams.numel() > 0, "No engrams stored; call store_engrams first."
        if self.engrams.device != wave.device:
            self.engrams = self.engrams.to(wave.device)
        r = self._flatten(wave)
        r = F.normalize(r, p=2, dim=-1)
        sim = r @ self.engrams.T  # [..., M]
        if top_k == 1:
            idx = sim.argmax(dim=-1)
            conf = sim.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            return idx, conf
        idx = sim.topk(top_k, dim=-1).indices
        conf = sim.gather(-1, idx)
        return idx, conf

    def _flatten(self, wave: torch.Tensor) -> torch.Tensor:
        """Complex [.., D/2] -> real [.., D]; real passes through."""
        if wave.is_complex():
            return torch.view_as_real(wave).reshape(*wave.shape[:-1], -1).to(torch.float32)
        return wave.to(torch.float32)

    @staticmethod
    def _complex_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Hermitian cosine similarity Re(<a,b>)/(||a|| ||b||) for complex waves."""
        num = torch.real(torch.vdot(a.flatten(), b.flatten()))
        den = torch.norm(a) * torch.norm(b) + 1e-12
        return (num / den).item()


class DualScaleAnalogLexicalSnap(nn.Module):
    """Phase 8.35 T2 (Miller et al. 2026): dual-scale analog lexical snap.

    The document formula (HENRI-SYNTHESIS-MILLER-LEE-2026):

        Psi_gated = Psi_micro ⊙ Softmax(Re(Psi_macro^† W_gate) / tau)

    A top-down macro-option wave Psi_macro (low-D) gates which micro
    dimensions of Psi_micro participate in Hopfield codebook snapping. The
    mask is a softmax over micro dimensions (temperature tau); the gated
    wave is renormalized, then lexical_snap runs inside the active
    macro-option subspace. This prevents unbinding logit scrambling across
    non-relevant vocabulary dimensions (the egress bottleneck).

    Controls:
      - macro_wave = 0  -> uniform mask -> gated == raw wave -> byte-identical
        to plain lexical_snap (true control arm).
      - tau -> 0        -> one-hot mask (deterministic subspace restriction).

    Default-OFF: no production consumer; tests and the 8.35 benchmark
    activate it.
    """

    def __init__(
        self,
        dim_micro: int = 65536,
        dim_macro: int = 2048,
        tau: float = 1.0,
        beta: float = None,
        seed: int = 835,
        top_k: Optional[int] = None,
    ):
        super().__init__()
        self.dim_micro = dim_micro
        self.dim_macro = dim_macro
        self.tau = float(tau)
        self.top_k = top_k
        self.cleanup = ContinuousHopfieldCleanup(dim=dim_micro, beta=beta)
        g = torch.Generator().manual_seed(seed)
        w_init = torch.randn(dim_macro, dim_micro, generator=g) / math.sqrt(dim_macro)
        self.register_buffer("W_gate", F.normalize(w_init, p=2, dim=-1))

    @torch.no_grad()
    def store_engrams(self, waves: torch.Tensor) -> int:
        return self.cleanup.store_engrams(waves)

    def gate_mask(self, macro_wave: torch.Tensor) -> torch.Tensor:
        """Softmax(Re(macro^† W_gate) / tau) over micro dims. [micro]

        top_k=None: pure document formula (flat at D=65,536, tau=1).
        top_k=k:    sparse ensemble gate — softmax over the top-k logits
        only, zeros elsewhere, renormalized. This is the Miller et al.
        "macro wave gates which local ensembles synchronize" mechanism;
        a flat softmax over 65,536 dims cannot restrict any subspace.
        macro_wave=0 (no top-down control) always returns the uniform mask
        (byte-identical to plain lexical_snap).
        """
        m = macro_wave.view(-1)
        if m.is_complex():
            m = torch.view_as_real(m).reshape(-1).to(torch.float32)
        m = m / (torch.norm(m) + 1e-12)
        logits = m @ self.W_gate  # [micro]
        if torch.norm(m) < 1e-9:
            return torch.full_like(logits, 1.0 / self.dim_micro)
        if self.top_k is None:
            return torch.softmax(logits / self.tau, dim=-1)
        k = min(int(self.top_k), self.dim_micro)
        vals, _ = torch.topk(logits, k)
        thresh = vals[-1]
        keep = logits >= thresh
        sel = torch.where(keep, logits, torch.full_like(logits, -1e30))
        soft = torch.softmax(sel / self.tau, dim=-1)
        return soft

    def gated_wave(self, wave: torch.Tensor, macro_wave: torch.Tensor) -> torch.Tensor:
        """Psi_micro ⊙ mask, renormalized (complex/real preserved family)."""
        mask = self.gate_mask(macro_wave).to(wave.device)
        was_complex = wave.is_complex()
        r = wave.view(-1)
        if was_complex:
            r = torch.view_as_real(r).reshape(-1).to(torch.float32)
        g = F.normalize(r * mask, p=2, dim=-1)
        if was_complex:
            g = torch.view_as_complex(g.reshape(-1, 2).contiguous())
        return g.reshape(wave.shape)

    def snap(self, wave: torch.Tensor, macro_wave: torch.Tensor, top_k: int = 1):
        """Macro-gated lexical snap. Returns (indices, confidences) like
        ContinuousHopfieldCleanup.lexical_snap."""
        assert self.cleanup.engrams.numel() > 0, "No engrams stored; call store_engrams first."
        return self.cleanup.lexical_snap(self.gated_wave(wave, macro_wave), top_k=top_k)


class HopfieldActionDecoder(nn.Module):
    """
    Drop-in replacement for HolographicActionDecoder.

    Canonical action waves are stored as Hopfield engrams drawn from a
    pseudo-orthogonal random basis (same RNG discipline as the O-VSA
    canonical basis) instead of correlated linspace phase ramps. Decoding is
    a hard Hopfield retrieval: the policy wave is snapped to the nearest
    action attractor and its engram index maps to the GameAction.
    """

    def __init__(self, d_model: int = 4096, action_enum_class=None, seed: int = 1234):
        super().__init__()
        self.d_model = d_model
        self.action_to_id = {}
        self.id_to_action = {}

        if action_enum_class is not None:
            for idx, action in enumerate([a for a in action_enum_class]):
                self.action_to_id[action] = idx
                self.id_to_action[idx] = action
        else:
            for idx, action in enumerate(["UP", "DOWN", "LEFT", "RIGHT", "ACTION1", "ACTION2"]):
                self.action_to_id[action] = idx
                self.id_to_action[idx] = action

        num_actions = len(self.action_to_id)
        # Pseudo-orthogonal complex action engrams on the unit hypersphere
        g = torch.Generator().manual_seed(seed)
        theta = torch.rand(num_actions, d_model, generator=g) * 2 * math.pi
        basis = torch.complex(torch.cos(theta), torch.sin(theta))
        basis = basis / torch.norm(basis, p=2, dim=-1, keepdim=True)

        self.cleanup = ContinuousHopfieldCleanup(dim=2 * d_model)
        self.cleanup.store_engrams(basis)

    def get_action_wave(self, action) -> torch.Tensor:
        idx = self.action_to_id.get(action, 0)
        real = self.cleanup.engrams[idx]
        return torch.view_as_complex(real.reshape(-1, 2).contiguous())

    def decode_wave_to_action(self, policy_wave: torch.Tensor):
        """
        Snap the policy wave to the nearest action attractor.
        Returns (action, confidence) where confidence is the softmax weight
        of the winning engram (1.0 = perfect resonance).
        """
        flat = policy_wave.view(-1)
        if not flat.is_complex():
            flat = flat.to(torch.complex64)
        flat = flat / torch.norm(flat, p=2).clamp(min=1e-12)
        _, idx, conf = self.cleanup.hard_retrieve(flat)
        return self.id_to_action[int(idx)], float(conf)

