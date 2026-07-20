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


if __name__ == "__main__":
    torch.manual_seed(0)
    D = 4096
    M = 16

    cleanup = ContinuousHopfieldCleanup(dim=2 * D)
    # Random unit-modulus complex engrams
    th = torch.rand(M, D) * 2 * math.pi
    engrams = torch.complex(torch.cos(th), torch.sin(th))
    engrams = engrams / torch.norm(engrams, p=2, dim=-1, keepdim=True)
    cleanup.store_engrams(engrams)

    # Corrupt engram 3 with moderate noise (relative amplitude 0.35 of the signal)
    noise = torch.complex(torch.randn(D), torch.randn(D))
    noise = noise / torch.norm(noise)
    noisy = engrams[3] + 0.35 * noise
    noisy = noisy / torch.norm(noisy)

    clean, weights = cleanup.retrieve(noisy, return_weights=True)
    clean_hard, idx, conf = cleanup.hard_retrieve(noisy)

    sim_soft = ContinuousHopfieldCleanup._complex_cosine(engrams[3], clean)
    print(f"[Hopfield] stored M={cleanup.num_engrams()} engrams, D={D}")
    print(f"[Hopfield] soft retrieval cosine to target: {sim_soft:.4f}")
    print(f"[Hopfield] hard retrieval index: {idx} (target 3), confidence: {conf:.4f}")
    assert int(idx) == 3, "Hopfield cleanup failed to recover the correct engram"
    assert sim_soft > 0.95, "Soft retrieval cosine too low"

    # Decoder smoke test
    dec = HopfieldActionDecoder(d_model=D)
    wave = dec.get_action_wave("ACTION1")
    noisy_wave = wave + 0.3 * torch.complex(torch.randn(D), torch.randn(D))
    action, conf = dec.decode_wave_to_action(noisy_wave)
    print(f"[Decoder] decoded '{action}' from noisy wave (confidence {conf:.4f})")
    assert action == "ACTION1"
    print("[Hopfield] ALL CHECKS PASSED")
