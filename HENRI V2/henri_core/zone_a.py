"""Zone A: modality ingress into the 4-slot Cl(3,0) wave.

Document anchors:
  doc p32 : Zone A Lean Parallel Transduction (henri_clifford_vla_encoder.py);
            maps raw inputs into 8,192 Cl(3,0) blocks on S^{D-1};
            enforces the 4-Slot Structural Subspace Rule
  doc p35 : unused slots set to ZERO to prevent the 99.85% noise-fill defect
  doc p6  : slot block ranges
  doc p20 : Slot 0 Entity, Slot 1 Action, Slot 2 Object, Slot 3 Context

Mechanism
    Each token id addresses one flat dimension of its slot and writes a unit
    phasor there. A learned slot router assigns every token to one slot. Slot
    accumulation is a complex sum, so equal phases add and opposed phases
    cancel - the interference the document describes.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn

from . import substrate as sub


class CliffordVLASlotEncoder(nn.Module):
    """Encode text, image patches, and joint vectors into one unit-norm wave.

    Args:
        dim:        wave dimension D (65536 full, small in tests)
        vocab:      tokenizer vocabulary size
        img_patch:  patch edge in pixels (doc: 16)
        n_patches:  patch count for one frame (doc: 256 -> 16x16 grid)
        action_dim: SE(3) joint vector width (doc: position 3 + quat 3 + grip 1)
    """

    def __init__(self, dim: int = sub.DEFAULT_DIM, vocab: int = 512,
                 img_patch: int = 16, n_patches: int = 256,
                 action_dim: int = 7, positional: bool = False,
                 pos_omega: float = math.pi / 2.0,
                 pos_multifreq: bool = False,
                 pos_block: int = 16, pos_rope_theta: float = 5.0e5,
                 ingress_seed: int | None = None):
        super().__init__()
        # D127: positional algebra. OFF by default so every committed receipt
        # reproduces byte-for-byte. ON rotates token t's phasor by t * pos_omega.
        self.positional = bool(positional)
        # D141 (self-caught): D131 showed the single-frequency rotation
        # (omega = pi/2) aliases mod 4, so positions 0 and 4 are identical and
        # 'IRIRIR' collapses back onto 'IR'. This is the multi-frequency remedy:
        # replicate each token write over a BLOCK of K addresses and rotate entry
        # j by t * omega_j with RoPE-style geometric frequencies
        #     omega_j = theta ^ (-2j/K).
        # Default OFF, so every committed receipt reproduces byte-for-byte.
        self.pos_multifreq = bool(pos_multifreq)
        self.pos_block = int(pos_block)
        self.pos_rope_theta = float(pos_rope_theta)
        self.pos_omega = float(pos_omega)
        self.dim = int(dim)
        self.vocab = int(vocab)
        self.img_patch = int(img_patch)
        self.n_patches = int(n_patches)
        self.action_dim = int(action_dim)
        self.n_slots = sub.N_SLOTS
        self.slot_dim = self.dim // self.n_slots
        # D143 (self-caught): this set-up MUST run AFTER slot_dim exists. The
        # first version read self.slot_dim 13 lines early -> AttributeError.
        if self.pos_multifreq:
            # D144 (self-caught by the diagnostic): the FIRST version used the
            # document's RoPE geometric spectrum, omega_j = theta^(-2j/K) with
            # theta = 5e5. That is tuned for 4096-token contexts. On a 4-5 token
            # corpus 14 of 16 frequencies evaluate to ~0, so position vectors are
            # nearly parallel: measured cos('ab','ba') = +0.970052 (SINGLE was
            # 0.000000), and max alias over shifts 1..8 = +0.990126. VERDICT was
            # MULTIFREQ_FAIL on 2 of 5 pre-registered properties.
            #
            # Analytic cause. The reversal cosine reduces to
            #     cos('ab','ba') = 2 Re sum_j exp(i omega_j) / (2K)
            # which is 0 exactly when the sum over the block vanishes, i.e. when
            # the frequencies are UNIFORMLY spread. Use the DFT basis
            #     omega_j = 2 pi j / K
            # so sum_j exp(i omega_j) = 0 and <p(t),p(t')> = 0 for t != t' (mod K).
            # No aliasing for any shift < K. This keeps multi-frequency phase
            # encoding while fixing the spectrum to the corpus's length regime.
            j = torch.arange(self.pos_block, dtype=torch.float64)
            om = (2.0 * math.pi * j / self.pos_block).to(torch.float32)
            self.register_buffer("pos_omega_vec", om)
            self.register_buffer("pos_offsets",
                                 torch.arange(self.pos_block, dtype=torch.long))
            self.n_addr = max(1, self.slot_dim // self.pos_block)
        else:
            self.register_buffer("pos_omega_vec",
                                 torch.zeros(1, dtype=torch.float32))
            self.register_buffer("pos_offsets", torch.zeros(1, dtype=torch.long))
            self.n_addr = self.slot_dim

        # Learned slot router (doc: 4-slot rule enforced on every write).
        # D128: token_emb AND slot_router draw from the global torch RNG, and
        # both sit in the text path (slot_router consumes token_emb). The ingress
        # is frozen by the zero-D_c contract, so routing was a random draw that
        # training never corrected: G-U4 redrew its own verdict across
        # construction seeds (range 0.150874, 2 pass / 6 fail).
        # ingress_seed=None keeps legacy behaviour, so every committed receipt
        # reproduces byte-for-byte. An int forks the RNG and seeds the two layers,
        # leaving the caller's global RNG state untouched.
        self.ingress_seed = ingress_seed
        with (torch.random.fork_rng(devices=[]) if ingress_seed is not None
              else contextlib.nullcontext()):
            if ingress_seed is not None:
                torch.manual_seed(int(ingress_seed))
            self.token_emb = nn.Embedding(self.vocab, 32)
            self.slot_router = nn.Linear(32, self.n_slots)
            # D129 (self-caught by the unit check): joint_proj was created
            # OUTSIDE this fork, so it still drew from the global RNG and the
            # "global RNG untouched" claim was FALSE. Moving it inside makes the
            # pin side-effect-free. The relative global order
            # token_emb -> slot_router -> joint_proj is preserved, so legacy
            # (ingress_seed=None) receipts still reproduce byte-for-byte.
            self.joint_proj = nn.Linear(self.action_dim, self.slot_dim,
                                        bias=False)

        # Per-slot, per-position phase address. Standard normal, seeded, frozen.
        g = torch.Generator().manual_seed(20261004)
        self.register_buffer(
            "angle", torch.rand(self.dim, generator=g) * 2.0 * math.pi)

        # Patch phase address (vision). The joint projection is created with the
        # other random layers above, inside the same fork, so the reset stays
        # side-effect-free on the global RNG (D129).
        self.register_buffer(
            "patch_phase", torch.rand(self.n_patches, generator=g) * 2.0 * math.pi)

    # ---------------------------------------------------------------- helpers
    def _token_writes(self, ids: torch.Tensor):
        """Return per-slot unit phasors for one sequence. ids: [T] long."""
        emb = self.token_emb(ids)                       # [T, 32]
        logits = self.slot_router(emb)                  # [T, 4]
        routes = logits.argmax(dim=-1)                  # [T] one slot per token
        acc = [torch.zeros(self.slot_dim, dtype=torch.complex64)
               for _ in range(self.n_slots)]
        for t in range(ids.shape[0]):
            tok = int(ids[t])
            if tok >= self.dim:
                tok = tok % self.dim                    # address wraps, deterministic
            s = int(routes[t])
            phase0 = self.angle[tok].to(torch.float32)
            if self.pos_multifreq:
                # D141: replicate the write over a BLOCK of K addresses, one per
                # RoPE frequency. This is the D131 fix -- a single omega aliases
                # modulo 4, so 'IRIRIR' collapsed back onto 'IR'.
                K = self.pos_block
                base = (tok % self.n_addr) * K
                for j in range(K):
                    ph = phase0 + float(t) * float(self.pos_omega_vec[j])
                    acc[s][base + j] += torch.polar(torch.tensor(1.0), ph)
                continue
            local = tok % self.slot_dim
            phase = phase0
            if self.positional:
                # D127: token ORDER was discarded. The address depends on token
                # identity only, so 'ab' and 'ba' wrote identical phasors and
                # cos('ab','ba') was exactly 1.000000. Rotate by a relative
                # position angle, as RoPE applies relative position to rotors.
                phase = phase + float(t) * self.pos_omega
            acc[s][local] += torch.polar(torch.tensor(1.0), phase)
        return acc, routes

    # ------------------------------------------------------------------ public
    def encode_text(self, text: str, tokenizer) -> torch.Tensor:
        """Map a string to a unit-norm complex wave [D]. Unused slots stay zero."""
        ids = tokenizer.encode(text)
        if not ids:
            return torch.zeros(self.dim, dtype=torch.complex64)
        acc, _ = self._token_writes(torch.tensor(ids, dtype=torch.long))
        return self._assemble(acc)

    def encode_scene(self, text: str | None = None, tokenizer=None,
                     image: torch.Tensor | None = None,
                     action: torch.Tensor | None = None) -> torch.Tensor:
        """Multi-modal ingress. Absent modalities contribute exactly zero.

        image:  [n_patches, patch*patch] or [n_patches, C, h, w] floats in [0,1]
        action: [action_dim] floats
        """
        acc = [torch.zeros(self.slot_dim, dtype=torch.complex64)
               for _ in range(self.n_slots)]
        if text:
            if tokenizer is None:
                raise ValueError("tokenizer required for text ingress")
            tw, _ = self._token_writes(
                torch.tensor(tokenizer.encode(text), dtype=torch.long))
            for s in range(self.n_slots):
                acc[s] = acc[s] + tw[s]
        if image is not None:
            acc[sub.SLOT_NAMES.index("context")] = \
                acc[sub.SLOT_NAMES.index("context")] + self._patch_writes(image)
        if action is not None:
            acc[sub.SLOT_NAMES.index("action")] = \
                acc[sub.SLOT_NAMES.index("action")] + self._joint_writes(action)
        return self._assemble(acc)

    def _assemble(self, acc) -> torch.Tensor:
        """Per-slot normalize, then global unit norm. Zero slots remain zero."""
        parts = []
        for a in acc:
            n = a.abs().sum()
            parts.append(a / n.clamp_min(1e-12) if float(n) > 0 else a)
        psi = torch.cat(parts)
        return sub.unit_norm(psi)

    def _patch_writes(self, image: torch.Tensor) -> torch.Tensor:
        p = image.reshape(self.n_patches, -1).mean(dim=-1).to(torch.float32)
        p = torch.tanh(p)
        ph = self.patch_phase.to(p.device)
        z = torch.polar(p.abs().clamp_min(1e-6), self.angle[:self.n_patches].to(p.device) + ph)
        out = torch.zeros(self.slot_dim, dtype=torch.complex64)
        n = min(self.slot_dim, self.n_patches)
        out[:n] = z[:n]
        return out

    def _joint_writes(self, action: torch.Tensor) -> torch.Tensor:
        a = action.to(torch.float32).reshape(-1)[:self.action_dim]
        if a.numel() < self.action_dim:
            pad = torch.zeros(self.action_dim - a.numel())
            a = torch.cat([a, pad])
        base = self.joint_proj(a)                       # [slot_dim] real
        return torch.polar(base.abs().clamp_min(1e-6),
                           self.angle[:self.slot_dim].to(base.device))

    # ------------------------------------------------------------------ purity
    def slot_energy(self, psi: torch.Tensor) -> torch.Tensor:
        """Energy fraction per slot. Foundation for gate G-U1."""
        e = (psi.abs() ** 2).reshape(self.n_slots, self.slot_dim).sum(dim=-1)
        return e / e.sum().clamp_min(1e-12)

    def noise_hash_rows(self, psi: torch.Tensor, active: set[int]) -> int:
        """Count samples in slots that should be zero. Gate G-U1: must be 0."""
        e = (psi.abs() ** 2).reshape(self.n_slots, self.slot_dim)
        bad = 0
        for s in range(self.n_slots):
            if s not in active and float(e[s].abs().sum()) > 1e-12:
                bad += 1
        return bad

    def extra_repr(self) -> str:
        return (f"dim={self.dim} vocab={self.vocab} slots={self.n_slots} "
                f"patch={self.img_patch} n_patches={self.n_patches}")
