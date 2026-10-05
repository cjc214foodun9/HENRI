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
                 action_dim: int = 7):
        super().__init__()
        self.dim = int(dim)
        self.vocab = int(vocab)
        self.img_patch = int(img_patch)
        self.n_patches = int(n_patches)
        self.action_dim = int(action_dim)
        self.n_slots = sub.N_SLOTS
        self.slot_dim = self.dim // self.n_slots

        # Learned slot router (doc: 4-slot rule enforced on every write)
        self.token_emb = nn.Embedding(self.vocab, 32)
        self.slot_router = nn.Linear(32, self.n_slots)

        # Per-slot, per-position phase address. Standard normal, seeded, frozen.
        g = torch.Generator().manual_seed(20261004)
        self.register_buffer(
            "angle", torch.rand(self.dim, generator=g) * 2.0 * math.pi)

        # Patch phase address (vision) and joint phase address (action)
        self.register_buffer(
            "patch_phase", torch.rand(self.n_patches, generator=g) * 2.0 * math.pi)
        self.joint_proj = nn.Linear(self.action_dim, self.slot_dim, bias=False)

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
            local = tok % self.slot_dim
            acc[s][local] += torch.polar(
                torch.tensor(1.0), self.angle[tok].to(torch.float32))
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
