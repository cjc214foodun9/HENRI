"""MODEL 2 - HENRI-Mem-65M Holographic Memory Manager.

Document anchors:
  doc p1  : third dedicated micro-model; in-database neuromorphic engine
  doc p5  : layer breakdown - Clifford spectral ingress 8,192 blocks -> 256 features;
            6-layer Gram manifold transformer (d_mem=512, 8 heads, Complex GeLU);
            viscoelastic decay predictor; prefetch generation head;
            closed-form Stiefel projector
  doc p3-4: dual Gram matrix G = X^H X; off-diagonal interference metric;
            Gram-Schmidt retraction onto the Stiefel manifold
  doc p6  : 4-Slot Clifford subspace rule; cross-slot bleed projection mask
  doc p10 : gates G-M1..G-M6

Cost rule: everything is dual. The Gram matrix is K x K with K <= 64. No D x D.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import substrate as sub
from .complex_ops import cgelu
from .viscoelastic import ViscoelasticMemoryKernel


class SpectralIngress(nn.Module):
    """8,192 Cl(3,0) blocks -> 256 compact spectral features.  doc p5."""

    def __init__(self, dim: int = sub.DEFAULT_DIM, n_features: int = 256,
                 n_blocks: int | None = None, hidden: int = 512):
        super().__init__()
        self.dim = int(dim)
        # D73 (self-caught): n_blocks was pinned at 8192, so a small dim (4096)
        # gave 4096 // 8192 = 0 and reshape raised. Derive it: one block per
        # Cl(3,0) blade group. dim=65536 -> 8192 blocks, exactly as doc p5.
        self.n_blocks = int(n_blocks) if n_blocks else max(1, self.dim // sub.BLADES)
        self.n_features = int(n_features)
        # block energy [n_blocks] and block phase coherence [n_blocks] -> features
        self.energy_proj = nn.Linear(self.n_blocks, hidden)
        self.phase_proj = nn.Linear(self.n_blocks, hidden)
        self.fuse = nn.Linear(hidden, self.n_features)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        b = psi.shape[0]
        blk = psi.reshape(b, self.n_blocks, self.dim // self.n_blocks)
        energy = (blk.abs() ** 2).sum(dim=-1)                       # [B, n_blocks]
        phase = blk.mean(dim=-1).angle()                            # [B, n_blocks]
        h = F.gelu(self.energy_proj(energy)) + F.gelu(self.phase_proj(phase))
        return self.fuse(h)                                          # [B, 256]


class GramManifoldTransformer(nn.Module):
    """6 layers of dual-domain self-attention over a [64 x 64] Gram space.  doc p5."""

    def __init__(self, n_layers: int = 6, d_mem: int = 512, n_heads: int = 8,
                 gram_size: int = 64, d_ffn: int = 2048):
        super().__init__()
        self.gram_size = int(gram_size)
        self.n_tokens = self.gram_size * self.gram_size
        self.token_proj = nn.Linear(1, d_mem)          # one scalar per Gram cell
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_mem))
        self.blocks = nn.ModuleList([
            _GramBlock(d_mem, n_heads, d_ffn) for _ in range(int(n_layers))])
        self.norm = nn.LayerNorm(d_mem)

    def forward(self, gram: torch.Tensor) -> torch.Tensor:
        """gram: [B, K, K] real. Returns [B, n_tokens, d_mem]."""
        b, k, _ = gram.shape
        flat = gram.reshape(b, k * k, 1)
        if flat.shape[1] != self.n_tokens:
            flat = F.interpolate(
                flat.transpose(1, 2), size=self.n_tokens,
                mode="nearest").transpose(1, 2)
        x = self.token_proj(flat) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class _GramBlock(nn.Module):
    def __init__(self, d: int, heads: int, d_ffn: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.w_gate = nn.Linear(d, d_ffn)
        self.w_up = nn.Linear(d, d_ffn)
        self.w_down = nn.Linear(d_ffn, d)

    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        h = self.n2(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class DecayPredictor(nn.Module):
    """Predict P relaxation coefficients {gamma_j}.  doc p5."""

    def __init__(self, d_mem: int = 512, n_slices: int = 4, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_mem, hidden), nn.GELU(),
            nn.Linear(hidden, n_slices))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        pooled = h.mean(dim=1)
        return torch.sigmoid(self.net(pooled))          # [B, P] in (0,1)


class PrefetchHead(nn.Module):
    """Estimate the forward wave Psi(t + dt).  doc p5, p7."""

    def __init__(self, d_mem: int = 512, n_features: int = 256,
                 hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_mem, hidden), nn.GELU(),
            nn.Linear(hidden, n_features))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h.mean(dim=1))                  # [B, n_features]


class HenriMem65M(nn.Module):
    """HENRI-Mem-65M. Zone C consolidation, pruning, and prefetch.

    forward(psi, bank):
        psi   [B, D] complex unit-norm incoming wave
        bank  [N, K, K] or None; when None the bank Gram is built from psi
    Returns a dict of predictions and diagnostics. Nothing here writes to
    TimescaleDB: this module is the cognitive governor, the store is external.
    """

    def __init__(self, dim: int = sub.DEFAULT_DIM, n_features: int = 256,
                 n_layers: int = 6, d_mem: int = 512, n_heads: int = 8,
                 gram_size: int = 64, n_slices: int = 4, d_ffn: int = 2048,
                 stiefel_iters: int = 5):
        super().__init__()
        self.dim = int(dim)
        self.gram_size = int(gram_size)
        self.stiefel_iters = int(stiefel_iters)
        self.ingress = SpectralIngress(dim, n_features)
        self.transformer = GramManifoldTransformer(
            n_layers, d_mem, n_heads, gram_size, d_ffn)
        self.decay = DecayPredictor(d_mem, n_slices)
        self.prefetch = PrefetchHead(d_mem, n_features)
        self.kernel = ViscoelasticMemoryKernel(n_slices, dim)

    # ------------------------------------------------------------------ Gram
    @staticmethod
    def gram_matrix(rows: torch.Tensor) -> torch.Tensor:
        """Dual Gram G = X^H X for unit-norm rows.  doc p3. K x K, never D x D."""
        return rows.conj() @ rows.transpose(-1, -2)

    @staticmethod
    def offdiag_max(gram: torch.Tensor) -> torch.Tensor:
        """Max |off-diagonal| interference.  doc p3. Gate G-M1 / G-U2."""
        k = gram.shape[-1]
        mask = ~torch.eye(k, dtype=torch.bool, device=gram.device)
        mag = gram.abs()
        return (mag.masked_fill(~mask, 0.0)).amax(dim=(-1, -2))

    def compact(self, rows: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt retraction onto the Stiefel manifold.  doc p4.

        Returns orthonormal rows whose off-diagonal Gram magnitude drops.
        """
        return sub.stiefel_retract(rows, iters=self.stiefel_iters)

    # ---------------------------------------------------------------- forward
    def forward(self, psi: torch.Tensor, bank: torch.Tensor | None = None,
                dt: float = 1e-3):
        b = psi.shape[0]
        feats = self.ingress(psi)                         # [B, 256]
        if bank is None:
            # self-Gram of the wave split into gram_size rows
            rows = psi.reshape(b, self.gram_size, self.dim // self.gram_size)
            rows = sub.unit_norm(rows)
            g = self.gram_matrix(rows).real               # [B, K, K]
            rows_in = rows
        else:
            g = bank.real if bank.is_complex() else bank
            rows_in = None
        h = self.transformer(g)                           # [B, n_tokens, d_mem]
        gamma = self.decay(h)                             # [B, P]
        prefetch = self.prefetch(h)                       # [B, 256]
        return {
            "spectral": feats,
            "gram": g,
            "offdiag_max": self.offdiag_max(g.real if g.is_complex() else g),
            "gamma": gamma,
            "prefetch": prefetch,
            "rows_in": rows_in,
            "hidden": h,
        }

    @torch.no_grad()
    def consolidate(self, bank: torch.Tensor, eps_delta: float = 0.12):
        """One Zone C maintenance pass. Returns the retracted bank and metrics.

        doc p6-7: compute Gram over the chunk, orthogonalize overlapping engrams,
        drop rows whose viscoelastic weight has decayed.
        """
        rows = sub.unit_norm(bank)
        before = self.offdiag_max(self.gram_matrix(rows)).mean().item()
        out = self.compact(rows)
        after = self.offdiag_max(self.gram_matrix(out)).mean().item()
        return {
            "bank": out,
            "offdiag_before": float(before),
            "offdiag_after": float(after),
            "within_tolerance": bool(after <= eps_delta),
        }

    def prune(self, weights: torch.Tensor, threshold: float = 1e-4):
        """doc p7: dead engrams where w_i(t) < 1e-4. Returns a keep mask."""
        return weights >= threshold
