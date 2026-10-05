"""MODEL 3 - HENRI-Dec-450M Multi-Modal Readout Decoder.

Document anchors:
  doc p3  : 448M-parameter Hopfield-cross-pooled 24-layer GQA transformer;
            readout crystallizer for text, action SE(3), vision patches
  doc p17 : HENRI-SPEC-2026-DEC-SUB1B-VLA
  doc p18 : four-stage pipeline - isometric slice, Hopfield memory, egress, veto
  doc p20 : 4 orthogonal Clifford slots keep A_signal constant (not 1/sqrt(L))
  doc p21 : d_model 1024, N 24, heads 16, d_k 64, N_kv 4 (GQA-4),
            d_ffn 2816 SwiGLU, L 4096, V 32768; target 448,624,640 params
  doc p22 : Stage 1 P_inv filtering, Stage 2 Hopfield cross-pooling K=256
  doc p23 : Stage 3 HRM routing 0..31 intent, 32..127 relational, 128..255 actuation
  doc p24 : RoPE theta 500,000; text head temperature T* = 0.038 (beta* = 26.10)

Gate G-D2 guards the 400M-500M envelope. No off-the-shelf module is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import substrate as sub


@dataclass
class DecoderConfig:
    """Doc p21 sizing. Defaults reproduce the documented 448.6M envelope."""
    dim: int = sub.DEFAULT_DIM
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ffn: int = 2816
    ctx: int = 4096
    vocab: int = 32768
    n_macro: int = 256
    n_mem: int = 0                   # D85/D86: Hopfield memory slots; 0 = auto
    n_invariants: int = 256          # P_inv column count
    n_vq: int = 8                    # audio RVQ stages (doc p25)
    theta_rope: float = 500_000.0
    text_temperature: float = 0.038  # doc p24 T*


class PinnedInvariantProjector(nn.Module):
    """P_inv in R^{D x 256}; removes unstructured phase drift.  doc p22 stage 1.

    Psi_filtered = Psi - P_inv (P_inv^H Psi).  Columns are orthonormal, so the
    operator is idempotent and costs D*256, never D^2.
    """

    def __init__(self, dim: int = sub.DEFAULT_DIM, n_invariants: int = 256,
                 seed: int = 20261004):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        raw = torch.randn(dim, n_invariants, generator=g).to(torch.complex64)
        cols = sub.stiefel_retract(raw.transpose(0, 1), iters=4).transpose(0, 1)
        self.register_buffer("P_inv", cols)          # [D, 256] orthonormal cols

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        coeff = psi @ self.P_inv.conj()              # [B, 256]
        recon = coeff @ self.P_inv.conj().transpose(0, 1)
        return sub.unit_norm(psi - recon)


def resolve_n_mem(d_model: int, n_mem: int = 0) -> int:
    """Memory slot count for the Hopfield cross-pooling. 0 means auto.

    D85: draft 1 held ONE memory slot. softmax over one key is identically 1,
    so every macro-token was the same value vector times a near-constant
    scalar -- the defect the operator described.
    D86 (self-caught): pinning the default to 256 broke small test configs,
    because d_k = d_model // n_mem = 128 // 256 = 0 gives zero-width tensors.
    Auto resolves to min(256, d_model // 4), so d_k stays >= 4 at every scale.
    """
    n = int(n_mem) if n_mem and n_mem > 0 else min(256, max(1, d_model // 4))
    if d_model % n != 0:
        raise ValueError(f"n_mem={n} must divide d_model={d_model}")
    return n


class HopfieldCrossPooling(nn.Module):
    """Compress Psi into K=256 macro-tokens via learned queries.  doc p22 stage 2.

    S = softmax(Q W_Q (Psi W_K)^H / sqrt(d_k)) (Psi W_V)
    Query count 256 is FIXED. The pooling is training-free in the sense that the
    single CCCP step is closed form; the projections are learned.
    """

    def __init__(self, dim: int = sub.DEFAULT_DIM, d_model: int = 1024,
                 n_macro: int = 256, beta: float = 26.10, n_mem: int = 0):
        super().__init__()
        self.n_macro = int(n_macro)
        self.n_mem = resolve_n_mem(d_model, n_mem)
        self.beta = float(beta)
        # D85 (the G-U4 root cause): d_k is d_model // n_mem, NOT a fixed /16.
        # Draft 1 held a SINGLE memory slot, so softmax over the memory was
        # identically 1 for every query and all 256 macro-tokens were one vector
        # times a near-constant scalar. With n_mem = 256 and d_model = 1024 the
        # memory holds 256 distinct slots of 4 channels, matching the document's
        # N-pattern Hopfield update instead of a degenerate N=1 case.
        self.d_k = d_model // self.n_mem
        self.q_macro = nn.Parameter(torch.randn(n_macro, self.d_k) * 0.02)
        # ONE complex cross-projector wave -> d_model. The doc budgets
        # "65,536 x 1,024 x 2 (Complex Re/Im) ~ 134.2M". Sharing this single
        # projection for key and value is what keeps the 448M envelope.
        self.wave_proj = nn.Linear(2 * dim, d_model, bias=False)
        # D85: per-token slot-channel mixing, shape [M, d_k, d_model]. This
        # replaces token_mod, which added a FIXED learned offset that could not
        # carry input-dependent routing. Each macro-token now owns a d_k -> d
        # map applied to its d_k-channel slot mixture, so tokens differ by INPUT
        # CONTENT. Cost M * d_k * d_model = 1.05M, inside the 134.2M budget.
        self.mod = nn.Parameter(torch.randn(n_macro, self.d_k, d_model) * 0.02)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        b = psi.shape[0]
        pair = torch.cat([psi.real, psi.imag], dim=-1)     # [B, 2D] real
        kv = self.wave_proj(pair)                          # [B, d_model]
        kv = kv.view(b, self.n_mem, self.d_k)              # [B, N, d_k]   (D85)
        # D82 (self-caught): the Hopfield energy (doc p19) assumes UNIT-NORM
        # patterns -- M = max_i ||x_i||_2, and the separation threshold is
        # defined on x_i^T x_j. Unnormalized keys made the logits O(1e-3), so
        # softmax sat uniform and the routing channel carried NO per-input
        # information (measured: logits std 1.1e-3, routing std 4.8e-6). The doc
        # calls for the single-step "Lexical Snap", which needs beta to act on a
        # cosine in [-1, 1]. Normalize both sides; beta stays at the doc value
        # 26.10, so this aligns the scale to the spec rather than tuning a bound.
        q = self.q_macro                                   # [M, d_k]
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        kn = kv / kv.norm(dim=-1, keepdim=True).clamp_min(1e-6)   # [B, N, d_k]
        logits = self.beta * torch.einsum("mk,bnk->bmn", q, kn)   # [B, M, N]
        att = torch.softmax(logits, dim=-1)                # [B, M, N] over MEMORY
        sl = torch.einsum("bmn,bnk->bmk", att, kv)         # [B, M, d_k]
        return torch.einsum("bmk,mkj->bmj", sl, self.mod)  # [B, M, d_model]


def _rope(q: torch.Tensor, theta: float, offset: int = 0) -> torch.Tensor:
    """Rotary position embedding.  doc p24: theta base 500,000.

    q: [B, H, T, d_k]. Rotates consecutive component pairs by position phase.
    """
    b, h, t, dk = q.shape
    half = dk // 2
    pos = torch.arange(offset, offset + t, device=q.device, dtype=torch.float32)
    inv = theta ** (-torch.arange(half, device=q.device, dtype=torch.float32) / half)
    ang = pos[:, None] * inv[None, :]                      # [T, half]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    q1, q2 = q[..., :half], q[..., half:]
    return torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)


class GQABlock(nn.Module):
    """Grouped-query attention + SwiGLU.  doc p24: 16 query heads, 4 KV heads."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.d_k = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.n1 = nn.LayerNorm(cfg.d_model)
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * self.d_k, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_k, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_k, bias=False)
        self.o = nn.Linear(cfg.n_heads * self.d_k, cfg.d_model, bias=False)
        self.n2 = nn.LayerNorm(cfg.d_model)
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.w_down = nn.Linear(cfg.d_ffn, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        b, t, d = x.shape
        h = self.n1(x)
        q = self.q(h).view(b, t, self.cfg.n_heads, self.d_k).transpose(1, 2)
        k = self.k(h).view(b, t, self.cfg.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.v(h).view(b, t, self.cfg.n_kv_heads, self.d_k).transpose(1, 2)
        q = _rope(q, self.cfg.theta_rope)
        k = _rope(k, self.cfg.theta_rope)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
        a = a.transpose(1, 2).reshape(b, t, d)
        x = x + self.o(a)
        h = self.n2(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class HRMRouter(nn.Module):
    """Tier routing over macro-tokens.  doc p23: three structural bands.

    The split is structural, not learned: tokens 0..31 intent, 32..127
    relational, 128..255 actuation. No parameters, no silent reordering.
    """

    def __init__(self, n_macro: int = 256):
        super().__init__()
        self.bands = {
            "intent": (0, 32),
            "relational": (32, 128),
            "actuation": (128, n_macro),
        }

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: tokens[:, lo:hi] for name, (lo, hi) in self.bands.items()}


class HenriDec450M(nn.Module):
    """HENRI-Dec-450M. Wave to text, action, and vision."""

    def __init__(self, cfg: DecoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or DecoderConfig()
        c = self.cfg
        self.projector = PinnedInvariantProjector(c.dim, c.n_invariants)
        self.pooling = HopfieldCrossPooling(c.dim, c.d_model, c.n_macro, n_mem=c.n_mem)
        # cross-projector from macro-tokens into the backbone width
        self.cross_proj = nn.Linear(c.n_macro, c.n_macro, bias=False)
        self.router = HRMRouter(c.n_macro)
        self.layers = nn.ModuleList([GQABlock(c) for _ in range(c.n_layers)])
        self.norm = nn.LayerNorm(c.d_model)
        # typed egress heads
        self.head_text = nn.Linear(c.d_model, c.vocab, bias=False)
        self.head_action = nn.Linear(c.d_model, 7, bias=True)      # SE(3)+gripper
        self.head_vision = nn.Linear(c.d_model, 256, bias=True)    # 16x16 patches
        self.head_audio = nn.Linear(c.d_model, c.n_vq, bias=True)  # RVQ stages

    # ------------------------------------------------------------------ stages
    def encode_wave(self, psi: torch.Tensor):
        """Stages 1-3: filter, Hopfield cross-pool, route. Returns routed bands."""
        filt = self.projector(psi)
        tokens = self.pooling(filt)                     # [B, M, d_model]
        tokens = torch.einsum("bmd,nm->bnd", tokens, self.cross_proj.weight)
        return self.router(tokens), tokens

    def forward(self, psi: torch.Tensor, return_all: bool = False):
        bands, tokens = self.encode_wave(psi)
        h = tokens
        for blk in self.layers:
            h = blk(h)
        h = self.norm(h)
        logits = self.head_text(h)                      # [B, M, vocab]
        out = {
            "text_logits": logits[:, -1, :],            # last macro-token
            "action": self.head_action(bands["actuation"].mean(dim=1)),
            "vision": self.head_vision(bands["relational"].mean(dim=1)),
            "audio": self.head_audio(h[:, -1, :]),
            "tokens": h,
        }
        if return_all:
            out["bands"] = {k: v.mean(dim=1) for k, v in bands.items()}
        return out

    # ------------------------------------------------------- training-free snap
    @torch.no_grad()
    def snap_text(self, psi: torch.Tensor, temperature: float | None = None):
        """Single-step lexical snap.  doc p19: the attractor is reached in t=1.

        Returns (token_ids, probabilities). Temperature defaults to T* = 0.038.
        """
        t = self.cfg.text_temperature if temperature is None else float(temperature)
        logits = self.forward(psi)["text_logits"] / max(t, 1e-6)
        p = torch.softmax(logits, dim=-1)
        return p.argmax(dim=-1), p


def decoder_param_formula(cfg: DecoderConfig) -> dict:
    """Exact parameter count from the config, WITHOUT instantiating.

    Why: the full config holds ~440M parameters. Materializing it to count them
    costs ~1.8 GB of CPU RAM. The formula is validated against a real small
    instantiation in verify_tri_model.py, then applied to the full config.
    """
    d, D, V, M = cfg.d_model, cfg.dim, cfg.vocab, cfg.n_macro
    dk = d // cfg.n_heads
    kv = cfg.n_kv_heads * dk
    per_block = (
        d * (cfg.n_heads * dk)      # q
        + d * kv                    # k
        + d * kv                    # v
        + (cfg.n_heads * dk) * d    # o
        # D78 (self-caught): LayerNorm(d) holds weight AND bias = 2d, and there
        # are TWO norms per block (n1, n2). The first formula charged 2*d, so it
        # undercounted 2d per block: 2 layers x 2d = 512, exactly the delta
        # measured against a real instantiation.
        + 4 * d                     # n1, n2 LayerNorm (2d each)
        + 3 * d * cfg.d_ffn         # w_gate, w_up, w_down
    )
    dk = d // resolve_n_mem(d, cfg.n_mem)
    parts = {
        # D87 (self-caught): mod is a [M, d_k, d_model] tensor, so its cost is
        # M*d_k*d_model, NOT the M*d_model I first wrote. The mismatch showed up
        # as formula 3,436,303 vs instantiated 3,484,431, delta 48,128, which is
        # exactly (M*d_k*d - d_k*d - M*d) at the small config: 65,536 - 1,024 -
        # 16,384 = 48,128. The d_k*d and M*d terms are the v1 leftovers, now gone.
        "pooling": 2 * D * d + M * dk + M * dk * d,       # D85: N memories + mod
        "cross_proj": M * M,
        "blocks": cfg.n_layers * per_block,
        "final_norm": 2 * d,
        "heads": d * V + (d * 7 + 7) + (d * 256 + 256) + (d * cfg.n_vq + cfg.n_vq),
    }
    return {"total": sum(parts.values()), "parts": parts,
            "doc_target": 448_624_640}


@torch.no_grad()
def count_params(module: nn.Module) -> dict:
    """Count trainable and total parameters, grouped by top-level child."""
    groups, total, trainable = {}, 0, 0
    for name, child in module.named_children():
        n = sum(p.numel() for p in child.parameters())
        groups[name] = n
        total += n
        trainable += sum(p.numel() for p in child.parameters() if p.requires_grad)
    direct = sum(p.numel() for p in module.parameters(recurse=False))
    if direct:
        groups["_direct"] = direct
        total += direct
        trainable += sum(p.numel() for p in module.parameters(recurse=False)
                         if p.requires_grad)
    return {"total": total, "trainable": trainable, "groups": groups}
