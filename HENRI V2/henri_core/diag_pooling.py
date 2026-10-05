"""Diagnose the macro-token routing: does it carry per-input information?

D85 background
    Draft 1's HopfieldCrossPooling held ONE memory key. softmax over a single
    key is identically 1 for every query, so every macro-token was the same
    value vector times a near-constant scalar. Measured then: logits std
    1.1e-3, routing std 4.8e-6. That is the symptom the operator described as
    "tokens differ by ... a near-constant scalar".

    The fix gives the pool N = 256 distinct memories (n_mem), each key/value a
    d_k = d_model // n_mem = 4 channel slice of the projected wave, plus a
    per-token d_k -> d_model mixer so tokens differ by INPUT CONTENT.

Controls (a gate that cannot fail is worthless -- D72):
    * the SAME class built with n_mem = 1 MUST reproduce the draft-1 defect
      and fail the routing-diversity check. That arm IS the old mechanism.
    * token features are computed from the tokens alone, never from the target
      wave, so G-U4 cannot pass by leaking.

Usage:  python henri_core/diag_pooling.py
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.model3_decoder import (                 # noqa: E402
    DecoderConfig, HenriDec450M, HopfieldCrossPooling, decoder_param_formula,
)


def token_diversity(tokens: torch.Tensor) -> dict:
    """tokens [B, M, d]. Mean pairwise |cos| and the participation ratio.

    A rank-1 token matrix (every token a scalar multiple of one vector) has
    mean pairwise |cos| = 1 and effective rank = 1. That is the v1 defect.
    """
    b, m, _ = tokens.shape
    x = tokens.float()
    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    sim = torch.einsum("bmd,bnd->bmn", xn, xn)            # [B, M, M]
    off = sim - torch.eye(m).unsqueeze(0)                 # drop the diagonal
    mean_abs_cos = float(off.abs().sum() / (b * m * (m - 1)))
    s = torch.linalg.svdvals(x[0].double())
    s2 = s ** 2
    pr = float((s2.sum() ** 2) / s2.pow(2).sum().clamp_min(1e-30))
    return {"mean_abs_cos": mean_abs_cos, "eff_rank": pr, "n_macro": m}


class LegacyRank1Pool(nn.Module):
    """The draft-1 pooling, reproduced EXACTLY, as the G-U4a failing control.

    D88 (self-caught): the first G-U4a control was "the same new class with
    n_mem=1". That arm still carried the per-token mixer, so it produced random
    full-rank tokens and PASSED the gate. Both arms passed, so the gate was
    vacuous. The control must be the literal v1 math.

    v1 computed tokens[b, m] = att[b, m] * v[b] * (1 + token_mod[m]) where att is
    a softmax over M queries against ONE key. Every token is a scalar multiple
    of the single vector v[b], so the M x d_model token matrix has rank 1.
    """

    def __init__(self, dim: int, d_model: int, n_macro: int, beta: float = 26.10):
        super().__init__()
        self.d_k = d_model
        self.n_mem = 1                # v1 held exactly ONE memory slot
        self.beta = float(beta)
        self.n_macro = int(n_macro)
        self.q_macro = nn.Parameter(torch.randn(n_macro, d_model) * 0.02)
        self.wave_proj = nn.Linear(2 * dim, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.token_mod = nn.Parameter(torch.randn(n_macro, d_model) * 0.1)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        b = psi.shape[0]
        pair = torch.cat([psi.real, psi.imag], dim=-1)
        v = self.wave_proj(pair)                          # [B, d]
        q = self.w_q(self.q_macro)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        kn = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos = q.unsqueeze(0) @ kn.unsqueeze(-1)            # [B, M, 1]
        logits = (self.beta * cos).squeeze(-1)             # [B, M]
        att = torch.softmax(logits, dim=-1)                # [B, M] over macros
        return (att.unsqueeze(-1) * v.unsqueeze(1)
                * (1.0 + self.token_mod).unsqueeze(0))     # [B, M, d] rank-1


def measure(pool, filt: torch.Tensor) -> dict:
    with torch.no_grad():
        tokens = pool(filt)
        pair = torch.cat([filt.real, filt.imag], dim=-1)
        kv = pool.wave_proj(pair).view(filt.shape[0], pool.n_mem, pool.d_k)
        q = pool.q_macro
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        kn = kv / kv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = pool.beta * torch.einsum("mk,bnk->bmn", q, kn)   # [B, M, N]
        att = torch.softmax(logits, dim=-1)
        ent = float(-(att * (att + 1e-12).log()).sum(-1).mean())
        routing = tokens.mean(dim=-1)                     # [B, M]
        div = token_diversity(tokens)
    return {
        "logits_std": float(logits.std()),
        "att_entropy": ent,
        "att_max_frac": math.exp(-ent) / pool.n_mem,      # ~1/N when uniform
        "routing_std_across_inputs": float(routing.std(dim=0).mean()),
        **div,
    }


def main() -> int:
    ok = True

    print("=== 1. parameter formula vs a real instantiation (D78/D84 lesson) ===")
    small_cfg = DecoderConfig(dim=4096, d_model=256, n_layers=2, n_heads=4,
                              n_kv_heads=1, d_ffn=512, n_macro=64,
                              n_invariants=64, vocab=512)
    real = sum(p.numel() for p in HenriDec450M(small_cfg).parameters())
    formula = decoder_param_formula(small_cfg)["total"]
    print(f"  instantiated {real:,}   formula {formula:,}   delta {real - formula}")
    ok &= (real == formula)
    print(f"  formula exact: {real == formula}")

    print("\n=== 2. full-config parameter envelope (G-D2) ===")
    full = decoder_param_formula(DecoderConfig())
    print(f"  total {full['total']:,}")
    print(f"  parts {full['parts']}")
    env = 400_000_000 <= full["total"] <= 500_000_000
    ok &= env
    print(f"  in 400M-500M envelope: {env}")

    print("\n=== 3. routing diversity, N=1 (draft-1 defect) vs N=256 (fix) ===")
    cfg = DecoderConfig(dim=4096, d_model=1024, n_layers=1, n_heads=8,
                        n_kv_heads=2, d_ffn=2048, n_macro=256,
                        n_invariants=256, vocab=256)
    dec = HenriDec450M(cfg)
    g = torch.Generator().manual_seed(20261004)
    texts = 32
    psi = torch.randn(texts, cfg.dim, generator=g, dtype=torch.complex64)
    psi = psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    with torch.no_grad():
        filt = dec.projector(psi)

    arms = {
        "N=1 rank-1 (draft-1 defect)": LegacyRank1Pool(cfg.dim, 1024, 256),
        "N=256 (fix)":                dec.pooling,
    }
    res = {}
    for name, pool in arms.items():
        r = measure(pool, filt)
        res[name] = r
        print(f"  {name}   d_k={pool.d_k}  n_mem={pool.n_mem}")
        print(f"     mean pairwise |cos|      {r['mean_abs_cos']:.6f}")
        print(f"     effective rank of M x d  {r['eff_rank']:.2f}  (of {r['n_macro']})")
        print(f"     logits std               {r['logits_std']:.6e}")
        print(f"     attention entropy        {r['att_entropy']:.6f} nats")
        print(f"     routing std across input {r['routing_std_across_inputs']:.6e}")

    fix = res["N=256 (fix)"]
    leg = res["N=1 rank-1 (draft-1 defect)"]
    fix_pass = fix["mean_abs_cos"] <= 0.65 and fix["eff_rank"] >= 64
    leg_fail = not (leg["mean_abs_cos"] <= 0.65 and leg["eff_rank"] >= 64)
    print("\n=== G-U4a routing diversity ===")
    print(f"  fix    pass={fix_pass}   |cos|={fix['mean_abs_cos']:.4f} "
          f"rank={fix['eff_rank']:.2f}")
    print(f"  legacy pass={not leg_fail}  |cos|={leg['mean_abs_cos']:.4f} "
          f"rank={leg['eff_rank']:.2f}")
    gate_ok = fix_pass and leg_fail
    print(f"  verdict: {'GATE OK (fix passes, control fails)' if gate_ok else 'GATE BROKEN'}")
    ok &= gate_ok

    print(f"\nOVERALL {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
