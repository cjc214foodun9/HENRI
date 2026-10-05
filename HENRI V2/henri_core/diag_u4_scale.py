"""Resolve the G-U4 scale question. DIAGNOSTIC ONLY -- no gate, no bound change.

The contradiction to settle
    The gate (build(small=True): dim=4096, d_model=128, n_macro=16) reports
    held-out slot-profile R^2 = 0.9060.
    A full-config diagnostic (dim=65536, d_model=1024, n_macro=256) reports
    R^2 = -0.09 on the same corpus.
    Two different architectures, so this is not necessarily a bug -- but the
    gate's number is then a SMALL-CONFIG PROXY, not the 440M decoder. That must
    be stated, or the gate overstates what it measured.

This tool measures BOTH scales with one code path and prints:
    * the oracle ceiling (256 |Psi|^2 bins; these aggregate to the target)
    * the pooling-path ceiling (the gate's feature recipe)
    * the target variation (coefficient of variation)
    * token routing diversity (mean pairwise |cos|, effective rank)

A near-zero coefficient of variation means the target is nearly constant, so
R^2 is noise-dominated and NO feature set can predict it.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import substrate as sub                 # noqa: E402
from henri_core.tokenizer import ByteBPE                # noqa: E402
from henri_core.system import TriModelSystem            # noqa: E402

CORPUS = [
    "the rotor advances the entity across the plane",
    "a stack pushes and pops the nested frame",
    "recursion unfolds the copied subtree",
    "the gripper closes on the object edge",
]


def r2(F, Y, n_comp=32, ridge=1.0):
    half = len(Y) // 2
    tr, te = slice(0, half), slice(half, len(Y))
    mu = F[tr].mean(0, keepdim=True)
    sd = F[tr].std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (F[tr] - mu) / sd, (F[te] - mu) / sd
    ybar = Y[tr].mean(0, keepdim=True)
    Ytrc, Yte = Y[tr] - ybar, Y[te]
    k = max(1, min(int(n_comp), Xtr.shape[0] - 1, Xtr.shape[1]))
    _, _, vh = torch.linalg.svd(Xtr, full_matrices=False)
    P = vh[:k]
    Ztr, Zte = Xtr @ P.T, Xte @ P.T
    gram = Ztr.T @ Ztr + ridge * torch.eye(k)
    w = torch.linalg.solve(gram, Ztr.T @ Ytrc)
    pred = Zte @ w + ybar
    ss_res = ((Yte - pred) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum().clamp_min(1e-12)
    return float(1.0 - ss_res / ss_tot)


def token_diversity(tokens):
    x = tokens.float()
    m = x.shape[1]
    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    sim = torch.einsum("bmd,bnd->bmn", xn, xn)
    off = sim - torch.eye(m).unsqueeze(0)
    # sample one batch to keep memory bounded
    s = torch.linalg.svdvals(x[0].double())
    s2 = s ** 2
    return (float(off.abs().sum() / (x.shape[0] * m * (m - 1))),
            float((s2.sum() ** 2) / s2.pow(2).sum().clamp_min(1e-30)))


def run(scale: str, small: bool, n: int) -> dict:
    texts = [f"retrieval transfer result {i} on the ladder" for i in range(n)]
    tok = ByteBPE().train(CORPUS, vocab_size=512)
    system = TriModelSystem(vocab=tok.vocab_size, small=small)
    system.eval()
    c = system.decoder.cfg
    with torch.no_grad():
        psi = torch.stack([system.wave_of(t, tok) for t in texts])
        prof = (psi.abs() ** 2).reshape(n, sub.N_SLOTS, -1).sum(-1)
        prof = prof / prof.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        _, tokens = system.decoder.encode_wave(psi)
        content = tokens.mean(dim=1).float()
        routing = tokens.mean(dim=-1).float()
        base = torch.cat([content, routing], dim=-1)
        feats = torch.cat([base, base ** 2], dim=-1)
        # oracle: |Psi|^2 split into 256 bins. For small dim these bins are tiny,
        # so this is a legitimate "features that generate the target" control.
        oracle = (psi.abs() ** 2).reshape(n, 256, -1).sum(-1)
        cos, rank = token_diversity(tokens)
    cv = float((prof.std(0) / prof.mean(0).clamp_min(1e-9)).mean())
    out = {
        "scale": scale, "dim": c.dim, "d_model": c.d_model, "n_macro": c.n_macro,
        "n_texts": n, "n_features": feats.shape[1],
        "slot_mean": [round(v, 5) for v in prof.mean(0).tolist()],
        "slot_std": [round(v, 6) for v in prof.std(0).tolist()],
        "coeff_var": cv,
        "oracle_r2": r2(oracle, prof),
        "pooling_r2": r2(feats, prof),
        "content_only_r2": r2(content, prof),
        "token_mean_abs_cos": cos,
        "token_eff_rank": rank,
    }
    print(f"\n=== scale={scale}  dim={c.dim} d_model={c.d_model} "
          f"n_macro={c.n_macro} n_mem={system.decoder.pooling.n_mem} ===")
    print(f"  n_texts {n}   features {out['n_features']}")
    print(f"  slot mean {out['slot_mean']}")
    print(f"  slot std  {out['slot_std']}   coeff_var {cv:.5f}")
    print(f"  oracle R2 (must be ~1.0)     {out['oracle_r2']:.6f}")
    print(f"  pooling R2 @32 (the gate)    {out['pooling_r2']:.6f}")
    print(f"  content-only R2 @32          {out['content_only_r2']:.6f}")
    print(f"  tokens |cos| {cos:.5f}  eff_rank {rank:.2f}")
    return out


def main() -> int:
    a = run("small (gate proxy)", True, 512)
    b = run("full (440M config)", False, 192)
    print("\n=== CONCLUSION ===")
    if a["pooling_r2"] > 0.8 and b["pooling_r2"] < 0.2:
        print("  The gate measures a SMALL-CONFIG PROXY. Retention does not")
        print("  transfer to the full config. The gate must say so.")
    elif abs(a["pooling_r2"] - b["pooling_r2"]) < 0.2:
        print("  Both scales agree. The earlier full-scale number was a script bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
