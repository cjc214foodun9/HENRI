"""Diagnose G-U4: is the estimator wrong, or is the pooling genuinely lossy?

A gate needs a POSITIVE control, not only a negative one. If the estimator
cannot recover a perfect signal, the estimator is at fault. If it can, and the
real features still fail, the finding is about the mechanism.

Also measures Hopfield attention saturation: if the attention is uniform, the
routing signature is constant and carries no per-wave information by
construction.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from henri_core import substrate as sub
from henri_core.system import TriModelSystem
from henri_core.tokenizer import ByteBPE

CORPUS = ["entity acts on object", "the axiom holds", "phase clears the port",
          "memory decays over time", "the swarm finds lower energy"]


def r2_raw(F, Y, half, ridge=1.0, n_comp=32):
    tr, te = slice(0, half), slice(half, len(Y))
    mu = F[tr].mean(0, keepdim=True)
    sd = F[tr].std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (F[tr] - mu) / sd, (F[te] - mu) / sd
    ybar = Y[tr].mean(0, keepdim=True)          # D80 fix: intercept
    _, _, vh = torch.linalg.svd(Xtr, full_matrices=False)
    k = max(1, min(n_comp, vh.shape[0]))
    P = vh[:k]
    Ztr, Zte = Xtr @ P.T, Xte @ P.T
    w = torch.linalg.solve(Ztr.T @ Ztr + ridge * torch.eye(k),
                           Ztr.T @ (Y[tr] - ybar))
    pred = Zte @ w + ybar
    ss_res = ((Y[te] - pred) ** 2).sum()
    ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot), float(ss_tot)


def main():
    N = 1024
    tok = ByteBPE().train(CORPUS, vocab_size=256)
    system = TriModelSystem(vocab=tok.vocab_size, small=True)
    system.eval()
    half = N // 2

    texts = [f"retrieval transfer result {i} on the ladder" for i in range(N)]
    with torch.no_grad():
        psi = torch.stack([system.wave_of(t, tok) for t in texts])
        prof = (psi.abs() ** 2).reshape(N, sub.N_SLOTS, -1).sum(-1)
        prof = prof / prof.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        filt = system.decoder.projector(psi)
        tokens = system.decoder.pooling(filt)
        att_logits = system.decoder.pooling.beta * (
            system.decoder.pooling.w_q(system.decoder.pooling.q_macro)
            @ (system.decoder.pooling.wave_proj(
                torch.cat([filt.real, filt.imag], -1)).mean(0)).unsqueeze(-1)
        ).squeeze(-1) / (system.decoder.pooling.d_k ** 0.5)
        content = tokens.mean(dim=1).float()
        routing = tokens.mean(dim=-1).float()

    print("=== profile statistics ===")
    print(f"  prof per-slot mean   {prof.mean(0).tolist()}")
    print(f"  prof per-slot std    {prof.std(0).tolist()}")

    print("\n=== attention saturation ===")
    # recompute attention weights from the real forward path
    with torch.no_grad():
        pair = torch.cat([filt.real, filt.imag], -1)
        kv = system.decoder.pooling.wave_proj(pair)
        q = system.decoder.pooling.w_q(system.decoder.pooling.q_macro)
        lg = system.decoder.pooling.beta * (q @ kv.unsqueeze(-1)).squeeze(-1) \
            / (system.decoder.pooling.d_k ** 0.5)
        att = torch.softmax(lg, dim=-1)
    print(f"  logits std across texts {lg.std(dim=0).mean():.6e}")
    print(f"  logits abs mean         {lg.abs().mean():.6e}")
    print(f"  attention per-token std across texts {att.std(dim=0).mean():.6e}")
    print(f"  routing(input) std      {routing.std(dim=0).mean():.6e}")

    print("\n=== estimator controls ===")
    F_real = torch.cat([content, routing, content ** 2, routing ** 2], dim=-1)
    r_real, ss = r2_raw(F_real, prof, half)
    print(f"  REAL features            R2={r_real:+.6f}  (ss_tot={ss:.6f})")

    g = torch.Generator().manual_seed(7)
    r_ctl, _ = r2_raw(F_real[torch.randperm(N, generator=g)], prof, half)
    print(f"  NEGATIVE control (shuffled) R2={r_ctl:+.6f}")

    # POSITIVE control: estimator must recover a perfectly-encoded target
    r_pos, _ = r2_raw(prof.clone(), prof, half)
    print(f"  POSITIVE control (target as feature) R2={r_pos:+.6f}")
    g2 = torch.Generator().manual_seed(11)
    noisy = prof + 0.05 * torch.randn(prof.shape, generator=g2)
    r_posn, _ = r2_raw(noisy, prof, half)
    print(f"  POSITIVE control (+5% noise)         R2={r_posn:+.6f}")

    # does the raw wave itself (its own slot energies) predict the profile?
    with torch.no_grad():
        blk = (psi.abs() ** 2).reshape(N, 64, -1).sum(-1)
    r_wave, _ = r2_raw(blk, prof, half)
    print(f"  wave block energies      R2={r_wave:+.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
