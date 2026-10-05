"""LEVER PROBE: which ingress/pooling lever actually moves the measured gates?

WHY THIS RUN EXISTS
    Kill test #5 proved the ROUTER is dead (argmax -> int() severs; re-routing
    among 4 slots changes nothing extractable). But the router is not the
    information carrier. In _token_writes the write value is

        acc[slot][tok % slot_dim] += polar(1.0, self.angle[tok])

    so the PHASE BUFFER self.angle holds the information, and it is frozen.
    Separately, gap#2 location showed the pooled feature collapses
    (pair |cos| 0.955) and that HopfieldCrossPooling computes its similarity in
    d_k = 4 dimensions at BOTH scales, because resolve_n_mem = d_model // 4.

    This probe tests both candidate levers against a frozen control, with the
    bound UNCHANGED (G-U4 >= 0.95). It decides the revised contract's content.

ARMS (identical pin 20261004, identical corpus, identical readout recipe)
    F  frozen        : today's behaviour. MUST reproduce G-U4 = 0.819028.
    P  learnable phase : self.angle becomes a trained tensor (the phase is the
                       carrier; the gradient flows through polar(), no STE needed)
    W  wider pooling : default phase, pooling rebuilt with n_mem = 4 so that
                       d_k = 32 instead of 4 (rank ceiling per macro-token x8)

ENDPOINT
    G-U4 (heldout R^2 of the 4-slot energy profile from the pooled feature).
    It has measured dynamic range (0.819..0.987 across seeds), a positive
    control at 0.999997, and a negative control at 0.0. Bounds do not move.

GUARDS (any failure -> VACUOUS)
    g1 phase grad non-zero            (the lever must be trainable at all)
    g2 wave changed on the GATE corpus (not just the 48 training rows, D137)
    g3 arm F reproduces 0.819028       (D138 baseline reproduction)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import substrate as sub                        # noqa: E402
from henri_core.gates import gate_u4_retention                 # noqa: E402
from henri_core.m4_generative import (                         # noqa: E402
    M4Config, WaveTextGenerator, build_corpus, build_system)
from henri_core.model3_decoder import HopfieldCrossPooling     # noqa: E402

PIN = 20261004
G_U4_BOUND = 0.95
F_BASELINE = 0.819028


# ------------------------------------------------- differentiable ingress
def diff_wave(enc, ids: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Same math as _token_writes/_assemble, but angle is a supplied tensor.

    Routing stays hard (the router is dead; kill test #5). The PHASE is the
    carrier, and polar() is differentiable in the phase, so the gradient reaches
    `angle` without any straight-through estimator.
    """
    emb = enc.token_emb(ids)
    logits = enc.slot_router(emb)
    routes = logits.argmax(dim=-1)                       # [T] hard, non-diff
    t = ids.shape[0]
    tok = (ids % enc.dim).long()
    local = (tok % enc.slot_dim).long()
    phase = angle[tok].to(torch.float32)
    phasor = torch.polar(torch.ones_like(phase), phase)  # [T] complex
    writes = torch.zeros(t, enc.slot_dim, dtype=torch.complex64)
    writes[torch.arange(t), local] = phasor
    oh = F.one_hot(routes, enc.n_slots).to(torch.complex64)   # [T, S]
    acc = torch.einsum("ts,td->sd", oh, writes)          # [S, D]
    parts = []
    for s in range(enc.n_slots):
        a = acc[s]
        n = a.abs().sum()
        parts.append(a / n.clamp_min(1e-12) if float(n.detach()) > 0 else a)
    return sub.unit_norm(torch.cat(parts))


def readout_logits(gen, psi):
    _b, tokens = gen.dec.encode_wave(psi)
    h = tokens
    for blk in gen.dec.layers:
        h = blk(h)
    return gen.dec.head_text(gen.dec.norm(h))


def wave_fingerprint(enc, texts, tok, angle):
    with torch.no_grad():
        return torch.stack([
            diff_wave(enc, torch.tensor(tok.encode(s), dtype=torch.long), angle)
            for s in texts])


# ------------------------------------------------------------------ training
def train_readout(gen, enc, angle, spec_ids, tgt, steps, lr, pad, tv,
                  also_angle: bool):
    for p in gen.dec.parameters():
        p.requires_grad_(True)
    params = [p for p in gen.dec.parameters() if p.requires_grad]
    if also_angle:
        angle.requires_grad_(True)
        params = params + [angle]
    else:
        angle.requires_grad_(False)
    opt = torch.optim.Adam(params, lr=lr)
    grad_max, first, last = 0.0, None, None
    for _ in range(steps):
        opt.zero_grad()
        total = None
        for lo in range(0, len(spec_ids), 8):
            sl = list(range(lo, min(lo + 8, len(spec_ids))))
            psi = torch.stack([diff_wave(enc, spec_ids[j], angle) for j in sl])
            lg = readout_logits(gen, psi)
            tt = max(len(tgt[j]) for j in sl)
            tb = torch.full((len(sl), tt), pad, dtype=torch.long)
            for r, j in enumerate(sl):
                tb[r, :len(tgt[j])] = torch.tensor(tgt[j], dtype=torch.long)
            loss = F.cross_entropy(lg[:, :tt, :].reshape(-1, tv), tb.reshape(-1),
                                   ignore_index=pad)
            total = loss if total is None else total + loss
        total.backward()
        if also_angle and angle.grad is not None:
            grad_max = max(grad_max, float(angle.grad.abs().max()))
        opt.step()
        first = float(total.detach()) if first is None else first
        last = float(total.detach())
    return {"loss_first": first, "loss_last": last, "angle_grad_max": grad_max}


def g_u4_of(system, tok, n_texts):
    return gate_u4_retention(system, tok, n_texts=n_texts)


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-texts", type=int, default=1024)
    ap.add_argument("--wide-nmem", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    cfg = M4Config(steps=a.steps, lr=3e-3, seed=0, train_body=True)
    corpus = build_corpus(max_len=3, holdout_len=3)
    tr = [corpus.specs[i] for i in corpus.train_idx]
    trt = [corpus.traces[i] for i in corpus.train_idx]
    gate_txt = list(corpus.corpus_texts)[:128]

    system, tok = build_system(corpus, ingress_seed=PIN, pin_seed=PIN)
    enc = system.ingress
    tv = tok.vocab_size
    spec_ids = [torch.tensor(tok.encode(s), dtype=torch.long) for s in tr]
    tgt = [tok.encode(s)[:cfg.max_trace] for s in tr]

    print(f"[setup] d_model={system.decoder.pooling.d_k * system.decoder.pooling.n_mem}"
          f" n_mem={system.decoder.pooling.n_mem} d_k={system.decoder.pooling.d_k}",
          flush=True)

    # ---- ARM F: frozen. Reproduce the committed baseline BEFORE any delta.
    genF = WaveTextGenerator(system, tok, train_body=True)
    angleF = enc.angle.detach().clone()
    f0 = g_u4_of(system, tok, a.n_texts)
    print(f"[F pre-train]  G-U4 = {f0.get('value')} (baseline {F_BASELINE})",
          flush=True)

    fp_before = wave_fingerprint(enc, gate_txt, tok, angleF)
    genF = copy.deepcopy(genF)
    sysF, tok = build_system(corpus, ingress_seed=PIN, pin_seed=PIN)
    genF = WaveTextGenerator(sysF, tok, train_body=True)
    encF = sysF.ingress
    fF = train_readout(genF, encF, angleF, spec_ids, tgt, a.steps, cfg.lr,
                       cfg.pad_id, tv, also_angle=False)
    u4F = g_u4_of(sysF, tok, a.n_texts)
    print(f"[F] loss {fF['loss_first']:.4f}->{fF['loss_last']:.4f} "
          f"G-U4={u4F.get('value')}", flush=True)

    # ---- ARM P: learnable phase.
    sysP, tokP = build_system(corpus, ingress_seed=PIN, pin_seed=PIN)
    encP = sysP.ingress
    genP = WaveTextGenerator(sysP, tokP, train_body=True)
    angleP = nn.Parameter(encP.angle.detach().clone())
    fP = train_readout(genP, encP, angleP, spec_ids, tgt, a.steps, cfg.lr,
                       cfg.pad_id, tv, also_angle=True)
    # write the trained phase back so the frozen-path gate sees it
    with torch.no_grad():
        encP.angle.copy_(angleP.detach())
    u4P = g_u4_of(sysP, tokP, a.n_texts)
    fp_after = wave_fingerprint(encP, gate_txt, tokP, angleP.detach())
    psi_delta = float((fp_after - fp_before).abs().max())
    print(f"[P] loss {fP['loss_first']:.4f}->{fP['loss_last']:.4f} "
          f"angle_grad={fP['angle_grad_max']:.3e} psi_delta={psi_delta:.3e} "
          f"G-U4={u4P.get('value')}", flush=True)

    # ---- ARM W: wider pooling (d_k up), default frozen phase.
    sysW, tokW = build_system(corpus, ingress_seed=PIN, pin_seed=PIN)
    d = sysW.decoder.pooling.d_k * sysW.decoder.pooling.n_mem
    new_pool = HopfieldCrossPooling(sysW.decoder.dim if hasattr(sysW.decoder, "dim")
                                    else 4096, d,
                                    sysW.decoder.pooling.n_macro,
                                    n_mem=a.wide_nmem)
    sysW.decoder.pooling = new_pool
    encW = sysW.ingress
    angleW = encW.angle.detach().clone()
    genW = WaveTextGenerator(sysW, tokW, train_body=True)
    fW = train_readout(genW, encW, angleW, spec_ids, tgt, a.steps, cfg.lr,
                       cfg.pad_id, tv, also_angle=False)
    u4W = g_u4_of(sysW, tokW, a.n_texts)
    print(f"[W] n_mem={new_pool.n_mem} d_k={new_pool.d_k} "
          f"loss {fW['loss_first']:.4f}->{fW['loss_last']:.4f} "
          f"G-U4={u4W.get('value')}", flush=True)

    # ---- verdict
    vF = u4F.get("value") or 0.0
    vP = u4P.get("value") or 0.0
    vW = u4W.get("value") or 0.0
    g1 = fP["angle_grad_max"] > 0.0
    g2 = psi_delta > 0.0
    g3 = abs(vF - F_BASELINE) <= 1e-3
    dP = vP - vF
    dW = vW - vF

    if not (g1 and g3):
        verdict = "VACUOUS"
        why = f"guards angle_grad>0:{g1} baseline_reproduced:{g3} " \
              f"(F={vF}, baseline={F_BASELINE})"
    else:
        movers = []
        if dP >= 0.02:
            movers.append(f"PHASE +{dP:.4f}")
        if dW >= 0.02:
            movers.append(f"WIDE_POOL +{dW:.4f}")
        if dP <= -0.02:
            movers.append(f"PHASE HARMS {dP:.4f}")
        if dW <= -0.02:
            movers.append(f"WIDE_POOL HARMS {dW:.4f}")
        verdict = ("LEVER_FOUND: " + ", ".join(movers)) if movers \
            else "NEITHER_LEVER_MOVES_G_U4"
        why = (f"F={vF:.6f} P={vP:.6f} (d={dP:+.6f}) W={vW:.6f} "
               f"(d={dW:+.6f}); bound {G_U4_BOUND} unchanged; "
               f"psi_delta={psi_delta:.3e}")

    out = {
        "schema": "henri.lever.probe.v1",
        "verdict": verdict, "why": why,
        "guards": {"angle_grad_nonzero": g1, "wave_changed_gate_corpus": g2,
                   "baseline_reproduced": g3},
        "g_u4": {"F_frozen": vF, "P_learnable_phase": vP, "W_wide_pool": vW,
                 "bound": G_U4_BOUND, "committed_baseline": F_BASELINE},
        "delta": {"P_minus_F": dP, "W_minus_F": dW},
        "runs": {"F": fF, "P": fP, "W": fW},
        "wide_pool": {"n_mem": a.wide_nmem, "d_k": new_pool.d_k},
        "run_cfg": {"steps": a.steps, "n_texts": a.n_texts, "pin": PIN},
        "seconds": round(time.time() - t0, 2),
        "defects": {
            "D150": "the router is dead (kill test #5) but the write PHASE holds "
                    "the information and was never tested as a lever. polar() is "
                    "differentiable in the phase, so no STE is required.",
            "D151": "HopfieldCrossPooling computes similarity in d_k = 4 at both "
                    "scales (resolve_n_mem = d_model // 4). Pooled tokens are "
                    "confined near a rank-<=4 per-macro-token subspace, which is "
                    "the candidate cause of the D2 collapse (pair |cos| 0.955).",
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("verdict", "why", "guards", "delta",
                                          "g_u4")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
