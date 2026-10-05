"""KILL TEST #5 (DIAGNOSTIC ONLY): can a TRAINABLE ingress router move the gates?

SPEC_A premise
    The text path's ONLY trainable lever is slot_router (132 params at full
    scale). zone_a._token_writes routes via argmax -> int(), which severs the
    gradient, so the router never trains. SPEC_A proposes an STE around that
    argmax so the representation can be optimized.

THIS DOES NOT IMPLEMENT SPEC_A. zone_a.py is untouched. Default path untouched.

D135 (self-caught): the first criterion fired on |dG-U4| >= 0.02. It reported
    ROUTER_MOVES_GATES while G-U4 FELL by 0.110 and M4-EM moved by exactly
    +0.000000. A criterion that rewards harmful movement is the same defect
    class as a gate that cannot fail. The criterion is now SIGNED: the router
    must IMPROVE at least one gate and degrade neither.

D136 (self-caught): in the first design arm B trained the decoder AND the
    router, so the decoder trajectory diverged on step 1 and every delta was
    confounded. Now the decoder is trained ONCE in phase 1, then copied into
    both arms. Phase 2 trains the ROUTER ONLY (decoder frozen) in arm B; arm A
    does nothing. The router is then the only difference between the arms.

VACUITY GUARD (D120/D121 class). system.wave_of is @no_grad, so this harness
    carries its own STE forward. Three guards must hold or the verdict is
    VACUOUS: (1) router grad non-zero; (2) routing actually changed in arm B;
    (3) arm A's routing is byte-identical before and after.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import substrate as sub                      # noqa: E402
from henri_core.gates import gate_u4_retention               # noqa: E402
from henri_core.m4_generative import (                       # noqa: E402
    M4Config, WaveTextGenerator, build_corpus, build_system)

PIN = 20261004
G_U4_BOUND = 0.95        # UNCHANGED
M4_BOUND = 0.246         # UNCHANGED


# ------------------------------------------------------------------- STE wave
def ste_wave(enc, ids: torch.Tensor) -> torch.Tensor:
    """Differentiable Zone A ingress. Hard routing forward, soft backward."""
    emb = enc.token_emb(ids)
    logits = enc.slot_router(emb)
    soft = F.softmax(logits, dim=-1)
    idx = soft.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
    w = hard + soft - soft.detach()

    t = ids.shape[0]
    tok = (ids % enc.dim).long()
    local = (tok % enc.slot_dim).long()
    phase = enc.angle[tok].to(torch.float32)
    phasor = torch.polar(torch.ones_like(phase), phase)
    writes = torch.zeros(t, enc.slot_dim, dtype=torch.complex64)
    writes[torch.arange(t), local] = phasor

    acc = torch.einsum("ts,td->sd", w.to(torch.complex64), writes)
    parts = []
    for s in range(enc.n_slots):
        a = acc[s]
        n = a.abs().sum()
        parts.append(a / n.clamp_min(1e-12) if float(n.detach()) > 0 else a)
    return sub.unit_norm(torch.cat(parts))


def route_ids(enc, ids: torch.Tensor) -> list[int]:
    with torch.no_grad():
        return enc.slot_router(enc.token_emb(ids)).argmax(-1).tolist()


def readout_logits(gen, psi: torch.Tensor) -> torch.Tensor:
    _bands, tokens = gen.dec.encode_wave(psi)
    h = tokens
    for blk in gen.dec.layers:
        h = blk(h)
    return gen.dec.head_text(gen.dec.norm(h))


def batches(spec_ids, tgt, traces, bsz: int, tv: int, pad: int):
    for lo in range(0, len(spec_ids), bsz):
        sl = list(range(lo, min(lo + bsz, len(spec_ids))))
        psi = torch.stack([ste_wave_holder["fn"](spec_ids[j]) for j in sl])
        tt = max(len(tgt[j]) for j in sl)
        tb = torch.full((len(sl), tt), pad, dtype=torch.long)
        for r, j in enumerate(sl):
            tb[r, :len(tgt[j])] = torch.tensor(tgt[j], dtype=torch.long)
        yield psi, tb.reshape(-1), tv, pad


ste_wave_holder: dict = {}


def phase1_train_decoder(gen, enc, spec_ids, tgt, cfg: M4Config) -> dict:
    """Train the readout with the router FROZEN. Same for both arms."""
    ste_wave_holder["fn"] = lambda ids: ste_wave(enc, ids)
    for p in enc.parameters():
        p.requires_grad_(False)
    params = [p for p in gen.dec.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    tv = gen.tok.vocab_size
    first = last = None
    for step in range(cfg.steps):
        opt.zero_grad()
        total = None
        for psi, tgb, tvb, pad in batches(spec_ids, tgt, None, 8, tv, cfg.pad_id):
            lg = readout_logits(gen, psi)
            tt = tgb.numel() // len(psi)
            loss = F.cross_entropy(lg[:, :tt, :].reshape(-1, tvb), tgb,
                                   ignore_index=pad)
            total = loss if total is None else total + loss
        total.backward()
        opt.step()
        first = float(total.detach()) if first is None else first
        last = float(total.detach())
    return {"loss_first": first, "loss_last": last}


def phase2_train_router(gen, enc, spec_ids, tgt, cfg: M4Config) -> dict:
    """Train the ROUTER ONLY (decoder frozen). The isolated lever."""
    ste_wave_holder["fn"] = lambda ids: ste_wave(enc, ids)
    for p in gen.dec.parameters():
        p.requires_grad_(False)
    for p in enc.parameters():
        p.requires_grad_(True)
    params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    tv = gen.tok.vocab_size
    grad_max = 0.0
    first = last = None
    for step in range(cfg.steps):
        opt.zero_grad()
        total = None
        for psi, tgb, tvb, pad in batches(spec_ids, tgt, None, 8, tv, cfg.pad_id):
            lg = readout_logits(gen, psi)
            tt = tgb.numel() // len(psi)
            loss = F.cross_entropy(lg[:, :tt, :].reshape(-1, tvb), tgb,
                                   ignore_index=pad)
            total = loss if total is None else total + loss
        total.backward()
        if enc.slot_router.weight.grad is not None:
            grad_max = max(grad_max, float(enc.slot_router.weight.grad.abs().max()))
        opt.step()
        first = float(total.detach()) if first is None else first
        last = float(total.detach())
    return {"loss_first": first, "loss_last": last, "router_grad_max": grad_max}


def heldout_scores(gen, tok, specs, traces, cfg) -> dict:
    gen.eval()
    ok = tot = em = 0
    with torch.no_grad():
        for s, tr in zip(specs, traces):
            ids = torch.tensor(tok.encode(s), dtype=torch.long)
            psi = ste_wave(gen.system.ingress, ids).unsqueeze(0)
            lg = readout_logits(gen, psi)[0]
            t = tok.encode(tr)[:cfg.max_trace]
            pred = lg[:len(t)].argmax(-1).tolist()
            hit = sum(1 for a, b in zip(pred, t) if a == b)
            ok += hit
            tot += len(t)
            em += int(hit == len(t))
    return {"token_acc": ok / max(1, tot),
            "exact_match": em / max(1, len(specs)), "n": len(specs)}


def unigram_floor(tok, train_traces, hold_traces) -> float:
    from collections import Counter
    col = Counter()
    for tr in train_traces:
        for p, i in enumerate(tok.encode(tr)):
            col[(p, i)] += 1
    best = {}
    for (p, i), c in col.items():
        if p not in best or c > best[p][1]:
            best[p] = (i, c)
    ok = tot = 0
    for tr in hold_traces:
        t = tok.encode(tr)
        for p, i in enumerate(t):
            if p in best and best[p][0] == i:
                ok += 1
            tot += 1
    return ok / max(1, tot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300,
                    help="phase-1 (decoder) steps; phase 2 uses the same count")
    ap.add_argument("--n-texts", type=int, default=1024)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    t0 = time.time()
    cfg = M4Config(steps=a.steps, lr=3e-3, seed=0, train_body=True)
    corpus = build_corpus(max_len=3, holdout_len=3)
    tr = [corpus.specs[i] for i in corpus.train_idx]
    trt = [corpus.traces[i] for i in corpus.train_idx]
    ho = [corpus.specs[i] for i in corpus.heldout_idx]
    hot = [corpus.traces[i] for i in corpus.heldout_idx]

    # D138 (self-caught): the first version passed pin_seed but OMITTED
    # ingress_seed, so arm C read 0.941940 where the committed pinned receipt
    # reads 0.819028. build_system re-forks the ingress when ingress_seed is
    # given, so the two constructions differ. Reproduce the published baseline
    # exactly, or the deltas are measured against the wrong system.
    system, tok = build_system(corpus, ingress_seed=PIN, pin_seed=PIN)
    gen0 = WaveTextGenerator(system, tok, train_body=True)
    enc = system.ingress
    tv = tok.vocab_size

    spec_ids = [torch.tensor(tok.encode(s), dtype=torch.long) for s in tr]
    tgt = [tok.encode(s)[:cfg.max_trace] for s in tr]
    print(f"[corpus] train={len(tr)} heldout={len(ho)} vocab={tv}", flush=True)

    # ARM C (D137): the UNTRAINED baseline. Quantifies what decoder training
    # alone does to G-U4 at this seed. Doubles as a reproduction check: the
    # pinned receipt (pin 20261004) recorded G-U4 = 0.819028 with no training.
    genC = copy.deepcopy(gen0)
    u4C = gate_u4_retention(genC.system, tok, n_texts=a.n_texts)
    reproduces_baseline = abs(float(u4C.get("value")) - 0.819028) < 1e-4
    print(f"[C_untrained] g_u4={u4C.get('value')} "
          f"(pinned receipt 0.819028, reproduces={reproduces_baseline})",
          flush=True)

    # PHASE 1: train the readout once, router frozen. Shared by both arms.
    print(f"[phase1] decoder {a.steps} steps (router frozen) ...", flush=True)
    p1 = phase1_train_decoder(gen0, enc, spec_ids, tgt, cfg)
    print(f"[phase1] loss {p1['loss_first']:.4f}->{p1['loss_last']:.4f}",
          flush=True)

    res = {}
    # ARM A: router untouched. Deepcopy so arm B gets the identical start.
    genA = copy.deepcopy(gen0)
    encA = genA.system.ingress
    routed_before_A = [route_ids(encA, i) for i in spec_ids]

    # ARM B: train the router ONLY from the identical phase-1 state.
    genB = copy.deepcopy(gen0)
    encB = genB.system.ingress
    routed_before_B = [route_ids(encB, i) for i in spec_ids]
    print(f"[phase2] router {a.steps} steps (decoder frozen) ...", flush=True)
    p2 = phase2_train_router(genB, encB, spec_ids, tgt, cfg)
    print(f"[phase2] loss {p2['loss_first']:.4f}->{p2['loss_last']:.4f} "
          f"grad_max={p2['router_grad_max']:.3e}", flush=True)

    routed_after_A = [route_ids(encA, i) for i in spec_ids]
    routed_after_B = [route_ids(encB, i) for i in spec_ids]

    for name, gen2, enc2, before, after in (
            ("A_frozen_router", genA, encA, routed_before_A, routed_after_A),
            ("B_ste_router", genB, encB, routed_before_B, routed_after_B)):
        ho_s = heldout_scores(gen2, tok, ho, hot, cfg)
        floor = unigram_floor(tok, trt, hot)
        u4 = gate_u4_retention(gen2.system, tok, n_texts=a.n_texts)
        changed = sum(1 for x, y in zip(before, after)
                      for u, v in zip(x, y) if u != v)
        total = sum(len(x) for x in before)
        res[name] = {
            "heldout_token_acc": ho_s["token_acc"],
            "heldout_exact_match": ho_s["exact_match"],
            "unigram_floor": floor,
            "g_u4": u4.get("value"), "g_u4_status": u4.get("status"),
            "g_u4_positive_control": u4.get("positive_control"),
            "tokens_routing_changed": changed, "tokens_total": total,
        }
        print(f"[{name}] ho_em={ho_s['exact_match']:.4f} floor={floor:.4f} "
              f"g_u4={u4.get('value')} routed_changed={changed}/{total}",
              flush=True)

    # ARM C row (untrained). See D137.
    hoC = heldout_scores(genC, tok, ho, hot, cfg)
    res["C_untrained"] = {
        "heldout_token_acc": hoC["token_acc"],
        "heldout_exact_match": hoC["exact_match"],
        "unigram_floor": unigram_floor(tok, trt, hot),
        "g_u4": u4C.get("value"), "g_u4_status": u4C.get("status"),
        "g_u4_positive_control": u4C.get("positive_control"),
        "tokens_routing_changed": 0, "tokens_total": 0,
    }
    print(f"[C_untrained] ho_em={hoC['exact_match']:.4f} "
          f"g_u4={u4C.get('value')}", flush=True)

    A, B = res["A_frozen_router"], res["B_ste_router"]
    C = res["C_untrained"]
    # D137 (self-caught): guard g2 was measured on the 96 TRAINING specs, where
    # argmax flipped 0/96 because phase 1 had already memorized the objective.
    # G-U4 reads a 1024-text corpus, so the guard must fingerprint the WAVE on
    # that population. A guard on the wrong population reports VACUOUS while the
    # router DID change the measured path (A 0.971294 -> B 0.844739).
    gate_txt = list(corpus.corpus_texts)[:128]

    def _fp(enc2):
        with torch.no_grad():
            return torch.stack([ste_wave(enc2, torch.tensor(
                tok.encode(t), dtype=torch.long)) for t in gate_txt])

    psi_delta = float((_fp(encB) - _fp(encA)).abs().max())
    g1 = p2["router_grad_max"] > 0.0
    g2 = psi_delta > 1e-6          # D137: wave change on the GATE population
    g3 = A["tokens_routing_changed"] == 0
    d_u4 = (B["g_u4"] or 0.0) - (A["g_u4"] or 0.0)
    d_m4 = B["heldout_exact_match"] - A["heldout_exact_match"]
    d_decode = (A["g_u4"] or 0.0) - (C["g_u4"] or 0.0)   # decoder-training effect

    if not (g1 and g2 and g3):
        verdict = "VACUOUS"
        why = f"guards grad>0:{g1} routing_changed:{g2} frozen_unchanged:{g3}"
    else:
        # D135: SIGNED. The router must improve at least one gate and degrade
        # neither. Mere movement, in either direction, is not support.
        helps_m4 = d_m4 > 0.02
        hurts_m4 = d_m4 < -0.02
        helps_u4 = d_u4 > 0.02
        hurts_u4 = d_u4 < -0.02
        if hurts_m4 or hurts_u4:
            verdict = "ROUTER_HARMS"
        elif helps_m4 or helps_u4:
            verdict = "ROUTER_HELPS"
        else:
            verdict = "ROUTER_MOVES_NOTHING"
        why = (f"dG-U4={d_u4:+.6f} dM4-EM={d_m4:+.6f}; "
               f"helps_m4={helps_m4} helps_u4={helps_u4} "
               f"hurts_m4={hurts_m4} hurts_u4={hurts_u4} "
               f"(bounds unchanged: G-U4 {G_U4_BOUND}, M4-EM {M4_BOUND})")

    out = {
        "schema": "henri.ste-router.kill5.v2",
        "diagnostic_only": True,
        "zone_a_untouched": True,
        "default_path_untouched": True,
        "spec_a_implemented": False,
        "pin_seed": PIN,
        "bounds": {"G-U4": G_U4_BOUND, "M4-EM": M4_BOUND},
        "run": {"steps_p1": a.steps, "steps_p2": a.steps, "n_texts": a.n_texts},
        "phase1": p1,
        "phase2": p2,
        "arms": res,
        "guards": {"router_grad_nonzero": g1, "routing_changed": g2,
                   "frozen_unchanged": g3},
        "delta": {"g_u4": d_u4, "m4_exact_match": d_m4},
        "delta_decoder_training_g_u4": d_decode,
        "psi_delta_AB_gate_corpus": psi_delta,
        "verdict": verdict,
        "why": why,
        "seconds": round(time.time() - t0, 2),
        "defects": {
            "D133": "isolate the router: STE bridges only the argmax. int() "
                    "indexing and the frozen angle buffer stay as designed.",
            "D135": "kill criterion was |d|; now SIGNED (ROUTER_HARMS on "
                    "degradation). The first version rewarded harmful movement.",
            "D136": "first design trained decoder+router in arm B, confounding "
                    "every delta. Now phase 1 trains the decoder once and both "
                    "arms deepcopy that identical state; phase 2 isolates the "
                    "router.",
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("verdict", "why", "guards", "delta",
                                          "seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
