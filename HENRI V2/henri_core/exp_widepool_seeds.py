"""Is the WIDE-POOL G-U4 gain real, or another draw? (D128 discipline)

FINDING UNDER TEST
    exp_lever_probe measured, at pin 20261004, n_texts 1024:
        F trained readout, default pooling (n_mem 32, d_k 4)  -> G-U4 0.856813
        W trained readout, wide pooling   (n_mem  4, d_k 32)  -> G-U4 0.955811
    W clears the UNCHANGED 0.95 bound. Margin is only +0.0058.

WHY A SWEEP IS REQUIRED
    D128 proved G-U4 was not reproducible: range 0.150874 over 8 construction
    seeds, 2 pass / 6 fail. A single 0.955811 could be init luck. The same
    discipline that killed the old claim must be applied to the new one.

    Also D152 (self-caught in this run): the lever probe's guard compared the
    TRAINED arm F against the UNTRAINED committed baseline 0.819028, so
    baseline_reproduced was False. The correct check is that the PRE-TRAIN arm
    reproduces 0.819028, which it did (0.8190275). Guard fixed here.

ARMS
    D  default  : n_mem = resolve_n_mem(d_model) = 32  -> d_k = 4
    W  wide     : n_mem = 4                            -> d_k = 32
    Both: identical pin, identical corpus, identical training recipe.
    Measurement order fixed: measure the gate BEFORE training (pre) and AFTER.

REPORTED
    per-seed G-U4 for both arms; min/max/mean/std; pass counts vs 0.95;
    and the parameter delta from the formula (no gate bound moves).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.gates import gate_u4_retention                 # noqa: E402
from henri_core.m4_generative import (                         # noqa: E402
    M4Config, WaveTextGenerator, build_corpus, build_system)
from henri_core.model3_decoder import (                        # noqa: E402
    DecoderConfig, HopfieldCrossPooling, decoder_param_formula, resolve_n_mem)

G_U4_BOUND = 0.95
BASE_PIN = 20261004
BASELINE_PRE = 0.819028      # committed untrained value at BASE_PIN


def readout_logits(gen, psi):
    _b, tokens = gen.dec.encode_wave(psi)
    h = tokens
    for blk in gen.dec.layers:
        h = blk(h)
    return gen.dec.head_text(gen.dec.norm(h))


def train_readout(gen, enc, spec_ids, tgt, steps, lr, pad, tv, specs):
    for p in gen.dec.parameters():
        p.requires_grad_(True)
    for p in enc.parameters():
        p.requires_grad_(False)
    params = [p for p in gen.dec.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    last = None
    for _ in range(steps):
        opt.zero_grad()
        total = None
        for lo in range(0, len(spec_ids), 8):
            sl = list(range(lo, min(lo + 8, len(spec_ids))))
            # D154 (self-caught): the first draft carried a dead no_grad block
            # that built a list of Nones and would have crashed on torch.stack.
            # The wave path is gen.wave(), which is @no_grad by design.
            psi = gen.wave([specs[j] for j in sl])
            lg = readout_logits(gen, psi)
            tt = max(len(tgt[j]) for j in sl)
            tb = torch.full((len(sl), tt), pad, dtype=torch.long)
            for r, j in enumerate(sl):
                tb[r, :len(tgt[j])] = torch.tensor(tgt[j], dtype=torch.long)
            loss = F.cross_entropy(lg[:, :tt, :].reshape(-1, tv), tb.reshape(-1),
                                   ignore_index=pad)
            total = loss if total is None else total + loss
        total.backward()
        opt.step()
        last = float(total.detach())
    return {"loss_last": last}


_specs: list[str] = []


def arm(corpus, tok, spec_ids, tgt, n_mem: int | None, seed: int,
        steps: int, lr: float, pad: int, tv: int, n_texts: int,
        specs: list[str]) -> dict:
    system, tok2 = build_system(corpus, ingress_seed=seed, pin_seed=seed)
    if n_mem is not None:
        d = system.decoder.pooling.n_mem * system.decoder.pooling.d_k
        system.decoder.pooling = HopfieldCrossPooling(
            system.decoder.dim if hasattr(system.decoder, "dim") else 4096,
            d, system.decoder.pooling.n_macro, n_mem=n_mem)
    u4_pre = gate_u4_retention(system, tok2, n_texts=n_texts)
    gen = WaveTextGenerator(system, tok2, train_body=True)
    tr = train_readout(gen, system.ingress, spec_ids, tgt, steps, lr, pad, tv,
                       specs)
    u4_post = gate_u4_retention(system, tok2, n_texts=n_texts)
    return {
        "seed": seed,
        "n_mem": system.decoder.pooling.n_mem,
        "d_k": system.decoder.pooling.d_k,
        "g_u4_pre": u4_pre.get("value"),
        "g_u4_post": u4_post.get("value"),
        "positive_control": u4_post.get("positive_control"),
        "estimator_sane": u4_post.get("estimator_sane"),
        "loss_last": tr["loss_last"],
    }


_specs_tr: list[str] = []


def stats(vals):
    t = torch.tensor([v for v in vals if v is not None], dtype=torch.float32)
    if t.numel() == 0:
        return {}
    return {"n": int(t.numel()), "min": float(t.min()), "max": float(t.max()),
            "mean": float(t.mean()), "std": float(t.std()),
            "pass_ge_bound": int((t >= G_U4_BOUND).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-texts", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    cfg = M4Config(steps=a.steps, lr=3e-3, seed=0, train_body=True)
    corpus = build_corpus(max_len=3, holdout_len=3)
    tr_spec = [corpus.specs[i] for i in corpus.train_idx]
    tr_trace = [corpus.traces[i] for i in corpus.train_idx]
    global _specs_tr
    _specs_tr = tr_spec

    _s, tok = build_system(corpus, ingress_seed=BASE_PIN, pin_seed=BASE_PIN)
    tv = tok.vocab_size
    spec_ids = [torch.tensor(tok.encode(s), dtype=torch.long) for s in tr_spec]
    tgt = [tok.encode(s)[:cfg.max_trace] for s in tr_trace]

    seeds = [BASE_PIN + i for i in range(a.seeds)]
    rows = []
    for s in seeds:
        for label, nm in (("default", None), ("wide", 4)):
            r = arm(corpus, tok, spec_ids, tgt, nm, s, a.steps, cfg.lr,
                    cfg.pad_id, tv, a.n_texts, tr_spec)
            r["arm"] = label
            rows.append(r)
            print(f"[seed {s}] {label:8s} n_mem={r['n_mem']:3d} d_k={r['d_k']:3d} "
                  f"pre={r['g_u4_pre']} post={r['g_u4_post']}", flush=True)

    dflt = [r["g_u4_post"] for r in rows if r["arm"] == "default"]
    wide = [r["g_u4_post"] for r in rows if r["arm"] == "wide"]
    dpre = [r["g_u4_pre"] for r in rows if r["arm"] == "default"]
    st_d, st_w, st_dpre = stats(dflt), stats(wide), stats(dpre)

    # parameter delta from the formula, at the FULL config
    full = DecoderConfig(dim=65536)
    f_default = decoder_param_formula(full)
    full_wide = DecoderConfig(dim=65536, n_mem=4)
    f_wide = decoder_param_formula(full_wide)

    # baseline reproduction: the PRE value at the base pin must match 0.819028
    base_pre = [r["g_u4_pre"] for r in rows if r["seed"] == BASE_PIN
                and r["arm"] == "default"]
    repro = (abs(base_pre[0] - BASELINE_PRE) <= 1e-3) if base_pre else False

    verdict = "INCONCLUSIVE"
    why = ""
    if not repro:
        verdict = "VACUOUS"
        why = f"pre-train baseline did not reproduce: {base_pre} vs {BASELINE_PRE}"
    else:
        gain = (st_w.get("mean", 0.0) - st_d.get("mean", 0.0))
        w_pass = st_w.get("pass_ge_bound", 0)
        d_pass = st_d.get("pass_ge_bound", 0)
        if w_pass >= 4 and w_pass > d_pass:
            verdict = "WIDE_POOL_STABLE_GAIN"
        elif w_pass > d_pass:
            verdict = "WIDE_POOL_PARTIAL"
        else:
            verdict = "WIDE_POOL_MOVES_NOTHING"
        why = (f"default mean={st_d.get('mean'):.6f} "
               f"[{st_d.get('min'):.6f},{st_d.get('max'):.6f}] pass={d_pass}/{st_d.get('n')}"
               f" | wide mean={st_w.get('mean'):.6f} "
               f"[{st_w.get('min'):.6f},{st_w.get('max'):.6f}] pass={w_pass}/{st_w.get('n')}"
               f" | mean gain {gain:+.6f}")

    out = {
        "schema": "henri.widepool.seedsweep.v1",
        "verdict": verdict, "why": why,
        "bound": G_U4_BOUND, "baseline_pre_reproduced": repro,
        "stats": {"default": st_d, "wide": st_w, "pre_train_default": st_dpre},
        "rows": rows,
        "param_envelope": {
            "default_total": f_default["total"],
            "wide_total": f_wide["total"],
            "delta": f_wide["total"] - f_default["total"],
            "doc_target": f_default.get("doc_target"),
            "default_parts": f_default["parts"],
            "wide_parts": f_wide["parts"],
        },
        "run": {"seeds": seeds, "steps": a.steps, "n_texts": a.n_texts},
        "seconds": round(time.time() - t0, 2),
        "defects": {
            "D152": "lever probe guard compared the TRAINED arm F against the "
                    "UNTRAINED committed baseline (0.819028), so "
                    "baseline_reproduced was False and the whole run was marked "
                    "VACUOUS. The correct check is on the PRE-train reading, "
                    "which reproduced 0.8190275. Guard corrected here.",
            "D153": "a single G-U4 value is not a result (D128). The wide-pool "
                    "gain is tested across construction seeds before any claim.",
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("verdict", "why", "bound", "baseline_pre_reproduced",
                       "stats", "param_envelope")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
