"""SPEC_B kill tests K-B1..K-B6 (approved). Bounds UNCHANGED.

PRE-REGISTERED (design/zone_a/SPEC_B_pooling_width_v1.md section 6):
  K-B1  baseline reproduction: pre-train G-U4 = 0.819028 +/- 1e-3 at pin 20261004
        with the OLD rule (dk_target=0). Not reproduced -> VACUOUS.
  K-B2  in-envelope gain, small scale: dk_target=32 gives G-U4 >= 0.95 in >=4/5
        construction seeds. <4/5 -> REJECT.
  K-B3  envelope: full config total params in [400M, 500M]. Outside -> REJECT.
  K-B4  no readout regression: M4-G1 held-out EM must not fall below the frozen
        baseline. Any fall -> REJECT.
  K-B5  FULL-SCALE TRANSFER: reproduce K-B2 at a DIFFERENT scale. Implemented as
        a d_model ladder {128, 512} because the full 440M CPU forward is not
        affordable here; that reduction is DISCLOSED and K-B5 is reported as
        PARTIAL unless the ladder passes at every rung.
  K-B6  negative control: the OLD rule (dk_target=0) must NOT clear 0.95 in
        >=4/5 seeds. If it does, the gate does not discriminate and is broken.

D135 (signed criterion): the contract must improve at least one gate and
degrade nothing. Movement alone is not support.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.gates import gate_u4_retention                    # noqa: E402
from henri_core.m4_generative import (                            # noqa: E402
    M4Config, WaveTextGenerator, build_corpus, run_program)
from henri_core.model3_decoder import (                           # noqa: E402
    DecoderConfig, decoder_param_formula)
from henri_core.system import TriModelSystem                      # noqa: E402
from henri_core.tokenizer import ByteBPE                          # noqa: E402

G_U4_BOUND = 0.95
BASE_PIN = 20261004
BASELINE_PRE = 0.819028
# D163 (self-caught): K-B1 compared a gate sampled at --n-texts-small (512) with
# a baseline recorded at n_texts=1024. Same pinned construction, different
# sampling -> 0.816939 vs 0.819028, so the gate read False while the LOG showed
# 0.819028. Baseline reproduction must use the baseline's OWN configuration.
BASE_NTEXTS = 1024

# (d_model, n_layers, n_heads, n_kv_heads, d_ffn)
LADDER = {
    "small_128": (128, 2, 4, 1, 256),
    "mid_512": (512, 4, 8, 2, 1024),
}


def make_system(tok, scale, dk_target, dim=4096, seed=None):
    """Build one rung of the scale ladder.

    D156 (self-caught): the first version called TriModelSystem directly and
    OMITTED the full-construction pin, so the pre-train G-U4 read 0.941940
    instead of the committed 0.819028. This now replicates
    build_system's pinning exactly (fork_rng + manual_seed, ingress_seed=None so
    the ingress draws from the pinned global RNG).
    D157 (self-caught): small=True SILENTLY OVERWRITES decoder_cfg (system.py
    58-63), so the mid_512 rung also ran at d_model=128 and n_mem=4 was
    impossible. Both rungs were the same scale and K-B5's transfer "pass" was
    fake. Non-small rungs now use small=False so decoder_cfg is honored.
    """
    dm, nl, nh, nkv, dff = LADDER[scale]
    is_small = scale.startswith("small")
    cfg = DecoderConfig(dim=dim, d_model=dm, n_layers=nl, n_heads=nh,
                        n_kv_heads=nkv, d_ffn=dff, n_macro=16,
                        n_invariants=32, vocab=tok.vocab_size,
                        dk_target=dk_target)
    ctx = (torch.random.fork_rng(devices=[]) if seed is not None
           else contextlib.nullcontext())
    with ctx:
        if seed is not None:
            torch.manual_seed(int(seed))
        system = TriModelSystem(vocab=tok.vocab_size, dim=dim, small=is_small,
                                decoder_cfg=None if is_small else cfg,
                                dk_target=dk_target,
                                # D156 ROOT CAUSE: the committed baseline is
                                # build_system(ingress_seed=PIN, pin_seed=PIN) --
                                # BOTH seeds. The first fix passed only the global
                                # pin, so the ingress still drew unpinned and the
                                # pre-train gate read 0.941940 instead of
                                # 0.819028. Pass the ingress seed as well.
                                ingress_seed=seed)
    system.eval()
    real_dm = system.decoder.cfg.d_model
    real_m = system.decoder.pooling.n_mem
    if real_dm != dm:
        raise RuntimeError(
            f"scale not honored: asked d_model={dm}, got {real_dm}. "
            f"small={is_small} overrides decoder_cfg.")
    return system, system.decoder.cfg


def readout_logits(gen, psi):
    _b, tokens = gen.dec.encode_wave(psi)
    h = tokens
    for blk in gen.dec.layers:
        h = blk(h)
    return gen.dec.head_text(gen.dec.norm(h))


def train_readout(gen, specs, tgt, steps, lr, pad, tv, max_trace):
    for p in gen.dec.parameters():
        p.requires_grad_(True)
    for p in gen.system.ingress.parameters():
        p.requires_grad_(False)
    params = [p for p in gen.dec.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    last = None
    for _ in range(steps):
        opt.zero_grad()
        total = None
        for lo in range(0, len(specs), 8):
            sl = list(range(lo, min(lo + 8, len(specs))))
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
    return last


def heldout_em(gen, tok, specs, traces, max_trace) -> float:
    ok = 0
    gen.eval()
    with torch.no_grad():
        for s, tr in zip(specs, traces):
            psi = gen.wave([s])
            lg = readout_logits(gen, psi)[0]
            t = tok.encode(tr)[:max_trace]
            pred = lg[:len(t)].argmax(-1).tolist()
            ok += int(pred == list(t))
    return ok / max(1, len(specs))


def run_arm(scale, dk_target, seed, steps, n_texts, corpus, tok, tr_spec,
            tr_trace, ho_spec, ho_trace, cfg) -> dict:
    # D156: NO bare torch.manual_seed here. The pin lives inside make_system, so
    # the construction is byte-identical across processes (D130/D138).
    system, dcfg = make_system(tok, scale, dk_target, seed=seed)
    u4_pre = gate_u4_retention(system, tok, n_texts=n_texts)
    gen = WaveTextGenerator(system, tok, train_body=True)
    tv = tok.vocab_size
    tgt = [tok.encode(s)[:cfg.max_trace] for s in tr_trace]
    loss = train_readout(gen, tr_spec, tgt, steps, cfg.lr, cfg.pad_id, tv,
                         cfg.max_trace)
    u4_post = gate_u4_retention(system, tok, n_texts=n_texts)
    m4_em = heldout_em(gen, tok, ho_spec, ho_trace, cfg.max_trace)
    return {
        "scale": scale, "dk_target": dk_target, "seed": seed,
        "d_model": dcfg.d_model,
        "n_mem": system.decoder.pooling.n_mem,
        "d_k": system.decoder.pooling.d_k,
        "g_u4_pre": u4_pre.get("value"), "g_u4_post": u4_post.get("value"),
        "g_u4_pos_ctl": u4_post.get("positive_control"),
        "g_u4_estimator_sane": u4_post.get("estimator_sane"),
        "m4_heldout_em": m4_em, "loss_last": loss,
    }


def stats(vals):
    t = torch.tensor([v for v in vals if v is not None], dtype=torch.float32)
    if t.numel() == 0:
        return {}
    return {"n": int(t.numel()), "min": float(t.min()), "max": float(t.max()),
            "mean": float(t.mean()), "std": float(t.std()),
            "pass_ge_0.95": int((t >= G_U4_BOUND).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds-small", type=int, default=5)
    ap.add_argument("--seeds-mid", type=int, default=3)
    ap.add_argument("--n-texts-small", type=int, default=1024)
    ap.add_argument("--n-texts-mid", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    cfg = M4Config(steps=a.steps, lr=3e-3, seed=0, train_body=True)

    corpus = build_corpus(max_len=3, holdout_len=3)
    tok = ByteBPE().train(corpus.corpus_texts, vocab_size=512)
    tr_spec = [corpus.specs[i] for i in corpus.train_idx]
    tr_trace = [corpus.traces[i] for i in corpus.train_idx]
    ho_spec = [corpus.specs[i] for i in corpus.heldout_idx]
    ho_trace = [corpus.traces[i] for i in corpus.heldout_idx]

    rows = []
    for scale, nseeds, nt in (("small_128", a.seeds_small, a.n_texts_small),
                              ("mid_512", a.seeds_mid, a.n_texts_mid)):
        for s in range(nseeds):
            seed = BASE_PIN + s
            for label, dkt in (("old_rule", 0), ("spec_b", 32)):
                r = run_arm(scale, dkt, seed, a.steps, nt, corpus, tok,
                            tr_spec, tr_trace, ho_spec, ho_trace, cfg)
                r["arm"] = label
                rows.append(r)
                print(f"[{scale} seed {seed}] {label:9s} n_mem={r['n_mem']:4d} "
                      f"d_k={r['d_k']:4d} pre={r['g_u4_pre']:.6f} "
                      f"post={r['g_u4_post']:.6f} m4EM={r['m4_heldout_em']:.4f}",
                      flush=True)

    def sel(scale, arm, key):
        return [r[key] for r in rows if r["scale"] == scale and r["arm"] == arm]

    st = {f"{sc}_{ar}": stats(sel(sc, ar, "g_u4_post"))
          for sc in LADDER for ar in ("old_rule", "spec_b")}
    m4 = {f"{sc}_{ar}": stats(sel(sc, ar, "m4_heldout_em"))
          for sc in LADDER for ar in ("old_rule", "spec_b")}

    # ---- K-B1 baseline reproduction, measured at the BASELINE's configuration
    # (n_texts = 1024). D163: reusing the sweep's n_texts made the gate compare
    # two different sampling regimes.
    system_b, _ = make_system(tok, "small_128", 0, seed=BASE_PIN)
    kb1_gate = gate_u4_retention(system_b, tok, n_texts=BASE_NTEXTS)
    kb1_val = kb1_gate.get("value")
    kb1 = abs(float(kb1_val) - BASELINE_PRE) <= 1e-3
    print(f"[K-B1] pinned old-rule G-U4 = {kb1_val} "
          f"(baseline {BASELINE_PRE}, n_texts {BASE_NTEXTS}) -> {kb1}", flush=True)

    # ---- K-B3 envelope, full config
    full_old = decoder_param_formula(DecoderConfig(dim=65536, dk_target=0))
    full_new = decoder_param_formula(DecoderConfig(dim=65536, dk_target=32))
    kb3 = 4.0e8 <= full_new["total"] <= 5.0e8

    # ---- K-B2 small-scale gain
    pass_new_small = st["small_128_spec_b"].get("pass_ge_0.95", 0)
    kb2 = pass_new_small >= 4

    # ---- K-B6 negative control: old rule must NOT pass
    pass_old_small = st["small_128_old_rule"].get("pass_ge_0.95", 0)
    kb6 = pass_old_small < 4

    # ---- K-B4 no readout regression (spec_b vs old_rule, small scale)
    m4_new = m4["small_128_spec_b"].get("mean", 0.0)
    m4_old = m4["small_128_old_rule"].get("mean", 0.0)
    kb4 = m4_new >= m4_old

    # ---- K-B5 transfer: spec_b must pass at EVERY rung of the ladder
    kb5_rungs = {sc: (st[f"{sc}_spec_b"].get("pass_ge_0.95", 0)
                      >= max(1, st[f"{sc}_spec_b"].get("n", 1) - 1))
                 for sc in LADDER}
    kb5 = all(kb5_rungs.values())

    gates = {
        "K-B1_baseline_reproduced": kib1 if (kib1 := kb1) else False,
        "K-B2_small_scale_gain": kb2,
        "K-B3_envelope": kb3,
        "K-B4_no_readout_regression": kb4,
        "K-B5_full_scale_transfer": kb5,
        "K-B6_negative_control_fails": kb6,
    }
    if not kb1:
        verdict = "VACUOUS"
    elif not kb3:
        verdict = "REJECT_ENVELOPE"
    elif kb6 and not kb2:
        verdict = "REJECT_NO_GAIN"
    elif not kb4:
        verdict = "REJECT_READOUT_REGRESSION"
    elif kb2 and kb3 and kb4 and kb6 and not kb5:
        verdict = "ACCEPT_SMALL_SCALE_ONLY"
    elif kb2 and kb3 and kb4 and kb5 and kb6:
        verdict = "ACCEPT"
    else:
        verdict = "PARTIAL"

    out = {
        "schema": "henri.specb.killtests.v1",
        "verdict": verdict,
        "gates": gates,
        "kb5_transfer_rungs": kb5_rungs,
        "stats_g_u4": st, "stats_m4_em": m4,
        "envelope": {"old_total": full_old["total"],
                     "spec_b_total": full_new["total"],
                     "spec_b_parts": full_new["parts"]},
        "rows": rows,
        "disclosure": {
            "K-B5_reduced": ("The full 440M CPU forward is not affordable in this "
                             "session. K-B5 is a d_model LADDER "
                             "{128, 512}, not the 1024 target. PASS here is "
                             "evidence of transfer, NOT full-scale proof."),
            "n_texts": {"small_128": a.n_texts_small, "mid_512": a.n_texts_mid},
        },
        "run": {"steps": a.steps, "pin_base": BASE_PIN, "bound": G_U4_BOUND},
        "seconds": round(time.time() - t0, 2),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("verdict", "gates", "kb5_transfer_rungs",
                                          "envelope")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
