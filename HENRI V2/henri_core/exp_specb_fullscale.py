"""SPEC_B full-scale transfer rung: K-B5 at d_model=1024, dim=65536.

NEW FILE. exp_specb_killtests.py is NOT modified, so the committed small/mid
receipts keep their exact code state (co-scientist rigor: a receipt's number
belongs to the exact code state it names).

QUESTION
    Does SPEC_B (dk_target=32 -> d_k=32) hold G-U4 >= 0.95 at the FULL
    d_model=1024 / dim=65536 configuration, where the old rule gives d_k=4?

ARMS
    spec_b   dk_target=32  -> n_mem=32,  d_k=32
    control  dk_target=0   -> n_mem=256, d_k=4   (K-B6 negative control)

PRE-REGISTERED (frozen before launch)
    bound            G-U4 >= 0.95
    pin              BASE_PIN = 20261004, seeds BASE_PIN + k
    n_texts          1024  (the baseline's own configuration, D163)
    spec_b pass rule >= 4/5 seeds clear 0.95
    control rule     old rule must NOT clear 0.95 in >= 4/5 seeds

DEVICE DISCLOSURE
    henri_core has NO device plumbing: no .cuda(), no .to(device). Tensors are
    CPU. This run executes on the Vast host's CPU (192 cores, 503 GB RAM). A
    CUDA-tensor run would require a code change (a new mechanism) and is out of
    scope for this rung. The GPU is not used by this code path.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.gates import gate_u4_retention              # noqa: E402
from henri_core.m4_generative import build_corpus           # noqa: E402
from henri_core.model3_decoder import DecoderConfig         # noqa: E402
from henri_core.system import TriModelSystem                # noqa: E402
from henri_core.tokenizer import ByteBPE                    # noqa: E402

G_U4_BOUND = 0.95
BASE_PIN = 20261004
DIM = 65536
DMODEL = 1024


def build_full(tok, dk_target, seed, dim=DIM, d_model=DMODEL):
    cfg = DecoderConfig(dim=dim, d_model=d_model, n_layers=24, n_heads=16,
                        n_kv_heads=4, d_ffn=2816, n_macro=256, n_invariants=256,
                        vocab=tok.vocab_size, dk_target=dk_target)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        system = TriModelSystem(vocab=tok.vocab_size, dim=dim, small=False,
                                decoder_cfg=cfg, dk_target=dk_target,
                                ingress_seed=seed)
    system.eval()
    return system


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def stats(vals):
    t = torch.tensor([v for v in vals if v is not None], dtype=torch.float32)
    if t.numel() == 0:
        return {}
    return {"n": int(t.numel()), "min": float(t.min()), "max": float(t.max()),
            "mean": float(t.mean()), "std": float(t.std()),
            "pass_ge_0.95": int((t >= G_U4_BOUND).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--nseeds", type=int, default=5)
    ap.add_argument("--nseeds-control", type=int, default=5)
    ap.add_argument("--n-texts", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.set_num_threads(min(96, os.cpu_count() or 8))
    t0 = time.time()
    corpus = build_corpus(max_len=3, holdout_len=3)
    tok = ByteBPE().train(corpus.corpus_texts, vocab_size=512)
    print(f"[tok] vocab={tok.vocab_size} threads={torch.get_num_threads()} "
          f"cpus={os.cpu_count()} t={time.time()-t0:.1f}s", flush=True)

    if a.probe:
        tb = time.time()
        s = build_full(tok, 32, BASE_PIN)
        npar = sum(p.numel() for p in s.decoder.parameters())
        print(f"[probe] build dk=32 {time.time()-tb:.1f}s rss={rss_mb():.0f}MB "
              f"decoder_params={npar:,} n_mem={s.decoder.pooling.n_mem} "
              f"d_k={s.decoder.pooling.d_k}", flush=True)
        tp = time.time()
        g = gate_u4_retention(s, tok, n_texts=256)
        print(f"[probe] gate256 {time.time()-tp:.1f}s value={g.get('value')} "
              f"pos={g.get('positive_control')} sane={g.get('estimator_sane')} "
              f"rss={rss_mb():.0f}MB", flush=True)
        return 0

    rows = []
    for dkt, ns in ((32, a.nseeds), (0, a.nseeds_control)):
        for k in range(ns):
            seed = BASE_PIN + k
            tb = time.time()
            s = build_full(tok, dkt, seed)
            tg = time.time()
            g = gate_u4_retention(s, tok, n_texts=a.n_texts)
            row = {"dk_target": dkt, "seed": seed,
                   "n_mem": int(s.decoder.pooling.n_mem),
                   "d_k": int(s.decoder.pooling.d_k),
                   "g_u4": g.get("value"),
                   "pos": g.get("positive_control"),
                   "sane": g.get("estimator_sane"),
                   "build_s": round(tg - tb, 1),
                   "gate_s": round(time.time() - tg, 1)}
            rows.append(row)
            print(f"[full] dk={dkt} seed={seed} n_mem={row['n_mem']} "
                  f"d_k={row['d_k']} g_u4={row['g_u4']:.6f} "
                  f"pos={row['pos']:.6f} sane={row['sane']} "
                  f"build={row['build_s']}s gate={row['gate_s']}s "
                  f"rss={rss_mb():.0f}MB", flush=True)
            del s

    st = {lab: stats([r["g_u4"] for r in rows if r["dk_target"] == dkt])
          for lab, dkt in (("spec_b", 32), ("control_old_rule", 0))}
    pass_specb = st["spec_b"].get("pass_ge_0.95", 0) >= 4
    pass_control = st["control_old_rule"].get("pass_ge_0.95", 0) >= 4
    if pass_specb and not pass_control:
        verdict = "ACCEPT_FULL_SCALE"
    elif pass_specb and pass_control:
        verdict = "NOT_DISCRIMINATING"
    elif not pass_specb and not pass_control:
        verdict = "REJECT_NO_GAIN_FULL_SCALE"
    else:
        verdict = "PARTIAL"

    out = {
        "schema": "henri.specb.fullscale.v1",
        "verdict": verdict,
        "scale": "full_1024",
        "dim": DIM, "d_model": DMODEL,
        "gates": {"KB5_fullscale_spec_b": pass_specb,
                  "KB6_control_old_rule_fails": (not pass_control)},
        "stats_g_u4": st,
        "rows": rows,
        "run": {"pin_base": BASE_PIN, "n_texts": a.n_texts,
                "nseeds": a.nseeds, "nseeds_control": a.nseeds_control,
                "bound": G_U4_BOUND, "threads": torch.get_num_threads()},
        "device": {"plumbing": "none (cpu tensors)",
                   "peak_rss_mb": round(rss_mb(), 1)},
        "disclosure": {
            "GPU_NOT_USED": ("henri_core has no device plumbing; this run is "
                             "CPU-only on the Vast host. The GPU is not used."),
            "K_B5_SCOPE": ("This is the d_model=1024 rung of the K-B5 ladder. "
                           "A PASS adds full scale to the transfer evidence; it "
                           "is NOT an AAII capability result."),
            "G_U4_MEANING": "information-retention proxy, not task accuracy",
        },
        "seconds": round(time.time() - t0, 1),
    }
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("verdict", "gates", "stats_g_u4",
                                          "device", "seconds")}, indent=2),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
