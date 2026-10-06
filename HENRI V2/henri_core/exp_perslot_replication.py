"""Replication + artifact control for the paired per-slot result.

Problem to resolve: at full_1024 the old_rule arm reported collapse=0.9851
(pooled features nearly IDENTICAL across texts) yet ksg_mi=0.7357 (high
MI to the per-text profile). Those two cannot both be true. Either the MI
estimator is biased at 32 dims / N=256, or the collapse statistic is wrong.

This script adds the control the first gate lacked: a SHUFFLED-target MI.
If mi_shuffled ~= mi_real, the estimator is saturated and no MI claim is valid.

Also sweeps seeds, so a single-seed artifact cannot masquerade as a finding.
CPU-only. Compact JSON out.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import substrate as sub                       # noqa: E402
from henri_core.exp_gate_v2_paired import SCALES, build       # noqa: E402
from henri_core.gates_v2 import ksg_mi, ksg_mi_selftest       # noqa: E402
from henri_core.m4_generative import build_corpus             # noqa: E402
from henri_core.tokenizer import ByteBPE                      # noqa: E402


def measure(system, tok, n_texts, n_comp, seed):
    texts = [f"retrieval transfer result {i} on the ladder" for i in range(n_texts)]
    with torch.no_grad():
        psi = torch.stack([system.wave_of(t, tok) for t in texts])
        prof = (psi.abs() ** 2).reshape(n_texts, sub.N_SLOTS, -1).sum(-1)
        prof = prof / prof.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        _, tokens = system.decoder.encode_wave(psi)
        feat = tokens.mean(dim=1).float()

    fn = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    m = int(min(n_texts, 256))
    gram = (fn[:m] @ fn[:m].transpose(0, 1)).abs()
    collapse = float((gram.sum() - torch.diagonal(gram).sum()) / max(1, m * (m - 1)))

    half = n_texts // 2
    mu = feat[:half].mean(0, keepdim=True)
    sd = feat[:half].std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (feat[:half] - mu) / sd
    _, _, vh = torch.linalg.svd(Xtr, full_matrices=False)
    P = vh[:max(1, min(n_comp, vh.shape[0]))]
    proj = ((feat - mu) / sd) @ P.T

    g = torch.Generator().manual_seed(seed + 4242)
    prof_shuf = prof[torch.randperm(n_texts, generator=g)]
    return {"collapse": collapse,
            "mi_real": ksg_mi(proj, prof),
            "mi_shuffled": ksg_mi(proj, prof_shuf),
            "collapse_shuffled_proxy": collapse}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="full_1024")
    ap.add_argument("--seeds", default="20261004,20261005,20261006")
    ap.add_argument("--n-texts", type=int, default=256)
    ap.add_argument("--n-comp", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    st = ksg_mi_selftest(seed=20261004)
    corpus = build_corpus(max_len=3, holdout_len=3)
    tok = ByteBPE().train(corpus.corpus_texts, vocab_size=512)

    rows = []
    for scale in a.scales.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            for label, dkt in (("old_rule", 0), ("spec_b", 32)):
                try:
                    system = build(tok, scale, dkt, seed)
                    r = measure(system, tok, a.n_texts, a.n_comp, seed)
                    r.update({"scale": scale, "arm": label, "seed": seed,
                              "n_mem": int(system.decoder.pooling.n_mem),
                              "d_k": int(system.decoder.pooling.d_k)})
                    del system
                    gc.collect()
                except Exception as e:                    # noqa: BLE001
                    r = {"scale": scale, "arm": label, "seed": seed,
                         "status": "BLOCKED", "error": f"{type(e).__name__}: {e}"}
                rows.append(r)
                print(json.dumps(r), flush=True)

    def agg(arm):
        v = [r for r in rows if r.get("arm") == arm and "collapse" in r]
        if not v:
            return {}
        def mm(k):
            xs = [r[k] for r in v if r[k] == r[k]]
            return {"mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs), "n": len(xs)}
        return {"n_mem": v[0]["n_mem"], "d_k": v[0]["d_k"],
                "collapse": mm("collapse"), "mi_real": mm("mi_real"),
                "mi_shuffled": mm("mi_shuffled")}

    A = {"old_rule": agg("old_rule"), "spec_b": agg("spec_b")}
    ctrl_ok = all(A[k].get("mi_shuffled", {}).get("mean", 1.0) <= 0.15
                  for k in ("old_rule", "spec_b") if A[k])
    out = {"schema": "henri.perslot_replication.v1", "pin": 20261005,
           "purpose": "resolve collapse-vs-MI contradiction; add shuffled-target MI control",
           "ksg_selftest": st,
           "aggregate": A,
           "mi_shuffle_control_passes": ctrl_ok,
           "reading": ("if mi_shuffled ~= mi_real the MI is an artifact; "
                       "if mi_shuffled ~= 0 the MI is measured"),
           "rows": rows, "n_texts": a.n_texts, "n_comp": a.n_comp}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"aggregate": A, "mi_shuffle_control_passes": ctrl_ok,
                      "ksg_selftest_sane": st["estimator_sane"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
