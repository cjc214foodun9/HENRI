"""G4 ARM-H — Hopfield lexical-snap calibration over K5 codec waves (v3, batched).

Read-only: IN-MEMORY engrams from corpus vocabulary; never writes DB,
never saves a checkpoint, never enables a flag.

Metrics per M:
  exact           : P(engram i snaps to i)  [codec->Hopfield word egress recall]
  noise_eps{X}    : P(after adding eps*|unit noise|, snaps back to original word)
  distinct_word   : P(w+s snaps to w)  [edit-local discrimination; report as measured]
  random_neg_conf : mean top-1 raw sim for 50 random unit vectors (null floor)
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from zone_c_world_knowledge_codec import CompositionalTextCodec  # noqa: E402
from hopfield_cleanup import ContinuousHopfieldCleanup  # noqa: E402

SOURCE_GLOBS = [
    "/workspace/k5-sources/computing/*.rst",
    "/workspace/k5-sources/*.txt",
]
DIM = 65536


def load_corpus_vocab() -> list[str]:
    seen: list[str] = []
    uniq: set[str] = set()
    for pat in SOURCE_GLOBS:
        for f in sorted(glob.glob(pat)):
            try:
                txt = open(f, encoding="utf-8", errors="replace").read().lower()
            except Exception:
                continue
            for w in txt.split():
                if w.isalpha() and len(w) >= 4 and w not in uniq:
                    uniq.add(w)
                    seen.append(w)
    return seen


def wave_of(codec: CompositionalTextCodec, text: str) -> torch.Tensor:
    w_bytes, _ = codec.encode(text)
    w = torch.from_numpy(np.frombuffer(w_bytes, dtype=np.float32).copy()).reshape(1, -1)
    return w / (w.norm() + 1e-12)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = CompositionalTextCodec()
    vocab = load_corpus_vocab()
    print(f"[g4-armh] corpus_unique_words={len(vocab)} device={device}", flush=True)
    results: dict = {"corpus_unique_words": len(vocab), "device": device, "M": {}}

    for M in (250, 1000, 2500, 5000, 10000):
        n = min(M, len(vocab))
        sel = vocab[:n]
        engr = torch.cat([wave_of(codec, w) for w in sel], dim=0).to(device)  # [n, D]
        h = ContinuousHopfieldCleanup(dim=DIM, beta=8.0)
        t0 = time.time()
        h.store_engrams(engr)
        store_ms = (time.time() - t0) * 1000

        # (a) exact recall
        idxs, confs = h.lexical_snap(engr, top_k=1)
        exact_p = float((idxs.reshape(-1) == torch.arange(n, device=device)).float().mean().item())

        # (b) noise robustness (batched, seeded)
        B = min(300, n)
        qs = engr[:B]
        noise_stats = {}
        for eps in (0.05, 0.20, 0.50):
            g = torch.Generator(device="cpu").manual_seed(20260905 + n + int(eps * 100))
            noise = torch.randn(B, DIM, generator=g, device="cpu")
            noise = F.normalize(noise, p=2, dim=-1).to(device)
            jit = F.normalize(qs + eps * noise, p=2, dim=-1)
            idx, _ = h.lexical_snap(jit, top_k=1)
            ok = float((idx.reshape(-1) == torch.arange(B, device=device)).float().mean().item())
            noise_stats[f"eps{eps}"] = round(ok, 4)

        # (c) edit-local discrimination: w+s must snap back to w (measured)
        B2 = min(100, n)
        alt = [sel[i] + "s" for i in range(B2)]
        qalt = torch.cat([wave_of(codec, w) for w in alt], dim=0).to(device)
        aidx, aconf = h.lexical_snap(qalt, top_k=1)
        disc_p = float((aidx.reshape(-1) == torch.arange(B2, device=device)).float().mean().item())

        # (d) random-negative null floor
        rng = np.random.default_rng(20260905 + n)
        neg = []
        for _ in range(50):
            v = rng.normal(size=DIM).astype(np.float32)
            v = torch.from_numpy(v)
            v = v / (v.norm() + 1e-12)
            nidx, nconf = h.lexical_snap(v.to(device).unsqueeze(0), top_k=1)
            neg.append(float(nconf.reshape(-1)[0].item()))
        neg_mean = float(np.mean(neg))

        results["M"][str(M)] = {
            "n_engrams": n,
            "store_ms": round(store_ms, 2),
            "p_exact": round(exact_p, 4),
            "p_noise": noise_stats,
            "p_edit1_disc": round(disc_p, 4),
            "random_neg_mean_conf": round(neg_mean, 6),
        }
        print(f"[g4-armh] M={n} exact={exact_p:.4f} noise={noise_stats} edit1={disc_p:.4f} neg={neg_mean:.6f}", flush=True)

    with open("/tmp/g4_armh_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
