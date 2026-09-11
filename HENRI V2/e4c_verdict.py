"""E4c VERDICT RUN — construct viability + 2x2 factorial, single clean path.

RESOLVES (all previously measured this session, disclosed):
  * e4c_extract_features.py applied model.model.norm to hidden_states[-1]. Probe 1
    showed that path diverges from the model's own logits by max|diff|=5.81, so
    the produced features.pt is INVALID. This script determines the correct path
    EMPIRICALLY (compare lm_head(h) against o.logits) and uses it.
  * e4c_oracle_probe.py crashed on a zero-length C2 slice (texts[:3000] did not
    yield enough tokens for the chosen region). Regions here are guarded.
  * C1 (sentence-window) ORACLE P@1 = 0.2690 vs its context-free marginal 0.440:
    the real model's own argmax cannot beat "always emit ' .'". This script
    re-measures both constructs and decides which is admissible.

DELIVERABLE
  1. correct hidden-state path (A raw / B extra-norm), empirically
  2. C2 (token-stream) baselines + oracle on a FRESH region
  3. C1 oracle reconfirm
  4. if C2 is viable: extract frozen-backbone features and run the 2x2 factorial
     (a untied-trainable, b tied-frozen, c tied-trainable, d untied-frozen)
  5. verdict for E4c

Zero backbone training. Backbone is eval-only, torch.no_grad(), zero trainable.

Usage (remote):
  HENRI_BACKBONE=1 python e4c_verdict.py --model-dir /root/e4c/qwen05b \
      --corpus /root/e2-corpus/wikitext2_train.parquet \
      --teacher /root/e1-calib/teacher_embeddings.pt --out /root/e4c/e4c_verdict.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECT_SHARD_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
EXPECT_SHARD_BYTES = 988097824
EXPECT_CORPUS_SHA_PREFIX = "e83889ba"
VOCAB, HIDDEN = 151936, 896
MAX_WORDS = 24
CTX = 128
C1_EVAL = (21000, 22000)
C2_CAL = (300_000, 320_000)
C2_EVAL = (600_000, 601_000)
STEPS, BATCH, LR, WD = 1000, 32, 3e-4, 1e-4
SEED = 20260910
MOVE, ATTR_TOL = 0.05, 0.02
FLAG = "HENRI_BACKBONE"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sentence_split(t: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", t.replace("\n", " ").replace("  ", " "))
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if os.environ.get(FLAG) != "1":
        print("E4C_VERDICT=BLOCKED_INFRA reason=FLAG_OFF")
        raise SystemExit(2)

    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    import pyarrow.parquet as pq

    shard = args.model_dir / "model.safetensors"
    if (sha256_file(shard) != EXPECT_SHARD_SHA
            or shard.stat().st_size != EXPECT_SHARD_BYTES):
        print("E4C_VERDICT=BLOCKED_INFRA reason=SHARD_SHA_OR_BYTES")
        raise SystemExit(2)
    csha = sha256_file(args.corpus)
    if not csha.startswith(EXPECT_CORPUS_SHA_PREFIX):
        print("E4C_VERDICT=BLOCKED_INFRA reason=CORPUS_SHA")
        raise SystemExit(2)
    if not args.teacher.exists():
        print("E4C_VERDICT=BLOCKED_INFRA reason=TEACHER_MISSING")
        raise SystemExit(2)
    tok_sha = sha256_file(args.model_dir / "tokenizer.json")
    print(f"[e4c] shard {EXPECT_SHARD_SHA[:16]} OK  corpus {csha[:16]}  "
          f"tokenizer {tok_sha[:16]}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    print(f"[e4c] backbone params={sum(p.numel() for p in model.parameters())} "
          f"device={dev}", flush=True)

    @torch.no_grad()
    def run(ids_list: list[list[int]]):
        """(lg_model, hA_raw, hB_norm, lgA, lgB) at each sequence's last real token."""
        B = len(ids_list)
        L = max(len(x) for x in ids_list)
        inp = torch.zeros((B, L), dtype=torch.long, device=dev)
        for j, x in enumerate(ids_list):
            inp[j, :len(x)] = torch.tensor(x, device=dev)
        o = model(input_ids=inp, output_hidden_states=True)
        ar = torch.arange(B, device=dev)
        last = torch.tensor([len(x) - 1 for x in ids_list], device=dev)
        lg_model = o.logits[ar, last, :].float()
        hs = o.hidden_states[-1][ar, last, :]
        hA = hs.float()
        hB = model.model.norm(hs.float())
        return (lg_model, hA, hB,
                model.lm_head(hA.to(torch.bfloat16)).float(),
                model.lm_head(hB.to(torch.bfloat16)).float())

    rec: dict = {"carrier": "E4c", "tokenizer_sha256": tok_sha,
                 "shard_sha256": EXPECT_SHARD_SHA, "corpus_sha256": csha}

    # ---------- 1. correct hidden path -------------------------------------
    prompts = ["The capital of France is",
               "In 1865 the Union general Gordon Granger arrived in Galveston ,",
               "The church was dedicated in April 1860 to",
               "Water is composed of hydrogen and",
               "The quick brown fox jumps over the",
               "The population of Texas in 1860 was"]
    lg, hA, hB, lgA, lgB = run([tok.encode(t).ids for t in prompts])
    dA, dB = float((lg - lgA).abs().max()), float((lg - lgB).abs().max())
    icA = int((lg.argmax(-1) == lgA.argmax(-1)).sum())
    icB = int((lg.argmax(-1) == lgB.argmax(-1)).sum())
    path = "A_raw" if (dA <= dB and icA >= icB) else "B_extra_norm"
    print(f"[1] raw-path max|diff|={dA:.4f} argmax={icA}/6 | "
          f"extra-norm max|diff|={dB:.4f} argmax={icB}/6 -> use {path}", flush=True)
    for t, i in zip(prompts, lg.argmax(-1).tolist()):
        print(f"    {t!r:62} -> {tok.decode([i])!r}", flush=True)
    rec["1_hidden_path"] = {"raw_max_diff": round(dA, 4), "raw_argmax": f"{icA}/6",
                            "extra_norm_max_diff": round(dB, 4),
                            "extra_norm_argmax": f"{icB}/6", "chosen": path}
    use_raw = path == "A_raw"

    def feats_o(ids_list):
        _, hA_, hB_, _, _ = run(ids_list)
        return hA_ if use_raw else hB_

    # ---------- stream ------------------------------------------------------
    rows = pq.read_table(str(args.corpus)).column("text").to_pylist()
    need = C2_EVAL[1] + 8
    stream: list[int] = []
    used = 0
    for r in rows:
        stream.extend(tok.encode(r).ids)
        used += 1
        if len(stream) >= need:
            break
    print(f"[2] stream={len(stream)} tokens from {used} rows (need {need})", flush=True)
    if len(stream) < C2_EVAL[1]:
        print(f"E4C_VERDICT=BLOCKED_CONSTRUCT reason=STREAM_SHORT {len(stream)}")
        raise SystemExit(2)

    # ---------- C2 baselines + oracle --------------------------------------
    base = stream[:C2_CAL[0]]
    uni = Counter(base)
    uni_top = [t for t, _ in uni.most_common(8)]
    m1: dict = {}
    m3: dict = {}
    for i in range(3, len(base)):
        m1.setdefault(base[i - 1], Counter())[base[i]] += 1
        m3.setdefault((base[i - 3], base[i - 2], base[i - 1]), Counter())[base[i]] += 1

    idxs = list(range(max(C2_EVAL[0], CTX), C2_EVAL[1]))
    o1 = o5 = u1 = u5 = l1 = l5 = g1 = g5 = 0
    for i in range(0, len(idxs), 32):
        ch = idxs[i:i + 32]
        lgc, _, _, _, _ = run([stream[j - CTX:j] for j in ch])
        gold = torch.tensor([stream[j] for j in ch], device=dev)
        t5 = torch.topk(lgc, 5, dim=-1).indices
        o1 += int((t5[:, 0] == gold).sum())
        o5 += int((t5 == gold.unsqueeze(1)).any(1).sum())
        for j in ch:
            g = stream[j]
            u1 += int(uni_top[0] == g)
            u5 += int(g in uni_top[:5])
            c1_ = m1.get(stream[j - 1])
            l_ = ([t for t, _ in c1_.most_common(8)] if c1_ else []) + uni_top
            l1 += int(l_[0] == g)
            l5 += int(g in l_[:5])
            c3_ = m3.get((stream[j - 3], stream[j - 2], stream[j - 1]))
            g_ = ([t for t, _ in c3_.most_common(8)] if c3_ else []) + l_
            g1 += int(g_[0] == g)
            g5 += int(g in g_[:5])
    n = len(idxs)
    c2 = {"n": n, "ctx": CTX, "eval_region": list(C2_EVAL), "cal_region": [0, C2_CAL[0]],
          "oracle_p1": round(o1 / n, 4), "oracle_p5": round(o5 / n, 4),
          "unigram_p1": round(u1 / n, 4), "unigram_p5": round(u5 / n, 4),
          "lasttok_p1": round(l1 / n, 4), "lasttok_p5": round(l5 / n, 4),
          "trigram_p1": round(g1 / n, 4), "trigram_p5": round(g5 / n, 4)}
    print(f"[3] C2 ORACLE   P@1={c2['oracle_p1']:.4f} P@5={c2['oracle_p5']:.4f}",
          flush=True)
    print(f"    unigram    P@1={c2['unigram_p1']:.4f} | lasttok {c2['lasttok_p1']:.4f}"
          f" | trigram {c2['trigram_p1']:.4f}", flush=True)
    rec["2_C2_token_stream"] = c2

    # ---------- C1 oracle reconfirm ----------------------------------------
    sents: list[str] = []
    for r in rows:
        sents.extend(sentence_split(r))
        if len(sents) >= 26000:
            break
    win = []
    for s in sents:
        w = s.split()[:MAX_WORDS]
        if len(w) < 2:
            continue
        prefix, window = " ".join(w[:-1]), " ".join(w)
        ia, ip = tok.encode(window).ids, tok.encode(prefix).ids
        if ia[:len(ip)] == ip and len(ia) > len(ip):
            win.append((prefix, ia[len(ip)]))
    c1s = win[C1_EVAL[0]:C1_EVAL[1]]
    p1 = p5 = 0
    golds = Counter()
    for i in range(0, len(c1s), 32):
        ch = c1s[i:i + 32]
        lgc, _, _, _, _ = run([tok.encode(p).ids for p, _ in ch])
        g = torch.tensor([x for _, x in ch], device=dev)
        t5 = torch.topk(lgc, 5, dim=-1).indices
        p1 += int((t5[:, 0] == g).sum())
        p5 += int((t5 == g.unsqueeze(1)).any(1).sum())
        for _, x in ch:
            golds[x] += 1
    n1 = len(c1s)
    top3 = [(int(t), round(c / n1, 4), tok.decode([t])) for t, c in golds.most_common(3)]
    print(f"[4] C1 ORACLE   P@1={p1/n1:.4f} P@5={p5/n1:.4f} n={n1} top3={top3}",
          flush=True)
    rec["3_C1_sentence_window"] = {
        "n": n1, "oracle_p1": round(p1 / n1, 4), "oracle_p5": round(p5 / n1, 4),
        "e4a_marginal_p1": 0.440, "gold_top3": top3}

    c2_base = max(c2["unigram_p1"], c2["trigram_p1"], c2["lasttok_p1"])
    c2_viable = c2["oracle_p1"] >= c2_base + MOVE
    c1_viable = (p1 / n1) >= 0.440 + MOVE
    rec["4_construct"] = {
        "C2_strongest_baseline_p1": c2_base, "C2_oracle_viable": bool(c2_viable),
        "C1_oracle_viable": bool(c1_viable),
        "reading": ("C1 is DEGENERATE: the real LM's own argmax cannot beat the "
                    "context-free marginal (the removed 'word' is a bare period "
                    "~44% of the time, which a constant predictor wins). C2 "
                    "(token-stream next-token) is the admissible construct."
                    if (not c1_viable and c2_viable) else "see fields")}
    print(f"[4] construct: C1_viable={c1_viable} C2_viable={c2_viable} "
          f"C2_strongest_base={c2_base:.4f}", flush=True)

    if not c2_viable:
        rec["verdict"] = "BLOCKED_CONSTRUCT"
        args.out.write_text(json.dumps(rec, indent=2))
        print("\nE4C_VERDICT=BLOCKED_CONSTRUCT")
        return

    # ---------- extract features (correct path) ---------------------------
    def extract(rng):
        idx = list(range(max(rng[0], CTX), rng[1]))
        H = []
        for i in range(0, len(idx), 64):
            ch = idx[i:i + 64]
            H.append(feats_o([stream[j - CTX:j] for j in ch]).cpu())
        return torch.cat(H, 0), torch.tensor([stream[j] for j in idx], dtype=torch.long)

    t0 = time.time()
    Xc, gc = extract(C2_CAL)
    Xe, ge = extract(C2_EVAL)
    print(f"[5] features calib {tuple(Xc.shape)} eval {tuple(Xe.shape)} "
          f"({time.time()-t0:.1f}s)", flush=True)
    if Xc.shape[1] != HIDDEN:
        print(f"E4C_VERDICT=BLOCKED_INFRA reason=HIDDEN {Xc.shape[1]}")
        raise SystemExit(2)

    # ---------- 2x2 factorial ---------------------------------------------
    teacher = torch.load(str(args.teacher), map_location="cpu", weights_only=True)
    E = teacher["weight"] if isinstance(teacher, dict) and "weight" in teacher else teacher
    E = torch.as_tensor(E).float()
    if tuple(E.shape) != (VOCAB, HIDDEN):
        print(f"E4C_VERDICT=BLOCKED_INFRA reason=TEACHER_SHAPE {tuple(E.shape)}")
        raise SystemExit(2)

    def mk(init):
        lin = nn.Linear(HIDDEN, VOCAB, bias=False).to(dev)
        with torch.no_grad():
            if init == "teacher":
                lin.weight.data = E.to(dev).clone()
            else:
                g = torch.Generator().manual_seed(SEED)
                lin.weight.data = (torch.randn(VOCAB, HIDDEN, generator=g)
                                   * (1.0 / math.sqrt(HIDDEN))).to(dev)
        return lin

    @torch.no_grad()
    def ev(lin):
        lin.eval()
        p1 = p5 = k = 0
        nll = 0.0
        for i in range(0, Xe.shape[0], 512):
            xb, gb = Xe[i:i + 512].to(dev), ge[i:i + 512].to(dev)
            l = xb @ lin.weight.t()
            if not torch.isfinite(l).all():
                raise RuntimeError("NON_FINITE_LOGITS")
            t5 = torch.topk(l, 5, dim=-1).indices
            p1 += int((t5[:, 0] == gb).sum())
            p5 += int((t5 == gb.unsqueeze(1)).any(1).sum())
            nll += float(F.cross_entropy(l.float(), gb, reduction="sum"))
            k += xb.shape[0]
        return {"p_at_1": round(p1 / k, 6), "p_at_5": round(p5 / k, 6),
                "ce": round(nll / k, 4), "n": k}

    arms = {}
    for tag, init, trainable in (("a_untied_trainable", "rand", True),
                                 ("b_tied_frozen", "teacher", False),
                                 ("c_tied_trainable", "teacher", True),
                                 ("d_untied_frozen", "rand", False)):
        lin = mk(init)
        lin.weight.requires_grad_(bool(trainable))
        info = {"init": init, "trainable": bool(trainable),
                "trainable_params": int(lin.weight.numel()) if trainable else 0}
        if trainable:
            opt = torch.optim.AdamW([lin.weight], lr=LR, weight_decay=WD)
            losses = []
            lin.train()
            for s in range(STEPS):
                g = torch.Generator().manual_seed(SEED + s)
                b = torch.randint(0, Xc.shape[0], (BATCH,), generator=g)
                xb, gb = Xc[b].to(dev), gc[b].to(dev)
                opt.zero_grad(set_to_none=True)
                l = xb @ lin.weight.t()
                if not torch.isfinite(l).all():
                    info["aborted"] = "NON_FINITE_LOGITS"
                    break
                F.cross_entropy(l.float(), gb).backward()
                torch.nn.utils.clip_grad_norm_([lin.weight], 1.0)
                opt.step()
                losses.append(float(F.cross_entropy(
                    (xb @ lin.weight.t()).float(), gb).detach()))
                if s % 250 == 0:
                    print(f"    {tag} step {s:4d} loss {losses[-1]:.4f}", flush=True)
            if losses:
                info.update({"steps": len(losses),
                             "loss_first": round(losses[0], 4),
                             "loss_last": round(losses[-1], 4)})
        info.update(ev(lin))
        arms[tag] = info
        print(f"[6] {tag}: P@1={info['p_at_1']:.4f} P@5={info['p_at_5']:.4f} "
              f"CE={info['ce']:.4f}", flush=True)

    A, B_, C_, D_ = (arms["a_untied_trainable"], arms["b_tied_frozen"],
                     arms["c_tied_trainable"], arms["d_untied_frozen"])
    best = max(A["p_at_1"], C_["p_at_1"])
    g_move = best >= c2_base + MOVE
    g_attr = C_["p_at_1"] >= A["p_at_1"] - ATTR_TOL
    attribution = ("TRAINABILITY_DOMINANT (c>=a-tol): E3's failure was trainability, "
                   "not weight tying" if g_attr else
                   "UNTYING_HELPFUL: (a) exceeds (c) beyond tolerance")
    verdict = "E4C_GATES_PASS" if g_move else "E4C_MOVEMENT_FAIL"

    rec.update({
        "5_input_representation": "frozen backbone final hidden state [896] "
                                  f"({path})",
        "6_arms": arms,
        "7_gates": {"G-A_finite": True,
                    "G-B_C2_oracle_viable": bool(c2_viable),
                    "G-C_movement_above_C2_baseline": bool(g_move),
                    "G-D_attribution_parity": bool(g_attr)},
        "8_attribution": attribution,
        "9_kill_bound": {"p_at_1": round(c2_base + MOVE, 4),
                         "basis": "C2 strongest trivial baseline + margin"},
        "10_lean_ledger": {
            "untied_readout_trainable_params": A["trainable_params"],
            "untied_readout_fp32_bytes": A["trainable_params"] * 4,
            "untied_readout_bf16_bytes": A["trainable_params"] * 2,
            "adamw_state_fp32_bytes": A["trainable_params"] * 8,
            "tied_matrix_frozen_params": int(E.numel()),
            "frozen_backbone_params": sum(p.numel() for p in model.parameters()),
        },
        "verdict": verdict,
        "labels": ["CONDITIONAL_FRESH_SPLIT", "CONDITIONAL_CONTAMINATION_UNDISCLOSED"],
    })
    args.out.write_text(json.dumps(rec, indent=2))
    print("\n" + json.dumps(rec["7_gates"], indent=2))
    print(f"attribution: {attribution}")
    print(f"E4C_VERDICT={verdict}")
    print(f"wrote {args.out}")
    print(f"sha256 {sha256_file(args.out)}")


if __name__ == "__main__":
    sys.exit(main())
