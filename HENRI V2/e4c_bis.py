"""E4c-bis — ATTRIBUTION DISAMBIGUATION: is the E4c degradation fundamental or
a destructive-hyperparameter artifact?

E4c measured (OBSERVED):
  arm b (tied, ZERO training)  P@1 0.4410   <- best
  arm c (tied, teacher init, TRAINED lr3e-4 wd1e-4): 0.441 -> 0.321, loss 0.99 -> 2.55
  arm a (untied, rand init, TRAINED):                0.1910
Two readings are possible and E4c cannot separate them:
  (H_FUND) training on frozen-backbone features cannot help at this layer
  (H_HYPER) my hyperparameters were destructive (wd shrinks the pretrained
            solution; lr too high; 1000 steps is destructive fine-tuning)

THIS CARRIER separates them on the SAME cached representation:
  b        tied teacher, frozen                       (reference)
  c0       tied teacher, trained wd=0 lr=1e-5         (gentle)
  c1       tied teacher, trained wd=0 lr=1e-4         (gentle, faster)
  c2       tied teacher, trained wd=1e-4 lr=3e-4      (exact E4c arm c, repro)
  r0       RESIDUAL identity adapter: logits = h@E^T + h@D^T, D zero-init, wd=0
           (starts mathematically EQUAL to arm b, so it cannot degrade below it
            by construction at step 0; tests whether ANY trainable gain exists)
The residual arm is the decisive one: it is initialized so that training can
only change things by leaving the frozen solution.

Gates:
  ATTRIB_HYPER   if any gentle/residual arm BEATS arm b
  ATTRIB_FUND    if NO trainable configuration beats arm b within noise
Zero backbone training (features cached). Fail-closed on SHAs.

Usage (remote):
  HENRI_BACKBONE=1 python e4c_bis.py --model-dir /root/e4c/qwen05b \
     --corpus /root/e2-corpus/wikitext2_train.parquet \
     --teacher /root/e1-calib/teacher_embeddings.pt --out /root/e4c/e4c_bis.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECT_SHARD_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
EXPECT_SHARD_BYTES = 988097824
CORPUS_PREFIX = "e83889ba"
VOCAB, HIDDEN = 151936, 896
CTX = 128
C2_CAL = (300_000, 320_000)
C2_EVAL = (600_000, 601_000)
BATCH, SEED = 32, 20260910
FLAG = "HENRI_BACKBONE"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if os.environ.get(FLAG) != "1":
        print("E4CBIS_VERDICT=BLOCKED_INFRA reason=FLAG_OFF")
        raise SystemExit(2)

    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    import pyarrow.parquet as pq

    shard = args.model_dir / "model.safetensors"
    if (sha256_file(shard) != EXPECT_SHARD_SHA
            or shard.stat().st_size != EXPECT_SHARD_BYTES):
        print("E4CBIS_VERDICT=BLOCKED_INFRA reason=SHARD")
        raise SystemExit(2)
    csha = sha256_file(args.corpus)
    if not csha.startswith(CORPUS_PREFIX):
        print("E4CBIS_VERDICT=BLOCKED_INFRA reason=CORPUS")
        raise SystemExit(2)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    print(f"[bis] backbone loaded device={dev}", flush=True)

    @torch.no_grad()
    def feats(ids_list):
        """Correct path (A_raw, verified in E4c: max|diff| = 0.0)."""
        B = len(ids_list)
        L = max(len(x) for x in ids_list)
        inp = torch.zeros((B, L), dtype=torch.long, device=dev)
        for j, x in enumerate(ids_list):
            inp[j, :len(x)] = torch.tensor(x, device=dev)
        o = model(input_ids=inp, output_hidden_states=True)
        ar = torch.arange(B, device=dev)
        last = torch.tensor([len(x) - 1 for x in ids_list], device=dev)
        return o.hidden_states[-1][ar, last, :].float()

    rows = pq.read_table(str(args.corpus)).column("text").to_pylist()
    stream: list[int] = []
    for r in rows:
        stream.extend(tok.encode(r).ids)
        if len(stream) >= C2_EVAL[1] + 8:
            break
    print(f"[bis] stream {len(stream)}", flush=True)

    def extract(rng):
        idx = list(range(max(rng[0], CTX), rng[1]))
        H = []
        for i in range(0, len(idx), 64):
            H.append(feats([stream[j - CTX:j] for j in idx[i:i + 64]]).cpu())
        return torch.cat(H, 0), torch.tensor([stream[j] for j in idx],
                                             dtype=torch.long), idx

    Xc, gc, _ = extract(C2_CAL)
    Xe, ge, idxe = extract(C2_EVAL)
    print(f"[bis] calib {tuple(Xc.shape)} eval {tuple(Xe.shape)}", flush=True)

    t = torch.load(str(args.teacher), map_location="cpu", weights_only=True)
    E = (t["weight"] if isinstance(t, dict) and "weight" in t else t).float()
    assert tuple(E.shape) == (VOCAB, HIDDEN), "TEACHER_SHAPE"

    @torch.no_grad()
    def ev(logits_fn, params):
        p1 = p5 = n = 0
        nll = 0.0
        for i in range(0, Xe.shape[0], 256):
            xb, gb = Xe[i:i + 256].to(dev), ge[i:i + 256].to(dev)
            l = logits_fn(xb)
            if not torch.isfinite(l).all():
                raise RuntimeError("NON_FINITE")
            t5 = torch.topk(l, 5, dim=-1).indices
            p1 += int((t5[:, 0] == gb).sum())
            p5 += int((t5 == gb.unsqueeze(1)).any(1).sum())
            nll += float(F.cross_entropy(l.float(), gb, reduction="sum"))
            n += xb.shape[0]
        return {"p_at_1": round(p1 / n, 6), "p_at_5": round(p5 / n, 6),
                "ce": round(nll / n, 4), "n": n}

    Edev = E.to(dev)
    results = {}

    # ---- reference: frozen tied ------------------------------------------
    results["b_frozen_tied_REFERENCE"] = ev(lambda x: x @ Edev.t(), None)
    print(f"[bis] b  frozen tied            P@1={results['b_frozen_tied_REFERENCE']['p_at_1']:.4f}",
          flush=True)

    def train_generic(tag, params, logits_fn, lr, wd, steps, notes):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        losses = []
        for s in range(steps):
            g = torch.Generator().manual_seed(SEED + s)
            b = torch.randint(0, Xc.shape[0], (BATCH,), generator=g)
            xb, gb = Xc[b].to(dev), gc[b].to(dev)
            opt.zero_grad(set_to_none=True)
            l = logits_fn(xb)
            if not torch.isfinite(l).all():
                print(f"[bis] {tag} NON_FINITE at {s}", flush=True)
                break
            F.cross_entropy(l.float(), gb).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            with torch.no_grad():
                losses.append(float(F.cross_entropy(logits_fn(xb).float(), gb)))
        r = ev(logits_fn, params)
        r.update({"lr": lr, "wd": wd, "steps": steps,
                  "loss_first": round(losses[0], 4) if losses else None,
                  "loss_last": round(losses[-1], 4) if losses else None,
                  "notes": notes})
        results[tag] = r
        print(f"[bis] {tag:34s} P@1={r['p_at_1']:.4f} P@5={r['p_at_5']:.4f} "
              f"loss {r['loss_first']}->{r['loss_last']}", flush=True)
        return r

    # ---- gentle tied variants (the H_HYPER test) -------------------------
    for tag, lr, wd, steps in (("c0_tied_gently_lr1e-5_wd0", 1e-5, 0.0, 1000),
                               ("c1_tied_gently_lr1e-4_wd0", 1e-4, 0.0, 1000),
                               ("c2_tied_E4C_repro_lr3e-4_wd1e-4", 3e-4, 1e-4, 1000)):
        W = nn.Parameter(Edev.clone())
        train_generic(tag, [W], lambda x, W=W: x @ W.t(), lr, wd, steps,
                      "tied teacher init, trained")

    # ---- RESIDUAL identity adapter (the decisive test) -------------------
    D = nn.Parameter(torch.zeros(VOCAB, HIDDEN, device=dev))
    dd = train_generic("r0_residual_zeroinit_lr1e-3_wd0",
                       [D], lambda x, D=D: x @ Edev.t() + x @ D.t(),
                       1e-3, 0.0, 1000,
                       "logits = h@E^T + h@D^T, D=0 at init => EQUAL to frozen b")
    dd2 = train_generic("r1_residual_zeroinit_lr1e-4_wd0",
                        [D := nn.Parameter(torch.zeros(VOCAB, HIDDEN, device=dev))],
                        lambda x, D=D: x @ Edev.t() + x @ D.t(),
                        1e-4, 0.0, 1000, "same, gentler lr")
    with torch.no_grad():
        dnorm = float(D.norm())

    ref = results["b_frozen_tied_REFERENCE"]["p_at_1"]
    best_other = max((v["p_at_1"], k) for k, v in results.items()
                     if k != "b_frozen_tied_REFERENCE")
    beats = best_other[0] > ref + 0.005

    verdict = ("ATTRIB_HYPER: a trainable configuration DOES beat the frozen "
               "readout -> the E4c degradation was destructive hyperparameters, "
               f"not a fundamental limit. best={best_other[1]} {best_other[0]:.4f} "
               f"vs frozen {ref:.4f}" if beats else
               "ATTRIB_FUND: NO trainable configuration (including a zero-init "
               "residual that starts EXACTLY equal to the frozen readout) beats "
               f"it. best={best_other[1]} {best_other[0]:.4f} vs frozen {ref:.4f} "
               "-> at this layer and scale, training on frozen-backbone features "
               "adds nothing.")

    rec = {
        "carrier": "E4c-bis",
        "purpose": "disambiguate fundamental-vs-hyperparameter attribution of E4c",
        "stream": len(stream), "calib_region": list(C2_CAL),
        "eval_region": list(C2_EVAL),
        "results": results,
        "residual_final_norm": round(dnorm, 6),
        "reference_frozen_p1": ref,
        "best_nonfrozen": {"tag": best_other[1], "p1": best_other[0]},
        "VERDICT": verdict,
        "labels": ["OBSERVED", "CONDITIONAL_FRESH_SPLIT",
                   "CONDITIONAL_CONTAMINATION_UNDISCLOSED"],
        "note": ("the residual arm is identity-initialized: at step 0 its logits "
                 "EQUALS the frozen readout, so it can only help by leaving that "
                 "solution. If it still does not beat frozen, the layer is "
                 "saturated by the backbone."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    print("\n" + verdict)
    print(f"wrote {args.out}")
    print(f"sha256 {sha256_file(args.out)}")


if __name__ == "__main__":
    sys.exit(main())
