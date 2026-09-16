#!/usr/bin/env python3
"""M1 REPLACEMENT METRIC — the gate that a near-orthogonal control CANNOT win.

WHY THIS EXISTS
  The audit's M1 gate ("count of distinct top-1 tokens > floor") was measured on
  2026-09-16 and FAILED, for an instructive reason: the RANDOM-WAVE control
  out-scored the treatment (71/120 and 84/120 vs 28/120 and 14/120). Distinctness
  measures DISPERSION, not semantics -- random vectors in 2048 real dims are
  mutually near-orthogonal so they spread, while prompts from 12 templates
  collapse. That metric is retired.

DESIGN OF THE REPLACEMENT (declared BEFORE measurement)

  Metric: In-Prompt Word Recovery (IPWR).
    For prompt p, take its wave, read the codebook logits, and ask whether the
    arg-max manifest word is one of the CONTENT words of p.

  THE TRAP THIS DESIGN AVOIDS: every prompt contains "the", "to", "a". If the
  candidate set included those, a CONSTANT top-1 would score 1.0 forever and the
  gate would be winnable by a constant. So candidates are restricted to words
  whose prompt-frequency is BELOW 50%. A constant answer over that set is
  capped below 0.5 BY CONSTRUCTION, so it cannot reach the floor.

  Pre-registered:
    R1 deterministic              same prompt twice -> identical top-1  == 1.00
    R2 IPWR (treatment)                                              >= 0.50
    R3 RANDOM-WAVE control IPWR                                     <= 0.15
    R4 SHUFFLED-LABEL null IPWR (labels permuted across prompts)     <= 0.15
    R5 treatment - max(control, null)                                >= 0.35
    R6 paired margin: mean logit(in-prompt cands) - mean(out)        >  0
  FALSIFIER: if R3 or R4 reaches the R2 floor, the metric is declared INVALID.

  Runs both position_binding modes (evidence for the open default decision).
"""
import json
import os
import random
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

import henri_vla_tokenizer as vt
from m1_open_answer_gate import (TEMPLATES, VOCAB, SEED, build_prompts,
                                 build_config)

N_PROMPTS = 120
R2_FLOOR = 0.50
R3_CEIL = 0.15
R4_CEIL = 0.15
R5_MARGIN = 0.35
FREQ_CUTOFF = 0.50


def content_words(prompts, n):
    """Words present in SOME but not most prompts. Frequency < FREQ_CUTOFF."""
    toks = [set(p.split()) for p in prompts]
    freq = {}
    for s in toks:
        for w in s:
            freq[w] = freq.get(w, 0) + 1
    cands = {w for w, c in freq.items() if (c / n) < FREQ_CUTOFF}
    return cands, [s & cands for s in toks], freq


def measure(cfg, tok, code, prompts, targets, manifest, waves=None):
    with torch.no_grad():
        if waves is None:
            waves = tok.encode_text(list(prompts))
            waves_b = tok.encode_text(list(prompts))   # true same-input-twice
        else:
            waves_b = waves                            # control: same waves again
        logits = code.logits(waves)
        top1 = logits.argmax(dim=-1).tolist()
        logits2 = code.logits(waves_b)
        top1b = logits2.argmax(dim=-1).tolist()

    hits, margins = 0, []
    idx_of = {w: i for i, w in enumerate(manifest)}
    for i, tset in enumerate(targets):
        if not tset:
            continue
        pred = manifest[top1[i]]
        hits += int(pred in tset)
        in_idx = [idx_of[w] for w in tset if w in idx_of]
        out_idx = [j for j, w in enumerate(manifest)
                   if w not in tset and w != "the"]
        if in_idx and out_idx:
            margins.append(float(logits[i][in_idx].mean().item()
                                 - logits[i][out_idx].mean().item()))
    n = sum(1 for t in targets if t)
    det = sum(1 for a, b in zip(top1, top1b) if a == b) / len(prompts)
    return dict(
        n_scored=n,
        ipwr=(hits / n if n else 0.0),
        determinism=det,
        paired_margin=(sum(margins) / len(margins) if margins else None),
        distinct=len(set(top1)),
    )


def run_arm(mode, prompts, targets, manifest, rng):
    cfg = build_config(len(VOCAB), mode)
    tok = vt.HoloVLATokenizer(cfg)
    code = vt.HoloEgressCodebook(cfg, tok, list(manifest))

    treat = measure(cfg, tok, code, prompts, targets, manifest)

    # RANDOM-WAVE control: same codebook, same shape of input, no content.
    g = torch.Generator().manual_seed(SEED + 11)
    rw = torch.randn(len(prompts), cfg.ambient_dim_D, generator=g).to(torch.complex64)
    rw = rw / rw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rand = measure(cfg, tok, code, prompts, targets, manifest, waves=rw)

    # SHUFFLED-LABEL null: keep waves, permute the label sets across prompts.
    perm = list(range(len(prompts)))
    rng.shuffle(perm)
    shuffled_targets = [targets[p] for p in perm]
    null = measure(cfg, tok, code, prompts, shuffled_targets, manifest)

    return dict(position_binding=mode, treatment=treat, random_control=rand,
                shuffled_null=null,
                chance=sum(len(t) for t in targets if t) /
                       (sum(1 for t in targets if t) * len(manifest)))


def verdict(a):
    t, rc, nl = a["treatment"], a["random_control"], a["shuffled_null"]
    r1 = t["determinism"] >= 1.0
    r2 = t["ipwr"] >= R2_FLOOR
    r3 = rc["ipwr"] <= R3_CEIL
    r4 = nl["ipwr"] <= R4_CEIL
    worst = max(rc["ipwr"], nl["ipwr"])
    r5 = (t["ipwr"] - worst) >= R5_MARGIN
    r6 = (t["paired_margin"] or 0.0) > 0.0
    invalid = (rc["ipwr"] >= R2_FLOOR) or (nl["ipwr"] >= R2_FLOOR)
    if invalid:
        label = "METRIC_INVALID_CONTROL_REACHED_FLOOR"
    elif r1 and r2 and r3 and r4 and r5 and r6:
        label = "M1_REPLACEMENT_GATE_PASS"
    else:
        failed = [k for k, v in (("R1", r1), ("R2", r2), ("R3", r3),
                                 ("R4", r4), ("R5", r5), ("R6", r6)) if not v]
        label = "M1_REPLACEMENT_GATE_FAIL:" + ",".join(failed)
    return dict(R1_determinism=r1, R2_ipwr_floor=r2, R3_random_ceiling=r3,
                R4_null_ceiling=r4, R5_separation=r5, R6_margin_positive=r6,
                control_max=worst, verdict=label)


def main():
    rng = random.Random(SEED)
    prompts = build_prompts(N_PROMPTS, rng)
    cands, targets, freq = content_words(prompts, N_PROMPTS)
    manifest = sorted(VOCAB)

    print("PRE-REGISTERED (replacement metric)")
    print(f"  N_prompts           = {N_PROMPTS}")
    print(f"  manifest size       = {len(manifest)}")
    print(f"  candidate words     = {len(cands)} (prompt-frequency < {FREQ_CUTOFF})")
    print(f"  R2 IPWR treatment   >= {R2_FLOOR}")
    print(f"  R3 random control   <= {R3_CEIL}")
    print(f"  R4 shuffled null    <= {R4_CEIL}")
    print(f"  R5 separation       >= {R5_MARGIN}")
    print("  R6 paired margin    >  0")
    print("  FALSIFIER: control or null reaching the R2 floor => METRIC INVALID")
    print()
    print(f"  NOTE stopwords excluded so a CONSTANT top-1 cannot pass:")
    top_const = sorted(freq.items(), key=lambda kv: -kv[1])[:6]
    print(f"    highest-frequency tokens: {top_const}")
    print()

    out = {}
    for mode in ("fractional_shift", "phasor_bind"):
        a = run_arm(mode, prompts, targets, manifest, random.Random(SEED))
        a.update(verdict(a))
        out[mode] = a
        print(f"=== ARM: {mode} ===")
        print(f"  {'metric':18s} {'treatment':>10s} {'random':>10s} {'null':>10s}")
        for k in ("ipwr", "determinism", "distinct"):
            print(f"  {k:18s} {a['treatment'][k]:>10.4f} "
                  f"{a['random_control'][k]:>10.4f} {a['shuffled_null'][k]:>10.4f}")
        print(f"  {'paired_margin':18s} {a['treatment']['paired_margin']:>10.4f}")
        print(f"  chance (analytic)  = {a['chance']:.4f}")
        print(f"  VERDICT            = {a['verdict']}")
        print()

    print("=== DECISION EVIDENCE (position_binding) ===")
    fs, pb = out["fractional_shift"], out["phasor_bind"]
    print(f"  IPWR           fractional_shift={fs['treatment']['ipwr']:.4f}  "
          f"phasor_bind={pb['treatment']['ipwr']:.4f}")
    print(f"  paired_margin  fractional_shift={fs['treatment']['paired_margin']:.4f}  "
          f"phasor_bind={pb['treatment']['paired_margin']:.4f}")
    print()
    print("=== M1 REPLACEMENT OVERALL ===")
    for m, a in out.items():
        print(f"  {m:18s} {a['verdict']}")
    passed = [m for m, a in out.items()
              if a["verdict"] == "M1_REPLACEMENT_GATE_PASS"]
    print(f"  arms passing = {passed}")

    dest = Path(os.environ.get("LOCALAPPDATA", ".")) / "Temp" / "m1_replacement_receipt.json"
    dest.write_text(json.dumps(dict(
        preregistration=dict(N=N_PROMPTS, R2=R2_FLOOR, R3=R3_CEIL, R4=R4_CEIL,
                             R5=R5_MARGIN, freq_cutoff=FREQ_CUTOFF, seed=SEED,
                             note="distinct-count metric retired 2026-09-16"),
        n_candidates=len(cands), arms=out), indent=2), encoding="utf-8")
    print(f"  receipt: {dest}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
