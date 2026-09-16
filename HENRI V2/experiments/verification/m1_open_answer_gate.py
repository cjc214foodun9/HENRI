#!/usr/bin/env python3
"""M1 GATE — the pre-registered acceptance test that was never run.

FROM THE AUDIT (verbatim):
  "M1 - Calibrated open-answer egress (unlocks 40%). Measured state:
   BLOCKED_SEMANTIC_CAPACITY; top1_token_unique = 1 across 16 distinct waves.
   Gate: on a held-out probe set of N>=100 distinct prompts, the count of distinct
   top-1 tokens must exceed a pre-registered floor, and an independent equivalence
   checker must agree above chance."

The prior verdict `top1_token_unique = 1` was VACUOUS: the random control scored
37/128 distinct vs treatment 39/128 -- statistically indistinguishable. So a
distinct-token COUNT alone cannot be the gate. This script therefore makes the
verdict a function of FOUR arms, one of which is designed to falsify a naive pass.

PRE-REGISTERED (declared before measurement):
  P1 DETERMINISM   same prompt twice            -> identical top-1 for 100% of N
  P2 DISTINCT      distinct_top1 / N            -> >= 0.50            [floor]
  P3 ORDER         shuffled chars (SAME multiset) -> different top-1 for >= 0.50 of N
  P4 EQUIVALENCE   near view (whitespace-normalized) -> top-1 agrees for >= 0.50 of N
  P5 VACUITY       random waves through the SAME codebook must NOT satisfy P2,
                   otherwise distinct-count is not evidence of content.

FALSIFIER BUILT IN: if the RANDOM-WAVE arm also reaches the P2 floor, the gate is
declared VACUOUS no matter how good the treatment looks. That is the exact defect
that made the original 39-vs-37 comparison meaningless.

Runs BOTH position_binding modes, because that is the evidence for the open
decision on the tokenizer default.
"""
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import henri_vla_tokenizer as vt

# ------------------------------------------------------------ pre-registration
P1_DETERMINISM = 1.00
P2_DISTINCT_FLOOR = 0.50
P3_ORDER_FLOOR = 0.50
P4_EQUIV_FLOOR = 0.50
N_PROMPTS = 120
SEED = 20260916

WORDS = """the a an and or but if then with without into from over under above below
move grasp place push pull lift lower open close rotate turn align insert remove
block cube sphere cylinder peg hole slot tray bin table shelf drawer red blue
green yellow purple orange gray black white large small left right front back
up down near far toward away slowly quickly carefully precisely gently firmly
object target goal state action plan step phase reward path grasp_pose place_pose
success failure retry observe predict verify measure compute estimate compare
robot arm gripper joint wrist base camera image pixel frame scene context task
instruction language vision action policy model latent wave phase amplitude
frequency resonance coherence invariant symmetry conservation locality quantity
rigidity containment information entropy bound constraint manifold prior engram
encode decode bind unbind retrieve store memory knowledge backbone holographic
vector symbolic algebra clifford rotor bivector quaternion spinor sagnac
interference homodyne lattice torus koopman edmd stiefel cholesky gradient
""".split()
VOCAB = sorted(set(w for w in WORDS if w.isascii() and len(w) > 1))

TEMPLATES = [
    "move the {a} block to the {b} tray",
    "grasp the {a} cube and place it on the {b} shelf",
    "push the {a} cylinder into the {b} hole",
    "rotate the {a} peg until it aligns with the {b} slot",
    "lift the {a} object and lower it near the {b} bin",
    "verify that the {a} state matches the {b} target",
    "predict the next {a} phase from the {b} observation",
    "encode the {a} scene into a {b} wave representation",
    "measure the {a} amplitude and compare with the {b} bound",
    "plan a {a} path toward the {b} goal",
    "remove the {a} part from the {b} assembly",
    "align the {a} gripper with the {b} object",
]


def build_prompts(n, rng):
    out, seen = [], set()
    i = 0
    while len(out) < n:
        t = TEMPLATES[i % len(TEMPLATES)]
        a = rng.choice(VOCAB)
        b = rng.choice([w for w in VOCAB if w != a])
        p = t.format(a=a, b=b)
        i += 1
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def shuffled(s, rng):
    """Multiset-preserving permutation: same characters, different arrangement."""
    c = list(s)
    rng.shuffle(c)
    return "".join(c)


def near_view(s):
    """Same-origin re-rendering: a GENUINELY different byte string that carries
    the same content (extra spacing + trailing space), then re-normalized.

    DEFECT FIXED HERE (disclosed): the first version returned
    `" ".join(s.split())`, which is the IDENTITY on these already-single-spaced
    prompts -- so P4 compared every string with itself and passed at 1.0
    vacuously. A same-origin test whose two arms are the same bytes tests
    nothing. This version produces a different byte string with identical
    semantics, so the arm can actually fail.
    """
    perturbed = s.replace(" ", "  ") + "   "
    assert perturbed != s, "near view must differ from the original bytes"
    assert " ".join(perturbed.split()) == s, "near view must preserve semantics"
    return perturbed


def build_config(vocab_size, position_binding):
    return vt.HoloVLAConfig(
        ambient_dim_D=2048, num_blocks=256, block_slots=8, grid_size_S=16,
        vocab_size_V=vocab_size, feat_dim=256, position_binding=position_binding,
        seed=SEED)


def run_arm(position_binding, prompts, rng):
    cfg = build_config(len(VOCAB), position_binding)
    tok = vt.HoloVLATokenizer(cfg)
    code = vt.HoloEgressCodebook(cfg, tok, list(VOCAB))
    V = len(VOCAB)

    def top1(strings):
        with torch.no_grad():
            w = tok.encode_text(list(strings))
            lg = code.logits(w)
            return lg.argmax(dim=-1).tolist(), lg

    ids_a, lg_a = top1(prompts)
    ids_b, _ = top1(prompts)                                   # P1 determinism
    ids_shuf, _ = top1([shuffled(p, rng) for p in prompts])     # P3 order
    ids_near, _ = top1([near_view(p) for p in prompts])         # P4 equivalence

    # P5 vacuity control: random waves through the SAME codebook
    g = torch.Generator().manual_seed(SEED + 7)
    rand_waves = torch.randn(len(prompts), cfg.ambient_dim_D,
                             generator=g).to(torch.complex64)
    rand_waves = rand_waves / rand_waves.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    with torch.no_grad():
        lgr = code.logits(rand_waves)
        ids_rand = lgr.argmax(dim=-1).tolist()

    n = len(prompts)
    ent = []
    with torch.no_grad():
        pr = torch.softmax(lg_a.real, dim=-1)
        ent = (-(pr * pr.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()

    return dict(
        position_binding=position_binding, n=n, vocab=V,
        determinism=sum(1 for x, y in zip(ids_a, ids_b) if x == y) / n,
        distinct_ratio=len(set(ids_a)) / n,
        distinct_count=len(set(ids_a)),
        order_sensitivity=sum(1 for x, y in zip(ids_a, ids_shuf) if x != y) / n,
        equivalence=sum(1 for x, y in zip(ids_a, ids_near) if x == y) / n,
        rand_distinct_ratio=len(set(ids_rand)) / n,
        rand_distinct_count=len(set(ids_rand)),
        mean_entropy_nats=ent, ln_vocab=float(torch.tensor(float(V)).log()))


def verdict(a):
    p1 = a["determinism"] >= P1_DETERMINISM
    p2 = a["distinct_ratio"] >= P2_DISTINCT_FLOOR
    p3 = a["order_sensitivity"] >= P3_ORDER_FLOOR
    p4 = a["equivalence"] >= P4_EQUIV_FLOOR
    vacuous = a["rand_distinct_ratio"] >= P2_DISTINCT_FLOOR
    p5 = not vacuous
    if vacuous:
        label = "VACUOUS_DISTINCT_COUNT_NOT_INFORMATIVE"
    elif p1 and p2 and p3 and p4 and p5:
        label = "M1_GATE_PASS"
    else:
        failed = [k for k, v in (("P1", p1), ("P2", p2), ("P3", p3),
                                 ("P4", p4), ("P5", p5)) if not v]
        label = "M1_GATE_FAIL:" + ",".join(failed)
    return dict(P1_determinism=p1, P2_distinct=p2, P3_order=p3, P4_equivalence=p4,
                P5_nonvacuous=p5, verdict=label)


def main():
    rng = random.Random(SEED)
    prompts = build_prompts(N_PROMPTS, rng)
    assert len(set(prompts)) == N_PROMPTS, "prompts must be distinct"

    print("PRE-REGISTERED CRITERIA")
    print(f"  N_prompts          = {N_PROMPTS}  (gate requires >=100)")
    print(f"  vocab (manifest)   = {len(VOCAB)} real words")
    print(f"  P1 determinism     >= {P1_DETERMINISM}")
    print(f"  P2 distinct top1   >= {P2_DISTINCT_FLOOR}")
    print(f"  P3 order-sensitive >= {P3_ORDER_FLOOR}")
    print(f"  P4 equivalence     >= {P4_EQUIV_FLOOR}")
    print("  P5 random-wave arm must NOT reach the P2 floor (else VACUOUS)")
    print()

    out = {}
    for mode in ("fractional_shift", "phasor_bind"):
        a = run_arm(mode, prompts, random.Random(SEED))
        v = verdict(a)
        a.update(v)
        out[mode] = a
        print(f"=== ARM: {mode} ===")
        for k in ("n", "vocab", "determinism", "distinct_count", "distinct_ratio",
                  "order_sensitivity", "equivalence", "rand_distinct_count",
                  "rand_distinct_ratio", "mean_entropy_nats", "ln_vocab"):
            print(f"  {k:22s} = {a[k]}")
        print(f"  {'VERDICT':22s} = {v['verdict']}")
        print()

    print("=== DECISION EVIDENCE (position_binding) ===")
    fs, pb = out["fractional_shift"], out["phasor_bind"]
    print(f"  order_sensitivity  fractional_shift={fs['order_sensitivity']:.4f}  "
          f"phasor_bind={pb['order_sensitivity']:.4f}")
    print(f"  distinct_ratio     fractional_shift={fs['distinct_ratio']:.4f}  "
          f"phasor_bind={pb['distinct_ratio']:.4f}")
    better = "phasor_bind" if pb["order_sensitivity"] > fs["order_sensitivity"] else "fractional_shift"
    print(f"  MORE order-sensitive: {better}")

    print()
    print("=== M1 OVERALL ===")
    for m, a in out.items():
        print(f"  {m:18s} {a['verdict']}")
    ok = all(a["verdict"] == "M1_GATE_PASS" for a in out.values())
    print(f"  M1_GATE_CLOSED = {ok}")

    dest = Path(os.environ.get("LOCALAPPDATA", ".")) / "Temp" / "m1_gate_receipt.json"
    dest.write_text(json.dumps(dict(
        preregistration=dict(N=N_PROMPTS, P1=P1_DETERMINISM,
                             P2=P2_DISTINCT_FLOOR, P3=P3_ORDER_FLOOR,
                             P4=P4_EQUIV_FLOOR, seed=SEED),
        arms=out), indent=2), encoding="utf-8")
    print(f"  receipt: {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
