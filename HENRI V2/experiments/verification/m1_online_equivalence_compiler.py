#!/usr/bin/env python3
"""M1 PART 2 — ONLINE EQUIVALENCE COMPILATION (v3: diversity-matched classes).

CONTEXT
  M1 Part 2 (commit 82fc26c) measured that the holographic encoder adds EXACTLY ZERO
  discriminative information beyond raw byte similarity on an equivalence judgement
  (wave 0.8273 vs byte 0.8273, diff +0.0000), and named the constructive requirement:
  an equivalence source that is NOT the codec -- test-time compilation from
  demonstration pairs (X_i, Y_i). This file BUILDS that mechanism and tests it.

VERSION HISTORY -- v1 and v2 are VOID, each with a measured signature
  v1: balanced accuracy at a midpoint threshold calibrated on TRAIN templates. The
      threshold does not transfer to unseen templates, so both arms scored off
      chance and the shuffled controls were themselves off chance
      (wave compiled 0.6792 vs shuffled 0.6917 -- the CONTROL BEAT THE TREATMENT).
      C3 fired -> VOID. v1 also printed a substantive claim from a void test: fixed.
  v2: threshold-free AUC (chance 0.50). C3/C4 still fired:
        wave compiled 0.6868 | shuffled 0.5456 | RANDOM-DIRECTION 0.6654
      Diagnosed by adding spread_stats: the POSITIVE class is DEGENERATE and the
      NEGATIVE class is not --
        wave: pos eff_dirs 8.7/120 (cos +0.0937) vs neg 32.2/120 (cos +0.0010)
      When one class is concentrated and the other dispersed, ALMOST ANY direction
      separates them, including a random one. The apparent wave advantage
      (+0.2306 over the byte arm) was CLASS-DIVERSITY MISMATCH, not semantics.
      Root cause: positives are drawn from only 5 usable synonym pairs, so the
      positive delta distribution has ~8 effective directions.

V3 FIX (principled, not a re-patch)
  Match class DIVERSITY, not just byte distance:
    * negatives are drawn from the SAME substitution-motif pool as positives
      (a in the synonym first-element set), so both classes share the motif count;
    * a construction-integrity assertion requires the participation-ratio RATIO
      between classes to be <= PR_RATIO_MAX for EVERY arm, else VOID.
  Byte distance and length are matched as before.

PRE-REGISTERED
  Train on templates[0:6], test on templates[6:12] (template-level holdout).
  Arms dimension-matched at 64-d; wave uses a FIXED seeded projection.
  Metric: AUC (threshold-free); chance = 0.50.
  Classifier: difference-of-means direction in difference space (compilation,
  no gradient descent, no parameter persistence).
  Integrity: length matched; byte distance delta REPORTED; PR ratio <= 2.0 per arm.
  Controls (both arms): shuffled-label AUC and random-direction AUC within 0.10
  of 0.50.
  C1 wave AUC - byte AUC >= +0.05   (comparative; the absolute floor is reported,
                                     NOT gated, to avoid moving goalposts)
  Verdicts branch BY MEASUREMENT. A VOID verdict prints ONLY the void statement.
"""
import json
import os
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

import torch
import henri_vla_tokenizer as vt
from m1_open_answer_gate import TEMPLATES, VOCAB, SEED, build_config

SYN_SLOTS = [("move", "push"), ("move", "pull"), ("push", "pull"),
             ("grasp", "place"), ("block", "brick")]
ANTONYMS = {frozenset(("large", "small"))}
N_TRAIN, N_TEST = 120, 120
C1_MARGIN, CONTROL_TOL, PR_RATIO_MAX = 0.05, 0.10, 2.0
EMB_DIM = 64


def render(t, a, b):
    return t.format(a=a, b=b)


def usable_slots():
    v = set(VOCAB)
    return [(a, e) for a, e in SYN_SLOTS
            if a in v and e in v and len(a) == len(e)
            and frozenset((a, e)) not in ANTONYMS]


def build_set(templates, n, rng, slots, by_len):
    """Positives from the synonym slots; negatives from the SAME motif pool."""
    partners = {}
    for a, e in slots:
        partners.setdefault(a, set()).add(e)
        partners.setdefault(e, set()).add(a)
    a_pool = sorted(partners)
    pos, neg, seen = [], [], set()
    guard = 0
    while (len(pos) < n or len(neg) < n) and guard < 800000:
        guard += 1
        t = rng.choice(templates)
        if len(pos) < n:
            a, e = rng.choice(slots)
            b = rng.choice([w for w in VOCAB if w not in (a, e)])
            X, Y = render(t, a, b), render(t, e, b)
            if len(X) == len(Y) and (X, Y) not in seen:
                seen.add((X, Y)); pos.append((X, Y))
        if len(neg) < n:
            a = rng.choice(a_pool)
            cands = [w for w in by_len.get(len(a), [])
                     if w != a and w not in partners.get(a, set())
                     and frozenset((a, w)) not in ANTONYMS]
            if not cands:
                continue
            c = rng.choice(cands)
            b = rng.choice([w for w in VOCAB if w not in (a, c)])
            X, Y = render(t, a, b), render(t, c, b)
            if len(X) == len(Y) and (X, Y) not in seen:
                seen.add((X, Y)); neg.append((X, Y))
    return pos[:n], neg[:n]


def byte_embed(s, dim=EMB_DIM):
    v = torch.zeros(dim)
    b = s.encode("utf-8")[:dim]
    if b:
        v[:len(b)] = torch.tensor([float(x) for x in b]) / 255.0
    return v


def make_emb_fns(tok, wave_dim):
    proj = torch.randn(wave_dim, EMB_DIM,
                       generator=torch.Generator().manual_seed(SEED)) / (wave_dim ** 0.5)

    def wave_fn(ss):
        with torch.no_grad():
            w = tok.encode_text(list(ss))
        return torch.view_as_real(w).reshape(len(ss), -1).to(torch.float32) @ proj

    def byte_fn(ss):
        return torch.stack([byte_embed(s) for s in ss])

    return wave_fn, byte_fn


def diffs(emb_fn, pairs):
    A, B = emb_fn([p[0] for p in pairs]), emb_fn([p[1] for p in pairs])
    d = B - A
    return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def auc(s_pos, s_neg):
    sp = torch.tensor(s_pos, dtype=torch.float64).unsqueeze(1)
    sn = torch.tensor(s_neg, dtype=torch.float64).unsqueeze(0)
    return float(((sp > sn).double() + 0.5 * (sp == sn).double()).mean().item())


def spread_stats(deltas):
    d = torch.stack([x.to(torch.float64).flatten() for x in deltas])
    n = d.shape[0]
    G = d @ d.t()
    off = (G.sum() - G.diag().sum()) / max(n * (n - 1), 1)
    ev = torch.linalg.eigvalsh(G).clamp_min(0.0)
    pr = float((ev.sum() ** 2) / ev.pow(2).sum().clamp_min(1e-30))
    return dict(n=int(n), mean_pairwise_cosine=float(off), participation_ratio=pr)


def compile_and_auc(emb_fn, pos_tr, neg_tr, pos_te, neg_te, *,
                    shuffle_seed=None, random_seed=None):
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        allp = pos_tr + neg_tr
        rng.shuffle(allp)
        pos_tr, neg_tr = allp[:len(pos_tr)], allp[len(pos_tr):]
    if random_seed is not None:
        g = torch.Generator().manual_seed(random_seed)
        w = torch.randn(EMB_DIM, generator=g)
        w = w / w.norm().clamp_min(1e-12)
    else:
        dp, dn = diffs(emb_fn, pos_tr), diffs(emb_fn, neg_tr)
        w = dp.mean(dim=0) - dn.mean(dim=0)
        w = w / w.norm().clamp_min(1e-12)
    sp = (diffs(emb_fn, pos_te) @ w).tolist()
    sn = (diffs(emb_fn, neg_te) @ w).tolist()
    return auc(sp, sn)


def main():
    rng = random.Random(SEED)
    vocab = set(VOCAB)
    slots = usable_slots()
    by_len = {}
    for w in VOCAB:
        by_len.setdefault(len(w), []).append(w)

    tr_t, te_t = TEMPLATES[:6], TEMPLATES[6:]
    pos_tr, neg_tr = build_set(tr_t, N_TRAIN, rng, slots, by_len)
    pos_te, neg_te = build_set(te_t, N_TEST, rng, slots, by_len)

    cfg = build_config(len(VOCAB), "phasor_bind")
    tok = vt.HoloVLATokenizer(cfg)
    wave_fn, byte_fn = make_emb_fns(tok, 2 * cfg.ambient_dim_D)

    print("PRE-REGISTERED (v3: diversity-matched classes)")
    print(f"  usable synonym slots = {slots}")
    print(f"  motif pool (both classes) = {sorted({a for a, _ in slots})}")
    print(f"  train templates={len(tr_t)} test templates={len(te_t)} "
          f"(template-level holdout)")
    print(f"  train pairs={len(pos_tr)}+{len(neg_tr)}  test={len(pos_te)}+{len(neg_te)}")
    print(f"  metric=AUC (threshold-free, chance 0.50); arms matched at {EMB_DIM}-d")
    print(f"  integrity: length matched + PR ratio <= {PR_RATIO_MAX} per arm")
    print(f"  controls: shuffled-label and random-direction within {CONTROL_TOL} of 0.50")
    print(f"  C1: wave AUC - byte AUC >= +{C1_MARGIN} (comparative; absolute AUC reported)")
    print()

    out, integrity_ok = {}, True
    for arm, fn in (("wave", wave_fn), ("byte", byte_fn)):
        real = compile_and_auc(fn, pos_tr, neg_tr, pos_te, neg_te)
        shuf = compile_and_auc(fn, pos_tr, neg_tr, pos_te, neg_te, shuffle_seed=SEED + 99)
        rand = compile_and_auc(fn, pos_tr, neg_tr, pos_te, neg_te, random_seed=SEED + 7)
        sp = spread_stats(diffs(fn, pos_te))
        sn = spread_stats(diffs(fn, neg_te))
        pr_ratio = max(sp["participation_ratio"], sn["participation_ratio"]) / \
            max(min(sp["participation_ratio"], sn["participation_ratio"]), 1e-9)
        this_ok = pr_ratio <= PR_RATIO_MAX
        integrity_ok = integrity_ok and this_ok
        out[arm] = dict(compiled=real, shuffled=shuf, random=rand,
                        spread_pos=sp, spread_neg=sn, pr_ratio=pr_ratio,
                        diversity_ok=this_ok)
        print(f"=== ARM: {arm} ===")
        print(f"  COMPILED         AUC = {real:.4f}")
        print(f"  SHUFFLED-label   AUC = {shuf:.4f}   <- control")
        print(f"  RANDOM-direction AUC = {rand:.4f}   <- control")
        print(f"  spread pos eff_dirs = {sp['participation_ratio']:.1f}/{sp['n']}  "
              f"neg = {sn['participation_ratio']:.1f}/{sn['n']}  "
              f"PR ratio = {pr_ratio:.2f}  ok={this_ok}")
        print()

    w, b = out["wave"], out["byte"]
    diff = w["compiled"] - b["compiled"]
    controls_ok = all(abs(v - 0.5) <= CONTROL_TOL for arm in out.values()
                      for v in (arm["shuffled"], arm["random"]))

    if not integrity_ok:
        verdict = "VOID_DIVERSITY_UNMATCHED"
    elif not controls_ok:
        verdict = "VOID_CONTROL_NOT_AT_CHANCE"
    elif diff >= C1_MARGIN:
        verdict = "M1_ONLINE_COMPILATION_PASS"
    else:
        verdict = "M1_ONLINE_COMPILATION_FAIL:C1_beats_byte_arm"

    print("=== DECISION ===")
    print(f"  wave AUC={w['compiled']:.4f}  byte AUC={b['compiled']:.4f}  "
          f"wave-byte={diff:+.4f}")
    print(f"  controls: wave shuf {w['shuffled']:.4f} rand {w['random']:.4f} | "
          f"byte shuf {b['shuffled']:.4f} rand {b['random']:.4f}")
    print(f"  integrity_ok={integrity_ok}  controls_ok={controls_ok}")
    print(f"  VERDICT = {verdict}")
    print()
    if verdict.startswith("VOID"):
        print("  => INCONCLUSIVE. This construction cannot distinguish compiled")
        print("     signal from construction artifact. NO claim about the model.")
    elif verdict.endswith("PASS"):
        print("  => online compilation SATISFIES the clause: the direction compiled")
        print("     from demonstration pairs transfers to unseen templates and beats")
        print(f"     the matched byte arm by {diff:+.4f}.")
    else:
        print("  => online compilation does NOT unlock the clause on this substrate:")
        print(f"     wave {w['compiled']:.4f} vs byte {b['compiled']:.4f} "
              f"(diff {diff:+.4f} < +{C1_MARGIN}).")
        print("     The compiled direction is available to the byte arm identically.")

    dest = Path(os.environ.get("LOCALAPPDATA", ".")) / "Temp" / \
        "m1_online_compilation_receipt.json"
    dest.write_text(json.dumps(dict(
        preregistration=dict(n_train=N_TRAIN, n_test=N_TEST, C1_margin=C1_MARGIN,
                             control_tol=CONTROL_TOL, pr_ratio_max=PR_RATIO_MAX,
                             seed=SEED, metric="AUC", chance=0.5, emb_dim=EMB_DIM,
                             slots=slots, train_templates=tr_t, test_templates=te_t,
                             mechanism="difference-of-means in difference space",
                             v1_void="bal acc at train-calibrated midpoint; shuffled "
                                     "control beat the treatment (0.6917 > 0.6792), and "
                                     "a substantive claim was printed from a void test",
                             v2_void="threshold-free AUC; random-direction control "
                                     "0.6654 revealed positive-class degeneracy "
                                     "(pos eff_dirs 8.7/120 vs neg 32.2/120) because "
                                     "only 5 synonym pairs existed"),
        arms=out, wave_minus_byte=diff, integrity_ok=integrity_ok,
        controls_ok=controls_ok, verdict=verdict), indent=2), encoding="utf-8")
    print(f"  receipt: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
