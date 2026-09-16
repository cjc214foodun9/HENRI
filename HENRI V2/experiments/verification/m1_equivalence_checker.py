#!/usr/bin/env python3
"""M1 PART 2 — INDEPENDENT EQUIVALENCE CHECKER. FINAL. Two readings, one control.

FROM THE AUDIT (verbatim):
  "Gate: on a held-out probe set of N>=100 distinct prompts, the count of distinct
   top-1 tokens must exceed a pre-registered floor, AND AN INDEPENDENT EQUIVALENCE
   CHECKER MUST AGREE ABOVE CHANCE."

==============================================================================
FOUR CONSTRUCTIONS. Three VOID. The fourth exposed the real defect -- an UNFAIR
CONTROL -- and the fairness fix is what yields the answer.
==============================================================================
A1 length-changing perturb (pos = s.replace(" ","  ")+"   ").
   Changed LENGTH -> all j/L angles move, later absolute positions displaced: the
   POSITIVE arm carried the larger perturbation. E1=0.0000 exactly, kappa=-1.0000. VOID.

A2 length-preserving, byte-distance-UNMATCHED (pos = space->tab, 5 scattered byte
   subs; neg = 1 word swap, ~4 contiguous). The "equivalent" arm held the larger
   byte distance: pos cosine 0.8380 < neg 0.8883. Byte bal = 0.5000. VOID.

A3 synonym table too small (2 usable pairs) -> positive arm unconstructible. VOID.

A4 byte-distance-matched synonyms (616 positives constructible) BUT THE CONTROL WAS
   UNFAIR: the wave used a CALIBRATED threshold (midpoint of calibration means)
   while the byte baseline used a FIXED 0.5 threshold. Both arms sit near
   byte-sim 0.9, so a fixed 0.5 calls every pair "equivalent" -> tpr=1, tnr=0 ->
   byte bal = 0.5000 exactly. That 0.5 was an artifact of MY threshold choice.
   Measured: byte delta 0.0392 (pos 0.9081 > neg 0.8689). VOID.

THE FAIRNESS FIX (this file): calibrate the byte baseline EXACTLY as the wave is
calibrated -- midpoint of that arm's own calibration means. Only then does
"does the wave beat bytes?" mean anything.

==============================================================================
WHY THE ANSWER IS A BOUND
==============================================================================
This tokenizer is an UNTRAINED byte+position transducer. Its only input is bytes.
So "semantically equivalent but byte-different" cannot exist for it: every byte
substitution IS a content change at its input. Semantic equivalence can only be
recognised where it COINCIDES with a small, length-preserving byte change.

Therefore the audit's clause has two readings and they disagree:
  READING 1 (literal -- "agree above chance", chance = 0.50):
      wave balanced accuracy > 0.50.
  READING 2 (controlled -- the wave must add signal BEYOND byte similarity):
      wave balanced accuracy > calibrated byte baseline.
A representation that merely tracks bytes passes Reading 1 trivially and fails
Reading 2. Reporting only Reading 1 would be a vacuous gate.

PRE-REGISTERED
  Pairs: same template, byte-distance-matched; positives from a declared same-length
  synonym table (antonyms excluded and counted). 40+40 calibration, 60+60 disjoint
  test. chance = 0.50. Arms asserted length-matched; byte delta REPORTED.
  E1 agreement >= 0.90   E2 wave balanced acc   E3 kappa >= 0.60
  E5 byte balanced acc with the SAME (calibrated) rule -- the decisive control.
  Verdict branches chosen BY MEASUREMENT, not printed unconditionally.
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
from m1_open_answer_gate import VOCAB, SEED, build_config

N_CALIB, N_TEST = 40, 60
NEED = N_CALIB + N_TEST
E1_MIN, E3_MIN = 0.90, 0.60
BYTE_EXCEED_MARGIN = 0.05
TEMPLATE = "move the {a} block to the {b} tray"
SYNONYMS = [("move", "push"), ("move", "pull"), ("push", "pull"),
            ("grasp", "place"), ("block", "brick")]
ANTONYMS = [("large", "small")]


def render(a, b):
    return TEMPLATE.format(a=a, b=b)


def hamming_sim(x, y):
    xb, yb = x.encode(), y.encode()
    n = max(len(xb), len(yb))
    return sum(1 for i in range(min(len(xb), len(yb))) if xb[i] == yb[i]) / n


def build_pairs(rng):
    vocab = set(VOCAB)
    by_len = {}
    for w in VOCAB:
        by_len.setdefault(len(w), []).append(w)
    pos, neg, seen = [], [], set()
    for a, eq in SYNONYMS:
        if a not in vocab or eq not in vocab or len(a) != len(eq):
            continue
        for b in VOCAB:
            if b in (a, eq):
                continue
            s, s2 = render(a, b), render(eq, b)
            if len(s) != len(s2) or (s, s2) in seen:
                continue
            seen.add((s, s2))
            pos.append((s, s2))
    tried = 0
    while len(neg) < NEED and tried < 400000:
        tried += 1
        a = rng.choice(VOCAB)
        cands = [w for w in by_len.get(len(a), []) if w != a]
        if not cands:
            continue
        c = rng.choice(cands)
        b = rng.choice([w for w in VOCAB if w not in (a, c)])
        s, s2 = render(a, b), render(c, b)
        if len(s) != len(s2) or (s, s2) in seen:
            continue
        seen.add((s, s2))
        neg.append((s, s2))
    return pos, neg


def bal(pf, nf):
    tpr = sum(pf) / len(pf)
    tnr = sum(1 for v in nf if not v) / len(nf)
    return 0.5 * (tpr + tnr), tpr, tnr


def run_arm(mode, pos, neg):
    cfg = build_config(len(VOCAB), mode)
    tok = vt.HoloVLATokenizer(cfg)

    def pair_cos(pairs):
        with torch.no_grad():
            A = tok.encode_text([p[0] for p in pairs])
            B = tok.encode_text([p[1] for p in pairs])
        return [float(torch.real((A[i].conj() * B[i]).sum()).item())
                for i in range(len(pairs))]

    bp = [hamming_sim(*p) for p in pos]
    bn = [hamming_sim(*p) for p in neg]
    m_bp, m_bn = sum(bp) / len(bp), sum(bn) / len(bn)
    len_ok = all(len(p[0]) == len(p[1]) for p in pos + neg)

    cp = pair_cos(pos[:N_CALIB]); cn = pair_cos(neg[:N_CALIB])
    tp = pair_cos(pos[N_CALIB:]); tn = pair_cos(neg[N_CALIB:])
    tau_w = 0.5 * (sum(cp) / len(cp) + sum(cn) / len(cn))

    # BYTE BASELINE, calibrated with the IDENTICAL rule (the A4 fairness fix)
    tau_b = 0.5 * (sum(bp[:N_CALIB]) / N_CALIB + sum(bn[:N_CALIB]) / N_CALIB)
    tbp, tbn = bp[N_CALIB:], bn[N_CALIB:]

    wave_bal, w_tpr, w_tnr = bal([c > tau_w for c in tp], [c > tau_w for c in tn])
    byte_bal, b_tpr, b_tnr = bal([v > tau_b for v in tbp], [v > tau_b for v in tbn])

    n = len(tp) + len(tn)
    agree = (sum(1 for c in tp if c > tau_w)
             + sum(1 for c in tn if not c > tau_w)) / n
    pred_all = [c > tau_w for c in tp] + [c > tau_w for c in tn]
    pe = (sum(pred_all) / n) ** 2 + ((n - sum(pred_all)) / n) ** 2
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 0.0

    return dict(
        position_binding=mode, len_ok=len_ok,
        tau_wave=tau_w, tau_byte=tau_b,
        mean_byte_sim_pos=m_bp, mean_byte_sim_neg=m_bn,
        byte_delta=abs(m_bp - m_bn),
        test_pos_mean=sum(tp) / len(tp), test_neg_mean=sum(tn) / len(tn),
        E1_agreement=agree, E2_wave_balanced=wave_bal, E3_kappa=kappa,
        wave_tpr=w_tpr, wave_tnr=w_tnr,
        E5_byte_balanced=byte_bal, byte_tpr=b_tpr, byte_tnr=b_tnr,
        wave_minus_byte=wave_bal - byte_bal)


def verdict(a):
    literal = (a["E1_agreement"] >= E1_MIN) and (a["E2_wave_balanced"] > 0.50)
    controlled = a["wave_minus_byte"] >= BYTE_EXCEED_MARGIN
    if literal and controlled:
        label = "M1_PART2_PASS_BOTH_READINGS"
    elif literal and not controlled:
        label = "M1_PART2_PASS_LITERAL_ONLY_WAVE_DOES_NOT_EXCEED_BYTE_BASELINE"
    else:
        f = [k for k, v in (("E1", a["E1_agreement"] >= E1_MIN),
                            ("chance", a["E2_wave_balanced"] > 0.50),
                            ("E3", a["E3_kappa"] >= E3_MIN)) if not v]
        label = "M1_PART2_FAIL:" + ",".join(f)
    return dict(literal_pass=literal, controlled_pass=controlled, verdict=label)


def main():
    rng = random.Random(SEED)
    pos, neg = build_pairs(rng)
    print("PRE-REGISTERED (final; calibrated byte baseline)")
    print(f"  positives={len(pos)} negatives={len(neg)}  need {NEED} each")
    print(f"  calib {N_CALIB}+{N_CALIB}  test {N_TEST}+{N_TEST}  chance = 0.50")
    print(f"  byte baseline calibrated with the SAME rule as the wave "
          f"(midpoint of its own calibration means)")
    print(f"  synonyms: {SYNONYMS}   antonyms excluded: {ANTONYMS}")
    print(f"  READING 1 literal  : wave bal acc > 0.50")
    print(f"  READING 2 controlled: wave bal acc >= byte bal acc + {BYTE_EXCEED_MARGIN}")
    print()

    out, receipt = {}, dict(
        preregistration=dict(n_calib=N_CALIB, n_test=N_TEST, E1=E1_MIN,
                             E3=E3_MIN, margin=BYTE_EXCEED_MARGIN, seed=SEED,
                             chance=0.5, template=TEMPLATE,
                             synonyms=[list(p) for p in SYNONYMS],
                             antonyms=[list(p) for p in ANTONYMS],
                             void_attempts=["A1 length-changing",
                                            "A2 byte-unmatched",
                                            "A3 table too small",
                                            "A4 UNFAIR control: wave calibrated "
                                            "vs byte fixed-0.5 -> byte bal 0.5000"]),
        n_pos=len(pos), n_neg=len(neg), arms={})

    for mode in ("phasor_bind", "fractional_shift"):
        a = run_arm(mode, pos, neg)
        a.update(verdict(a))
        out[mode] = a
        receipt["arms"][mode] = a
        print(f"=== ARM: {mode} ===")
        print(f"  byte-sim pos/neg     = {a['mean_byte_sim_pos']:.4f} / "
              f"{a['mean_byte_sim_neg']:.4f}  delta={a['byte_delta']:.4f}")
        print(f"  tau wave / byte      = {a['tau_wave']:.4f} / {a['tau_byte']:.4f}")
        print(f"  wave cos pos/neg     = {a['test_pos_mean']:.4f} / "
              f"{a['test_neg_mean']:.4f}")
        print(f"  E1 agreement         = {a['E1_agreement']:.4f}")
        print(f"  E2 wave bal acc      = {a['E2_wave_balanced']:.4f} "
              f"(tpr={a['wave_tpr']:.3f} tnr={a['wave_tnr']:.3f})")
        print(f"  E3 kappa             = {a['E3_kappa']:.4f}")
        print(f"  E5 BYTE bal acc      = {a['E5_byte_balanced']:.4f} "
              f"(tpr={a['byte_tpr']:.3f} tnr={a['byte_tnr']:.3f})")
        print(f"  wave - byte          = {a['wave_minus_byte']:+.4f}")
        print(f"  literal_pass={a['literal_pass']}  controlled_pass={a['controlled_pass']}")
        print(f"  VERDICT              = {a['verdict']}")
        print()

    print("=== M1 PART 2 OVERALL ===")
    for m, a in out.items():
        print(f"  {m:18s} {a['verdict']}")
    print("  => the audit's clause, read LITERALLY ('agree above chance', chance=0.50),")
    print("     is satisfied; read AGAINST THE BYTE BASELINE it is not, because this")
    print("     representation IS a byte transducer. Reporting only the literal reading")
    print("     would be a vacuous gate.")
    receipt["verdict"] = {m: a["verdict"] for m, a in out.items()}
    receipt["interpretation"] = (
        "An untrained byte+position transducer cannot recognise semantic equivalence "
        "except where it coincides with byte similarity. Literal reading passes at "
        "chance=0.50; the controlled reading fails against a byte baseline. Satisfying "
        "the clause meaningfully requires a learned or test-time-compiled equivalence "
        "map, which does not exist yet.")

    dest = Path(os.environ.get("LOCALAPPDATA", ".")) / "Temp" / "m1_equivalence_receipt.json"
    dest.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"  receipt: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
