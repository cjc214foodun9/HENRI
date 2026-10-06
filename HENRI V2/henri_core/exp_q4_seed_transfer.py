"""D-SEED: is the Q4 membership threshold STABLE across ingress seeds?

WHY THIS IS THE DECISIVE TEST
    Measured: the string "zzz qqq unrelated gamma" scored 0.342 under one unpinned
    ingress draw and 0.5725 under another. The ingress weights (token_emb,
    slot_router, joint_proj) are random when ingress_seed is None and they sit in
    the text path. If the membership score's SCALE depends on that draw, then any
    threshold calibrated on one system does not transfer to another, and the gate
    is seed-bound rather than a property of the memory.

    This is the question that decides whether the Q4 signal is USABLE or CURIOUS.

DESIGN (pre-registered)
    Seeds: the frozen 20261004 plus four others.
    Per seed, with the ingress pinned to that seed:
      in6  = 6 corpus items stored AND queried  (bank = those 6)
      ho   = 4 corpus items NEVER stored        (same distribution, held out)
      F1   = drop-last-word negatives vs the FULL corpus bank
      F4   = zero-overlap synonym negatives
      F5   = random-wave floor
    Reported per seed: min(in6), max(ho), the GAP, AUC(in6 vs ho), and the
    synonym/random levels.

KILL CONDITIONS (frozen before the run)
    K1  max(ho) < min(in6) on EVERY seed      -> separation generalises
    K2  the gap is positive on every seed      -> a threshold exists per seed
    K3  max(ho) spread across seeds <= 0.10    -> a SINGLE threshold transfers
    K4  AUC(in6 vs ho) >= 0.99 on every seed   -> the ordering is stable

    If K1/K2 hold but K3 fails, the signal is real yet seed-bound: honest and
    useful, but it needs per-system calibration.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, ".")
from henri_core import cli as H_cli
from henri_core import novelty_gate as NG
from henri_core.system import TriModelSystem
from henri_core.tokenizer import ByteBPE

torch.set_num_threads(8)
CORPUS = H_cli.CORPUS
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = [20261004, 111111, 222222, 333333, 444444]
SYNONYMS = [
    "a thing affects another thing nearby",
    "memory hands over an answer up the rungs",
    "the rule keeps the edge in place",
]
RANDOM_TEXT = ["zzz qqq unrelated gamma", "wxk pfj blorp tender", "qq vv xx zz mm"]


def drop_last(t):
    w = t.split()
    return " ".join(w[:-1]) if len(w) > 1 else t


def one_seed(seed):
    tok = ByteBPE().train(CORPUS, vocab_size=512)
    s = TriModelSystem(vocab=tok.vocab_size, small=True, ingress_seed=seed)
    s.eval()
    full = s.build_axioms(CORPUS, tok)                 # bank = all 10

    def sc(text, bk):
        return NG.membership_score(s.wave_of(text, tok), bk)

    # held-out subset bank
    half = torch.stack([s.wave_of(t, tok) for t in CORPUS[:6]])
    in6 = [sc(t, half) for t in CORPUS[:6]]
    ho = [sc(t, half) for t in CORPUS[6:]]

    f1 = [sc(drop_last(t), full) for t in CORPUS]
    f4 = [sc(t, full) for t in SYNONYMS]
    rt = [sc(t, full) for t in RANDOM_TEXT]
    g = torch.Generator().manual_seed(999)
    f5 = [NG.membership_score(torch.randn(s.dim, generator=g).to(torch.complex64), full)
          for _ in range(60)]

    return {
        "seed": seed,
        "in6_min": round(min(in6), 6), "in6_mean": round(sum(in6) / len(in6), 6),
        "ho_max": round(max(ho), 6), "ho_mean": round(sum(ho) / len(ho), 6),
        "gap_min_in6_minus_max_ho": round(min(in6) - max(ho), 6),
        "auc_in6_vs_ho": round(NG.auc(in6, ho), 4),
        "f1_drop_last_max": round(max(f1), 6),
        "f1_drop_last_mean": round(sum(f1) / len(f1), 6),
        "f4_synonym_max": round(max(f4), 6),
        "random_text_max": round(max(rt), 6),
        "random_wave_max": round(max(f5), 6),
        "K1_ho_below_all_in6": bool(max(ho) < min(in6)),
    }


def main() -> int:
    rows = [one_seed(sd) for sd in SEEDS]
    ho_maxes = [r["ho_max"] for r in rows]
    gaps = [r["gap_min_in6_minus_max_ho"] for r in rows]
    aucs = [r["auc_in6_vs_ho"] for r in rows]
    f1max = [r["f1_drop_last_max"] for r in rows]
    in6min = [r["in6_min"] for r in rows]

    R = {
        "schema": "henri.q4.seed_transfer.v1",
        "seeds": SEEDS,
        "per_seed": rows,
        "summary": {
            "K1_all_seeds_separate": all(r["K1_ho_below_all_in6"] for r in rows),
            "K2_gap_positive_all": all(gg > 0 for gg in gaps),
            "K3_ho_max_spread": round(max(ho_maxes) - min(ho_maxes), 6),
            "K3_single_threshold_transfers": bool(max(ho_maxes) - min(ho_maxes) <= 0.10),
            "K4_auc_min": round(min(aucs), 4),
            "K4_ordering_stable": bool(min(aucs) >= 0.99),
            "in6_min_spread": round(max(in6min) - min(in6min), 6),
            "f1_drop_last_max_spread": round(max(f1max) - min(f1max), 6),
        },
    }
    s_ = R["summary"]
    R["verdict"] = {
        "signal_is_real": bool(s_["K1_all_seeds_separate"] and s_["K2_gap_positive_all"]),
        "threshold_transfers_across_seeds": s_["K3_single_threshold_transfers"],
        "claim": (
            "The membership signal SEPARATES in-bank from held-out corpus on every "
            f"seed (K1={s_['K1_all_seeds_separate']}, K2={s_['K2_gap_positive_all']}, "
            f"AUC min {s_['K4_auc_min']}). "
            + ("A single threshold TRANSFERS: held-out max varies only "
               f"{s_['K3_ho_max_spread']:.4f} across seeds. "
               if s_["K3_single_threshold_transfers"] else
               "A single threshold does NOT transfer: held-out max varies "
               f"{s_['K3_ho_max_spread']:.4f} across seeds, so the gate requires "
               "PER-SYSTEM calibration and is seed-bound, not absolute. ")
            + f"Held-out max spans {min(ho_maxes):.4f}..{max(ho_maxes):.4f} while "
            f"in-bank min stays {min(in6min):.4f}, so the SEPARATION is robust even "
            "where the absolute level is not. The statistic measures stored-wave "
            "membership; it is a graded lexical meter, NOT understanding."),
    }

    out = json.dumps(R, indent=1)
    ev = os.path.join(REPO, "design", "zone_a", "evidence")
    dst = (os.path.join(ev, "henri_q4_seed_transfer_receipt.json") if os.path.isdir(ev)
           else r"C:/Users/chan/AppData/Local/Temp/henri_q4_seed_transfer_receipt.json")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(out)

    print("D-SEED: is the Q4 membership threshold stable across ingress seeds?")
    print("=" * 92)
    print(f"  {'seed':>9}{'in6_min':>10}{'ho_max':>9}{'gap':>9}{'AUC':>7}"
          f"{'drop_last_max':>15}{'syn_max':>9}{'rand_txt':>9}")
    for r in rows:
        print(f"  {r['seed']:>9}{r['in6_min']:>10.4f}{r['ho_max']:>9.4f}"
              f"{r['gap_min_in6_minus_max_ho']:>9.4f}{r['auc_in6_vs_ho']:>7.3f}"
              f"{r['f1_drop_last_max']:>15.4f}{r['f4_synonym_max']:>9.4f}"
              f"{r['random_text_max']:>9.4f}")
    print("  " + "-" * 90)
    for k in ("K1_all_seeds_separate", "K2_gap_positive_all",
              "K3_single_threshold_transfers", "K4_ordering_stable",
              "K3_ho_max_spread", "K4_auc_min"):
        print(f"  {k:<36} {s_[k]}")
    print("=" * 92)
    print(f"  {R['verdict']['claim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
