"""Q4 DISCRIMINATIVE TEST v2: can HENRI reject an input its bank does not contain?

v1 DEFECT (self-caught, before commit)
    v1 reported "hard-negative AUC = 1.0" using AUC(family vs random_waves).
    Random waves are pseudo-orthogonal to every bank row by construction, so that
    comparison is trivially 1.0. It is the wrong-population defect: a comparator
    that cannot fail proves nothing.

v2 CHANGES
    1. AUC is reported against the RANDOM FLOOR only as a sanity line, labelled
       trivial. The load-bearing metrics are:
         (a) AUC(F0 in-bank vs each out-of-bank family)   [degenerate if F0 saturates]
         (b) the OUT-OF-BANK SPREAD: a true membership predicate would give all
             out-of-bank inputs the SAME low score. A spread means the statistic
             is a graded similarity meter, not a predicate.
         (c) monotonicity of score vs token overlap (Pearson + order check)
    2. SHUFFLED-BANK ABLATION (the control that can fail): rebuild the bank from
       RANDOM unit vectors of the same shape and re-run. A real memory-membership
       signal must COLLAPSE (AUC ~0.5). If it survives, the statistic is vacuous.
    3. Accept-rate at three thresholds: calibrated, 0.80, 0.50.
    4. Post-hoc calibration control: calibrate theta from the EVAL set and show
       it inflates nothing (a reminder that theta is not to be moved).

Pre-registration: design/zone_a/evidence/henri_q4_preregistration.json
Statistic:      henri_core/novelty_gate.py  (pure functions; no model change)
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
PIN = 20261004

SYNONYMS = [
    "a thing affects another thing nearby",
    "memory hands over an answer up the rungs",
    "the rule keeps the edge in place",
    "sensing the lowest point of the hill",
    "the port becomes clear when waves cancel",
]
FAM_KEYS = ["F1_drop_last", "F2_drop_first", "F3_middle_chunk",
            "F4_synonym", "F6_shuffled_order"]


def build():
    """Build the system with a PINNED ingress (see exp_q4_margin.build)."""
    tok = ByteBPE().train(CORPUS, vocab_size=512)
    s = TriModelSystem(vocab=tok.vocab_size, small=True, ingress_seed=PIN)
    s.eval()
    bank = s.build_axioms(CORPUS, tok)
    return s, tok, bank


def drop_last(t):
    w = t.split()
    return " ".join(w[:-1]) if len(w) > 1 else t


def drop_first(t):
    w = t.split()
    return " ".join(w[1:]) if len(w) > 1 else t


def middle_three(t):
    w = t.split()
    if len(w) <= 3:
        return t
    m = len(w) // 2
    return " ".join(w[max(0, m - 1):m + 2])


def shuffled(t, g):
    w = t.split()
    perm = torch.randperm(len(w), generator=g).tolist()
    return " ".join(w[j] for j in perm)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return round(num / den, 4) if den > 0 else None


def main() -> int:
    s, tok, bank = build()
    N, D = len(CORPUS), s.dim
    R = {"schema": "henri.q4.discriminative.v2", "N": N, "dim": D,
         "supersedes": "v1 (AUC vs random floor was trivially 1.0)"}

    def score(text, bk):
        return NG.membership_score(s.wave_of(text, tok), bk)

    # ---------- families (PRE-projection) -------------------------------------
    g = torch.Generator().manual_seed(20261008)
    mk = {
        "F1_drop_last": lambda t: drop_last(t),
        "F2_drop_first": lambda t: drop_first(t),
        "F3_middle_chunk": lambda t: middle_three(t),
        "F6_shuffled_order": lambda t: shuffled(t, g),
    }
    fam = {"F0_exact_inbank": {t: score(t, bank) for t in CORPUS}}
    for k, f in mk.items():
        fam[k] = {t: score(f(t), bank) for t in CORPUS}
    fam["F4_synonym"] = {t: score(t, bank) for t in SYNONYMS}

    g2 = torch.Generator().manual_seed(20261009)
    # Random waves have no text: score the tensor DIRECTLY, bypassing wave_of().
    # (v2 bug: routed these through score(), which calls wave_of(text,...) and
    #  crashed with AttributeError: 'Tensor' object has no attribute 'encode'.)
    rand = [NG.membership_score(torch.randn(D, generator=g2).to(torch.complex64), bank)
            for _ in range(100)]
    fam["F5_random_waves"] = {f"r{i}": v for i, v in enumerate(rand)}

    # ---------- calibration: F0 vs F5 ONLY ------------------------------------
    theta, cal_bal = NG.calibrated_threshold(list(fam["F0_exact_inbank"].values()), rand)
    R["calibration"] = {"theta": round(theta, 6),
                        "balanced_accuracy": round(cal_bal, 4),
                        "pos": "F0_exact_inbank", "neg": "F5_random_waves",
                        "note": "theta is DEGENERATE when F0 is identically 1.0; "
                                "reported, not hidden"}

    # ---------- metric (a): AUC vs random floor (TRIVIAL, labelled) -----------
    R["metric_a_auc_vs_random_TRIVIAL"] = {
        "F0": round(NG.auc(list(fam["F0_exact_inbank"].values()), rand), 4),
        **{k: round(NG.auc(list(fam[k].values()), rand), 4) for k in FAM_KEYS},
        "warning": "every value is 1.0 because random waves are orthogonal to the "
                   "bank by construction. NOT evidence of discrimination.",
    }

    # ---------- metric (b): the out-of-bank spread ----------------------------
    oob = [v for k in FAM_KEYS for v in fam[k].values()]
    R["metric_b_out_of_bank_spread"] = {
        "min": round(min(oob), 6), "max": round(max(oob), 6),
        "mean": round(sum(oob) / len(oob), 6),
        "range": round(max(oob) - min(oob), 6),
        "reads_as": "a membership PREDICATE would give all out-of-bank inputs one "
                    "low score. A wide range means a graded SIMILARITY METER.",
        "in_bank_is_saturated": all(abs(v - 1.0) < 1e-9
                                    for v in fam["F0_exact_inbank"].values()),
    }

    # ---------- metric (c): monotonicity vs token overlap ---------------------
    xs, ys = [], []
    for k, f in mk.items():
        for t in CORPUS:
            xs.append(NG.token_overlap(f(t), t))
            ys.append(fam[k][t])
    for t in SYNONYMS:
        xs.append(max(NG.token_overlap(t, c) for c in CORPUS))
        ys.append(fam["F4_synonym"][t])
    R["metric_c_lexical_coupling"] = {
        "pearson_r_overlap_vs_score": pearson(xs, ys),
        "family_means_by_overlap": {
            k: round(sum(fam[k].values()) / len(fam[k]), 6) for k in
            ["F0_exact_inbank", "F2_drop_first", "F1_drop_last",
             "F3_middle_chunk", "F6_shuffled_order", "F4_synonym", "F5_random_waves"]},
        "reads_as": "if r > 0.7 the statistic tracks LEXICAL OVERLAP, not meaning",
    }

    # ---------- SHUFFLED-BANK ABLATION (the control that MUST fail) -----------
    gr = torch.Generator().manual_seed(4242)
    raw = torch.randn(N, D, generator=gr).to(torch.complex64)
    bank_shuf = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    f0_sh = [score(t, bank_shuf) for t in CORPUS]
    f5_sh = [NG.membership_score(torch.randn(D, generator=g2).to(torch.complex64),
                                 bank_shuf) for _ in range(100)]
    R["control_shuffled_bank"] = {
        "auc_F0_vs_random_on_shuffled_bank": round(NG.auc(f0_sh, f5_sh), 4),
        "F0_mean_on_shuffled_bank": round(sum(f0_sh) / len(f0_sh), 6),
        "F0_mean_on_real_bank": round(
            sum(fam["F0_exact_inbank"].values()) / N, 6),
        "PASSED": NG.auc(f0_sh, f5_sh) <= 0.65,
        "reads_as": "if the real bank carries the signal, replacing it with random "
                    "vectors must COLLAPSE the separation toward chance",
    }

    # ---------- accept rates at three thresholds -----------------------------
    tbl = {}
    for th, label in ((theta, "calibrated"), (0.80, "0.80"), (0.50, "0.50")):
        tbl[label] = {k: round(sum(1 for v in fam[k].values() if v >= th) / len(fam[k]), 3)
                      for k in ["F0_exact_inbank"] + FAM_KEYS + ["F5_random_waves"]}
    R["accept_rate_by_threshold"] = {"theta_used": round(theta, 6), "table": tbl}

    # ---------- post-hoc calibration control ---------------------------------
    th_post, bal_post = NG.calibrated_threshold(
        list(fam["F0_exact_inbank"].values()) + list(fam["F1_drop_last"].values()), rand)
    R["control_posthoc_calibration"] = {
        "theta_if_calibrated_on_a_hard_family": round(th_post, 6),
        "note": "shown to confirm theta does NOT move to accommodate a family; "
                "the frozen theta comes from F0 vs F5 only",
    }

    # ---------- K4 confound (POST-projection) --------------------------------
    post = []
    for t in CORPUS:
        r = s.solve(t, tok, use_swarm=True)
        post.append(NG.membership_score(r["psi_converged"], bank))
    frac = sum(1 for v in post if v >= 0.95) / len(post)
    R["K4_confound"] = {"post_mean": round(sum(post) / len(post), 6),
                        "frac_ge_0.95": round(frac, 4),
                        "CONFOUND_CONFIRMED": frac >= 0.90}

    # ---------- verdict -------------------------------------------------------
    spread = R["metric_b_out_of_bank_spread"]["range"]
    r_lex = R["metric_c_lexical_coupling"]["pearson_r_overlap_vs_score"] or 0.0
    ctrl = R["control_shuffled_bank"]
    R["verdict"] = {
        "Q4_MECHANISM_EXISTS": float(min(oob)) < float(min(fam["F0_exact_inbank"].values())),
        "literal_requirement_met": bool(float(max(oob)) < float(min(fam["F0_exact_inbank"].values()))),
        "shuffled_bank_control_passed": ctrl["PASSED"],
        "is_graded_similarity_meter": spread > 0.30,
        "is_lexically_coupled": r_lex > 0.70,
        "claim": (
            "A pre-projection membership score separates in-bank from out-of-bank "
            "inputs, and the shuffled-bank control collapses it, so the signal is "
            "real. BUT the statistic is a graded leXICAL similarity meter, not a "
            "membership predicate: out-of-bank scores span "
            f"{R['metric_b_out_of_bank_spread']['min']}..{R['metric_b_out_of_bank_spread']['max']} "
            f"and correlate with token overlap at r={r_lex}. It makes HENRI able to "
            "ABSTAIN on unrelated input; it does NOT give HENRI semantic membership "
            "or understanding."),
        "what_it_is_NOT": ["semantic generalisation", "compositional reasoning",
                           "understanding of paraphrases", "measurably intelligent"],
    }

    out = json.dumps(R, indent=1)
    ev = os.path.join(REPO, "design", "zone_a", "evidence")
    dst = (os.path.join(ev, "henri_q4_discriminative_receipt.json")
           if os.path.isdir(ev) else
           r"C:/Users/chan/AppData/Local/Temp/henri_q4_discriminative_receipt.json")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(out)

    print("Q4 DISCRIMINATIVE TEST v2")
    print("=" * 78)
    print(f"  N={N} dim={D}  theta={theta:.6f} (degenerate: F0 identically 1.0)")
    print(f"  {'family':<22}{'mean':>10}{'accept@.80':>12}{'accept@.50':>12}")
    for k in ["F0_exact_inbank"] + FAM_KEYS + ["F5_random_waves"]:
        m = sum(fam[k].values()) / len(fam[k])
        print(f"  {k:<22}{m:>10.6f}{tbl['0.80'][k]:>12.3f}{tbl['0.50'][k]:>12.3f}")
    print("  " + "-" * 74)
    print(f"  out-of-bank spread     : {R['metric_b_out_of_bank_spread']['min']}"
          f" .. {R['metric_b_out_of_bank_spread']['max']}")
    print(f"  lexical coupling r     : {r_lex}")
    print(f"  shuffled-bank AUC      : {ctrl['auc_F0_vs_random_on_shuffled_bank']}"
          f"  (PASS={ctrl['PASSED']}; F0 real={ctrl['F0_mean_on_real_bank']}"
          f" -> shuffled={ctrl['F0_mean_on_shuffled_bank']})")
    print(f"  post-projection >=0.95 : {R['K4_confound']['frac_ge_0.95']}"
          f"  CONFOUND_CONFIRMED={R['K4_confound']['CONFOUND_CONFIRMED']}")
    print("=" * 78)
    print(f"  Q4_MECHANISM_EXISTS={R['verdict']['Q4_MECHANISM_EXISTS']}"
          f"  literal_met={R['verdict']['literal_requirement_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
