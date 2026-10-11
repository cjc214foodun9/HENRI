"""EXECUTION-AXIOM PROBE -- can ANY content verdict separate correct from incorrect?

WHY THIS EXISTS
    The derandomized veto audit measured the Sagnac veto as index-neutral but
    ANTI-SELECTIVE (separation = -0.683414): it fires on 81.99% of CORRECT candidates
    and only 13.65% of INCORRECT ones. Receipt:
        design/zone_a/evidence/henri_derandomized_veto_observed.json
        sha256 5a80d1a105543b7e2b3131cd2ca3744276c3bd4fbb1320465daaea996df183e2

    The harness in that audit scores delta = 1 - S(candidate, QUERY WAVE). That is the
    INFORMATION-IDENTITY defect exp_attributable_veto.py already documented: the query
    wave is what the ridge ranker consumed to BUILD the ranking, so a monotone function
    of it cannot re-rank. The inversion is the predicted signature of that defect.

    This probe asks the one question that decides the selector path: does replacing the
    query-wave reference with an EXECUTION-DERIVED axiom flip the sign of the separation?

ARMS (all evaluated on the SAME held-out pairs; only the axiom differs)
    A0 query_wave        delta = 1 - S(cand, query). Reproduction of the inverted arm.
    A1 exec_centroid     delta = 1 - S(cand, A_exec), where A_exec is the normalised mean
                         wave of TRAIN candidates whose traces EXECUTE CORRECTLY.
                         LABEL-HONEST: built from train-split execution labels only.
    A2 cluster_centroid  delta = 1 - S(cand, nearest of K execution-cluster centroids),
                         a piecewise variant that does not assume one convex correctness
                         region on the sphere.
    A3 random_axiom      CONTROL. A fixed random unit axiom. MUST NOT separate. If it does,
                         the statistic is measuring something other than correctness.

STATISTIC (threshold-free, because epsilon_hard=0.35 SATURATES at this geometry)
    A thresholded trigger is degenerate here: two independent unit waves at D=8192 give
    delta ~ 0.5 > 0.35, so the veto fires on ~everything (measured: arm 2 trigger 1.0).
    Any threshold statistic would then measure the base rate, not the discriminator. So
    this probe uses the CONTINUOUS delta and reports:
      * mean delta for correct vs incorrect pairs
      * AUC of (-delta) predicting correctness, with a tie-aware rank AUC
      * separation = rejection_rate_on_incorrect - false_veto_rate_on_correct, computed
        at the train-set MEDIAN delta (reported as the operating point, not selected on
        held-out)
    Sign convention: a USEFUL axiom gives LOWER delta for correct candidates (they are
    closer to the correctness reference), so AUC(-delta) > 0.5 and separation > 0.

READING RULE (frozen before the run)
    A3 must land near 0.5 AUC. If it does not, verdict HARNESS_BROKEN.
    A0 must reproduce the ~inversion. If it does not, this harness is not that harness.
    A1/A2 separate iff AUC(-delta) >= 0.5 + 0.05 AND separation > 0 with a Clopper-Pearson
    lower bound above 0. Otherwise the content signal is absent, and the selector path
    needs a different verifier FEATURE SET -- not another axiom choice.

NOT DONE
    No dark-port wiring. No model-performance claim. No task attempted or solved. CPU,
    diagnostic only. The candidate waves are encoder waves, not EFE-planned waves.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.dirname(os.path.dirname(_HERE))
REPO = os.path.dirname(_V2)
for p in (_V2, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import henri_core.exp_executor_verified_selection as sel          # noqa: E402
from run_derandomized_veto import (clopper_pearson, _clopper_pearson_selftest,
                                   pearson_r)                       # noqa: E402

EXPECTED_M4_SHA256 = "fb46c19b8c15c27cb363eb91eed78f7a0707905426096f4c25d672f42b3ff316"
PIN = sel.PIN
ROWS = sel.ROWS
K = sel.K
AUC_BAR = 0.05
CLUSTERS = 8


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def rank_auc(scores, labels):
    """Tie-aware rank AUC. Returns nan when one class is absent."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    s_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] > 0)
    return (s_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def sep_at(deltas, labels, thr):
    """(false_veto_on_correct, reject_on_incorrect) at a fixed threshold."""
    n_ok = n_bad = f_ok = f_bad = 0
    for d, y in zip(deltas, labels):
        fired = d > thr                     # veto fires when delta EXCEEDS threshold
        if y == 0:
            n_ok += 1
            f_ok += int(fired)
        else:
            n_bad += 1
            f_bad += int(fired)
    return f_ok, n_ok, f_bad, n_bad


def report(name, deltas, labels, thr):
    """labels: 0 = CORRECT, 1 = INCORRECT.

    SIGN CONVENTION (SELF-CAUGHT): the first draft scored rank_auc(-delta, labels) with
    the positive class set to INCORRECT, then compared that number to a >0.5 "separates"
    bar. That flips the meaning of every AUC in the table and made the harness report
    HARNESS_BROKEN on an arm whose threshold separation was +0.798 with disjoint CIs.
    Fixed here: the positive class is CORRECT, so AUC > 0.5 means CORRECT candidates carry
    LOWER delta (closer to the correctness reference) -- the direction a useful veto needs.
    """
    y_correct = [1 - y for y in labels]
    auc = rank_auc([-d for d in deltas], y_correct)
    f_ok, n_ok, f_bad, n_bad = sep_at(deltas, labels, thr)
    fv = f_ok / max(1, n_ok)
    rj = f_bad / max(1, n_bad)
    fv_lo, fv_hi = clopper_pearson(f_ok, n_ok)
    rj_lo, rj_hi = clopper_pearson(f_bad, n_bad)
    d_ok = [d for d, y in zip(deltas, labels) if y == 0]
    d_bad = [d for d, y in zip(deltas, labels) if y == 1]
    return {
        "arm": name,
        "auc_neg_delta_predicts_correct": round(auc, 6) if math.isfinite(auc) else "UNDEFINED",
        "delta_mean_correct": round(sum(d_ok) / max(1, len(d_ok)), 6),
        "delta_mean_incorrect": round(sum(d_bad) / max(1, len(d_bad)), 6),
        "delta_gap_correct_minus_incorrect": round(
            sum(d_ok) / max(1, len(d_ok)) - sum(d_bad) / max(1, len(d_bad)), 6),
        "threshold": round(thr, 6),
        "false_veto_rate_on_correct": round(fv, 6),
        "false_veto_clopper_pearson_95": [round(fv_lo, 6), round(fv_hi, 6)],
        "rejection_rate_on_incorrect": round(rj, 6),
        "rejection_clopper_pearson_95": [round(rj_lo, 6), round(rj_hi, 6)],
        "separation": round(rj - fv, 6),
        "separation_ci_disjoint": bool(fv_hi < rj_lo),
        "n_correct": n_ok, "n_incorrect": n_bad,
    }


def main():
    _clopper_pearson_selftest()
    t0 = time.time()
    resolved = os.path.abspath(sys.modules["henri_core.m4_generative"].__file__)
    actual = sha256_file(resolved)
    R = {"schema": "henri.execution.axiom.probe.v1", "pin": PIN, "k": K,
         "auc_bar": AUC_BAR, "clusters": CLUSTERS,
         "dependency": {"module": resolved, "sha256": actual,
                        "expected_sha256": EXPECTED_M4_SHA256,
                        "match": actual == EXPECTED_M4_SHA256}}
    if not R["dependency"]["match"]:
        R["verdict"] = "HARNESS_BROKEN_DEPENDENCY_SHA_MISMATCH"
        _write(R)
        return 2

    corpus = sel.build_corpus(max_len=sel.MAX_LEN, holdout_len=sel.MAX_LEN,
                              inputs=sel.make_inputs(sel.N_INPUTS))
    system, tok = sel.build_system(corpus, pin_seed=PIN)
    tr_tgt = [tok.encode(corpus.traces[i]) for i in corpus.train_idx]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]
    tr_spec = [corpus.specs[i] for i in corpus.train_idx]
    ho_spec = [corpus.specs[i] for i in corpus.heldout_idx]
    Vv, M = tok.vocab_size, sel.M4Config().max_trace

    gen = sel.WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Ztr = F.normalize(torch.nan_to_num(sel.flat(gen.wave(tr_spec))), dim=-1)
        Zho = F.normalize(torch.nan_to_num(sel.flat(gen.wave(ho_spec))), dim=-1)
    DIM = int(Ztr.shape[1])

    def same(c, w):
        return list(c) == list(w)

    def ymat(tg):
        Y = torch.zeros(len(tg), M * Vv)
        for i, ids in enumerate(tg):
            for p, t in enumerate(ids[:M]):
                Y[i, p * Vv + int(t)] = 1.0
        return Y

    n = min(ROWS, len(Ztr))
    Ytr = ymat(tr_tgt[:n]).reshape(n, M, Vv)
    # train-side ranking, so train execution labels can be harvested
    sc_tr = sel.ridge_scores(Ztr[:n], ymat(tr_tgt[:n]), Ztr[:n]).reshape(n, M, Vv)
    C_tr = torch.einsum("qmv,jmv->qj", sc_tr, Ytr)
    topk_tr = [torch.topk(C_tr[q], min(K, n)).indices.tolist() for q in range(n)]

    ok_waves = [Ztr[j] for q in range(n) for j in topk_tr[q]
                if same(tr_tgt[j], tr_tgt[q])]
    R["train_execution_labels"] = {
        "n_train_queries": n, "n_correct_pairs": len(ok_waves),
        "n_total_pairs": int(n * K),
        "rate": round(len(ok_waves) / (n * K), 6),
    }
    if not ok_waves:
        R["verdict"] = "NO_TRAIN_EXECUTION_LABELS"
        _write(R)
        return 3
    A_exec = F.normalize(torch.stack(ok_waves).mean(0, keepdim=True), dim=-1).squeeze(0)

    # execution clusters: k-means on the correct waves (few steps is enough to partition)
    W = torch.stack(ok_waves)
    g = torch.Generator().manual_seed(PIN)
    cent = W[torch.randperm(len(W), generator=g)[:CLUSTERS]].clone()
    for _ in range(40):
        d = W @ cent.T
        assign = d.argmax(1)
        for c in range(CLUSTERS):
            m = assign == c
            if int(m.sum()) > 0:
                cent[c] = F.normalize(W[m].mean(0, keepdim=True), dim=-1).squeeze(0)
    cent = F.normalize(cent, dim=-1)

    gr = torch.Generator().manual_seed(PIN + 1)
    A_rand = F.normalize(torch.randn(DIM, generator=gr), dim=-1)

    # held-out ranking and labels
    sc = sel.ridge_scores(Ztr[:n], ymat(tr_tgt[:n]), Zho).reshape(len(Zho), M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc, Ytr)
    topk = [torch.topk(C[q], min(K, n)).indices.tolist() for q in range(len(Zho))]

    def sagnac_delta(a, b):
        s = 0.5 * (1.0 + float(torch.dot(a.reshape(-1), b.reshape(-1))))
        return 1.0 - max(0.0, min(1.0, s))

    d0, d1, d2, d3, labels, qids = [], [], [], [], [], []
    for q in range(len(Zho)):
        qw = Zho[q]
        for j in topk[q]:
            cw = Ztr[j]
            d0.append(sagnac_delta(cw, qw))
            d1.append(sagnac_delta(cw, A_exec))
            d2.append(min(sagnac_delta(cw, cent[c]) for c in range(len(cent))))
            d3.append(sagnac_delta(cw, A_rand))
            labels.append(0 if same(tr_tgt[j], ho_tgt[q]) else 1)
            qids.append(j)          # CANDIDATE IDENTITY, needed for the prevalence test

    n_ok = sum(1 for y in labels if y == 0)
    R["heldout_pairs"] = {"n": len(labels), "n_correct": n_ok,
                          "n_incorrect": len(labels) - n_ok, "dim": DIM}
    # operating point from the TRAIN median of the same statistic (never held-out)
    vals0 = []
    for q in range(n):
        for j in topk_tr[q]:
            vals0.append(sagnac_delta(Ztr[j], Ztr[q]))
    vals0.sort()
    thr0 = vals0[len(vals0) // 2]
    vals1 = [sagnac_delta(Ztr[j], A_exec) for q in range(n) for j in topk_tr[q]]
    vals1.sort()
    thr1 = vals1[len(vals1) // 2]
    vals2 = [min(sagnac_delta(Ztr[j], cent[c]) for c in range(len(cent)))
             for q in range(n) for j in topk_tr[q]]
    vals2.sort()
    thr2 = vals2[len(vals2) // 2]
    vals3 = [sagnac_delta(Ztr[j], A_rand) for q in range(n) for j in topk_tr[q]]
    vals3.sort()
    thr3 = vals3[len(vals3) // 2]

    R["arms"] = {
        "A0_query_wave": report("A0_query_wave", d0, labels, thr0),
        "A1_exec_centroid": report("A1_exec_centroid", d1, labels, thr1),
        "A2_cluster_centroid": report("A2_cluster_centroid", d2, labels, thr2),
        "A3_random_axiom": report("A3_random_axiom", d3, labels, thr3),
    }

    # ---- WHO IS CORRECT? (prevalence diagnostic; tests the null's validity) -------
    # The global permutation null came back INVALID: the random-axiom control sat at
    # z=-11.3, far outside it. A random direction cannot carry correctness information, so
    # the null's EXCHANGEABILITY assumption must be false. The likely cause is candidate
    # IDENTITY: there are only ~1092 train traces and 572 correct pairs, so the correct
    # labels concentrate on a few trace identities. A global shuffle destroys that
    # concentration; a random projection can nonetheless read it. This measures whether
    # the concentration is real before any AUC is believed.
    from collections import Counter
    corr_ids = Counter(j for j, y in zip(qids, labels) if y == 0)
    all_ids = Counter(qids)
    n_distinct_corr = len(corr_ids)
    top_share = (max(corr_ids.values()) / max(1, sum(corr_ids.values()))) if corr_ids else 0.0
    R["prevalence_diagnostic"] = {
        "n_pairs": len(labels), "n_correct": sum(1 for y in labels if y == 0),
        "n_distinct_candidates": len(all_ids),
        "n_distinct_correct_candidates": n_distinct_corr,
        "correct_per_distinct_candidate": round(
            sum(1 for y in labels if y == 0) / max(1, n_distinct_corr), 4),
        "max_single_candidate_share_of_correct": round(top_share, 6),
        "top5_correct_candidates": corr_ids.most_common(5),
        "verdict": ("CONCENTRATED" if n_distinct_corr and
                    (sum(1 for y in labels if y == 0) / n_distinct_corr) > 2.0
                    else "SPREAD"),
        "implication": ("If CONCENTRATED, global label permutation is NOT a valid null: "
                        "candidate identity leaks correctness. Any AUC must then be "
                        "reported against a within-query null instead."),
    }

    # PERMUTATION NULLS. Two of them, because the first was demonstrably invalid.
    #   global : shuffle labels across all 11,340 pairs. Preserves class counts only.
    #            INVALID here if the prevalence diagnostic says CONCENTRATED.
    #   within : shuffle labels WITHIN each query's K=5 candidates. Preserves per-query
    #            composition, so candidate-identity prevalence cannot leak into the null.
    def rank_auc_q(scores, y_correct):
        return rank_auc(scores, y_correct)

    def perm_null_global(deltas, n_perm=200):
        import random
        scores = [-d for d in deltas]
        base = [0] * n_ok + [1] * (len(labels) - n_ok)
        rng = random.Random(PIN)
        out = []
        for _ in range(n_perm):
            y = list(base)
            rng.shuffle(y)
            out.append(rank_auc_q(scores, [1 - v for v in y]))
        out.sort()
        return out

    def perm_null_within(deltas, n_perm=200):
        """Shuffle labels inside each query block of K=5."""
        import random
        scores = [-d for d in deltas]
        rng = random.Random(PIN)
        blocks = [list(range(b * K, (b + 1) * K)) for b in range(len(labels) // K)]
        out = []
        for _ in range(n_perm):
            y = list(labels)
            for b in blocks:
                vals = [y[i] for i in b]
                rng.shuffle(vals)
                for i, v in zip(b, vals):
                    y[i] = v
            out.append(rank_auc_q(scores, [1 - v for v in y]))
        out.sort()
        return out

    def summarize(nl, obs):
        mu = sum(nl) / len(nl)
        sd = math.sqrt(sum((v - mu) ** 2 for v in nl) / max(1, len(nl) - 1))
        lo, hi = nl[int(0.025 * len(nl))], nl[int(0.975 * len(nl))]
        z = (obs - mu) / sd if sd > 0 else float("nan")
        return {"auc_observed": obs, "null_mean": round(mu, 6),
                "null_sd": round(sd, 6), "null_p2.5": round(lo, 6),
                "null_p97.5": round(hi, 6),
                "z": round(z, 3) if math.isfinite(z) else "UNDEFINED",
                "outside_null_95": bool(obs < lo or obs > hi)}

    R["permutation_null_global"] = {}
    R["permutation_null_within_query"] = {}
    for name, dd in (("A0_query_wave", d0), ("A1_exec_centroid", d1),
                     ("A2_cluster_centroid", d2), ("A3_random_axiom", d3)):
        obs = R["arms"][name]["auc_neg_delta_predicts_correct"]
        R["permutation_null_global"][name] = summarize(perm_null_global(dd), obs)
        R["permutation_null_within_query"][name] = summarize(perm_null_within(dd), obs)
    R["permutation_null"] = R["permutation_null_within_query"]   # primary = valid one

    # ---- PER-QUERY TOP-1: the only statistic here that CANNOT be identity-confounded --
    # WHY THIS REPLACES THE AUC. Every arm scores delta(cand, A) for a FIXED A, so its
    # value depends on candidate identity alone -- not on the query. Prevalence is
    # CONCENTRATED (139 distinct candidates, 22 distinct correct, 26 correct pairs per
    # correct candidate), so the AUC is comparing IDENTITY SETS. That is why the
    # random-axiom control also lands outside both permutation nulls (within-query
    # z=-6.04): a fixed random direction reads which identities are correct. The AUC
    # therefore cannot certify a veto, and NO arm's AUC separation is promoted.
    # Per-query top-1 forces the comparison WITHIN one query's K candidates, which is the
    # actual decision the veto must make, so identity prevalence cannot inflate it.
    blocks = [list(range(b * K, (b + 1) * K)) for b in range(len(labels) // K)]
    pos_blocks = [b for b in blocks if any(labels[i] == 0 for i in b)]

    def per_query_top1(deltas):
        strict = tieset = 0
        for b in pos_blocks:
            vals = [deltas[i] for i in b]
            mn = min(vals)
            argmin_idx = [i for i in b if deltas[i] == mn]
            pick = argmin_idx[0]                      # first index on an exact tie
            strict += int(labels[pick] == 0)
            tieset += int(any(labels[i] == 0 for i in argmin_idx))
        n = max(1, len(pos_blocks))
        return {"n_queries_with_a_correct_candidate": len(pos_blocks),
                "top1_strict": round(strict / n, 6),
                "top1_any_in_argmin_set": round(tieset / n, 6),
                "chance_top1": round(sum(1 for i in pos_blocks[0]) / K, 6)}

    R["per_query_top1"] = {
        "A0_query_wave": per_query_top1(d0),
        "A1_exec_centroid": per_query_top1(d1),
        "A2_cluster_centroid": per_query_top1(d2),
        "A3_random_axiom": per_query_top1(d3),
    }
    t1 = R["per_query_top1"]

    # ANTI-SELECTION is now defined per-query, where it cannot be confounded: does the
    # veto's preferred candidate (lowest delta) beat the LEAST-preferred one (highest
    # delta) at being correct, within the same query?
    def within_query_preference(deltas):
        pref_ok = unpref_ok = 0
        for b in pos_blocks:
            lo = min(b, key=lambda i: deltas[i])     # veto prefers lowest delta
            hi = max(b, key=lambda i: deltas[i])     # veto least prefers highest delta
            pref_ok += int(labels[lo] == 0)
            unpref_ok += int(labels[hi] == 0)
        n = max(1, len(pos_blocks))
        return {"preferred_is_correct_rate": round(pref_ok / n, 6),
                "least_preferred_is_correct_rate": round(unpref_ok / n, 6),
                "preference_advantage": round((pref_ok - unpref_ok) / n, 6)}

    R["within_query_preference"] = {
        "A0_query_wave": within_query_preference(d0),
        "A1_exec_centroid": within_query_preference(d1),
        "A2_cluster_centroid": within_query_preference(d2),
        "A3_random_axiom": within_query_preference(d3),
    }
    wq = R["within_query_preference"]

    def auc_of(a):
        v = R["arms"][a]["auc_neg_delta_predicts_correct"]
        return v if isinstance(v, float) else float("nan")

    a0, a1, a2, a3 = auc_of("A0_query_wave"), auc_of("A1_exec_centroid"), \
        auc_of("A2_cluster_centroid"), auc_of("A3_random_axiom")
    # Reading rule now keys on per-query top-1 and within-query preference, both of which
    # are identity-safe. AUC values are reported but DO NOT gate.
    ctrl_safe = t1["A3_random_axiom"]["top1_strict"] <= 0.25
    R["checks"] = {
        "auc_invalid_identity_confounded": bool(R["prevalence_diagnostic"]["verdict"]
                                                == "CONCENTRATED"),
        "control_top1_near_chance": bool(ctrl_safe),
        "inverted_arm_reproduced": bool(a0 < 0.5
                                        and wq["A0_query_wave"]["preference_advantage"] <= 0),
        "exec_centroid_per_query_gain": bool(
            t1["A1_exec_centroid"]["top1_strict"] > 0.25
            and wq["A1_exec_centroid"]["preference_advantage"] > 0),
        "cluster_centroid_per_query_gain": bool(
            t1["A2_cluster_centroid"]["top1_strict"] > 0.25
            and wq["A2_cluster_centroid"]["preference_advantage"] > 0),
    }
    c = R["checks"]
    if not c["inverted_arm_reproduced"]:
        R["verdict"] = "HARNESS_MISMATCH_INVERSION_NOT_REPRODUCED"
    elif not ctrl_safe:
        R["verdict"] = "HARNESS_CONTROL_BEATS_CHANCE_PER_QUERY_STILL_UNEXPLAINED"
    elif c["exec_centroid_per_query_gain"] or c["cluster_centroid_per_query_gain"]:
        R["verdict"] = "EXECUTION_AXIOM_SEPARATES_PER_QUERY"
    else:
        R["verdict"] = "NO_PER_QUERY_SIGNAL_IN_THIS_FEATURE_SET"
    R["provisional_read"] = {
        "reading_rule": ("per-query top-1 and within-query preference gate; AUC is "
                         "reported but does not gate because it is identity-confounded"),
        "auc_invalidated_by": "prevalence_diagnostic.verdict == CONCENTRATED",
        "exec_centroid": t1["A1_exec_centroid"],
        "cluster_centroid": t1["A2_cluster_centroid"],
        "query_wave": t1["A0_query_wave"],
        "control": t1["A3_random_axiom"],
    }
    R["interpretation"] = {
        "inversion_is_the_information_identity_defect": (
            "A0 consumes the same query wave the ridge ranker used to build the ranking, "
            "so a monotone function of it cannot re-rank. Its inversion reproduces the "
            "predicted signature."),
        "threshold_avoided": (
            "epsilon_hard=0.35 sits below the ~0.5 delta of two independent unit waves, so "
            "a thresholded trigger saturates and measures the base rate. Deltas are "
            "therefore scored continuously (AUC) with the operating point taken from train."),
        "scope": "encoder waves, NOT EFE-planned candidates; no task attempted or solved.",
    }
    R["not_claims"] = ["no dark-port wiring", "no model-performance claim",
                       "no task attempted or solved"]
    R["elapsed_s"] = round(time.time() - t0, 1)
    _write(R)
    return 0


def _write(R):
    from receipt_path import resolve_receipt_path
    default = os.path.join(REPO, "design", "zone_a", "evidence",
                           "henri_execution_axiom_probe_observed.json")
    path = resolve_receipt_path(default)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    print(f"[receipt] {path}", file=sys.stderr)
    return path


if __name__ == "__main__":
    sys.exit(main())
