"""CIRCULARITY KILL TEST -- is the execution-axiom signal a VERIFIER or a LOOKUP TABLE?

WHY THIS EXISTS
    run_execution_axiom_probe.py measured per-query top-1 of 0.9846 for the execution
    centroid, versus 0.3745 for a random-axiom control. Those numbers LOOK decisive. They
    are not yet evidence, for two reasons this module tests directly.

    REASON 1 -- THE AXIOM IS BUILT FROM THE SET IT PREDICTS.
        A_exec is the normalised mean wave of TRAIN candidates whose traces execute
        correctly. Scoring a candidate by proximity to that centroid asks "is this
        candidate one of the known-correct identities?" -- which is the label, not a
        prediction of it. That is a lookup table by construction.
        KILL TEST: rebuild the axiom from train-correct traces EXCLUDING every identity that
        appears as a correct candidate on the held-out side. If top-1 collapses to the
        random-axiom control, the original number was circular.

    REASON 2 -- LABEL CONCENTRATION.
        The held-out surface has 572 correct pairs over only 22 distinct candidate
        identities (139 distinct candidates total). A fixed axiom scores a candidate
        identically in every query, so any per-candidate score inherits that structure.
        The random-axiom control reaching 0.3745 is the measured size of this effect.
        KILL TEST: report the identity overlap explicitly, and the chance level computed
        from the ACTUAL number of correct candidates per query.

    FIXED IN THIS MODULE: the earlier probe reported chance_top1 = len(block)/K = 1.0,
    which is not a chance level. Chance here is (mean correct candidates per query)/K.

READING RULE (frozen)
    If leave_identity_out top-1 is NOT clearly above the random-axiom control
    (margin > 0.10), the signal is CIRCULAR and no attribution claim is made.
    If it IS clearly above, the signal survives identity removal and is admitted as
    PROVISIONAL evidence pending the encoder rewrite.
    Either way: no dark-port wiring, no model-performance claim, no task attempted.
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

EXPECTED_M4_SHA256 = "fb46c19b8c15c27cb363eb91eed78f7a0707905426096f4c25d672f42b3ff316"
PIN = sel.PIN
ROWS = sel.ROWS
K = sel.K
MARGIN = 0.10


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    t0 = time.time()
    resolved = os.path.abspath(sys.modules["henri_core.m4_generative"].__file__)
    actual = sha256_file(resolved)
    R = {"schema": "henri.execution.axiom.circularity_test.v1", "pin": PIN, "k": K,
         "margin": MARGIN,
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
    sc_tr = sel.ridge_scores(Ztr[:n], ymat(tr_tgt[:n]), Ztr[:n]).reshape(n, M, Vv)
    C_tr = torch.einsum("qmv,jmv->qj", sc_tr, Ytr)
    topk_tr = [torch.topk(C_tr[q], min(K, n)).indices.tolist() for q in range(n)]

    sc = sel.ridge_scores(Ztr[:n], ymat(tr_tgt[:n]), Zho).reshape(len(Zho), M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc, Ytr)
    topk = [torch.topk(C[q], min(K, n)).indices.tolist() for q in range(len(Zho))]

    # ---- identity sets -----------------------------------------------------------
    train_correct = set()          # train candidate j that is correct for its OWN query
    for q in range(n):
        for j in topk_tr[q]:
            if same(tr_tgt[j], tr_tgt[q]):
                train_correct.add(j)
    held_correct = set()           # train candidate j that is correct for a HELD query
    for q in range(len(Zho)):
        for j in topk[q]:
            if same(tr_tgt[j], ho_tgt[q]):
                held_correct.add(j)
    overlap = train_correct & held_correct
    R["identity_surface"] = {
        "n_train_candidates": n, "n_held_queries": len(Zho),
        "n_distinct_train_correct_identities": len(train_correct),
        "n_distinct_held_correct_identities": len(held_correct),
        "n_overlap": len(overlap),
        "overlap_share_of_held_correct": round(len(overlap) / max(1, len(held_correct)), 6),
        "verdict": ("CIRCULAR_BY_CONSTRUCTION"
                    if len(overlap) == len(held_correct) and held_correct else
                    ("PARTIAL_OVERLAP" if overlap else "DISJOINT")),
    }

    def sagnac_delta(a, b):
        s = 0.5 * (1.0 + float(torch.dot(a.reshape(-1), b.reshape(-1))))
        return 1.0 - max(0.0, min(1.0, s))

    def centroid_of(ids):
        ids = sorted(ids)
        if not ids:
            return None
        return F.normalize(torch.stack([Ztr[j] for j in ids]).mean(0, keepdim=True),
                           dim=-1).squeeze(0)

    A_full = centroid_of(train_correct)
    A_lio = centroid_of(set(train_correct) - overlap)          # LEAVE-IDENTITY-OUT
    g = torch.Generator().manual_seed(PIN + 1)
    A_rand = F.normalize(torch.randn(DIM, generator=g), dim=-1)

    blocks = [list(range(b * K, (b + 1) * K)) for b in range(len(Zho))]
    labels = []
    for q in range(len(Zho)):
        for j in topk[q]:
            labels.append(0 if same(tr_tgt[j], ho_tgt[q]) else 1)
    pos_blocks = [b for b in blocks if any(labels[i] == 0 for i in b)]
    # chance from the ACTUAL correct-candidate count per query (earlier bug: len(block)/K)
    chance = sum(sum(1 for i in b if labels[i] == 0) for b in pos_blocks) / (len(pos_blocks) * K)

    def cand_at(b_i):
        q = b_i // K
        return topk[q][b_i - q * K]

    def top1_clean(axiom):
        """Per-query top-1: is a correct candidate in the argmin (veto-preferred) set?

        Identity-safe by construction: the comparison never leaves one query's K block, so
        the 572/22 label concentration cannot inflate it.
        """
        if axiom is None:
            return {"top1_any_in_argmin_set": None,
                    "note": "axiom empty after identity removal"}
        hits = 0
        for b in pos_blocks:
            vals = [sagnac_delta(Ztr[cand_at(i)], axiom) for i in b]
            mn = min(vals)
            hits += int(any(labels[i] == 0 for i, v in zip(b, vals) if v == mn))
        return {"top1_any_in_argmin_set": round(hits / max(1, len(pos_blocks)), 6)}

    R["arms"] = {
        "chance_from_actual_composition": round(chance, 6),
        "A_full_centroid_of_all_train_correct": top1_clean(A_full),
        "A_leave_identity_out": top1_clean(A_lio),
        "A_random_axiom_control": top1_clean(A_rand),
    }
    full_v = R["arms"]["A_full_centroid_of_all_train_correct"]["top1_any_in_argmin_set"]
    lio_v = R["arms"]["A_leave_identity_out"]["top1_any_in_argmin_set"]
    rnd_v = R["arms"]["A_random_axiom_control"]["top1_any_in_argmin_set"]
    margin_over_control = (None if lio_v is None else round(lio_v - rnd_v, 6))

    R["checks"] = {
        "held_correct_identities_subset_of_train_correct"
            : bool(len(overlap) == len(held_correct) and held_correct),
        "random_control_above_true_chance": bool(rnd_v is not None and rnd_v > chance + 0.05),
        "leave_identity_out_beats_control": bool(margin_over_control is not None
                                                 and margin_over_control > MARGIN),
        "full_centroid_exceeds_leave_identity_out": bool(
            full_v is not None and lio_v is not None and full_v - lio_v > 0.05),
    }
    c = R["checks"]
    if c["held_correct_identities_subset_of_train_correct"] and not c[
            "leave_identity_out_beats_control"]:
        R["verdict"] = "SIGNAL_IS_CIRCULAR_LOOKUP_TABLE"
    elif c["leave_identity_out_beats_control"]:
        R["verdict"] = "SIGNAL_SURVIVES_IDENTITY_REMOVAL_PROVISIONAL"
    else:
        R["verdict"] = "INDETERMINATE_NEEDS_OWN_PAIRED_BAR"
    R["interpretation"] = {
        "why_full_centroid_is_not_evidence": (
            "A_full is the centroid of the very identities it is scored against, so a high "
            "top-1 recovers the label set by construction. The leave-identity-out arm is the "
            "only admissible test."),
        "why_the_random_control_matters": (
            "A fixed axiom scores a candidate identically in every query, so it reads the "
            "identity concentration (22 distinct correct identities over 572 pairs). The "
            "random arm measures that floor; any arm must clear it."),
        "earlier_defect_corrected": (
            "the previous probe reported chance_top1 from len(block)/K, which is 1.0 and not a "
            "chance level. Chance is (mean correct candidates per query)/K."),
        "scope": "encoder waves, NOT EFE-planned candidates; no task attempted or solved.",
    }
    R["not_claims"] = ["no dark-port wiring", "no model-performance claim",
                       "no selection improvement claim", "no task attempted or solved"]
    R["elapsed_s"] = round(time.time() - t0, 1)
    _write(R)
    return 0


def _under_pytest() -> bool:
    """True when a test harness drives this module. Guards a tracked-receipt write."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST")
                or os.environ.get("PYTEST_VERSION"))


def _write(R):
    from receipt_path import resolve_receipt_path
    default = os.path.join(REPO, "design", "zone_a", "evidence",
                           "henri_execution_axiom_circularity_observed.json")
    path = resolve_receipt_path(default)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Guard propagated from sagnac_scale_defect.py; see run_execution_axiom_probe.py.
    if _under_pytest() and os.path.abspath(path) == os.path.abspath(default):
        print(json.dumps(R, indent=1, default=str))
        print(f"[receipt] NOT written under pytest (no override): {path}", file=sys.stderr)
        return path
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    print(f"[receipt] {path}", file=sys.stderr)
    return path


if __name__ == "__main__":
    sys.exit(main())
