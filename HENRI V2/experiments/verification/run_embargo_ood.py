#!/usr/bin/env python3
"""PILLAR 4 — out-of-distribution generalization under TASK-LEVEL embargo.

WHY THIS EXISTS
    The 0.7833 accuracy in calibration_eval_observed.json fits the operator W on
    each task's OWN demonstration pairs and then scores that same task's test
    input. That is per-task adaptation, not generalization: it never asks whether
    an operator learned on some tasks transfers to tasks it has never seen. The
    named pillar is OOD generalization under embargo, so this harness measures
    exactly that.

WHAT IS EMBARGOED
    A single batch operator is fitted on the TRAIN task IDs only. It is then
    evaluated on DISJOINT task IDs whose inputs never entered the fit. The split
    is computed from a frozen seed BEFORE any scoring, and the indices plus task
    IDs are written into the receipt, so the split cannot be re-chosen after
    seeing results.

HONEST SCOPING — WHAT "TEMPORAL" DOES AND DOES NOT MEAN HERE
    ARC-AGI tasks carry no timestamp. There is therefore no true time axis to
    embargo, and this harness does NOT claim a time-series embargo. What it
    implements is TASK-DISJOINT OOD: the evaluation tasks are unseen identities,
    and no demonstration pair from them touches the fit. Calling this "temporal"
    would be a labelling upgrade the data does not support.

PRE-REGISTERED CLAIMS (fixed before running)
    E1 EMBARGOED SIGNAL: embargoed accuracy > chance (1/K). If not, the operator
       transfers nothing and the pillar is unmet for this operator.
    E2 OPERATOR HEADROOM: report embargoed operator accuracy MINUS embargoed
       identity accuracy. Prior measurement puts the operator's gain near +0.05;
       this is re-measured under a task-disjoint split rather than a within-task
       one. A negative value means the operator HURTS transfer and is reported.
    E3 CONTENT CONTROL: a seeded random operator must not beat identity on the
       embargoed set by a margin comparable to the fitted operator's. If it does,
       the fit carries no task information and the arm is VOID.
    E4 BOTH SIDES REPORTED: in-sample (train-task) and embargoed numbers are
       always emitted together. An in-sample-only number overstates transfer.
"""
import json
import pathlib
import sys
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
VERIF = R / "experiments" / "verification"
# run_calibration_eval lives in experiments/verification, NOT the repo root.
sys.path.insert(0, str(VERIF))
sys.path.insert(0, str(R))

from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402
import run_calibration_eval as RCE  # noqa: E402

OUT = VERIF / "embargo_ood_observed.json"

N_TASKS = 24
HOLDOUT_FRACTION = 0.5
SPLIT_SEED = 20260918
RANDOM_OP_SEED = 777
D_MODEL = 65536
N_BLOCKS, BLOCK_DIM = 8192, 8
K = RCE.K_OPTIONS


def build_encoder():
    kind, bg = RCE.resolve_spatial_basis()
    return HENRIVisionEncoder(d_model=D_MODEL, k_blocks=N_BLOCKS,
                              block_dim=BLOCK_DIM, device="cpu",
                              spatial_basis_kind=kind, bg_mask=bg), kind, bg


def encode(enc, grid):
    try:
        w = enc.encode_spatial_grid([list(r) for r in grid])
    except Exception:
        return None
    if w is None:
        return None
    v = w.squeeze(0).reshape(-1).detach().to(torch.float32)
    if v.numel() != D_MODEL or not torch.isfinite(v).all():
        return None
    return F.normalize(v, p=2.0, dim=-1)


def fit_batch_operator(enc, tasks):
    """One operator over ALL demo pairs of ALL given tasks.

    This is the fit that makes the test OOD: the operator is shared across tasks,
    so an evaluation task contributes NOTHING to it.
    """
    num = torch.zeros(D_MODEL, dtype=torch.float32)
    den = torch.zeros(D_MODEL, dtype=torch.float32)
    n_pairs = 0
    skipped = []
    for tid, t in tasks:
        for pair in t["train"][:3]:
            x = encode(enc, pair["input"])
            y = encode(enc, pair["output"])
            if x is None or y is None:
                continue
            num = num + (x * y)
            den = den + (x * x)
            n_pairs += 1
        if n_pairs == 0:
            skipped.append(tid)
    W = num / (den + RCE.RIDGE_EPS)
    return W, n_pairs, skipped


def evaluate(enc, tasks, W, random_W=None):
    """Score the fitted arm, identity arm and (optionally) a random-operator arm."""
    n = acc_w = acc_id = acc_rnd = 0
    rows = []
    for tid, t in tasks:
        try:
            te = t["test"][0]
            x = encode(enc, te["input"])
            if x is None:
                continue
            cands, truth, _seed, kinds = RCE.build_candidates(te["output"], tid)
            cw = torch.stack([encode(enc, c) for c in cands])
            if not torch.isfinite(cw).all():
                continue
            pred = W * x
            sw = torch.tensor([float(F.cosine_similarity(pred, c, dim=0))
                               for c in cw])
            si = torch.tensor([float(F.cosine_similarity(x, c, dim=0)) for c in cw])
            cw_w = int(int(torch.argmax(sw).item()))
            cw_i = int(int(torch.argmax(si).item()))
            cw_r = None
            if random_W is not None:
                pr = random_W * x
                sr = torch.tensor([float(F.cosine_similarity(pr, c, dim=0))
                                   for c in cw])
                cw_r = int(int(torch.argmax(sr).item()))
                acc_rnd += cw_r == truth
            n += 1
            acc_w += cw_w == truth
            acc_id += cw_i == truth
            rows.append({
                "task_id": tid, "truth_index": int(truth),
                "op_choice": cw_w, "identity_choice": cw_i,
                "random_op_choice": cw_r,
                "op_correct": bool(cw_w == truth),
                "identity_correct": bool(cw_i == truth),
                "random_op_correct": (None if cw_r is None else bool(cw_r == truth)),
                "option_kinds": kinds,
            })
        except RCE.CandidateError:
            continue
        except Exception as exc:  # noqa: BLE001
            rows.append({"task_id": tid, "error": f"{type(exc).__name__}: {exc}"})
    usable = [r for r in rows if "error" not in r]
    n = len(usable)
    acc_w = sum(1 for r in usable if r["op_correct"])
    acc_id = sum(1 for r in usable if r["identity_correct"])
    acc_rnd = sum(1 for r in usable if r["random_op_correct"]) if random_W is not None else None
    return {
        "n": n,
        "acc_operator": (acc_w / n) if n else None,
        "acc_identity": (acc_id / n) if n else None,
        "acc_random_operator": (acc_rnd / n) if (n and acc_rnd is not None) else None,
        "operator_gain_over_identity": ((acc_w - acc_id) / n) if n else None,
        "chance": 1.0 / K,
        "rows": rows,
    }


def main():
    print("=" * 78)
    print("PILLAR 4 — TASK-LEVEL EMBARGO / OOD GENERALIZATION (CPU)")
    print("=" * 78)
    enc, kind, bg = build_encoder()
    print(f"encoder = HENRIVisionEncoder(kind={kind!r}, bg_mask={bg})  d_model={D_MODEL}")

    all_tasks = RCE.load_arc(RCE.ARC_ROOT, N_TASKS)
    print(f"tasks loaded: {len(all_tasks)} from {RCE.ARC_ROOT}")
    if len(all_tasks) < 8:
        print("BLOCKED_NO_CORPUS")
        return 1
    all_tasks = [(tid, t) for tid, t in all_tasks[:N_TASKS]]

    # ---- the split, frozen before any scoring and recorded in the receipt
    order = list(range(len(all_tasks)))
    torch.Generator().manual_seed(SPLIT_SEED)
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(all_tasks), generator=g).tolist()
    n_hold = max(2, int(round(len(all_tasks) * HOLDOUT_FRACTION)))
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
    train_tasks = [all_tasks[i] for i in train_idx]
    hold_tasks = [all_tasks[i] for i in hold_idx]
    print(f"split: train={len(train_tasks)} tasks  embargoed={len(hold_tasks)} tasks")
    print(f"  train IDs   : {[t[0] for t in train_tasks][:6]} ...")
    print(f"  embargo IDs : {[t[0] for t in hold_tasks]}")

    W, n_pairs, skipped = fit_batch_operator(enc, train_tasks)
    print(f"batch operator fitted on {n_pairs} demo pairs from train tasks "
          f"(skipped {len(skipped)})")

    gen = torch.Generator().manual_seed(RANDOM_OP_SEED)
    W_rand = torch.randn(D_MODEL, generator=gen)

    print("\n  --- IN-SAMPLE (train task IDs, included in the fit) ---")
    tr = evaluate(enc, train_tasks, W, random_W=W_rand)
    print(f"    n={tr['n']}  op={tr['acc_operator']}  identity={tr['acc_identity']}  "
          f"random={tr['acc_random_operator']}  gain={tr['operator_gain_over_identity']}")

    print("\n  --- EMBARGOED (disjoint task IDs, never in the fit) ---")
    ho = evaluate(enc, hold_tasks, W, random_W=W_rand)
    print(f"    n={ho['n']}  op={ho['acc_operator']}  identity={ho['acc_identity']}  "
          f"random={ho['acc_random_operator']}  gain={ho['operator_gain_over_identity']}")

    def num(d, k):
        v = d.get(k)
        return 0.0 if v is None else float(v)

    chance = 1.0 / K
    pre = {
        "E1_embargoed_beats_chance": bool(num(ho, "acc_operator") > chance),
        "E2_operator_gain_reported": True,
        "E2_operator_helps_on_embargoed": bool(
            num(ho, "operator_gain_over_identity") > 0.0),
        "E3_random_operator_does_not_match_gain": bool(
            num(ho, "acc_random_operator") <= chance + max(
                0.05, abs(num(ho, "operator_gain_over_identity")))),
        "E4_both_sides_reported": bool(tr["n"] and ho["n"]),
        "E5_split_frozen_and_recorded": True,
    }

    if not pre["E4_both_sides_reported"]:
        verdict = "BLOCKED — one side produced no usable tasks"
    elif not pre["E1_embargoed_beats_chance"]:
        verdict = ("PILLAR_4_UNMET — the batch operator transfers NOTHING to "
                   "embargoed task identities; accuracy at chance")
    elif not pre["E3_random_operator_does_not_match_gain"]:
        verdict = ("VOID_CONTROL — a random operator matches the fitted operator's "
                   "embargoed behaviour, so the fit carries no task information")
    elif pre["E2_operator_helps_on_embargoed"]:
        verdict = ("PILLAR_4_PARTIAL — the batch operator generalizes ABOVE chance "
                   "to unseen task identities and beats identity, but the gain is "
                   "small; report the number, not a capability claim")
    else:
        verdict = ("PILLAR_4_NEGATIVE — the operator transfers above chance but "
                   "does NOT beat the identity baseline out of distribution; the "
                   "learned operator is not the source of generalization")

    body = {
        "schema": "henri.embargo-ood.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "OBSERVED",
        "evidence_class_note": (
            "OBSERVED: production HENRIVisionEncoder encode + the production "
            "candidate construction. Accuracy counts are DERIVED from per-task rows."
        ),
        "pillar": "4 — out-of-distribution generalization under embargo",
        "scoping": (
            "ARC tasks carry no timestamp, so this is TASK-DISJOINT OOD, not a "
            "time-series embargo. No temporal claim is made."
        ),
        "design": {
            "n_tasks_total": len(all_tasks),
            "n_train_tasks": len(train_tasks),
            "n_embargoed_tasks": len(hold_tasks),
            "split_seed": SPLIT_SEED,
            "holdout_fraction": HOLDOUT_FRACTION,
            "train_task_ids": [t[0] for t in train_tasks],
            "embargoed_task_ids": [t[0] for t in hold_tasks],
            "fit": "single batch per-slot diagonal ridge LS over ALL train-task demo pairs",
            "arms": ["batch_operator", "identity", "random_operator"],
            "random_operator_seed": RANDOM_OP_SEED,
            "encoder": f"HENRIVisionEncoder(kind={kind!r}, bg_mask={bg})",
            "corpus": str(RCE.ARC_ROOT),
            "n_fit_pairs": n_pairs,
        },
        "in_sample_train_tasks": {k: v for k, v in tr.items() if k != "rows"},
        "embargoed_tasks": {k: v for k, v in ho.items() if k != "rows"},
        "embargoed_rows": ho["rows"],
        "pre_registered": pre,
        "verdict": verdict,
        "limits": [
            "Task-disjoint OOD, NOT temporal embargo (no time axis exists in ARC).",
            "Not a benchmark score; candidates are pre-built per task.",
            "One batch operator family (per-slot diagonal ridge LS) only.",
            "CPU only; no CUDA (Vast 50797414 EXITED, credit 0).",
            "The in-sample number is always reported beside the embargoed one.",
        ],
    }
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print()
    for k, v in pre.items():
        print(f"  {k:<44} {v}")
    print(f"  VERDICT: {verdict}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
