#!/usr/bin/env python3
"""S1: the FIRST empirical calibration receipt for a HENRI head, on 60 ARC tasks.

WHAT THIS MEASURES
    Whether the Zone A wave readout produces probabilities whose stated
    confidence TRACKS accuracy, on 60 ARC-AGI tasks.

THE TASK (a K-way choice, K=4, chance = 0.25)
    For each ARC task: fit a per-slot diagonal ridge least-squares operator W on
    the task's demonstration pairs (the same operator family as the sealed
    60-task harness, in the real UWE domain). Ask the wave readout which of 4
    candidates is the true test output:
        scores_k = cos(W . X_test, C_k)
    Candidates are the true test output plus 3 deterministic, mostly
    shape-preserving transforms of it (rotations, flips, colour cycles). The
    true option is NOT privileged in the scoring, and option ORDER is shuffled
    under a per-task seed, so there is no position bias. Truth is ARC corpus
    ground truth.

WHY THIS IS NOT CIRCULAR
    Probabilities come from the DEMO-fit operator; truth is the HELD-OUT test
    output. That is a generalisation question, not the system scoring itself.
    A null control (truth labels shuffled) is included: it must fall to chance.

WHAT IT DOES NOT ESTABLISH
    - NOT a task score and NOT a benchmark score.
    - NOT a capability claim about the planner.
    - NOT that HENRI "is calibrated": the label is reported exactly as
      `is_well_calibrated` computes it, never threshold-shopped.
    - The measured head is the wave readout over a choice task, not the planner.
    - ECE is a function of the stated temperature; a T sweep is reported so the
      number cannot be read as temperature-independent.

GROUND TRUTH SOURCE (and why it is not the receipts)
    The blueprint proposed taking (state, options, truth) triples from the
    sealed 60-task receipts. Probing all 21 receipts 2026-09-18 showed they
    store AGGREGATE statistics only (arms -> held_out_mean / gap /
    frac_tasks_beating_identity); NONE contains per-task rows, and the sealed
    runner derives truth at runtime without persisting it. So truth is taken
    from the local ARC corpus and THIS runner emits per-task rows, so the
    calibration dataset can be rebuilt from a local corpus at zero API cost.

FAIL-CLOSED
    A task is either ROWED, SKIPPED for a typed reason (degenerate candidate
    set / encoder refusal), or ERRORED. Nothing is silently dropped and no row
    is fabricated. Status is OK only when there are no errors at all.

Usage:
    python experiments/verification/run_calibration_eval.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
VERIF = REPO / "experiments" / "verification"

from arc_egress_contract import probe_from_logits, state_snapshot_id_of  # noqa: E402
from arc_spatial_basis import resolve_spatial_basis                      # noqa: E402
from henri_probe_calibration import build_calibration_receipt            # noqa: E402
from henri_vision_encoder import HENRIVisionEncoder                      # noqa: E402

ARC_ROOT = os.environ.get("ARC_CORPUS", "C:/Users/chan/henri_data/ARC-AGI/data")
N_TASKS = int(os.environ.get("ARC_N_TASKS", "60"))
K_OPTIONS = 4
TEMPERATURE = 1.0
T_SWEEP = (0.25, 0.5, 1.0, 2.0, 4.0)
RIDGE_EPS = 1e-9
D_MODEL = 65536
N_BLOCKS = 8192
BLOCK_DIM = 8
SHUFFLE_SEED = 20260918
RANDOM_ARM_SEED = 424242

OUT = VERIF / "calibration_eval_observed.json"


class CandidateError(RuntimeError):
    """Fewer than K distinct candidates could be built for this task."""


class EncoderRefused(RuntimeError):
    """The encoder legitimately refused a grid (fail-closed). Not a bug."""


def load_arc(root: str, n: int):
    """VERBATIM replica of the sealed 60-task loader. Do not alter.

    Same split order, same sort, same filters, so the task set is identical to
    the one in evaluate_60_task_koopman_gap_observed.json.
    """
    out = []
    for split in ("training", "evaluation"):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                try:
                    t = json.load(open(os.path.join(d, f), encoding="utf-8"))
                except Exception:
                    continue
                if t.get("train") and t.get("test"):
                    out.append((f[:-5], t))
                if len(out) >= n:
                    return out
    return out


# ---- deterministic candidate transforms ------------------------------------
# Shape-preserving transforms come FIRST so the choice is rarely decidable by
# output shape alone; transpose (potentially shape-changing) comes last.

def _t_rot(g, k):
    return np.rot90(np.asarray(g), k).tolist()


def _t_flipud(g):
    return np.flipud(np.asarray(g)).tolist()


def _t_fliplr(g):
    return np.fliplr(np.asarray(g)).tolist()


def _t_transpose(g):
    return np.asarray(g).T.tolist()


def _t_color_cycle(g, shift=1):
    """Cyclically remap the non-zero colours. None when <2 colours exist."""
    arr = np.asarray(g)
    vals = sorted({int(v) for v in arr.flatten() if int(v) != 0})
    if len(vals) < 2:
        return None
    lut = {v: vals[(i + shift) % len(vals)] for i, v in enumerate(vals)}
    out = arr.copy()
    for a, b in lut.items():
        out[arr == a] = b
    return out.tolist()


POOL = [
    ("rot90", lambda g: _t_rot(g, 1)),
    ("rot180", lambda g: _t_rot(g, 2)),
    ("rot270", lambda g: _t_rot(g, 3)),
    ("flipud", _t_flipud),
    ("fliplr", _t_fliplr),
    ("color_shift1", lambda g: _t_color_cycle(g, 1)),
    ("color_shift2", lambda g: _t_color_cycle(g, 2)),
    ("transpose", _t_transpose),
]


def build_candidates(y_true, task_id: str):
    """(candidates, truth_index, seed, kinds). Raises CandidateError if short.

    The true output is index 0 before shuffling, so truth_index is recovered
    from the permutation rather than hard-coded.
    """
    base = [list(r) for r in y_true]
    cands = [base]
    kinds = ["TRUE"]
    for name, fn in POOL:
        if len(cands) >= K_OPTIONS:
            break
        try:
            t = fn(base)
        except Exception:
            continue
        if t is None:
            continue
        if any(np.array_equal(np.asarray(t), np.asarray(c)) for c in cands):
            continue
        cands.append(t)
        kinds.append(name)
    if len(cands) < K_OPTIONS:
        raise CandidateError(
            f"only {len(cands)} distinct candidates from {len(POOL)} transforms "
            f"(grid {len(base)}x{len(base[0])})"
        )
    seed = SHUFFLE_SEED + (int(hashlib.sha256(task_id.encode()).hexdigest(), 16)
                           % 100000)
    perm = np.random.default_rng(seed).permutation(K_OPTIONS)
    cands = [cands[i] for i in perm]
    kinds = [kinds[i] for i in perm]
    truth_index = int(np.where(perm == 0)[0][0])
    return cands, truth_index, int(seed), kinds


def softmax_rows(score_rows, temperature: float):
    """Row-wise softmax over cosine scores. Numpy twin of the torch path."""
    out = []
    for row in score_rows:
        s = np.asarray(row, dtype=np.float64) / float(temperature)
        s = s - s.max()
        e = np.exp(s)
        out.append((e / e.sum()).tolist())
    return out


def main() -> int:
    t0 = time.time()
    resolve_kind, bg_mask = resolve_spatial_basis()
    print(f"device=cpu torch={torch.__version__}")
    print(f"basis={resolve_kind!r} bg_mask={bg_mask} K={K_OPTIONS} T={TEMPERATURE}")

    tasks = load_arc(ARC_ROOT, N_TASKS)
    print(f"tasks loaded: {len(tasks)} from {ARC_ROOT}")
    if not tasks:
        print("BLOCKED_NO_CORPUS")
        return 1

    enc = HENRIVisionEncoder(d_model=D_MODEL, k_blocks=N_BLOCKS, block_dim=BLOCK_DIM,
                             device="cpu", spatial_basis_kind=resolve_kind,
                             bg_mask=bg_mask)

    def encode(grid):
        try:
            with torch.no_grad():
                w = enc.encode_spatial_grid(grid)
            return w.squeeze(0).reshape(-1).to(torch.float32)
        except ValueError as exc:
            raise EncoderRefused(str(exc)[:140])

    def cos(a, b):
        return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())

    per_task, skipped, errors = [], [], []
    gen = torch.Generator().manual_seed(RANDOM_ARM_SEED)

    for idx, (tid, t) in enumerate(tasks):
        try:
            train = t["train"][:3]
            te = t["test"][0]
            Xtr = torch.stack([encode(p["input"]) for p in train])
            Ytr = torch.stack([encode(p["output"]) for p in train])

            # per-slot diagonal ridge LS in the real UWE domain (diag_ls family)
            W = (Xtr * Ytr).sum(0) / ((Xtr * Xtr).sum(0) + RIDGE_EPS)

            x_te = encode(te["input"])
            W_rand = torch.randn(x_te.numel(), generator=gen)

            cands, truth_index, shuf_seed, kinds = build_candidates(
                te["output"], tid)
            cw = torch.stack([encode(c) for c in cands])

            arms = {
                "functor": (W * x_te),
                "identity": x_te.clone(),
                "random": (W_rand * x_te),
            }
            row = {
                "task_id": tid,
                "truth_index": truth_index,
                "shuffle_seed": shuf_seed,
                "option_kinds": kinds,
                "snapshot": state_snapshot_id_of(x_te),
                "grid_in": [len(te["input"]), len(te["input"][0])],
                "grid_out": [len(cands[0]), len(cands[0][0])],
            }
            for arm, pred in arms.items():
                scores = torch.tensor([cos(pred, c) for c in cw], dtype=torch.float32)
                env = probe_from_logits(
                    scores / TEMPERATURE,
                    option_ids=tuple(range(K_OPTIONS)),
                    state_snapshot_id=row["snapshot"],
                    probe_id=idx,
                    wave_binding=(pred.reshape(N_BLOCKS, BLOCK_DIM)
                                  if pred.numel() == D_MODEL else None),
                )
                probs = list(env.probabilities)
                row[f"{arm}_probs"] = probs
                row[f"{arm}_status"] = env.status
                row[f"{arm}_scores"] = [float(s) for s in scores.tolist()]
                row[f"{arm}_correct"] = int(int(np.argmax(probs)) == truth_index)
            per_task.append(row)
        except CandidateError as exc:
            skipped.append({"task": tid, "reason": f"degenerate candidates: {exc}"})
        except EncoderRefused as exc:
            skipped.append({"task": tid, "reason": f"encoder refused: {exc}"})
        except Exception as exc:  # noqa: BLE001
            errors.append({"task": tid, "stage": "task",
                           "error": f"{type(exc).__name__}: {exc}"[:200]})
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(tasks)} tasks, {len(per_task)} rowed, "
                  f"{len(skipped)} skipped, {time.time() - t0:.1f}s")

    n_valid, n_skip, n_err = len(per_task), len(skipped), len(errors)
    status = "BLOCKED" if n_valid == 0 else ("PARTIAL" if n_err else "OK")
    print(f"rowed={n_valid} skipped={n_skip} errors={n_err} status={status}")

    receipt = {
        "schema": "henri.probe-calibration-receipt.v1",
        "blueprint_schema_alias": "henri.calibration-receipt.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_class": "DERIVED",
        "evidence_note": (
            "DERIVED from OBSERVED inputs: wave cosines measured on the live "
            "encoder over the local ARC corpus, combined with ARC ground truth "
            "by the stated rule (softmax over cosine scores, via "
            "arc_egress_contract.probe_from_logits)."
        ),
        "device_kind": "cpu",
        "torch": torch.__version__,
        "config": {
            "arc_root": ARC_ROOT,
            "n_tasks_requested": len(tasks),
            "n_tasks_rowed": n_valid,
            "n_tasks_skipped": n_skip,
            "n_tasks_errored": n_err,
            "k_options": K_OPTIONS,
            "chance_accuracy": 1.0 / K_OPTIONS,
            "temperature": TEMPERATURE,
            "t_sweep": list(T_SWEEP),
            "score_kind": "cosine",
            "operator": "per-slot diagonal ridge LS, real UWE domain",
            "ridge_eps": RIDGE_EPS,
            "encoder_d_model": D_MODEL,
            "encoder_blocks": N_BLOCKS,
            "spatial_basis_kind": resolve_kind,
            "bg_mask": bg_mask,
            "shuffle_seed_base": SHUFFLE_SEED,
            "random_arm_seed": RANDOM_ARM_SEED,
            "candidate_pool": [n for n, _ in POOL],
        },
        "status": status,
        "counts": {"rowed": n_valid, "skipped": n_skip, "errored": n_err},
        "skipped": skipped,
        "errors": errors,
        "codebook_kind": (
            "per-task demo-fit operator (W), NOT a frozen or trained codebook. "
            "The distractor set is a deterministic transform pool. An identity "
            "codebook recovers its own binding by construction, so this receipt "
            "is NOT evidence of representation generalisation."
        ),
        "anti_circularity": (
            "probabilities <- demonstration-fit W; truth <- held-out ARC test "
            "output. Scored against each other, never against the fit signal."
        ),
        "ground_truth_source": {
            "chosen": "local ARC-AGI corpus",
            "corpus_root": ARC_ROOT,
            "blueprint_proposed": "the sealed 60-task receipts themselves",
            "blueprint_premise_verdict": "FALSIFIED",
            "probe_date": "2026-09-18",
            "probe_evidence": (
                "All 21 experiments/verification/*_observed.json receipts store "
                "AGGREGATE statistics only (arms -> held_out_mean / gap / "
                "frac_tasks_beating_identity; n_solves; per-arm scored-set "
                "sizes). ZERO receipts expose per-task rows, and the sealed "
                "60-task runner derives ground truth at runtime without "
                "persisting per-task probabilities or logits. A runner consuming "
                "(state, options, truth) from the receipts would fail its own "
                "fail-closed gate at step 1."
            ),
            "consequence": (
                "ECE is measured against corpus ground truth. THIS runner emits "
                "per_task rows so the calibration dataset is rebuildable from a "
                "local corpus at zero API cost."
            ),
        },
    }

    for arm in ("functor", "identity", "random"):
        if per_task:
            receipt[f"arm_{arm}"] = build_calibration_receipt(
                [r[f"{arm}_probs"] for r in per_task],
                [r["truth_index"] for r in per_task],
                n_bins=10,
            )

    # control: shuffle the truth labels on the primary arm -> must fall to chance
    if per_task:
        perm = np.random.default_rng(SHUFFLE_SEED).permutation(n_valid)
        receipt["control_null_shuffled_truth"] = build_calibration_receipt(
            [r["functor_probs"] for r in per_task],
            [per_task[i]["truth_index"] for i in perm],
            n_bins=10,
        )

    # self-check: the torch softmax path (probe_from_logits) must agree with the
    # numpy twin used for the sweep, at T = TEMPERATURE.
    if per_task:
        twin = softmax_rows([r["functor_scores"] for r in per_task], TEMPERATURE)
        worst = max(
            abs(a - b)
            for row_a, row_b in zip(twin, [r["functor_probs"] for r in per_task])
            for a, b in zip(row_a, row_b)
        )
        receipt["self_check"] = {
            "torch_vs_numpy_max_abs_diff": worst,
            "agree_lt_1e-6": bool(worst < 1e-6),
        }

    # temperature sweep: ECE is a function of stated sharpness. Report the trade.
    if per_task:
        truth = [r["truth_index"] for r in per_task]
        scores = [r["functor_scores"] for r in per_task]
        receipt["temperature_sweep"] = []
        for T in T_SWEEP:
            rec = build_calibration_receipt(softmax_rows(scores, T), truth, n_bins=10)
            receipt["temperature_sweep"].append({
                "temperature": T,
                "ece": rec["ece"],
                "brier": rec["brier"],
                "accuracy": rec["accuracy"],
                "brier_skill_score": rec["brier_skill_score"],
                "sharpness_mean_peak": rec["sharpness_mean_peak"],
                "calibration_skew": rec["calibration_skew"],
                "is_well_calibrated": rec["is_well_calibrated"],
            })

    receipt["per_task"] = per_task
    receipt["limits"] = [
        "Measures the wave READOUT over a 4-way choice task, not the planner.",
        "Not a task score, not a benchmark score, not a capability claim.",
        "CPU only: no CUDA verification (Vast instance EXITED, SSH refused).",
        "The 'calibrated' label is reported as computed, never adjusted.",
        "ECE depends on the stated temperature; see temperature_sweep.",
        "Distractors are deterministic transforms of the true output; a",
        "different distractor policy would change difficulty and the numbers.",
    ]
    receipt["elapsed_secs"] = round(time.time() - t0, 2)

    OUT.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    body = OUT.read_bytes()
    print(f"wrote {OUT} ({len(body)} bytes)")
    print(f"sha256 {hashlib.sha256(body).hexdigest()}")
    for arm in ("functor", "identity", "random"):
        a = receipt.get(f"arm_{arm}")
        if a:
            print(f"  arm {arm:9s} acc={a['accuracy']:.4f} ECE={a['ece']:.4f} "
                  f"brier={a['brier']:.4f} skill={a['brier_skill_score']:+.4f} "
                  f"skew={a['calibration_skew']:+.4f} "
                  f"peak={a['sharpness_mean_peak']:.4f} "
                  f"well_cal={a['is_well_calibrated']}")
    nul = receipt.get("control_null_shuffled_truth")
    if nul:
        print(f"  null      acc={nul['accuracy']:.4f} ECE={nul['ece']:.4f}")
    sc = receipt.get("self_check")
    if sc:
        print(f"  self_check torch-vs-numpy maxdiff={sc['torch_vs_numpy_max_abs_diff']:.3e}")
    print(f"elapsed {receipt['elapsed_secs']}s")
    return 0 if status == "OK" else (3 if status == "PARTIAL" else 1)


if __name__ == "__main__":
    sys.exit(main())
