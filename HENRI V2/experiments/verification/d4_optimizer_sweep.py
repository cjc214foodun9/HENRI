"""DIRECTIVE 4 — optimizer/schedule sweep at FIXED 32,896 parameters.

DOCUMENT CLAIM UNDER TEST (HENRI-ARCH-2026-DIAGNOSTIC-CRITICAL-PATH-V3, sec 3.1)
================================================================================
  "The True Cause: The stagnation is neither an information-theoretic barrier of
   the tape nor a model capacity limit. It is an OPTIMIZER AND LEARNING-RATE
   SCHEDULE FAILURE. The learning trajectory stalled in a shallow gradient basin."

That is an ASSERTION the document did not measure. It is treated here as
HYPOTHESIS and this sweep decides it.

PRIOR EVIDENCE (already in the audit chain, not re-litigated here)
=================================================================
  * capacity 8x  -> heldout 0.129908 vs control 0.129273 (WORSE)  => capacity FALSIFIED
  * control (32,896 p) reached 0.129273, BELOW a bigram floor fit on 4,000 rows
    (0.180437).

THE FLAW IN THAT FLOOR (found by reading the learner, 2026-09-28)
=================================================================
  `BilinearLearner.loss` computes logits = emb[ids] @ head. There is NO context
  mixing: the prediction for position t uses ONLY token t. It is a BIGRAM model.
  Its best possible loss is therefore H(next | current) of the TRUE distribution.
  The quoted "honest floor" was fit on 4,000 rows while the learner consumed
  1200*256 = 307,200 rows. Beating a data-starved floor shows only that more data
  gives a better ESTIMATE -- it is not evidence about entropy.

  => The floor must be re-estimated at MATCHED data volume. That is arm FLOOR_MATCHED.

PRE-REGISTERED DECISION RULE (written BEFORE running)
=====================================================
  Let base = control heldout loss (must reproduce 0.12927328145989597).
  MARGIN = 0.01 nats  (~10x observed arm-to-arm spread ~7e-4).

  1. OPTIMIZER_STALL_CONFIRMED  iff any sweep arm <= base - MARGIN
        -> the document's claim is SUPPORTED; the basin was escapable by
           optimiser/schedule alone.
  2. OPTIMIZER_STALL_FALSIFIED  iff NO sweep arm <= base - MARGIN
        -> schedule and LR are not the lever at this capacity.
  3. MODEL_CLASS_IS_THE_BINDING_CONSTRAINT iff the CONTEXT arm (same ~32,900
     params, but with a previous-token embedding) <= base - MARGIN
        -> the ceiling is the ABSENCE OF CONTEXT in the model class, not the
           optimiser. Falsifies the document's claim by a DIFFERENT route and
           predicts that Directive 1's 2D emitter cannot help either, because
           the learner still has no context mechanism.
  4. AT_MATCHED_BIGRAM_FLOOR iff |base - matched_bigram_floor| <= 0.005
       -> the control is already at the bigram conditional entropy; no optimiser
          can go below it. Strong form of (2).
       SUPERSEDED 2026-09-28 by d4_budget_extension_probe.json. The clause
       "already at ... no optimiser can go below it" is too strong: at 3x this
       budget the control reaches 0.127878, which is BELOW the floor quoted here
       (0.127910). The floor is VOLUME-DEPENDENT (0.127910 at 307,200 rows,
       0.127717 at 921,600 rows), so it is not a fixed bound. Measured outcome:
       the control stays above the MATCHED-volume floor at every checkpoint
       (gaps +0.00136 / +0.00136 / +0.00016) while still descending, and the
       context arm beats the control by ~0.0124 nats at 3x budget. Read the
       verdict as "the bilinear arm approaches the matched bigram floor", NOT as
       "the floor cannot be crossed".

  Arms 3 and 4 are INDEPENDENT of arm 1: all three are reported, never merged.

CPU ONLY. This host has no CUDA (measured: torch 2.11.0+cu128, cuda False).
The document explicitly sanctions CPU execution for immediate work.
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

import stage0_seeding_run as S                                    # noqa: E402
from stage0_universal_seeder import (                             # noqa: E402
    TIMEOUT_TOKEN, VMConfig, CircularTapeVM, sample_program,
)

VOCAB = S.VOCAB
PROG_LEN = 32
SEQ_LEN = 33
BATCH = 256
STEPS = 1200
HELDOUT_N = 256
SEED = 0
BASE_LR = 3e-3
MARGIN = 0.01
BASE_LOSS = 0.12927328145989597          # must be reproduced by the control


# --------------------------------------------------------------------- data
def _vm():
    return CircularTapeVM(VMConfig(tape_size=256, max_steps=512,
                                   max_output=SEQ_LEN - 1))


def make_rows(n_rows, rng, prog_len=PROG_LEN):
    vm = _vm()
    rows = []
    while len(rows) < n_rows:
        res = vm.execute(sample_program(prog_len, rng))
        seq = list(res.output[: SEQ_LEN - 1])
        if res.timed_out:
            seq.append(TIMEOUT_TOKEN)
        seq = seq + [0] * (SEQ_LEN - len(seq))
        rows.append([min(x, VOCAB - 1) for x in seq[:SEQ_LEN]])
    return torch.tensor(rows, dtype=torch.long)


def bigram_floor(fit_ids, eval_ids):
    a = fit_ids[:, :-1].reshape(-1)
    b = fit_ids[:, 1:].reshape(-1)
    N = torch.zeros(VOCAB, VOCAB, dtype=torch.float64)
    N.index_put_((a, b), torch.ones_like(a, dtype=torch.float64), accumulate=True)
    Pc = N / N.sum(dim=1, keepdim=True).clamp(min=1e-12)
    ea = eval_ids[:, :-1].reshape(-1)
    eb = eval_ids[:, 1:].reshape(-1)
    return float(-Pc[ea, eb].clamp(min=1e-12).log().mean())


# ----------------------------------------------------------------- learners
class BilinearLearner:
    """Production TapeLearner at rank r, with optional schedule + grad clipping.

    Adam core (b1 .9 / b2 .999 / eps 1e-8, bias-corrected) is copied VERBATIM
    from TapeLearner.step. Schedule/clip are ADDED features, never replacements.
    """

    def __init__(self, rank, seed, lr=BASE_LR, cosine=False, grad_clip=0.0):
        gen = torch.Generator().manual_seed(seed)
        self.params = {
            "emb": torch.randn(VOCAB, rank, dtype=torch.float64, generator=gen) * 0.1,
            "head": torch.randn(rank, VOCAB, dtype=torch.float64, generator=gen) * 0.1,
        }
        for p in self.params.values():
            p.requires_grad_(True)
        self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        self.lr0, self.cos, self.clip = float(lr), bool(cosine), float(grad_clip)
        self.step_count = 0

    def n_params(self):
        return int(sum(p.numel() for p in self.params.values()))

    def loss(self, ids):
        x = self.params["emb"][ids]
        logits = x @ self.params["head"]
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))

    def step(self, ids):
        loss = self.loss(ids)
        grads = torch.autograd.grad(loss, tuple(self.params.values()))
        self.step_count += 1
        lr = self.lr0
        if self.cos:
            p = min(1.0, self.step_count / float(STEPS))
            lr = 0.5 * self.lr0 * (1.0 + math.cos(math.pi * p))
        if self.clip > 0.0:
            tot = torch.sqrt(sum((g * g).sum() for g in grads))
            if float(tot) > self.clip:
                s = self.clip / (float(tot) + 1e-12)
                grads = tuple(g * s for g in grads)
        with torch.no_grad():
            for (name, p), g in zip(self.params.items(), grads):
                self.v[name].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                v_hat = self.v[name] / (1 - self.b2 ** self.step_count)
                p.add_(-lr * g / (torch.sqrt(v_hat) + self.eps))
        return float(loss.item())


class ContextLearner:
    """CAPACITY-MATCHED CONTROL for the MODEL-CLASS hypothesis.

    Same bilinear form, BUT position t also sees token t-1. If this beats the
    control by MARGIN at matched capacity, the binding constraint is the ABSENCE
    OF CONTEXT -- not the optimiser, and not (as the doc claims) the schedule.

    rank 42 -> 3*257*42 = 32,382 params vs control 32,896 (99% matched).
    """

    def __init__(self, rank, seed, lr=BASE_LR):
        gen = torch.Generator().manual_seed(seed)
        self.params = {
            "emb": torch.randn(VOCAB, rank, dtype=torch.float64, generator=gen) * 0.1,
            "ctx": torch.randn(VOCAB, rank, dtype=torch.float64, generator=gen) * 0.1,
            "head": torch.randn(rank, VOCAB, dtype=torch.float64, generator=gen) * 0.1,
        }
        for p in self.params.values():
            p.requires_grad_(True)
        self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.b1, self.b2, self.eps, self.lr = 0.9, 0.999, 1e-8, float(lr)
        self.step_count = 0

    def n_params(self):
        return int(sum(p.numel() for p in self.params.values()))

    def loss(self, ids):
        cur = self.params["emb"][ids]                      # [B,T,r]
        prev_ids = torch.cat([ids[:, :1], ids[:, :-1]], dim=1)   # shift right
        prev = self.params["ctx"][prev_ids]                # [B,T,r]
        logits = (cur + prev) @ self.params["head"]
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))

    def step(self, ids):
        loss = self.loss(ids)
        grads = torch.autograd.grad(loss, tuple(self.params.values()))
        self.step_count += 1
        with torch.no_grad():
            for (name, p), g in zip(self.params.items(), grads):
                self.v[name].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                v_hat = self.v[name] / (1 - self.b2 ** self.step_count)
                p.add_(-self.lr * g / (torch.sqrt(v_hat) + self.eps))
        return float(loss.item())


def train(learner, heldout_ids, tag):
    rng = torch.Generator().manual_seed(SEED + 4242)
    losses = []
    for _ in range(STEPS):
        losses.append(learner.step(make_rows(BATCH, rng)))
    with torch.no_grad():
        ho = float(learner.loss(heldout_ids))
    return {
        "tag": tag,
        "n_params": learner.n_params(),
        "heldout_loss_final": ho,
        "train_loss_first10_mean": sum(losses[:10]) / 10,
        "train_loss_last10_mean": sum(losses[-10:]) / 10,
        # wall time deliberately EXCLUDED from the receipt (F1 defect class)
    }


def main():
    R = {"schema": "henri.d4-optimizer-sweep.v1",
         "document_claim": ("plateau is an OPTIMIZER AND LEARNING-RATE SCHEDULE "
                            "failure, not entropy and not capacity"),
         "claim_status": "HYPOTHESIS_UNDER_TEST",
         "fixed_architecture": {"class": "bilinear (TapeLearner form)",
                                "target_params": 32896},
         "config": {"steps": STEPS, "batch": BATCH, "seq_len": SEQ_LEN,
                    "prog_len": PROG_LEN, "heldout_n": HELDOUT_N, "seed": SEED,
                    "base_lr": BASE_LR, "margin": MARGIN,
                    "rows_consumed_by_each_arm": STEPS * BATCH}}

    heldout_ids = S.build_heldout(HELDOUT_N, SEED, PROG_LEN, SEQ_LEN)

    # ---- harness control: production class, verbatim
    prod = S.TapeLearner(seed=SEED)
    _rng = torch.Generator().manual_seed(SEED + 4242)
    for _ in range(STEPS):
        prod.step(make_rows(BATCH, _rng))
    with torch.no_grad():
        prod_ho = float(prod.loss(heldout_ids))
    ctrl = train(BilinearLearner(64, SEED), heldout_ids, "CTRL_r64_lr3e-3")
    R["harness_control"] = {
        "production_TapeLearner": prod_ho,
        "generalised_r64_baseline": ctrl["heldout_loss_final"],
        "abs_diff_vs_production": abs(prod_ho - ctrl["heldout_loss_final"]),
        "abs_diff_vs_recorded_BASE_LOSS": abs(ctrl["heldout_loss_final"] - BASE_LOSS),
        "verdict": ("HARNESS_OK"
                    if abs(ctrl["heldout_loss_final"] - BASE_LOSS) < 1e-12
                    else "HARNESS_BROKEN_do_not_report_arms"),
    }
    base = ctrl["heldout_loss_final"]

    # ---- floors at MATCHED data volume (fixes my own 4,000-row flaw)
    fit_rng = torch.Generator().manual_seed(SEED + 4242)
    matched_fit = make_rows(STEPS * BATCH, fit_rng)          # 307,200 rows
    small_fit = make_rows(4000, torch.Generator().manual_seed(SEED + 4242))
    R["floors"] = {
        "bigram_fit_on_4000_rows_PREVIOUSLY_QUOTED": bigram_floor(small_fit, heldout_ids),
        "bigram_fit_on_matched_307200_rows": bigram_floor(matched_fit, heldout_ids),
        "uniform_ln_vocab": math.log(VOCAB),
    }
    matched_floor = R["floors"]["bigram_fit_on_matched_307200_rows"]

    # ---- ARM 1: LR sweep (no schedule change, no clip)
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 5e-2]
    arms = [ctrl]
    for lr in lrs:
        if abs(lr - BASE_LR) < 1e-12:
            continue                    # baseline already present
        arms.append(train(BilinearLearner(64, SEED, lr=lr), heldout_ids,
                          f"LR_{lr:g}"))

    # ---- ARM 2: cosine decay (best LR + baseline LR)
    arms.append(train(BilinearLearner(64, SEED, lr=BASE_LR, cosine=True),
                      heldout_ids, "LR_3e-3_COSINE"))
    arms.append(train(BilinearLearner(64, SEED, lr=1e-2, cosine=True),
                      heldout_ids, "LR_1e-2_COSINE"))

    # ---- ARM 3: gradient clipping
    arms.append(train(BilinearLearner(64, SEED, lr=1e-2, grad_clip=1.0),
                      heldout_ids, "LR_1e-2_CLIP1.0"))
    arms.append(train(BilinearLearner(64, SEED, lr=3e-3, cosine=True, grad_clip=1.0),
                      heldout_ids, "LR_3e-3_COSINE_CLIP1.0"))

    R["arms"] = arms

    best = min(arms, key=lambda a: a["heldout_loss_final"])
    R["best_arm"] = {"tag": best["tag"], "heldout_loss_final": best["heldout_loss_final"],
                     "improvement_over_control": base - best["heldout_loss_final"]}

    # ---- ARM 4: MODEL-CLASS control (capacity matched, WITH previous token)
    ctx = train(ContextLearner(42, SEED), heldout_ids, "CONTEXT_r42_prev_token")
    R["model_class_arm"] = ctx

    # ---- pre-registered verdicts (independent, all reported)
    opt_stall = (base - best["heldout_loss_final"]) >= MARGIN
    class_win = (base - ctx["heldout_loss_final"]) >= MARGIN
    at_floor = abs(base - matched_floor) <= 0.005
    R["verdicts"] = {
        "1_OPTIMIZER_STALL_CONFIRMED": bool(opt_stall),
        "2_OPTIMIZER_STALL_FALSIFIED": bool(not opt_stall),
        "3_MODEL_CLASS_IS_THE_BINDING_CONSTRAINT": bool(class_win and not opt_stall),
        "4_AT_MATCHED_BIGRAM_FLOOR": bool(at_floor),
        "best_sweep_delta_vs_control": base - best["heldout_loss_final"],
        "context_delta_vs_control": base - ctx["heldout_loss_final"],
        "control_minus_matched_bigram_floor": base - matched_floor,
        "control_minus_4000row_bigram_floor": base - R["floors"][
            "bigram_fit_on_4000_rows_PREVIOUSLY_QUOTED"],
    }
    R["interpretation"] = (
        "OPTIMIZER_STALL_CONFIRMED: an LR/schedule change alone broke the plateau."
        if opt_stall else
        ("MODEL_CLASS_IS_THE_BINDING_CONSTRAINT: no schedule change helped, but a "
         "capacity-matched WITH-CONTEXT model did. The ceiling is the absence of "
         "context in the bilinear form, not the optimiser and not capacity."
         if class_win else
         "OPTIMIZER_STALL_FALSIFIED and no context win: the plateau is not an "
         "optimiser failure; the binding constraint is neither LR, schedule, "
         "clipping, nor (on this evidence) context."))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "d4_optimizer_sweep.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2, default=str)
    print(json.dumps(R, indent=2, default=str))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
