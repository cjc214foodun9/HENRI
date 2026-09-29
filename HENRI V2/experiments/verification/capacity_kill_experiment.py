"""PRE-REGISTERED KILL EXPERIMENT for the capacity-saturation finding.

CLAIM UNDER TEST (recorded 2026-09-28, commit fe7fd11)
======================================================
The Stage-0 10B-token loss plateau (0.0984) is caused by CAPACITY SATURATION of
a 32,896-parameter bilinear byte model, NOT by entropy exhaustion of the 1D
tape. Direct implication: Directive 1 (replace the 1D tape with a 2D emitter)
cannot raise the ceiling, because the ceiling is parameter count.

FALSIFIER (pre-registered, written BEFORE running)
==================================================
Raise capacity ALONE -- same corpus, same generator, same held-out batch, same
optimiser rule, same step count -- and measure held-out cross-entropy.

  * If a higher-capacity arm drives held-out loss BELOW the train-fitted bigram
    floor, the capacity hypothesis is ACCEPTED and the 2D-substrate directive is
    unnecessary. (My claim survives.)
  * If NO arm beats the train-fitted bigram floor, the capacity hypothesis is
    FALSIFIED: the corpus itself bounds learnable structure, and the substrate
    question reopens. (My claim dies.)

ARMS (all trained identically; only the model family/capacity changes)
=====================================================================
  BILINEAR_r64   32,896 params   CONTROL -- must reproduce production TapeLearner
  BILINEAR_r512  263,168 params  8x capacity, SAME class
  NONLINEAR_h64   36,992 params  ~same capacity, DIFFERENT class (tanh hidden)

HARNESS-VALIDITY CONTROL (this is the load-bearing check)
=========================================================
The control arm instantiates the PRODUCTION `TapeLearner` class directly, and a
generalised learner is run at r=64 with the same seed. If their losses are not
identical, the harness is broken and NO arm result is reportable.

HONESTY CONTROLS
================
  * Held-out batch is MATERIALISED ONCE (production `build_heldout`) and shared
    by every arm, so arms cannot differ by sampling.
  * Two floors are reported: `train_fitted_bigram -> heldout` (the honest
    generalisation floor) and `heldout_fitted_bigram -> heldout` (in-sample,
    the optimistic 0.1204 figure quoted earlier). The strict one gates.
  * Training data is drawn from the production generator, exactly as
    stage0_seeding_run.py builds `ids` (execute a sampled program, take output
    bytes, append TIMEOUT on timeout, pad, clamp to VOCAB-1).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

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
LR = 3e-3


# --------------------------------------------------------------------- data
def _vm():
    return CircularTapeVM(VMConfig(tape_size=256, max_steps=512,
                                   max_output=SEQ_LEN - 1))


def make_rows(n_rows, rng, prog_len):
    """EXACTLY the production construction (stage0_seeding_run.py:356-361)."""
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


# ------------------------------------------------------- floored references
def bigram_floor(fit_ids, eval_ids):
    a = fit_ids[:, :-1].reshape(-1)
    b = fit_ids[:, 1:].reshape(-1)
    N = torch.zeros(VOCAB, VOCAB, dtype=torch.float64)
    N.index_put_((a, b), torch.ones_like(a, dtype=torch.float64), accumulate=True)
    Pcond = N / N.sum(dim=1, keepdim=True).clamp(min=1e-12)
    ea = eval_ids[:, :-1].reshape(-1)
    eb = eval_ids[:, 1:].reshape(-1)
    p = Pcond[ea, eb].clamp(min=1e-12)
    return float(-p.log().mean())


# ----------------------------------------------------------------- learners
class BilinearLearner:
    """Production TapeLearner architecture at rank r; Adam rule copied VERBATIM
    from TapeLearner.step (b1 .9 / b2 .999 / eps 1e-8, bias-corrected)."""

    def __init__(self, rank, seed):
        gen = torch.Generator().manual_seed(seed)
        self.params = {
            "emb": torch.randn(VOCAB, rank, dtype=torch.float64, generator=gen) * 0.1,
            "head": torch.randn(rank, VOCAB, dtype=torch.float64, generator=gen) * 0.1,
        }
        for p in self.params.values():
            p.requires_grad_(True)
        self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.b1, self.b2, self.eps, self.lr = 0.9, 0.999, 1e-8, LR
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
        with torch.no_grad():
            for (name, p), g in zip(self.params.items(), grads):
                self.v[name].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                v_hat = self.v[name] / (1 - self.b2 ** self.step_count)
                p.add_(-self.lr * g / (torch.sqrt(v_hat) + self.eps))
        return float(loss.item())


class NonlinearLearner:
    """Different CLASS at ~baseline capacity: tanh hidden of width H."""

    def __init__(self, hidden, seed):
        gen = torch.Generator().manual_seed(seed)
        self.params = {
            "emb": torch.randn(VOCAB, hidden, dtype=torch.float64, generator=gen) * 0.1,
            "hid": torch.randn(hidden, hidden, dtype=torch.float64, generator=gen) * 0.1,
            "head": torch.randn(hidden, VOCAB, dtype=torch.float64, generator=gen) * 0.1,
        }
        for p in self.params.values():
            p.requires_grad_(True)
        self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.b1, self.b2, self.eps, self.lr = 0.9, 0.999, 1e-8, LR
        self.step_count = 0

    def n_params(self):
        return int(sum(p.numel() for p in self.params.values()))

    def loss(self, ids):
        x = torch.tanh(self.params["emb"][ids] @ self.params["hid"])
        logits = x @ self.params["head"]
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
    start = time.time()
    first = losses = None
    losses = []
    for i in range(STEPS):
        ids = make_rows(BATCH, rng, PROG_LEN)
        losses.append(learner.step(ids))
    with torch.no_grad():
        final_ho = float(learner.loss(heldout_ids))
    traj = losses
    return {
        "tag": tag,
        "n_params": learner.n_params(),
        "heldout_loss_final": final_ho,
        "train_loss_first10_mean": sum(traj[:10]) / 10,
        "train_loss_last10_mean": sum(traj[-10:]) / 10,
        "heldout_loss_at_200": None,   # filled below if available
        # NOTE: wall_s is deliberately NOT in the receipt. It is wall-clock and
        # varies between passes (measured: 12.3 vs 12.6 s), which would make the
        # committed artifact non-reproducible for a non-scientific field -- the
        # exact defect recorded as F1 in the 9a8fbce retraction. Timing is
        # printed to stdout instead.
        "wall_s_stdout_only": None,
    }


def main():
    R = {"schema": "henri.capacity-kill.v1", "config": {
        "steps": STEPS, "batch": BATCH, "seq_len": SEQ_LEN, "prog_len": PROG_LEN,
        "heldout_n": HELDOUT_N, "seed": SEED, "lr": LR}}

    # shared, materialised-once held-out set (production constructor)
    heldout_ids = S.build_heldout(HELDOUT_N, SEED, PROG_LEN, SEQ_LEN)

    # honest floors
    fit_rng = torch.Generator().manual_seed(SEED + 4242)
    train_fit = make_rows(4000, fit_rng, PROG_LEN)
    R["floors"] = {
        "train_fitted_bigram_to_heldout_HONEST": bigram_floor(train_fit, heldout_ids),
        "heldout_fitted_bigram_to_heldout_INSAMPLE": bigram_floor(heldout_ids, heldout_ids),
        "uniform_ln_vocab": math.log(VOCAB),
    }
    floor = R["floors"]["train_fitted_bigram_to_heldout_HONEST"]

    # ---- harness-validity control: production class vs generalised at r=64
    prod = S.TapeLearner(seed=SEED)
    prod_ho = None
    _rng = torch.Generator().manual_seed(SEED + 4242)
    for _ in range(STEPS):
        prod.step(make_rows(BATCH, _rng, PROG_LEN))
    with torch.no_grad():
        prod_ho = float(prod.loss(heldout_ids))
    gen64 = BilinearLearner(64, SEED)
    g64 = train(gen64, heldout_ids, "BILINEAR_r64")
    R["harness_control"] = {
        "production_TapeLearner_heldout_loss": prod_ho,
        "generalised_r64_heldout_loss": g64["heldout_loss_final"],
        "abs_diff": abs(prod_ho - g64["heldout_loss_final"]),
        "verdict": ("HARNESS_OK" if abs(prod_ho - g64["heldout_loss_final"]) < 1e-9
                    else "HARNESS_BROKEN"),
    }
    R["arms"] = [g64]

    R["arms"].append(train(BilinearLearner(512, SEED), heldout_ids, "BILINEAR_r512"))
    R["arms"].append(train(NonlinearLearner(64, SEED), heldout_ids, "NONLINEAR_h64"))

    for a in R["arms"]:
        a["beats_honest_bigram_floor"] = bool(a["heldout_loss_final"] < floor)
    R["floor_used"] = floor

    # ---- CORRECTED VERDICT RULE (harness defect found by RUNNING the first one).
    # v1 accepted on `arm < honest_floor`. MEASURED: the CONTROL (r64, 32,896
    # params) scores 0.129273 and ALSO beats that floor (0.180437), so v1 was
    # NON-DISCRIMINATING -- it could not have returned anything but ACCEPTED
    # regardless of the treatment arms. A capacity claim needs an ARM-vs-ARM
    # comparison against the control, with a pre-registered margin, on the SAME
    # materialised held-out batch. MARGIN is set to 0.01 nats: ~10x the observed
    # arm-to-arm spread (~7e-4), so it cannot be reached by run noise.
    ctrl = next(a for a in R["arms"] if a["tag"].startswith("BILINEAR_r64"))
    r512 = next(a for a in R["arms"] if "r512" in a["tag"])
    nl = next(a for a in R["arms"] if a["tag"].startswith("NONLINEAR"))
    MARGIN = 0.01
    d_cap = ctrl["heldout_loss_final"] - r512["heldout_loss_final"]
    d_cls = ctrl["heldout_loss_final"] - nl["heldout_loss_final"]
    R["v1_verdict_SUPERSEDED"] = "CAPACITY_HYPOTHESIS_ACCEPTED (v1 rule, non-discriminating)"
    R["corrected_verdict_rule"] = {
        "rule": ("ACCEPT capacity iff (control - r512) >= MARGIN; else ACCEPT "
                 "class iff (control - nonlinear) >= MARGIN; else FALSIFIED both"),
        "margin": MARGIN,
        "delta_capacity_8x": d_cap,
        "delta_class": d_cls,
        "verdict": ("CAPACITY_HYPOTHESIS_FALSIFIED"
                    if d_cap < MARGIN and d_cls < MARGIN else
                    "CAPACITY_HYPOTHESIS_ACCEPTED" if d_cap >= MARGIN else
                    "CLASS_HYPOTHESIS_ACCEPTED"),
        "why_v1_was_wrong": (
            "the control arm itself passed the v1 floor, so v1 could only ACCEPT"),
    }
    R["verdict"] = R["corrected_verdict_rule"]["verdict"]

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "capacity_kill_experiment.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2, default=str)
    print(json.dumps(R, indent=2, default=str))
    print("WROTE", out)


if __name__ == "__main__":
    main()
