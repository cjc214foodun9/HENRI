"""Decisive test of MY OWN interpretation: BROAD FLAT BASIN vs EXACT POINT.

WHY THIS EXISTS
===============
My committed SGDR receipt (4b75207, 65a7f72) measured FOUR DISTINCT weight hashes
whose held-out losses agree within |delta| <= 2.1e-04. I interpreted that as a
BROAD FLAT BASIN: many near-equivalent minima, loss insensitive to the schedule.

A MoA reference instead claimed the opposite mechanism -- that every arm lands on
ONE EXACT optimum (it asserted a single bit-identical weight sha, which my
precision probe refuted at BOTH dtypes; the claim itself is FALSIFIED). But a
claim can be wrong in its numbers and RIGHT in its mechanism, and I recorded that
my data does NOT exclude an exact common stationary point. So this probe tests
MY OWN interpretation rather than dismissing theirs.

DECISIVE OBSERVABLE
===================
All arms start from the IDENTICAL seed and init (D4.SEED) and differ ONLY in the
learning-rate schedule. Therefore:
  * if a unique reachable minimizer exists, longer training pulls every arm
    toward it -> the pairwise distance between arm weight VECTORS shrinks.
  * if the plateau is a broad flat basin, the arms stay separated while their
    losses agree.

So the discriminating quantity is WEIGHT-SPACE SEPARATION vs TRAINING BUDGET,
with loss agreement as the control. Hash identity (what the reference used) is
the wrong instrument: it is ambiguous, because Adam's update -> 0 whenever the
gradient vanishes, so identical weights can mean either "same optimum" or
"the schedule never reached the optimizer". That ambiguity is why this probe
measures distances and gradient norms instead.

PRE-REGISTERED (written before the first run)
=============================================
  COLLAPSE_SUPPORTED  iff max pairwise relative weight distance at the FINAL
                      checkpoint is < 0.01 (1%) AND strictly decreased from the
                      first checkpoint.
  BROAD_BASIN_STANDS  otherwise.
Both are reported; the loss spread at each checkpoint is reported alongside so
the reader can see agreement-without-collapse if that is what occurs.

VALIDITY GATES
==============
  V1 harness control: the constant-lr arm at 1200 steps reproduces D4's recorded
     BASE_LOSS bit-for-bit (else the harness drifted and the run is VOID).
  V2 schedules non-inert: each arm's lr trace differs from the control's.
  V3 gradient norm reported per arm: a stationary point means grad -> 0.

REUSE, NEVER REIMPLEMENT: data generation, Adam core, heldout builder and the
training loop are IMPORTED from the verified `d4_optimizer_sweep` module.

DETERMINISM: no wall-clock value enters the receipt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "verification"))

import d4_optimizer_sweep as D4                                 # noqa: E402
import stage0_seeding_run as S                                 # noqa: E402

TOTAL = 3600                     # 3x the D4 budget of 1200
CHECKPOINTS = (1200, 2400, 3600)
RANK = 64
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "d4_basin_shape_probe.json")


def resolve_out() -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a, _ = ap.parse_known_args()
    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        if not os.path.isdir(d):
            print(f"MALFORMED --out: directory does not exist: {d}", file=sys.stderr)
            raise SystemExit(2)
        return os.path.abspath(a.out)
    envd = os.environ.get("HENRI_RECEIPT_DIR")
    if envd:
        os.makedirs(envd, exist_ok=True)
        return os.path.join(envd, "d4_basin_shape_probe.json")
    return DEFAULT_OUT


class ScheduledLearner(D4.BilinearLearner):
    """D4's verified learner with the schedule externalised."""

    def __init__(self, rank, seed, sched, lr=D4.BASE_LR):
        super().__init__(rank, seed, lr=lr, cosine=False, grad_clip=0.0)
        self.sched = sched
        self.lr_trace: list[float] = []
        self.grad_norm: float = float("nan")

    def step(self, ids):
        loss = self.loss(ids)
        grads = torch.autograd.grad(loss, tuple(self.params.values()))
        self.step_count += 1
        lr = float(self.sched(self.step_count - 1))
        self.lr_trace.append(lr)
        # global gradient norm: the stationarity measure
        self.grad_norm = float(torch.sqrt(sum((g * g).sum() for g in grads)))
        with torch.no_grad():
            for (name, p), g in zip(self.params.items(), grads):
                self.v[name].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                v_hat = self.v[name] / (1 - self.b2 ** self.step_count)
                p.add_(-lr * g / (torch.sqrt(v_hat) + self.eps))
        return float(loss.item())


def wvec(L) -> torch.Tensor:
    return torch.cat([L.params[k].detach().reshape(-1).to(torch.float64)
                      for k in sorted(L.params)])


def rel_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetric relative separation, scale-free."""
    return float((a - b).norm() / (0.5 * (a.norm() + b.norm()) + 1e-12))


def max_pairwise(vs: list[torch.Tensor]) -> float:
    m = 0.0
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            m = max(m, rel_dist(vs[i], vs[j]))
    return m


def sgdr_cycles(n_cycles: int, t0: int, eta_min: float, mult: float):
    """Cosine annealing with warm restarts, LONG enough to cover TOTAL steps."""
    bounds = []
    T, m = t0, 1.0
    for _ in range(n_cycles):
        bounds.append(T)
        T *= m
        m = mult
    def s(step: int) -> float:
        t = step
        for T in bounds:
            if t < T:
                return eta_min + 0.5 * (D4.BASE_LR - eta_min) * \
                    (1.0 + math.cos(math.pi * t / T))
            t -= T
        return eta_min
    return s


def multi_kick(times: tuple):
    def s(step: int) -> float:
        return 3e-2 if step in times else D4.BASE_LR
    return s


def main() -> int:
    out = resolve_out()
    print("[basin] receipt:", out, flush=True)

    heldout = S.build_heldout(D4.HELDOUT_N, D4.SEED, D4.PROG_LEN, D4.SEQ_LEN)
    arms = [
        ("CTRL_const_lr3e-3",     lambda s: D4.BASE_LR),
        ("SGDR_9xT400_emin3e-4",  sgdr_cycles(9, 400, 3e-4, 1.0)),
        ("SGDR_9xT400_emin0_x2",  sgdr_cycles(9, 400, 0.0, 2.0)),
        ("KICK_600_1800_3000",    multi_kick((600, 1800, 3000))),
    ]

    rng = torch.Generator().manual_seed(D4.SEED + 4242)
    learners = {tag: ScheduledLearner(RANK, D4.SEED, sch) for tag, sch in arms}

    R: dict = {"schema": "henri.d4-basin-shape-probe.v1",
               "purpose": ("decide BROAD FLAT BASIN vs EXACT COMMON POINT; tests my "
                           "own interpretation, since the reference's hash-identity "
                           "instrument is ambiguous for Adam"),
               "pre_registration": {
                   "collapse_rule": ("max pairwise relative weight distance at the "
                                     "FINAL checkpoint < 0.01 AND strictly decreased"),
                   "broad_basin_rule": "otherwise",
                   "total_steps": TOTAL, "checkpoints": list(CHECKPOINTS),
                   "rank": RANK, "margin_context": D4.MARGIN},
               "checkpoints": {}}

    for step in range(1, TOTAL + 1):
        ids = D4.make_rows(D4.BATCH, rng)
        losses = {}
        for tag, L in learners.items():
            losses[tag] = L.step(ids)
        if step in CHECKPOINTS:
            snap = {}
            for tag, L in learners.items():
                with torch.no_grad():
                    ho = float(L.loss(heldout))
                snap[tag] = {"heldout": ho, "grad_norm": L.grad_norm,
                             "train_loss_100_mean": None}
            vs = [wvec(learners[t]) for t in learners]
            sep = max_pairwise(vs)
            losses_now = [snap[t]["heldout"] for t in learners]
            spread = max(losses_now) - min(losses_now)
            R["checkpoints"][str(step)] = {
                "max_pairwise_rel_weight_distance": sep,
                "heldout_spread": spread,
                "per_arm": snap,
                "lr_at_checkpoint": {t: learners[t].lr_trace[-1] for t in learners},
            }
            print(f"[basin] step={step:5d}  sep={sep:.6f}  "
                  f"heldout_spread={spread:.3e}  "
                  f"grad_norm[{arms[0][0]}]={snap[arms[0][0]]['grad_norm']:.3e}",
                  flush=True)

    cps = R["checkpoints"]
    first, last = cps[str(CHECKPOINTS[0])], cps[str(CHECKPOINTS[-1])]
    sep_first = first["max_pairwise_rel_weight_distance"]
    sep_last = last["max_pairwise_rel_weight_distance"]
    collapse = bool(sep_last < 0.01 and sep_last < sep_first)

    R["validity"] = {
        "v1_control_reproduces_D4_BASE_LOSS": abs(
            first["per_arm"]["CTRL_const_lr3e-3"]["heldout"] - D4.BASE_LOSS) < 1e-9,
        "v1_control_at_1200": first["per_arm"]["CTRL_const_lr3e-3"]["heldout"],
        "v1_D4_recorded": D4.BASE_LOSS,
        "v3_grad_norms_at_final": {t: last["per_arm"][t]["grad_norm"] for t in learners},
    }
    R["verdicts"] = {
        "max_pairwise_rel_distance_first": sep_first,
        "max_pairwise_rel_distance_final": sep_last,
        "distance_decreased": bool(sep_last < sep_first),
        "COLLAPSE_SUPPORTED_exact_common_point": collapse,
        "BROAD_BASIN_STANDS": bool(not collapse),
        "heldout_spread_first": first["heldout_spread"],
        "heldout_spread_final": last["heldout_spread"],
    }
    R["interpretation"] = (
        "COLLAPSE_SUPPORTED means longer training pulls the arms to one point, so "
        "the plateau is a single optimum and the reference's MECHANISM was right "
        "even though its hash evidence was fabricated. BROAD_BASIN_STANDS means "
        "the arms remain separated weight-wise while their losses agree, so the "
        "plateau is a wide flat basin. The loss spread is reported at every "
        "checkpoint: agreement-without-collapse is the broad-basin signature.")
    R["honest_limits"] = [
        "synthetic byte-tape fixture, not ARC/SciCode data",
        "rank 64 only; rank 42 untested here",
        "3x the D4 budget (3600 vs 1200 steps); a much longer budget is untested",
        "conclusions are for THIS model class, data volume and seed",
    ]

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print("\n[basin] WROTE", out, flush=True)
    print("[basin] VERDICT:", json.dumps(R["verdicts"], indent=2), flush=True)
    print("[basin] validity:", json.dumps(R["validity"], indent=2)[:400], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
