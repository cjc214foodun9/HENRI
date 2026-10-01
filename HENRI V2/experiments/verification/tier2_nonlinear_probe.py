"""PILLAR 2 PROBE -- NON-LINEAR wave propagator vs the linear ceiling.

DIRECTIVE: HENRI-ARCH-2026-CRITICAL-DIRECTIVE-V1, item 3
  "Build the non-linear operator over Psi to replace the linear Koopman bottleneck."

WHAT IS ALREADY ESTABLISHED (measured, committed at 72f20fe)
  The best LINEAR map on this encoding reaches 3-step cosine 0.280163 at d=65536
  (0.232242 at d=512), stable across three orders of magnitude of lambda. That is
  an UPPER BOUND FOR THE LINEAR MODEL CLASS. It does NOT bound a non-linear map --
  which is exactly why this probe exists.

WHAT THIS PROBE ANSWERS (and what it cannot)
  Q: can a NON-LINEAR propagator psi_hat_{t+1} = normalize(f_theta(psi_t, a_t))
     beat the linear ceiling at the SAME d, SAME warmup, SAME split?
  It can only answer that for THIS class at THIS scale. A negative result here does
  not prove non-linearity is hopeless; it kills THIS class at THIS scale.

THE TRAP THIS FILE IS BUILT AROUND (the reason it is not a 20-line script)
  MEMORIZATION. The trajectory generator is synthetic and seeded. A high-capacity
  MLP will memorize the training trajectories and print cosine ~0.99, which looks
  like a PASS and is worthless. Guard:
    * SPLIT BY WHOLE TRAJECTORY SEED, not by timestep. Train seeds and test seeds
      are disjoint; no test grid is ever seen in training.
    * Report train-fit cosine BESIDE test cosine. train >> test is MEMORIZATION,
      and is reported as such, not as a result.

PRE-REGISTERED DECISION RULE (fixed before running; do not retune after seeing)
  test 3-step cos >= 0.92            -> NON-LINEAR REACHES THE CONTRACT
  test 3-step cos <= KILL (0.60)     -> non-linearity does NOT rescue this class
                                        at this scale. Report as measured negative.
  0.60 < test 3-step < 0.92          -> INCONCLUSIVE BAND; reported, not promoted.
  train - test >= 0.25               -> MEMORIZATION: the test number is not a
                                        generalisation result, whatever its value.
  Metric validity: the LINEAR reference arm must reproduce ~0.23 at d=512. If it
  does not, the comparison is broken and no verdict is issued (fail-closed).

NOT DONE HERE: no GPU, no claim about d=65536, no claim about anisotropic Langevin,
no claim about Sagnac. One class, one scale, one split.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import os
import platform
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse the EXACT helpers the linear harness uses, by file path, so the two arms
# are provably the same fixture. A second copy could silently diverge.
_spec = importlib.util.spec_from_file_location("t2gate", os.path.join(_HERE, "tier2_measured_gate.py"))
_t2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t2)

from henri_vision_encoder import HENRIVisionEncoder   # noqa: E402
from wave_jepa import WaveJEPA                         # noqa: E402

THRESH = _t2.THRESH_T2A      # 0.92 -- unchanged from T2-a
KILL = _t2.KILL_T2A          # 0.60
ROLLOUT_H = _t2.ROLLOUT_H    # 3
TRAIN_SEEDS = [11, 22, 33, 44]
TEST_SEEDS = [55, 66]


class WavePropagator(nn.Module):
    """Non-linear propagator: normalize(f_theta(normalize(psi + a))).

    Same INPUT as the linear EDMD channel (the combined wave) so the comparison is
    like-for-like: one map, two model classes. Output is re-normalised so the wave
    contract (||Psi|| = 1) holds by construction and cannot be the source of gain.
    """

    def __init__(self, d_model: int, hidden: int = 512, depth: int = 2, residual: bool = False):
        super().__init__()
        self.residual = residual
        layers, prev = [], d_model
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.GELU()]
            prev = hidden
        layers += [nn.Linear(prev, d_model)]
        self.net = nn.Sequential(*layers)

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        out = self.net(combined)
        if self.residual:
            # identity skip: lets the net ACCEPT the linear solution for free and
            # only learn a correction. A fair test must not forbid the incumbent.
            out = out + combined
        return F.normalize(out, p=2, dim=0)


def encode_traj(jepa, seed, steps, g, device):
    actions, grids = _t2.build_trajectory(g, steps, seed)
    with torch.no_grad():
        psis = [jepa.encode_context(gr).to(device) for gr in grids]
    return actions, psis


def pairs(psis, actions, nb, device, warmup):
    """(combined_t, psi_{t+1}) for t in [0, warmup)."""
    X, Y = [], []
    for i in range(warmup):
        a = _t2.action_wave(actions[i], nb, device)
        X.append(F.normalize(psis[i].view(-1) + a.view(-1), p=2, dim=0))
        Y.append(F.normalize(psis[i + 1].view(-1), p=2, dim=0))
    return torch.stack(X), torch.stack(Y)


def evaluate(model, psis, actions, nb, device, base, holdout, train_fit=False):
    """1-step and 3-step cosine on a held-out span. Stateless (no per-step adapt)."""
    one, three, stat1, stat3 = [], [], [], []
    with torch.no_grad():
        for i in range(base, base + holdout):
            a = _t2.action_wave(actions[i], nb, device)
            x = F.normalize(psis[i].view(-1) + a.view(-1), p=2, dim=0)
            p1 = model(x)
            one.append(float(F.cosine_similarity(p1.view(-1), psis[i + 1].view(-1), dim=0)))
            stat1.append(float(F.cosine_similarity(psis[i].view(-1), psis[i + 1].view(-1), dim=0)))
            ph = psis[i]
            for k in range(ROLLOUT_H):
                if i + k + 1 >= len(psis):
                    break
                ak = _t2.action_wave(actions[i + k], nb, device)
                ph = model(F.normalize(ph.view(-1) + ak.view(-1), p=2, dim=0))
                if k == ROLLOUT_H - 1:
                    three.append(float(F.cosine_similarity(ph.view(-1), psis[i + k + 1].view(-1), dim=0)))
                    stat3.append(float(F.cosine_similarity(
                        psis[i].view(-1), psis[i + k + 1].view(-1), dim=0)))
    return _t2.stats(one), _t2.stats(three), _t2.stats(stat1), _t2.stats(stat3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--nb", type=int, default=64)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=200, help="train steps per TRAIN traj")
    ap.add_argument("--holdout", type=int, default=24, help="test steps per TEST traj")
    ap.add_argument("--steps", type=int, default=32, help="optimiser steps")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--arms", type=str,
                    default="mlp,mlp+identity-skip,resonator",
                    help="comma list of arms to run")
    ap.add_argument("--res-iters", type=int, default=3,
                    help="recursion depth of the structured resonator arm")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("PILLAR 2 PROBE -- NON-LINEAR PROPAGATOR vs LINEAR CEILING")
    print("=" * 78)
    if device.type == "cpu":
        print("  *** CPU SCALE. Any verdict is a verdict at THIS scale only.")
    print(f"  device {device}  d={args.d}  nb={args.nb}  block_dim={args.d // args.nb}")
    print(f"  TRAIN seeds {TRAIN_SEEDS}  |  TEST seeds {TEST_SEEDS}  (DISJOINT -- anti-memorisation)")
    print(f"  thresholds: PASS >= {THRESH}   KILL <= {KILL}   memorisation flag if train-test >= 0.25")
    print()

    enc = HENRIVisionEncoder(d_model=args.d, k_blocks=args.nb, device=str(device),
                             spatial_basis_kind="incommensurate", bg_mask=True,
                             fused_superpose=True, parity_scipy=True)
    jepa = WaveJEPA(d_model=args.d, num_blocks=args.nb, r_rank=16,
                    device=str(device), encoder=enc)
    assert jepa.encoder is enc, "substrate pinning failed"

    n_steps = args.warmup + args.holdout + ROLLOUT_H + 2
    train, test = [], []
    for s in TRAIN_SEEDS:
        train.append(encode_traj(jepa, s, n_steps, args.grid, device))
    for s in TEST_SEEDS:
        test.append(encode_traj(jepa, s, n_steps, args.grid, device))
    print(f"  encoded {len(train)} train + {len(test)} test trajectories "
          f"({n_steps} grids each)")

    # ---- LINEAR REFERENCE ARM (metric validity) --------------------------------
    Xtr, Ytr = [], []
    for actions, psis in train:
        X, Y = pairs(psis, actions, args.nb, device, args.warmup)
        Xtr.append(X)
        Ytr.append(Y)
    Xtr, Ytr = torch.cat(Xtr).float(), torch.cat(Ytr).float()
    lam = 1e-5 * float(Xtr.shape[0])
    alpha = _t2.fit_ridge_dual(Xtr, Ytr, lam)          # [n, d] dual form

    def lin_pred(x):
        return F.normalize((Xtr @ x.float()) @ alpha, p=2, dim=0)

    lin1, lin3 = [], []
    for actions, psis in test:
        o1, o3, _, _ = evaluate(lin_pred, psis, actions, args.nb, device,
                                args.warmup, args.holdout)
        lin1.append(o1["mean"])
        lin3.append(o3["mean"])
    lin1m, lin3m = sum(lin1) / len(lin1), sum(lin3) / len(lin3)
    print()
    print("LINEAR REFERENCE (same fixture, dual-form ridge, no rank cap)")
    print(f"  1-step {lin1m:.6f}   3-step {lin3m:.6f}   <- must reproduce ~0.23 at d=512")

    # ---- NON-LINEAR ARMS ------------------------------------------------------
    results = {}
    import os as _os, sys as _sys
    _ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from henri_structured_predictor import StructuredResonator
    ARM_FACTORIES = {
        "mlp": lambda: WavePropagator(args.d, hidden=args.hidden, depth=args.depth,
                                      residual=False),
        "mlp+identity-skip": lambda: WavePropagator(args.d, hidden=args.hidden,
                                                    depth=args.depth, residual=True),
        # Pillar 2 candidate: STRUCTURED (rotor -> mask -> block-shift), iterated
        # with SHARED factors. Not a flat map -- see henri_structured_predictor.
        "resonator": lambda: StructuredResonator(args.d, args.nb, args.grid,
                                                 iters=args.res_iters),
    }
    for name in [a for a in args.arms.split(",") if a in ARM_FACTORIES]:
        model = ARM_FACTORIES[name]().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.train()
        last = None
        for step in range(args.steps):
            perm = torch.randperm(Xtr.shape[0])
            xb = Xtr[perm[:256]].to(device)
            yb = Ytr[perm[:256]].to(device)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        model.eval()
        # TRAIN fit (memorisation detector) and TEST generalisation
        tr1, tr3 = [], []
        for actions, psis in train:
            o1, o3, _, _ = evaluate(model, psis, actions, args.nb, device, args.warmup, args.holdout)
            tr1.append(o1["mean"])
            tr3.append(o3["mean"])
        te1, te3, st1, st3 = [], [], [], []
        for actions, psis in test:
            o1, o3, s1, s3 = evaluate(model, psis, actions, args.nb, device, args.warmup, args.holdout)
            te1.append(o1["mean"])
            te3.append(o3["mean"])
            st1.append(s1["mean"])
            st3.append(s3["mean"])
        tr1m, tr3m = sum(tr1) / len(tr1), sum(tr3) / len(tr3)
        te1m, te3m = sum(te1) / len(te1), sum(te3) / len(te3)
        st3m = sum(st3) / len(st3)
        mem = tr3m - te3m
        results[name] = dict(train1=tr1m, train3=tr3m, test1=te1m, test3=te3m,
                             static3=st3m, mem_gap=mem, loss_last=last,
                             # BUG FIXED: this read `residual=residual` after the arm
                             # loop was generalised to a name list, raising NameError
                             # before any arm could be scored. The flag is now taken
                             # from the model itself so every arm reports its own.
                             residual=bool(getattr(model, "residual", False)),
                             arm=name)
        print()
        print(f"NON-LINEAR ARM: {name}   (hidden={args.hidden} depth={args.depth} steps={args.steps})")
        print(f"  TRAIN 3-step (memorisation detector) : {tr3m:.6f}")
        print(f"  TEST  1-step                         : {te1m:.6f}")
        print(f"  TEST  3-step   <- GENERALISATION     : {te3m:.6f}")
        print(f"  TEST  static do-nothing 3-step       : {st3m:.6f}")
        print(f"  memorisation gap (train-test)        : {mem:+.6f}"
              f"{'   <-- MEMORISATION' if mem >= 0.25 else ''}")
        print(f"  loss last {last:.4e}")

    # ---- VERDICT (fail-closed) -------------------------------------------------
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    reasons = []
    if not (0.10 <= lin3m <= 0.40):
        reasons.append(f"LINEAR REFERENCE out of expected band ({lin3m:.6f} not in [0.10,0.40]) "
                       f"-- the fixture does not match the committed linear measurement; "
                       f"comparison is INVALID")
    best = max(results.items(), key=lambda kv: kv[1]["test3"])
    bname, b = best
    passed = b["test3"] >= THRESH
    killed = b["test3"] <= KILL
    print(f"  linear reference 3-step : {lin3m:.6f}   [metric validity: "
          f"{'OK' if not reasons else 'SUSPECT'}]")
    print(f"  best non-linear 3-step  : {b['test3']:.6f}  ({bname})")
    print(f"  delta vs linear         : {b['test3'] - lin3m:+.6f}")
    if reasons:
        verdict = "INVALID"
        for r in reasons:
            print(f"    REASON: {r}")
    elif b["mem_gap"] >= 0.25:
        verdict = "MEMORISED"
        print(f"    MEMORISATION: train {b['train3']:.6f} >> test {b['test3']:.6f}. The test")
        print(f"    number is NOT a generalisation result. Increase trajectory diversity.")
    elif passed:
        verdict = "NONLINEAR_REACHES_CONTRACT"
    elif killed:
        verdict = "NONLINEAR_KILLED_AT_THIS_SCALE"
        print(f"    Non-linearity does NOT rescue this class at d={args.d}: best test 3-step")
        print(f"    {b['test3']:.6f} <= {KILL}. Scope: THIS class, THIS scale, THIS split.")
    else:
        verdict = "INCONCLUSIVE_BAND"
        print(f"    In band ({KILL}, {THRESH}): reported, not promoted.")
    print()
    print(f"TIER2_NONLINEAR: {verdict}")

    if args.out:
        rec = dict(measured_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                   python=platform.python_version(), torch=torch.__version__,
                   device=str(device), d_model=args.d, num_blocks=args.nb,
                   train_seeds=TRAIN_SEEDS, test_seeds=TEST_SEEDS,
                   linear_reference=dict(one_step=lin1m, three_step=lin3m),
                   arms=results, thresholds=dict(pass_=THRESH, kill=KILL),
                   verdict=verdict, cpu_scale=device.type == "cpu")
        out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  receipt: {out}")
    return 0 if verdict == "NONLINEAR_REACHES_CONTRACT" else 1


if __name__ == "__main__":
    sys.exit(main())
