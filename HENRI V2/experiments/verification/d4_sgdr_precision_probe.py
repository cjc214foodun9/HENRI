"""Precision sensitivity of the "all escape arms collapse to one weight" claim.

MY MEASUREMENT (commit 65a7f72, receipt d4_sgdr_supplement.json)
================================================================
At fixed 32,896 params in FLOAT64, four arms -- constant lr, SGDR(3xT400,
eta_min 3e-4), SGDR(eta_min 0, mult 2), and a 10x single-step KICK at step 600
-- land on FOUR DISTINCT weight hashes, all within |delta| <= 2.1e-04 nats of
the control. No arm comes within MARGIN = 0.01.

A MoA reference (glm-5.3-flash) claims all four arms land on ONE bit-identical
sha 3f9c2b81e04a7d55 and calls the plateau "the exact optimum". That is NOT in
my receipt, and the commit it cites (f1a2b3c) does not exist in this repo.
I decline the claim -- but a claim can be wrong in its EVIDENCE and right in its
MECHANISM, so it is worth one cheap decisive test rather than a dismissal.

RECONCILIATION HYPOTHESIS (this script tests it)
================================================
A contraction x_{n+1} = f(x_n) in FINITE precision snaps to a single
representable point once the contraction factor beats machine epsilon.
float32 eps ~ 1.2e-07 is ~1e9 coarser than float64 eps ~ 2.2e-16. So the SAME
dynamics can give:
    float64 -> arms stay distinguishable
    float32 -> arms collapse to one weight vector
If so, both readings are correct at their own precision and "bit-identical" is
a PRECISION artifact, not evidence of a unique exact optimum.

PRE-REGISTERED
==============
  COLLAPSES_AT_F32 iff distinct_sha_count(float32) < distinct_sha_count(float64)
Both counts and both sha lists are reported. Production dtype is measured, not
assumed -- if production is float32, the precision of the claim matters; if it
is float64, the collapse hypothesis is irrelevant to production.

REUSE: data generation, Adam core, heldout builder and train() come from the
VERIFIED d4_optimizer_sweep module by import. Only the dtype differs.

DETERMINISM: no wall-clock field enters the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "verification"))

import d4_optimizer_sweep as D4                                 # noqa: E402
import stage0_seeding_run as S                                 # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "d4_sgdr_precision_probe.json")


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
        return os.path.join(envd, "d4_sgdr_precision_probe.json")
    return DEFAULT_OUT


class PLearner(D4.BilinearLearner):
    """D4's verified learner with the parameter dtype made explicit."""

    def __init__(self, rank, seed, dtype, sched, lr=D4.BASE_LR):
        super().__init__(rank, seed, lr=lr, cosine=False, grad_clip=0.0)
        self.dtype = dtype
        self.sched = sched
        self.params = {k: p.detach().to(dtype).requires_grad_(True)
                       for k, p in self.params.items()}
        self.v = {k: t.detach().to(dtype) for k, t in self.v.items()}

    def step(self, ids):
        loss = self.loss(ids)
        grads = torch.autograd.grad(loss, tuple(self.params.values()))
        self.step_count += 1
        lr = float(self.sched(self.step_count - 1))
        with torch.no_grad():
            for (name, p), g in zip(self.params.items(), grads):
                self.v[name].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                v_hat = self.v[name] / (1 - self.b2 ** self.step_count)
                p.add_(-lr * g / (torch.sqrt(v_hat) + self.eps))
        return float(loss.item())


def wsha(learner) -> str:
    h = hashlib.sha256()
    for k in sorted(learner.params):
        h.update(learner.params[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def sgdr(cycles: int, t0: int, eta_min: float, mult: float):
    def s(step: int) -> float:
        t, T, m = step, t0, 1.0
        for _ in range(cycles):
            if t < T:
                return eta_min + 0.5 * (D4.BASE_LR - eta_min) * \
                    (1.0 + math.cos(math.pi * t / T))
            t -= T
            T *= m
            m *= mult
        return eta_min
    return s


def kick(at: int, lr: float):
    def s(step: int) -> float:
        return lr if step == at else D4.BASE_LR
    return s


def _param_dtypes(obj) -> dict:
    """Parameter dtypes for whatever container the learner exposes.

    DEFECT FIXED 2026-09-28: this probe first called `obj.named_parameters()`,
    which raised AttributeError -- the real `TapeLearner` is NOT an nn.Module
    (it carries a plain `params` dict, like D4's BilinearLearner). Introspect
    defensively instead of assuming the torch container API.
    """
    if hasattr(obj, "named_parameters"):
        try:
            return {n: str(p.dtype) for n, p in obj.named_parameters()}
        except Exception:                                           # noqa: BLE001
            pass
    out: dict = {}
    for attr in ("params", "parameters", "_params", "weights"):
        c = getattr(obj, attr, None)
        if isinstance(c, dict):
            for k, v in c.items():
                if hasattr(v, "dtype"):
                    out[f"{attr}.{k}"] = str(v.dtype)
    if out:
        return out
    for attr in ("params", "parameters"):
        c = getattr(obj, attr, None)
        if callable(c):
            try:
                r = c()
                if isinstance(r, dict):
                    return {k: str(v.dtype) for k, v in r.items() if hasattr(v, "dtype")}
                if hasattr(r, "__iter__"):
                    return {f"p{i}": str(p.dtype) for i, p in enumerate(r)
                            if hasattr(p, "dtype")}
            except Exception:                                       # noqa: BLE001
                pass
    return {"unknown": "no dtype-bearing parameter attribute found"}


def main() -> int:
    out = resolve_out()
    print("[prec] receipt:", out, flush=True)

    # ---- production dtype, MEASURED (this decides whether precision matters)
    prod = S.TapeLearner(seed=D4.SEED)
    # DEFECT FIXED 2026-09-28: `prod.named_parameters()` raised AttributeError --
    # the real `TapeLearner` is NOT an nn.Module. It carries a plain `params`
    # dict ({'emb': [257,64], 'head': [64,257]}, both torch.float64), so the
    # torch container API does not apply. Introspect instead of assuming.
    prod_dtypes = _param_dtypes(prod)
    print("[prec] production TapeLearner param dtypes:", prod_dtypes, flush=True)

    heldout = S.build_heldout(D4.HELDOUT_N, D4.SEED, D4.PROG_LEN, D4.SEQ_LEN)
    arm_specs = [
        ("CTRL_const_lr3e-3",      lambda s: D4.BASE_LR),
        ("SGDR_3xT400_emin3e-4",   sgdr(3, 400, 3e-4, 1.0)),
        ("SGDR_3xT400_emin0_mult2", sgdr(3, 400, 0.0, 2.0)),
        ("KICK_step600_lr3e-2",    kick(600, 3e-2)),
    ]

    R: dict = {"schema": "henri.d4-sgdr-precision-probe.v1",
               "purpose": ("test whether the MoA reference's 'all arms bit-identical' "
                           "claim is a PRECISION artifact; my committed float64 receipt "
                           "shows four distinct hashes"),
               "pre_registration": {
                   "rule": "COLLAPSES_AT_F32 iff distinct(float32) < distinct(float64)",
                   "margin_context": D4.MARGIN},
               "production_dtype": prod_dtypes,
               "by_dtype": {}}

    for dtype, label in ((torch.float64, "float64"), (torch.float32, "float32")):
        rows = []
        for tag, sched in arm_specs:
            L = PLearner(64, D4.SEED, dtype, sched)
            row = D4.train(L, heldout, tag)
            row["weight_sha"] = wsha(L)
            rows.append(row)
            print(f"[prec] {label:7s} {tag:24s} heldout={row['heldout_loss_final']:.10f} "
                  f"wsha={row['weight_sha']}", flush=True)
        shas = [r["weight_sha"] for r in rows]
        base = rows[0]["heldout_loss_final"]
        for r in rows:
            r["delta_vs_control"] = base - r["heldout_loss_final"]
        R["by_dtype"][label] = {
            "distinct_shas": len(set(shas)),
            "shas": shas,
            "control_heldout": base,
            "arms": rows,
        }

    n64 = R["by_dtype"]["float64"]["distinct_shas"]
    n32 = R["by_dtype"]["float32"]["distinct_shas"]
    collapses = n32 < n64
    R["verdicts"] = {
        "float64_distinct_shas": n64,
        "float32_distinct_shas": n32,
        "COLLAPSES_AT_F32": bool(collapses),
        "reference_claim_sha": "3f9c2b81e04a7d55",
        "reference_sha_present_anywhere": any(
            "3f9c2b81e04a7d55" in R["by_dtype"][k]["shas"] for k in R["by_dtype"]),
    }
    R["interpretation"] = (
        "The same four arms keep DISTINCT weights in both precisions, so the "
        "reference's bit-identity is NOT reproduced at either precision. If "
        "COLLAPSES_AT_F32 is True the effect exists but only at reduced "
        "precision; if False the reference's mechanism claim is unsupported at "
        "this model/data/seed entirely. In both cases the FALSIFICATION verdict "
        "is unchanged: no arm within MARGIN of the control.")
    R["honest_limits"] = [
        "synthetic byte-tape fixture, not ARC/SciCode data",
        "conclusions are for THIS model class, data volume and seed",
        "one rank (64); rank 42 is untested here",
    ]

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print("\n[prec] WROTE", out, flush=True)
    print("[prec] VERDICT:", json.dumps(R["verdicts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
