"""BUDGET-EXTENSION TEST: is the plateau converged, and does the context win persist?

WHAT FORCED THIS PROBE (my own measurement, not a reference claim)
=================================================================
`d4_basin_shape_probe.json` trained the SAME bilinear architecture at 1x, 2x and
3x the D4 budget (1200 -> 2400 -> 3600 steps) and the held-out loss kept falling:

    step 1200   control 0.12927328
    step 2400   control 0.12916915
    step 3600   control 0.12787813   <-- BELOW D4's matched bigram floor 0.12791051

Two consequences, both damaging to claims I ALREADY REPORTED:

1. D4 verdict `4_AT_MATCHED_BIGRAM_FLOOR` said the control "is already at the
   bigram conditional entropy; no optimiser can go below it". The control went
   below it. That interpretation clause is FALSIFIED by my own measurement.
2. The `CONTEXT_r42_prev_token` advantage of 0.0115403 nats -- the single piece of
   evidence behind `MODEL_CLASS_IS_THE_BINDING_CONSTRAINT` -- was measured at 1200
   steps ONLY. If the bilinear arm simply needed more steps, that verdict was a
   BUDGET ARTIFACT rather than a model-class effect.

So the plateau's status is reopened. This probe decides it.

PRE-REGISTERED (written before the first run)
=============================================
Arms, both at matched budget, identical init seed and identical batch order:
    CTRL   = D4.BilinearLearner(rank 64, 32,896 params)      lr 3e-3 constant
    CONTEXT= D4.ContextLearner(rank 42, 32,382 params)       prev-token embedding
Checkpoints: 1200, 2400, 3600. MARGIN = 0.01 (D4's, unchanged).

  CONTEXT_ADVANTAGE_PERSISTS   iff (ctrl - ctx) >= MARGIN at the FINAL checkpoint
  CONTEXT_ADVANTAGE_WAS_BUDGET iff (ctrl - ctx) <  MARGIN at the FINAL checkpoint
  FLOOR_BINDING_AT_FINAL       iff ctrl_final >= floor(matched volume) - 0.005
  FLOOR_FALSIFIED_BY_BUDGET    iff ctrl_final <  floor(matched volume) - 0.005

Floors are re-fit at MATCHED DATA VOLUME for each checkpoint (1200*256 = 307,200
rows and 3600*256 = 921,600 rows), because a floor quoted at one volume is not a
bound at another -- that was the exact defect I fixed in D4 (its predecessor
quoted a 4,000-row floor against 307,200 rows of training).

VALIDITY GATES (the run is VOID unless V1 and V2 pass)
======================================================
  V1 ctrl at 1200 reproduces D4's recorded BASE_LOSS bit-for-bit.
  V2 floor fit on 307,200 rows reproduces D4's recorded 0.12791051.
  V3 the loss trajectory per arm is reported at every checkpoint, so a reader can
     see the traffic, not just the endpoint.

REUSE, NEVER REIMPLEMENT: the learners, batch generator, bigram floor and heldout
builder are IMPORTED from the verified `d4_optimizer_sweep` module. The only new
code is the checkpoint loop.

DETERMINISM: no wall-clock value enters the receipt. Batch order is generator
seeded per arm exactly as D4's `train()` does, so each arm sees the same sequence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "verification"))

import d4_optimizer_sweep as D4                                 # noqa: E402
import stage0_seeding_run as S                                 # noqa: E402

CHECKPOINTS = (1200, 2400, 3600)
RANK_CTRL = 64                     # 32,896 params
RANK_CTX = 42                      # 32,382 params (99% matched)
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "d4_budget_extension_probe.json")


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
        return os.path.join(envd, "d4_budget_extension_probe.json")
    return DEFAULT_OUT


def main() -> int:
    out = resolve_out()
    print("[budget] receipt:", out, flush=True)

    heldout = S.build_heldout(D4.HELDOUT_N, D4.SEED, D4.PROG_LEN, D4.SEQ_LEN)

    # Each arm gets its OWN generator seeded exactly as D4.train() does, so both
    # arms consume the IDENTICAL batch sequence (D4's comparability contract).
    ctrl = D4.BilinearLearner(RANK_CTRL, D4.SEED, lr=D4.BASE_LR,
                              cosine=False, grad_clip=0.0)
    ctx = D4.ContextLearner(RANK_CTX, D4.SEED, lr=D4.BASE_LR)
    gens = {"CTRL_r64": torch.Generator().manual_seed(D4.SEED + 4242),
            "CONTEXT_r42_prev_token": torch.Generator().manual_seed(D4.SEED + 4242)}

    R: dict = {
        "schema": "henri.d4-budget-extension-probe.v1",
        "purpose": ("decide whether the 0.129273 plateau is CONVERGED or "
                    "BUDGET-LIMITED, and whether the context advantage measured "
                    "at 1200 steps persists at matched budget"),
        "forced_by": ("d4_basin_shape_probe measured control 0.12787813 at 3600 "
                      "steps, BELOW D4's matched bigram floor 0.12791051 -- which "
                      "falsifies my own verdict-4 interpretation clause"),
        "pre_registration": {
            "margin": D4.MARGIN,
            "checkpoints": list(CHECKPOINTS),
            "context_persists_rule": "(ctrl - ctx) >= MARGIN at FINAL checkpoint",
            "context_budget_artifact_rule": "(ctrl - ctx) < MARGIN at FINAL checkpoint",
            "floor_binding_rule": "ctrl_final >= floor(matched volume) - 0.005",
            "ranks": {"ctrl": RANK_CTRL, "ctx": RANK_CTX},
            "params": {"ctrl": ctrl.n_params(), "ctx": ctx.n_params()},
        },
        "checkpoints": {},
    }

    # FLOOR DATA, generated ONCE at the maximum volume and sliced by prefix.
    # `make_rows` draws sequentially from a seeded generator, so the first `vols`
    # rows of the max-volume draw are EXACTLY the draw of `vols` rows with the same
    # seed. Prefix-slicing is therefore the same data as a fresh draw, at 1/3 cost,
    # and it keeps the three floors mutually consistent by construction.
    _fit_rng = torch.Generator().manual_seed(D4.SEED + 4242)
    fit_rows_max = D4.make_rows(max(CHECKPOINTS) * D4.BATCH, _fit_rng)

    for step in range(1, max(CHECKPOINTS) + 1):
        for arm, learner, gen in (("CTRL_r64", ctrl, gens["CTRL_r64"]),
                                 ("CONTEXT_r42_prev_token", ctx,
                                  gens["CONTEXT_r42_prev_token"])):
            learner.step(D4.make_rows(D4.BATCH, gen))

        if step in CHECKPOINTS:
            vols = step * D4.BATCH
            floor = D4.bigram_floor(fit_rows_max[:vols], heldout)
            with torch.no_grad():
                c_ho = float(ctrl.loss(heldout))
                x_ho = float(ctx.loss(heldout))
            gap = c_ho - x_ho
            R["checkpoints"][str(step)] = {
                "rows_consumed": vols,
                "control_heldout": c_ho,
                "context_heldout": x_ho,
                "gap_control_minus_context": gap,
                "bigram_floor_at_matched_volume": floor,
                "control_minus_floor": c_ho - floor,
                "context_minus_floor": x_ho - floor,
            }
            print(f"[budget] step={step:5d} ctrl={c_ho:.11f} ctx={x_ho:.11f} "
                  f"gap={gap:+.11f} floor={floor:.11f} ctrl-floor={c_ho-floor:+.11f}",
                  flush=True)

    cps = R["checkpoints"]
    first = cps[str(CHECKPOINTS[0])]
    last = cps[str(CHECKPOINTS[-1])]

    persists = last["gap_control_minus_context"] >= D4.MARGIN
    floor_binding = last["control_minus_floor"] >= -0.005

    R["validity"] = {
        "v1_control_at_1200_reproduces_D4_BASE_LOSS":
            abs(first["control_heldout"] - D4.BASE_LOSS) < 1e-12,
        "v1_control_at_1200": first["control_heldout"],
        "v1_D4_recorded_BASE_LOSS": D4.BASE_LOSS,
        "v2_floor_at_307200_reproduces_D4_record":
            abs(first["bigram_floor_at_matched_volume"] - 0.12791051131738324) < 1e-9,
        "v2_floor_at_307200": first["bigram_floor_at_matched_volume"],
        "v2_D4_recorded_floor": 0.12791051131738324,
        "v3_loss_still_descending_control":
            bool(last["control_heldout"] < first["control_heldout"]),
        "v3_loss_still_descending_context":
            bool(last["context_heldout"] < first["context_heldout"]),
        "v3_floor_still_descending_with_volume":
            bool(last["bigram_floor_at_matched_volume"]
                 < first["bigram_floor_at_matched_volume"]),
    }
    R["verdicts"] = {
        "control_1200": first["control_heldout"],
        "control_3600": last["control_heldout"],
        "context_1200": first["context_heldout"],
        "context_3600": last["context_heldout"],
        "gap_1200": first["gap_control_minus_context"],
        "gap_3600": last["gap_control_minus_context"],
        "MARGIN": D4.MARGIN,
        "CONTEXT_ADVANTAGE_PERSISTS_AT_MATCHED_BUDGET": bool(persists),
        "CONTEXT_ADVANTAGE_WAS_BUDGET_ARTIFACT": bool(not persists),
        "FLOOR_BINDING_AT_FINAL": bool(floor_binding),
        "FLOOR_FALSIFIED_BY_BUDGET": bool(not floor_binding),
    }
    R["interpretation"] = (
        "CONTEXT_ADVANTAGE_PERSISTS: the model-class conclusion survives matched "
        "budget -- context, not the optimiser, is the binding constraint."
        if persists else
        "CONTEXT_ADVANTAGE_WAS_BUDGET_ARTIFACT: at matched budget the bilinear arm "
        "catches the context arm, so MODEL_CLASS_IS_THE_BINDING_CONSTRAINT does NOT "
        "survive budget matching. The binding constraint is then TRAINING BUDGET on "
        "this fixture, and my earlier 1200-step verdict must be withdrawn.")
    R["honest_limits"] = [
        "synthetic byte-tape fixture, not ARC/SciCode data",
        "3600 steps is still finite; the loss was descending at the last checkpoint "
        "in the basin probe, so no claim of a converged floor is made anywhere here",
        "two arms only; the full LR/clip sweep is not re-run at extended budget",
        "conclusions are for THIS model class, data volume and seed",
    ]

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print("\n[budget] WROTE", out, flush=True)
    print("[budget] validity:", json.dumps(R["validity"], indent=2), flush=True)
    print("[budget] VERDICT:", json.dumps(R["verdicts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
