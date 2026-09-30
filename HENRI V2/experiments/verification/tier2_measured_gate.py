"""TIER-2 MEASURED GATE -- the GPU half of item 5.

STATUS: harness preflighted at PREFLIGHT SCALE on CPU (d=512, r=16, 16x16 grid).
The GPU half is still NOT measured. Nothing here is a Tier-2 promotion.

WHY THIS FILE EXISTS (the instance is stopped and costs nothing)
  GPU windows are the scarce resource. The proven pattern is preflight-then-
  measure: write and CPU-preflight the harness BEFORE provisioning the 5090, so
  the paid window runs one command. The structural half of the Tier-2 gate PASSED
  (experiments/verification/test_tier2_wiring.py, 5/5).

TWO MEASUREMENTS, DELIBERATELY SEPARATED
  T2-a  LATENT PREDICTION. Open-loop 3-step cosine between the rolled-forward
        latent and the true encoded latent, on steps HELD OUT from the online
        warmup. Threshold 0.92. Teacher-forced one-step cosine is reported beside
        it so the two are never confused.

  T2-c  UNITARITY, measured ON THE SUBSPACE where unitarity is definable.

  THE TRAP THIS AVOIDS (two traps, both real, both hit)
  1. RecursiveDualEDMD.forward ends in F.normalize(...), so any norm read back
     from forward() is 1.0 BY CONSTRUCTION. Reporting that as "unitary preserved"
     is a manufactured pass -- symbolic proof by naming.
  2. FIRST DRAFT OF THIS FILE measured ||V A_sub V^T v|| on an AMBIENT unit vector
     and asserted "A_sub = I must give 1.0". That assertion is FALSE by geometry:
     V^T V = I_r (columns are F.normalize'd) but V V^T is a rank-r projector, so
     for random ambient v, ||V V^T v|| ~ sqrt(r/d). Measured at the CPU preflight:
     0.137 at r=16, d=512, against sqrt(16/512) = 0.177. The clause could NEVER
     print PASS. A gate that cannot pass is as worthless as one that cannot fail.
  3. CORRECTED TEST. Unitarity is definable only on the r-dim subspace:
         u in R^r, gain = ||A_sub u|| / ||u||,  orthogonality defect
         ||A_sub^T A_sub - I||_F / sqrt(r).
     A pure rotation gives gain 1 and defect 0. A [REF] orthogonal factor (QR of
     a random matrix) is measured alongside to prove the metric CAN report a clean
     pass. The ambient projection loss is printed as [DIAG] and labelled, so it can
     never be mistaken for the unitarity claim.

  CAVEAT CARRIED FORWARD: the production config uses r_rank=16 too (WaveJEPA
  default), so rank is NOT scaled up with d. This is why --rank-sweep exists: it
  separates "the subspace is too small" (capacity) from "the map is not learnable
  at any rank" (architecture/information). Both are real findings; they license
  different next actions.

PRE-REGISTERED DECISION RULE (fixed before the run)
  T2-a PASS  : open-loop 3-step cosine >= 0.92
  T2-a KILL  : min open-loop cosine  <  0.60   -> exit 2, hard stop, no PASS
  0.60 <= cos < 0.92 -> exit 1, "INCONCLUSIVE-BAND": reported, not promoted
  T2-c PASS  : gain within 1e-5 of 1 AND orthogonality defect <= 1e-5
  Any non-finite measurement -> exit 1, "INVALID".

FAIL-CLOSED
  The harness can NOT print TIER2_MEASURED: PASS unless every clause was measured
  and finite.

Run (CPU preflight -- what has actually been run):
  python experiments/verification/tier2_measured_gate.py --d 512 --nb 64
  python experiments/verification/tier2_measured_gate.py --d 512 --nb 64 --rank-sweep
Run (target GPU, the paid window -- NOT yet run):
  python experiments/verification/tier2_measured_gate.py --out receipts/tier2_measured_<date>.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from henri_vision_encoder import HENRIVisionEncoder      # noqa: E402
from wave_jepa import WaveJEPA                            # noqa: E402
from recursive_dual_edmd import RecursiveDualEDMD         # noqa: E402

# ---- contract (pre-registered; do not tune after seeing a result) ----------
THRESH_T2A = 0.92          # open-loop 3-step cosine that must be reached
KILL_T2A = 0.60            # below this the hypothesis is rejected
THRESH_T2C = 1e-5          # subspace gain and orthogonality defect
ROLLOUT_H = 3              # "3-step"

MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]     # right, left, down, up


def make_grid(g: int, y: int, x: int) -> list[list[int]]:
    grid = [[0] * g for _ in range(g)]
    grid[y % g][x % g] = 1
    return grid


def action_wave(a_idx: int, nb: int, device: torch.device) -> torch.Tensor:
    """Distinct, deterministic unit action wave per action index. [nb, 8]."""
    gen = torch.Generator(device="cpu").manual_seed(90210 + int(a_idx))
    w = torch.randn(nb, 8, generator=gen)
    return F.normalize(w.view(-1), p=2, dim=0).to(device).view(nb, 8)


def build_trajectory(g: int, steps: int, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    actions = torch.randint(0, 4, (steps,), generator=gen).tolist()
    grids, cur_y, cur_x = [], 0, 0
    grids.append(make_grid(g, cur_y, cur_x))
    for a in actions:
        dy, dx = MOVES[a % 4]
        cur_y, cur_x = cur_y + dy, cur_x + dx
        grids.append(make_grid(g, cur_y, cur_x))
    return actions, grids


def stats(xs):
    return dict(n=len(xs), mean=sum(xs) / len(xs), min=min(xs), max=max(xs))


def score_predictor(rank: int, psis, actions, args, device):
    """Train a FRESH predictor at `rank` on the warmup, score on the holdout.

    The encoder is rank-independent, so psis are reused -- only the predictor is
    rebuilt. This is what makes --rank-sweep cheap enough to be a diagnostic.
    """
    pred = RecursiveDualEDMD(d_model=args.d, r_rank=rank, lambda_forget=0.98)
    losses = []
    for i in range(args.warmup):
        a_i = action_wave(actions[i], args.nb, device)
        losses.append(pred.update_online_step(psis[i], a_i, psis[i + 1]))

    base = args.warmup
    one_step, open_loop = [], []
    with torch.no_grad():
        for i in range(base, base + args.holdout):
            a_i = action_wave(actions[i], args.nb, device)
            p1 = pred(psis[i], a_i)
            one_step.append(float(F.cosine_similarity(
                p1.view(-1), psis[i + 1].view(-1), dim=0)))
            psi_hat = psis[i]
            for k in range(ROLLOUT_H):
                if i + k + 1 >= len(psis):
                    break
                a_k = action_wave(actions[i + k], args.nb, device)
                psi_hat = pred(psi_hat, a_k)
                if k == ROLLOUT_H - 1:
                    open_loop.append(float(F.cosine_similarity(
                        psi_hat.view(-1), psis[i + k + 1].view(-1), dim=0)))
    return dict(rank=rank, one_step=stats(one_step), open_loop=stats(open_loop),
                loss_first=losses[0], loss_last=losses[-1], pred=pred)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=65536)
    ap.add_argument("--nb", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--holdout", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--rank-sweep", action="store_true",
                    help="diagnostic: separate capacity (rank) from architecture")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.d % args.nb != 0 or (args.d // args.nb) != 8:
        print(f"REFUSING: d={args.d} nb={args.nb} -> block_dim={args.d / args.nb}; "
              f"the encoder requires block_dim == 8 (d = nb * 8)")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half else (torch.bfloat16 if device.type == "cuda" else torch.float32)

    print("=" * 78)
    print("TIER-2 MEASURED GATE")
    print("=" * 78)
    if device.type == "cpu":
        print("  *** PREFLIGHT SCALE (CPU). A failure here is a failure at THIS scale,")
        print("  *** not a GPU-scale result. It is reported as measured, not as proof.")
    print(f"  device      : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  dtype       : {dtype}")
    print(f"  d_model     : {args.d}   num_blocks: {args.nb}   block_dim: 8")
    print(f"  rank        : {args.rank}   (production default is also 16)")
    print(f"  grid        : {args.grid}x{args.grid}   warmup: {args.warmup}   holdout: {args.holdout}")
    print(f"  thresholds  : T2-a cos >= {THRESH_T2A}  KILL < {KILL_T2A}  T2-c gain/defect <= {THRESH_T2C}")
    print()

    enc = HENRIVisionEncoder(d_model=args.d, k_blocks=args.nb, device=str(device),
                             spatial_basis_kind="incommensurate", bg_mask=True,
                             fused_superpose=True, parity_scipy=True)
    jepa = WaveJEPA(d_model=args.d, num_blocks=args.nb, r_rank=args.rank,
                    device=str(device), encoder=enc)
    assert jepa.encoder is enc, "substrate pinning failed"

    n_steps = args.warmup + args.holdout + ROLLOUT_H + 2
    actions, grids = build_trajectory(args.grid, n_steps, args.seed)
    psis = []
    with torch.no_grad():
        for g in grids:
            psis.append(jepa.encode_context(g).to(device))
    norms = [float(p.norm()) for p in psis]
    unique = len({tuple(g[0]) for g in grids})
    print(f"  trajectory  : {len(actions)} actions, {len(grids)} grids (seeded {args.seed})")
    print(f"  encoded     : {len(psis)} waves {tuple(psis[0].shape)}, "
          f"norm {min(norms):.7f}..{max(norms):.7f}")
    print(f"  distinct first-rows: {unique}  (a degenerate trajectory would make every "
          f"cosine 1.0; reported, not hidden)")
    if unique <= 1:
        print("REFUSING: the trajectory does not move -> T2-a would be tautological.")
        return 1
    print()

    # ---- T2-a ----
    res = score_predictor(args.rank, psis, actions, args, device)
    s1, s3 = res["one_step"], res["open_loop"]
    print("T2-a  LATENT PREDICTION (held-out)")
    print(f"  one-step (teacher-forced) : mean={s1['mean']:.6f} min={s1['min']:.6f} "
          f"max={s1['max']:.6f}  n={s1['n']}")
    print(f"  3-step OPEN-LOOP          : mean={s3['mean']:.6f} min={s3['min']:.6f} "
          f"max={s3['max']:.6f}  n={s3['n']}   <- GATED")
    print(f"  warmup loss first={res['loss_first']:.4e} last={res['loss_last']:.4e}")
    # chance level for a random unit vector in d dims
    chance_sd = 1.0 / math.sqrt(args.d)
    print(f"  CHANCE BASELINE: a random unit vector gives cos ~ N(0, 1/sqrt(d)), "
          f"sd={chance_sd:.4f}")
    print(f"  teacher-forced is reported for contrast; it is NOT the T2-a claim.")
    print()

    # ---- T2-c (corrected TWICE -- the second correction matters more) ----
    # CORRECTION 1 (geometry): the first draft measured ||V A V^T v|| on an AMBIENT
    # unit vector and asserted "A=I must give 1.0". False by geometry: V V^T is a
    # rank-r projector, so ||V V^T v|| ~ sqrt(r/d) (measured 0.137 at r=16,d=512 vs
    # sqrt(16/512)=0.177). Unitarity is definable only on the r-dim subspace.
    #
    # CORRECTION 2 (MANUFACTURED PASS): the corrected draft then read
    # `jepa.predictor`, which is NEVER TRAINED -- main() trains a FRESH predictor
    # inside score_predictor(). A_sub initialises to torch.eye(r), so the "measured"
    # gain was exactly 1.0 and the defect exactly 0.0 BY INITIALISATION. Reporting
    # the identity matrix as proof that a LEARNED operator is unitary is symbolic
    # proof by naming -- the same defect class as the hardcoded smoke marker.
    # T2-c now reads the TRAINED operator, and an untouched operator is INVALID
    # (fail-closed), not a PASS.
    trained = res["pred"]
    V = trained.V.to(device).to(torch.float32)
    A = trained.A_sub.to(device).to(torch.float32)
    r = args.rank
    eye_r = torch.eye(r, device=device)
    a_is_identity = bool(torch.allclose(A, eye_r, atol=1e-6))
    a_delta_init = float((A - eye_r).abs().max())
    gen = torch.Generator(device="cpu").manual_seed(777)
    u = F.normalize(torch.randn(r, generator=gen), p=2, dim=0).to(device)
    v_amb = F.normalize(torch.randn(args.d, generator=gen), p=2, dim=0).to(device)

    gain_sub = float((A @ u).norm() / u.norm())
    orth_defect = float((A.T @ A - torch.eye(r, device=device)).norm() / math.sqrt(r))
    R_ref = torch.linalg.qr(torch.randn(r, r, generator=gen).to(device))[0]
    gain_ref = float((R_ref @ u).norm() / u.norm())
    ref_defect = float((R_ref.T @ R_ref - torch.eye(r, device=device)).norm() / math.sqrt(r))
    proj_amb = float((V @ (torch.eye(r, device=device) @ (V.T @ v_amb))).norm())
    proj_theory = math.sqrt(r / args.d)

    print("T2-c  UNITARITY (subspace, TRAINED operator) + ambient projection (diagnostic)")
    print(f"  [CLAIM] ||A_sub u||/||u||          : {gain_sub:.8f}   |gain-1|={abs(gain_sub-1):.3e}")
    print(f"  [CLAIM] ||A^T A - I||_F/sqrt(r)    : {orth_defect:.8e}")
    print(f"  [GUARD] ||A_sub - I||_max          : {a_delta_init:.3e}   "
          f"{'UNLEARNED (identity) -- NOT EVIDENCE' if a_is_identity else 'operator moved from init'}")
    print(f"  [REF]   orthogonal factor: gain={gain_ref:.8f} defect={ref_defect:.2e} "
          f"(proves the metric CAN report a clean pass)")
    print(f"  [DIAG]  ||V V^T v||, A=I, ambient  : {proj_amb:.8f}  vs sqrt(r/d)={proj_theory:.8f}")
    print(f"          <- rank-deficient projection loss, NOT non-unitarity. Labelled so it")
    print(f"             cannot be reported as the claim (the first draft's error).")
    print(f"  NOTE: a norm read from forward() is 1.0 BY CONSTRUCTION (F.normalize) -- "
          f"not evidence.")
    print()

    if args.rank_sweep:
        r0 = max(2, args.rank)
        ranks = sorted({r0, min(args.d // 4, r0 * 2), min(args.d // 2, r0 * 4), min(args.d, r0 * 8)})
        print("RANK SWEEP (diagnostic: capacity vs architecture)")
        print(f"  {'rank':>6}  {'1-step cos':>11}  {'3-step cos':>11}  {'loss last':>11}")
        sweep = []
        for rr in ranks:
            o = score_predictor(rr, psis, actions, args, device)
            sweep.append(dict(rank=rr, one_step=o["one_step"]["mean"],
                              three_step=o["open_loop"]["mean"], loss_last=o["loss_last"]))
            print(f"  {rr:>6}  {o['one_step']['mean']:>11.6f}  "
                  f"{o['open_loop']['mean']:>11.6f}  {o['loss_last']:>11.4e}")
        three = [s["three_step"] for s in sweep]
        one = [s["one_step"] for s in sweep]
        spread3 = max(three) - min(three)
        spread1 = max(one) - min(one)
        mono3 = all(b >= a for a, b in zip(three, three[1:]))
        mono1 = all(b >= a for a, b in zip(one, one[1:]))
        print(f"  spread: 3-step={spread3:.6f}  1-step={spread1:.6f}  3*chance_sd={3*chance_sd:.4f}")
        print(f"  monotone increasing with rank: 3-step={mono3}  1-step={mono1}")
        above3 = [s["rank"] for s in sweep if s["three_step"] >= 3 * chance_sd]
        above1 = [s["rank"] for s in sweep if s["one_step"] >= 3 * chance_sd]
        print(f"  ranks with 3-step >= 3*chance : {above3 or 'none'}")
        print(f"  ranks with 1-step >= 3*chance : {above1 or 'none'}")
        # HONESTY NOTE: an earlier draft of this block tested ONLY the 3-step spread
        # against 3*chance_sd and then printed "the limit is NOT subspace capacity".
        # That overstates the data: the 1-step column rises monotonically and clears
        # 3*chance_sd at the top ranks. The curve is consistent with BOTH a capacity
        # limit and an architecture/information limit, and this diagnostic does not
        # separate them. Report the curve; do not over-read it.
        if not above1 and not above3:
            print("  READING: no rank clears 3*chance at either horizon -> not attributable")
            print("  to subspace capacity at this scale. Diagnostic only; not promoted.")
        elif mono1 or mono3:
            print("  READING: cosine rises with rank, so subspace capacity CONTRIBUTES. But")
            print("  even the best cell sits far below the 0.92 threshold, so capacity alone")
            print("  does NOT explain the miss. This curve does not separate capacity from an")
            print("  architecture/information limit. Do not tune the threshold to it.")
        else:
            print("  READING: no monotone trend and no cell clears 3*chance -> more consistent")
            print("  with an architecture/information limit than with capacity. Diagnostic only.")
        print("  CAVEAT: measured at PREFLIGHT SCALE only (see header). Production scale")
        print("  (d=65536) is NOT measured and may differ. Never compare across fixtures.")
        print()

    # ---- verdict (fail-closed) ----
    reasons = []
    for name, val in (("T2-a mean", s3["mean"]), ("T2-a min", s3["min"]),
                      ("T2-c gain", gain_sub), ("T2-c defect", orth_defect),
                      ("T2-c ref gain", gain_ref)):
        if val is None or not math.isfinite(val):
            reasons.append(f"{name} is not finite ({val})")
    if a_is_identity:
        reasons.append("T2-c read an UNLEARNED (identity) A_sub -- not evidence of unitarity")

    cos_ok = s3["mean"] >= THRESH_T2A
    killed = s3["min"] < KILL_T2A
    t2c_ok = (abs(gain_sub - 1.0) <= THRESH_T2C) and (orth_defect <= THRESH_T2C)

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  T2-a 3-step open-loop mean cos {s3['mean']:.6f} vs >= {THRESH_T2A}  -> "
          f"{'PASS' if cos_ok else 'NOT-REACHED'}")
    print(f"  T2-c |gain-1|={abs(gain_sub-1):.3e}  defect={orth_defect:.3e} vs <= {THRESH_T2C}  -> "
          f"{'PASS' if t2c_ok else 'NOT-REACHED'}")
    print(f"  [metric validity check] orthogonal reference defect={ref_defect:.2e} "
          f"-> {'metric OK' if ref_defect <= 1e-5 else 'METRIC SUSPECT'}")

    verdict, rc = None, 1
    if reasons:
        verdict = "INVALID"
        for x in reasons:
            print(f"    REASON: {x}")
    elif killed:
        verdict, rc = "KILLED", 2
        print(f"    PRE-REGISTERED KILL: min open-loop cos {s3['min']:.6f} < {KILL_T2A}.")
        print(f"    The Tier-2 latent-prediction hypothesis is REJECTED at this scale.")
    elif cos_ok and t2c_ok:
        verdict, rc = "PASS", 0
    else:
        verdict = "INCONCLUSIVE-BAND"
        if not cos_ok:
            print(f"    T2-a in band [{KILL_T2A}, {THRESH_T2A}): reported, not promoted.")
        if not t2c_ok:
            print(f"    T2-c not reached: the learned subspace operator is not orthogonal to")
            print(f"    {THRESH_T2C}. The 'unitary predictor' label is NOT earned.")

    print()
    print(f"TIER2_MEASURED: {verdict}")
    if verdict != "PASS":
        print("  (not a promotion. No Tier-2 claim is promoted without PASS.)")

    if args.out:
        rec = {
            "measured_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "dtype": str(dtype), "preflight_scale": device.type == "cpu",
            "d_model": args.d, "num_blocks": args.nb, "rank": args.rank,
            "grid": args.grid, "seed": args.seed,
            "warmup_steps": args.warmup, "holdout_steps": args.holdout,
            "substrate": {"spatial_basis_kind": "incommensurate", "bg_mask": True,
                          "fused_superpose": True, "parity_scipy": True},
            "t2a_one_step": s1, "t2a_open_loop_3step": s3,
            "t2a_chance_sd": chance_sd,
            "t2c_subspace_gain": gain_sub, "t2c_orthogonality_defect": orth_defect,
            "t2c_reference_gain": gain_ref, "t2c_reference_defect": ref_defect,
            "t2c_ambient_projection_diag": proj_amb, "t2c_ambient_projection_theory": proj_theory,
            "thresholds": {"t2a": THRESH_T2A, "kill": KILL_T2A, "t2c": THRESH_T2C},
            "warmup_loss_first": res["loss_first"], "warmup_loss_last": res["loss_last"],
            "distinct_first_rows": unique, "verdict": verdict,
        }
        out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  receipt: {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
