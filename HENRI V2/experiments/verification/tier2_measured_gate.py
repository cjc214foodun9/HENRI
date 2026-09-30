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


def learned_basis(psis, actions, rank, device, args):
    """Data-fit ORTHONORMAL basis: top-`rank` principal directions of the inputs.

    WHY THIS EXISTS -- the confound the rank sweep could NOT resolve
      RecursiveDualEDMD.V is a FIXED RANDOM buffer (registered, never learned).
      Sweeping `rank` therefore varies CAPACITY while holding basis QUALITY fixed,
      entangling three distinct causes:
        (a) capacity            -- r is too small
        (b) projection quality  -- random V discards the signal
        (c) model class / info  -- no rank-r LINEAR operator on this encoding works
      A flat curve over `rank` is consistent with ALL THREE, which is exactly why
      "capacity vs architecture" stayed unresolved. This arm fixes `rank` and
      changes ONLY the basis, isolating (b) from (a).

      Fit on the WARMUP inputs ONLY -- same data every arm gets, no holdout leakage.
    """
    X = []
    with torch.no_grad():
        for i in range(args.warmup):
            a_i = action_wave(actions[i], args.nb, device)
            X.append(F.normalize(psis[i].view(-1) + a_i.view(-1), p=2, dim=0))
    X = torch.stack(X).float()                     # [n, d], n = warmup
    # EFFECTIVE-RANK CAP (silent-mislabel guard). The basis cannot have more
    # principal directions than warmup inputs. Requesting rank 2048 with warmup
    # 120 would silently produce a 120-column basis while the arms table still
    # PRINTED "rank 2048" -- mislabelling a capacity arm. Return the effective
    # rank and let the caller refuse or report it; never relabel it silently.
    n_eff = int(min(rank, X.shape[0]))
    G = X @ X.T                                    # [n, n] (small; d never squared)
    evals, evecs = torch.linalg.eigh(G)
    idx = torch.argsort(evals, descending=True)[:n_eff]
    V = X.T @ evecs[:, idx]                        # [d, n_eff]
    V = torch.linalg.qr(V)[0]                      # orthonormal columns
    return V[:, :n_eff].contiguous(), n_eff


def fit_ridge_dual(X, Y, lam):
    """Closed-form ridge in DUAL form. Returns alpha [n,d].

    W = (X^T X + lam I)^-1 X^T Y  has shape [d,d] -- at d=65536 that is 1.7e10
    entries (68 GB fp32), so it is NEVER materialised. The identity

        x^T W = (X x)^T (K + lam I)^-1 Y,   K = X X^T  [n,n]

    needs only the [n,n] Gram matrix (n = warmup). This keeps the probe cheap
    enough to run inside the same window as the rank-bounded arms.
    """
    K = X @ X.T                                        # [n,n]
    n = K.shape[0]
    A = K + lam * torch.eye(n, device=K.device, dtype=K.dtype)
    return torch.linalg.solve(A, Y)                    # alpha [n,d]


def info_probe(psis, actions, args, device, lam_scale=1e-3):
    """INFORMATION CEILING -- the arm that separates INFORMATION from everything else.

    THE GAP THIS FILLS (the reason "capacity vs architecture" stayed unresolved)
      The rank ladder varies CAPACITY and BASIS QUALITY but never bounds what the
      ENCODING can support at all. Without that bound, a KILL verdict has no named
      cause: "rank too small", "basis wrong", "optimizer stuck" and "the next
      latent is simply not a linear function of the current one" are all consistent
      with the same flat curve.

      This probe fits the BEST POSSIBLE linear map combined_t -> psi_{t+1} with NO
      rank bottleneck (full d-dimensional, closed-form, no gradient descent). It is
      an UPPER BOUND on every linear world model on this encoding -- the rank-bounded
      arms can only do worse (their V is a projection of this).

    DECISION RULE (pre-registered, fixed before the run)
      probe 3-step >= 0.92  -> the information IS linearly present; the rank-bounded
                               failure is CAPACITY/OPTIMIZATION in the world model.
                               Blame the world model, not the encoder.
      probe 3-step <  0.60  -> the next latent is NOT a linear function of the
                               current one at ANY capacity. The limit is the
                               ENCODING (or the linear model class). Rank, basis and
                               optimizer changes cannot fix it. Stop tuning them.
      0.60 <= probe < 0.92  -> band. The information is partly there; the gap is
                               model class or noise. Reported, not promoted.

    NOTE ON HONESTY: this is a probe fit on the SAME warmup the arms get, scored on
    the SAME holdout. It is not given more data -- only more DEGREES OF FREEDOM.
    """
    n = args.warmup

    def comb(i):
        a = action_wave(actions[i], args.nb, device)
        return F.normalize(psis[i].view(-1) + a.view(-1), p=2, dim=0)

    X = torch.stack([comb(i) for i in range(n)]).float()                 # [n,d]
    Y = torch.stack([F.normalize(psis[i + 1].view(-1), p=2, dim=0)
                     for i in range(n)]).float()                         # [n,d]
    base = args.warmup
    _g = torch.Generator(device="cpu").manual_seed(31337)
    _perm = torch.randperm(n, generator=_g)

    def rollout(alpha):
        one, three = [], []

        def predict(combined):
            return F.normalize((X @ combined.float()) @ alpha, p=2, dim=0)

        with torch.no_grad():
            for i in range(base, base + args.holdout):
                one.append(float(F.cosine_similarity(
                    predict(comb(i)).view(-1), psis[i + 1].view(-1), dim=0)))
                ph = psis[i]
                for k in range(ROLLOUT_H):
                    if i + k + 1 >= len(psis):
                        break
                    ph = predict(comb(i + k) if k == 0 else
                                 F.normalize(ph.view(-1)
                                             + action_wave(actions[i + k], args.nb, device).view(-1),
                                             p=2, dim=0))
                    if k == ROLLOUT_H - 1:
                        three.append(float(F.cosine_similarity(
                            ph.view(-1), psis[i + k + 1].view(-1), dim=0)))
        return stats(one), stats(three)

    # LAM SWEEP -- WHY IT IS NECESSARY (a control caught my error)
    #   Draft 1 used ONE lam and required identity-recovery cos >= 0.99, but scored
    #   the identity control OUT OF SAMPLE. Ridge shrinks by construction
    #   (K(K+lamI)^-1 = I - lam(K+lamI)^-1), so at lam=1e-3*n=0.2 the identity map
    #   recovered only 0.9589 and the control correctly REFUSED to certify the probe.
    #   The same lam was ALSO suppressing the main probe -- so "the information is
    #   not in the encoding" may have been an over-regularization artifact.
    #   Fix: (a) test the machinery IN SAMPLE, where a working solve must be exact;
    #   (b) sweep lam so no verdict hinges on one arbitrary value. Every tunable
    #   stays visible in the output; nothing is hidden.
    records = []
    for ls in (1e-6, 1e-5, 1e-4, 1e-3):
        lam = ls * float(n)
        s1, s3 = rollout(fit_ridge_dual(X, Y, lam))
        # MACHINERY CONTROL, IN SAMPLE. rank(K) = min(n, d) = n here, so a small lam
        # must recover the identity almost exactly. A low value means a broken solve.
        a_self = fit_ridge_dual(X, X, lam)
        with torch.no_grad():
            self_in = [float(F.cosine_similarity(
                F.normalize((X @ X[j]) @ a_self, p=2, dim=0).view(-1),
                X[j].view(-1), dim=0)) for j in range(n)]
        a_null = fit_ridge_dual(X, Y[_perm], lam)
        with torch.no_grad():
            null = [float(F.cosine_similarity(
                F.normalize((X @ comb(i).float()) @ a_null, p=2, dim=0).view(-1),
                psis[i + 1].view(-1), dim=0))
                for i in range(base, base + args.holdout)]
        c_self, c_null = stats(self_in), stats(null)
        # fail-closed: identity must be exact to 1e-3 IN SAMPLE; shuffled ~ 0
        ok = bool(min(self_in) >= 0.999 and abs(c_null["mean"]) <= 0.15)
        records.append(dict(lam=lam, lam_scale=ls, one_step=s1, open_loop=s3,
                            control_self_in=c_self, control_null=c_null,
                            self_in_min=min(self_in), control_ok=ok))
    valid = [r for r in records if r["control_ok"]]
    # The probe is an UPPER BOUND on linear predictability, so the headline is the
    # MAX over lams whose machinery control passed -- not a cherry-pick: a higher
    # cosine is FRIENDLIER to the world model, and thus CONSERVATIVE for a KILL.
    best = (max(valid, key=lambda r: r["open_loop"]["mean"]) if valid else None)
    return dict(n=n, records=records, n_valid=len(valid),
                control_ok=bool(valid), best=best,
                one_step=(best or records[0])["one_step"],
                open_loop=(best or records[0])["open_loop"])


def score_predictor(rank: int, psis, actions, args, device, basis="random", tag=None):
    """Train a FRESH predictor at `rank` on the warmup, score on the holdout.

    basis="random"  -> the module's fixed random V (current production path)
    basis="learned" -> data-fit orthonormal basis, SAME rank (isolates basis quality)

    DEVICE: the module is moved to `device` explicitly. Without this the buffers
    (V, C_t, G_t, A_sub) stay on CPU while `psis` is on CUDA, and the first
    F.cosine_similarity raises a device mismatch -- a window-killing crash that the
    CPU preflight cannot reproduce. `device_ok` is reported so a silent host-side
    fallback can never be mistaken for a GPU measurement.
    """
    # EFFECTIVE RANK FIRST. The learned basis cannot exceed the warmup count, so
    # the MODULE must be built at that effective rank. Building at `rank` and then
    # copying a narrower V raises a shape error (RankError -> window killed).
    V = None
    eff_rank = rank
    if basis == "learned":
        V, eff_rank = learned_basis(psis, actions, rank, device, args)
    pred = RecursiveDualEDMD(d_model=args.d, r_rank=eff_rank, lambda_forget=0.98).to(device)
    if V is not None:
        pred.V.copy_(V.to(pred.V.dtype))
    device_ok = bool(pred.V.device == device and pred.A_sub.device == device)

    losses = []
    for i in range(args.warmup):
        a_i = action_wave(actions[i], args.nb, device)
        losses.append(pred.update_online_step(psis[i], a_i, psis[i + 1]))

    base = args.warmup
    one_step, open_loop, teacher3 = [], [], []
    static1, static3 = [], []
    with torch.no_grad():
        for i in range(base, base + args.holdout):
            a_i = action_wave(actions[i], args.nb, device)
            p1 = pred(psis[i], a_i)
            one_step.append(float(F.cosine_similarity(
                p1.view(-1), psis[i + 1].view(-1), dim=0)))
            # STATIC BASELINE (tautology guard on the T2-a threshold).
            # The do-nothing predictor psi_hat := psi_t. If adjacent encoded states
            # are highly collinear, this scores high WITHOUT any world model, and a
            # T2-a "PASS" would then be purchasable by doing nothing. The v8 control
            # hinted at this: an identity-like fit reconstructed held-out points at
            # cos 0.9589. Measured here, not assumed.
            static1.append(float(F.cosine_similarity(
                psis[i].view(-1), psis[i + 1].view(-1), dim=0)))
            psi_hat = psis[i]
            for k in range(ROLLOUT_H):
                if i + k + 1 >= len(psis):
                    break
                a_k = action_wave(actions[i + k], args.nb, device)
                psi_hat = pred(psi_hat, a_k)
                if k == ROLLOUT_H - 1:
                    open_loop.append(float(F.cosine_similarity(
                        psi_hat.view(-1), psis[i + k + 1].view(-1), dim=0)))
                    static3.append(float(F.cosine_similarity(
                        psis[i].view(-1), psis[i + k + 1].view(-1), dim=0)))
            # ORACLE: 3rd step with TEACHER FORCED inputs at steps 0,1.
            # Separates per-step error (this) from open-loop COMPOUNDING (above).
            psi_tf = psis[i]
            for k in range(ROLLOUT_H):
                if i + k + 1 >= len(psis):
                    break
                a_k = action_wave(actions[i + k], args.nb, device)
                psi_tf = pred(psi_tf, a_k)
                if k < ROLLOUT_H - 1:
                    psi_tf = psis[i + k + 1]
                else:
                    teacher3.append(float(F.cosine_similarity(
                        psi_tf.view(-1), psis[i + k + 1].view(-1), dim=0)))
    return dict(rank=rank, basis=basis, tag=tag, device_ok=device_ok,
                eff_rank=eff_rank, rank_capped=bool(eff_rank < rank),
                one_step=stats(one_step), open_loop=stats(open_loop),
                teacher3=stats(teacher3),
                static1=stats(static1), static3=stats(static3),
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
    ap.add_argument("--arms", action="store_true",
                    help="pre-registered attribution ladder: capacity vs projection vs model class")
    ap.add_argument("--arm-ranks", type=str, default=None,
                    help="TWO comma-separated ranks for --arms: 'r0,r1' (default: rank, 4*rank). "
                         "Production must scale rank with d: the d=512 preflight used r/d=1/8, "
                         "so d=65536 needs larger ranks to test the SAME relative capacity.")
    ap.add_argument("--info-probe", action="store_true",
                    help="closed-form ridge upper bound: separates INFORMATION (encoder) "
                         "from capacity/optimization (world model). No rank bottleneck.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    arms_records = None
    info_rec = None
    arm_r0, arm_r1 = ((int(x) for x in args.arm_ranks.split(",")) if args.arm_ranks
                      else (args.rank, args.rank * 4))

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
    # STATIC BASELINE -- MUST BE PRINTED (this was computed-and-dropped once already).
    # The do-nothing predictor psi_hat := psi_t. If this alone clears the gate, then
    # T2-a is TAUTOLOGICAL: a "PASS" would be purchasable by doing nothing.
    ss1, ss3 = res["static1"], res["static3"]
    print(f"  STATIC do-nothing 1-step  : mean={ss1['mean']:.6f} "
          f"min={ss1['min']:.6f} max={ss1['max']:.6f}")
    print(f"  STATIC do-nothing 3-step  : mean={ss3['mean']:.6f} "
          f"min={ss3['min']:.6f} max={ss3['max']:.6f}   <- TAUTOLOGY GUARD")
    tautology = bool(ss3["mean"] >= THRESH_T2A)
    print(f"  TAUTOLOGY GUARD           : "
          f"{'THRESHOLD IS PURCHASABLE BY DOING NOTHING -- gate invalid' if tautology else 'gate is non-trivial (static baseline below threshold)'}")
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
    A = trained.A_sub.to(device).to(torch.float32)
    # DIMENSION FROM THE OPERATOR, not from args.rank: a learned-basis arm may
    # legally run at eff_rank < args.rank, and reading args.rank here would size
    # the probe vectors wrongly (shape error) or silently probe the wrong subspace.
    r = int(A.shape[0])
    V = trained.V.to(device).to(torch.float32)
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

    # ---- ATTRIBUTION LADDER (pre-registered BEFORE the run) ----
    if args.arms:
        print("ATTRIBUTION LADDER -- pre-registered: capacity vs projection vs model class")
        print("  RecursiveDualEDMD.V is a FIXED RANDOM buffer that never learns, so a rank")
        print("  sweep varies CAPACITY while holding basis QUALITY fixed. It cannot separate")
        print("  (a) capacity, (b) projection quality, (c) model class / information.")
        print()
        cells = [("random", arm_r0), ("learned", arm_r0),
                 ("learned", arm_r1), ("random", arm_r1)]
        arms_records = []
        print(f"  {'basis':>8} {'rank':>6} {'eff':>6} {'1-step':>10} {'3-step':>10} "
              f"{'3-step TF':>10} {'loss last':>11}")
        for bs, rr in cells:
            o = score_predictor(rr, psis, actions, args, device, basis=bs, tag=f"{bs}-r{rr}")
            arms_records.append(dict(
                basis=bs, rank=rr, eff_rank=o["eff_rank"], rank_capped=o["rank_capped"],
                device_ok=o["device_ok"],
                one_step=o["one_step"], open_loop=o["open_loop"], teacher3=o["teacher3"],
                loss_first=o["loss_first"], loss_last=o["loss_last"]))
            print(f"  {bs:>8} {rr:>6} {o['eff_rank']:>6} {o['one_step']['mean']:>10.6f} "
                  f"{o['open_loop']['mean']:>10.6f} {o['teacher3']['mean']:>10.6f} "
                  f"{o['loss_last']:>11.4e}")
        print()
        A = arms_records[0]["open_loop"]["mean"]      # fixed basis,   rank r0
        B = arms_records[1]["open_loop"]["mean"]      # learned basis, rank r0
        C = arms_records[2]["open_loop"]["mean"]      # learned basis, rank r1
        tf_gap = arms_records[0]["teacher3"]["mean"] - arms_records[0]["open_loop"]["mean"]
        best = max(rec["open_loop"]["mean"] for rec in arms_records)
        basis_gain, cap_gain = B - A, C - B
        print("=" * 78)
        print("ATTRIBUTION RULE (fixed BEFORE the run; do not retune)")
        print(f"  A=fixed-r{arm_r0}={A:.6f}   B=learned-r{arm_r0}={B:.6f}   "
              f"C=learned-r{arm_r1}={C:.6f}")
        print(f"  basis gain (B-A) = {basis_gain:+.6f}    capacity gain (C-B) = {cap_gain:+.6f}")
        print(f"  teacher-forced gap (TF - open-loop | A) = {tf_gap:+.6f}")
        capped = [(rec["basis"], rec["rank"], rec["eff_rank"]) for rec in arms_records
                  if rec["rank_capped"]]
        if not all(rec["device_ok"] for rec in arms_records):
            print("  ATTRIBUTION: INVALID -- an arm did not sit on the compute device")
            print("  (device_ok False). Those cosines are host-side; not comparable.")
        elif capped:
            print(f"  ATTRIBUTION: INVALID -- rank CAPPED at warmup for {capped}. The arm")
            print("  is not the rank it claims, so the capacity comparison is meaningless.")
            print("  Re-run with --warmup >= the largest rank. Do not read the table below.")
        elif tf_gap >= 0.10:
            print("  ATTRIBUTION: COMPOUNDING. Teacher-forced >> open-loop at the SAME cell:")
            print("  per-step error is small but amplifies over 3 steps. The fix is in the")
            print("  rollout / optimization, NOT in rank and NOT in the basis.")
        elif basis_gain >= 0.10:
            print("  ATTRIBUTION: PROJECTION QUALITY. A learned basis at the SAME rank")
            print("  recovers signal, so the fixed random V is the defect. Rank is not the")
            print("  binding constraint.")
        elif cap_gain >= 0.10:
            print("  ATTRIBUTION: CAPACITY. Rank binds once the basis is data-fit.")
        elif best < 3 * chance_sd:
            print("  ATTRIBUTION: INFORMATION / MODEL CLASS. No rank-r LINEAR operator on this")
            print("  encoding predicts the next latent at any tested rank or basis: the limit")
            print("  is in what psi CONTAINS, not in the world model.")
        else:
            print("  ATTRIBUTION: MIXED / UNRESOLVED at this scale. Report the table; promote")
            print("  no cause. A separation >= 0.10 or a clear TF gap is required.")
        print("  Thresholds (0.10 separation, 3*chance_sd floor) were fixed BEFORE this run.")
        print()

    # ---- INFORMATION CEILING (the arm that names the cause of a KILL) ----
    if args.info_probe:
        print("INFORMATION CEILING PROBE -- best LINEAR map on this encoding, no rank cap")
        print("  Full-d closed-form ridge (dual form; the [d,d] operator is never built).")
        print(f"  {'rank cap':>9} {'1-step':>10} {'3-step':>10}")
        print(f"  {'none':>9} {0.0:>10} {0.0:>10}", end="\r")
        info_rec = info_probe(psis, actions, args, device)
        # LAM SWEEP TABLE. Every tunable is shown; nothing is hidden behind one
        # arbitrary regularization. The identity control is IN SAMPLE, where a
        # working solve must be exact.
        print(f"  {'lam_scale':>10} {'lam':>10} {'1-step':>10} {'3-step':>10} "
              f"{'self_in':>9} {'null':>9} {'ctrl':>6}")
        for r in info_rec["records"]:
            print(f"  {r['lam_scale']:>10.0e} {r['lam']:>10.2e} "
                  f"{r['one_step']['mean']:>10.6f} {r['open_loop']['mean']:>10.6f} "
                  f"{r['self_in_min']:>8.4f}* {r['control_null']['mean']:>+9.4f} "
                  f"{'ok' if r['control_ok'] else 'BAD':>6}")
        print(f"  (* = MIN in-sample identity cosine; must be >= 0.9990)")
        print(f"  lams with VALID machinery: {info_rec['n_valid']}/{len(info_rec['records'])}")
        if info_rec["best"]:
            b = info_rec["best"]
            print(f"  HEADLINE (max over valid lams, conservative for a KILL): "
                  f"1-step={b['one_step']['mean']:.6f} 3-step={b['open_loop']['mean']:.6f} "
                  f"at lam={b['lam']:.2e}")
        p3 = info_rec["open_loop"]["mean"]
        print()
        if not info_rec["control_ok"]:
            # The probe is a DIAGNOSTIC. Its failure does not invalidate the T2-a/T2-c
            # gate (which is independently measured), but it VOIDS the cause
            # attribution -- so the attribution is withheld rather than guessed.
            print("  CAUSE ATTRIBUTION: WITHHELD -- probe controls failed, so a low cosine")
            print("  cannot be separated from a broken solve. The primary verdict stands.")
            print()
        elif p3 >= THRESH_T2A:
            print("  INFORMATION PROBE: >= 0.92 -> the next latent IS linearly present in this")
            print("  encoding. The rank-bounded failure is therefore CAPACITY or OPTIMIZATION")
            print("  in the world model. The encoder is NOT implicated.")
        elif p3 < KILL_T2A:
            print("  INFORMATION PROBE: < 0.60 -> the next latent is NOT a linear function of")
            print("  the current combined wave at ANY capacity. The limit is the ENCODING (or")
            print("  the linear model class). More rank, a learned basis, or a better optimizer")
            print("  CANNOT fix this. This is the named cause of the KILL.")
        else:
            print(f"  INFORMATION PROBE: {p3:.6f} in band [{KILL_T2A}, {THRESH_T2A}) -> the")
            print("  information is PARTLY present. The residual gap is model class or noise;")
            print("  neither capacity nor the encoder alone explains it. Reported, not promoted.")
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
    if args.arms and arms_records and not all(rec["device_ok"] for rec in arms_records):
        reasons.append("an attribution arm ran OFF the compute device -- its cosines are invalid")
    if args.arms and arms_records and any(rec["rank_capped"] for rec in arms_records):
        reasons.append("an attribution arm was rank-CAPPED at warmup -- the capacity "
                       "comparison is invalid (raise --warmup above the largest rank)")
    if tautology:
        reasons.append("T2-a threshold is TAUTOLOGICAL: the do-nothing static baseline "
                       f"({ss3['mean']:.6f}) already clears {THRESH_T2A}. The gate cannot "
                       "distinguish a world model from doing nothing.")

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
            "dtype_declared_only": "dtype is DECLARED for context, not applied by this "
                                   "harness (no tensor is cast here); the encoder sets its own.",
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
            "arms": arms_records,
            "info_probe": info_rec,
            "predictor_device_ok": bool(res["pred"].V.device == device),
        }
        out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  receipt: {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
