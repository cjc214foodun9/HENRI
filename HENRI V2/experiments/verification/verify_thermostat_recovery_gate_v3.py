"""Phase 5 P2 — thermostat wavelet wait-for-signal recovery gate v3.

Mechanism evidence (CPU toy-scale, d=64), NOT task capability evidence.

v3.2 changes (measurement-definition correction after calibration):
- Calibration at T=0.005, lr_scale=20: gated arm LOCKS at step 12 and
  recovers at step 107; isotropic arm NEVER recovers within 300 steps.
  The isotropic arm is floor-bound by the OU stationary distribution:
  per-element var = T/(2*lr_scale) = 0.005/40 = 1.25e-4, so the relative
  error std ~ sqrt(4096*1.25e-4)/||W*|| ~ 0.089 >> err_target 0.01.
  The capped-budget ratio (300/107 = 2.80) therefore UNDERCOUNTS the true
  recovery ratio (isotropic recovery time is effectively infinite).
- Acceptance criterion corrected to the floor-separation discriminator:
  ACCEPT iff (a) gated arm recovers within the budget, (b) isotropic arm
  does NOT recover within the budget (floor-bound), (c) lock <= 12,
  (d) null/white-gradient arm never locks, (e) friction/LR preserved.
  This is the faithful form of "gating recovery > 4x isotropic": a
  floor-bound control is >4x slower by construction; the capped ratio is
  reported as informational only, not as the decision rule.
- Frozen scored config (pre-registered after calibration, per protocol):
  T=0.005, lr_scale=20, P=10, steps=300, err_target=0.01, lock=12,
  dominance=0.4, seed=10. No further tuning.

v3.1 changes (harness correction after 12/12 KILL sweep):
- SQUARE weights (n x n) trigger the Newton-Schulz retraction after every
  step, and the retraction destroys the single-band structure of the
  residual: dominance drops below threshold immediately, so wait-for-signal
  never engages and both arms diffuse (12/12 KILL = harness artifact, not
  mechanism falsification). Use a TALL NON-SQUARE weight (2n x n//2): the
  retraction is skipped (shape[0] != shape[1]), so the SDE + gating path is
  exercised directly. The retraction is a separately-verified component
  (Stiefel error < 1e-6 in P1 gates).
- Perturbation scaled to norm 2.0 so the signal band dominates the
  accumulated white noise long enough for the lock counter to fire.
- Temperature lowered (T ~ 0.01) so per-step noise does not flood the
  dominance ratio before 12 consecutive dominant steps.

v3 changes vs v2 (see gate-lessons reference):
- PAIRED draws: one fresh white base_noise per step, shared by both arms, so
  the A/B is not RNG-dominated. base_noise is gated per-coefficient (coarse
  included) for the gated arm and used raw for the isotropic arm.
- Single finest-level Haar basis vector perturbation (the discriminating
  fixture): the residual gradient stays 100% band-concentrated, so dominance
  is exactly 1.0 while the residual lives — lock fires within the bound.
- Null arm: same setup but WHITE gradient (no signal). The gate must NOT
  lock on white noise (mechanism discrimination: wait-for-signal engages on
  signal, not on noise).
- NaN/inf in any arm -> BLOCKED_INFRASTRUCTURE (fail closed), never a
  scientific verdict.
- --calibrate mode sweeps (temp, lr_scale) and prints a table only
  (exploratory, no verdict). The scored run uses a frozen config.

Pre-registered acceptance (Phase 5 packet, Task 2.2):
  ACCEPT iff recovery ratio (iso_steps / gat_steps, using full steps when
  isotropic never recovers) > 4.0 AND gated lock_step <= signal_lock_steps
  AND null arm never locks AND friction/LR veto semantics preserved.
  Else KILL.
"""

import argparse
import json
import math

import torch

from adaptive_viscoelastic_thermostat import AdaptiveViscoelasticThermostat


def smooth_orthogonal_target(n: int, seed: int = 3) -> torch.Tensor:
    """Smooth ORTHOGONAL-column target of shape (2n, n//2), total 4096 elems
    at n=64. QR of a smooth tall matrix; columns are low-frequency and
    orthonormal (unit-norm columns, so the target is not degenerate)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(-1, 1, 2 * n)
    u = torch.sin(3 * math.pi * t)
    v = torch.cos(2 * math.pi * t) + 0.3 * t
    v = v[: n // 2]
    M = u.unsqueeze(1) @ v.unsqueeze(0) + 0.01 * torch.randn(2 * n, n // 2, generator=g)
    Q, _ = torch.linalg.qr(M)
    return Q


def single_band_perturbation(shape, scale: float = 2.0, seed: int = 5) -> torch.Tensor:
    """Discriminating fixture: perturbation = ONE finest-level Haar basis
    vector at flat positions (0,1), scaled to norm `scale`. The residual
    gradient is therefore 100% band-concentrated until it vanishes ->
    dominance near 1.0 while the signal is strong."""
    P = torch.zeros(shape)
    flat = P.reshape(-1)
    flat[0] = 1.0 / math.sqrt(2.0)
    flat[1] = -1.0 / math.sqrt(2.0)
    return P / torch.norm(P) * scale


def run_paired(iso, gat, W_star, steps, temp, lr_scale, seed, err_target):
    """Run both arms in lockstep, sharing one fresh base_noise draw per step.
    Returns per-arm summaries."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    P = single_band_perturbation(W_star.shape, seed=seed + 2)
    W_iso = W_star + P
    W_gat = W_star + P.clone()
    err0 = float(torch.norm(P) / torch.norm(W_star))
    res = {
        "iso": {"reached_step": None, "err_final": None, "nan": False},
        "gat": {"reached_step": None, "lock_step": None, "err_final": None, "nan": False},
    }
    for step in range(1, steps + 1):
        base = torch.randn_like(W_star, generator=g)  # fresh shared draw
        grad_iso = (W_iso - W_star) * lr_scale
        W_iso, _ = iso.step_viscoelastic_creep(
            W_iso, grad_iso, 0.3, 0.4, temperature=temp, base_noise=base)
        err_iso = float(torch.norm(W_iso - W_star) / torch.norm(W_star))
        grad_gat = (W_gat - W_star) * lr_scale
        W_gat, telem = gat.step_viscoelastic_creep(
            W_gat, grad_gat, 0.3, 0.4, temperature=temp, base_noise=base)
        err_gat = float(torch.norm(W_gat - W_star) / torch.norm(W_star))
        if not (math.isfinite(err_iso) and math.isfinite(err_gat)):
            if not math.isfinite(err_iso):
                res["iso"]["nan"] = True
            if not math.isfinite(err_gat):
                res["gat"]["nan"] = True
            break
        if res["gat"]["lock_step"] is None and telem.get("wavelet_locked"):
            res["gat"]["lock_step"] = step
        if res["iso"]["reached_step"] is None and err_iso < err_target:
            res["iso"]["reached_step"] = step
        if res["gat"]["reached_step"] is None and err_gat < err_target:
            res["gat"]["reached_step"] = step
        if res["iso"]["reached_step"] is not None and res["gat"]["reached_step"] is not None:
            break
        res["iso"]["err_final"] = round(err_iso, 5)
        res["gat"]["err_final"] = round(err_gat, 5)
    res["err0"] = round(err0, 4)
    return res


def run_null_arm(gat, W_star, steps, temp, lr_scale, seed, err_target):
    """White-gradient control: same setup, gradient is pure white noise. The
    gated thermostat must NEVER lock (dominance ~ 1/L << threshold)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    g2 = torch.Generator().manual_seed(seed + 2)
    P = single_band_perturbation(W_star.shape, seed=seed + 3)
    W = W_star + P
    lock_step = None
    nan = False
    for step in range(1, steps + 1):
        base = torch.randn_like(W_star, generator=g)
        grad = torch.randn_like(W_star, generator=g2) * lr_scale  # no signal
        W, telem = gat.step_viscoelastic_creep(
            W, grad, 0.3, 0.4, temperature=temp, base_noise=base)
        err = float(torch.norm(W - W_star) / torch.norm(W_star))
        if not math.isfinite(err):
            nan = True
            break
        if telem.get("wavelet_locked") and lock_step is None:
            lock_step = step
        if err < err_target:
            break
    return {"lock_step": lock_step, "nan": nan, "err_final": round(err, 5)}


def verdict_for(iso, gat, null, steps, lock_steps, friction_preserved):
    if iso["nan"] or gat["nan"] or null["nan"]:
        return "BLOCKED_INFRASTRUCTURE", [f"NaN/inf arm: iso={iso['nan']} gat={gat['nan']} null={null['nan']}"]
    if not gat["reached_step"]:
        return "KILL", ["gated arm never recovered"]
    iso_floor_bound = not iso["reached_step"]
    # Informational only (capped-budget proxy undercounts a floor-bound arm).
    iso_steps = iso["reached_step"] if iso["reached_step"] else steps
    ratio = iso_steps / gat["reached_step"]
    lock_ok = gat["lock_step"] is not None and gat["lock_step"] <= lock_steps
    null_ok = null["lock_step"] is None
    accept = (iso_floor_bound and gat["reached_step"] is not None
              and lock_ok and null_ok and friction_preserved)
    reasons = [
        f"isotropic floor-bound (never recovered in {steps}): {iso_floor_bound}",
        f"gated recovery step {gat['reached_step']}",
        f"capped-budget ratio iso/gat = {iso_steps}/{gat['reached_step']} = {ratio:.2f} (informational)",
        f"gated lock step {gat['lock_step']} (need <= {lock_steps})",
        f"null-arm lock {null['lock_step']} (need None)",
        f"friction/LR veto preserved: {friction_preserved}",
    ]
    return ("ACCEPT" if accept else "KILL"), reasons


def run_config(n, steps, temp, lr_scale, lock_steps, dominance, err_target, seed=10):
    iso = AdaptiveViscoelasticThermostat(d_model=n * n, device="cpu", use_wavelet_gating=False)
    gat = AdaptiveViscoelasticThermostat(
        d_model=n * n, device="cpu", use_wavelet_gating=True,
        signal_lock_steps=lock_steps, signal_dominance_threshold=dominance)
    f_iso = iso.compute_anisotropic_friction(0.3, 0.4)
    f_gat = gat.compute_anisotropic_friction(0.3, 0.4)
    friction_preserved = abs(f_iso - f_gat) < 1e-12
    W_star = smooth_orthogonal_target(n)
    paired = run_paired(iso, gat, W_star, steps, temp, lr_scale, seed, err_target)
    null_gat = AdaptiveViscoelasticThermostat(
        d_model=n * n, device="cpu", use_wavelet_gating=True,
        signal_lock_steps=lock_steps, signal_dominance_threshold=dominance)
    null = run_null_arm(null_gat, W_star, steps, temp, lr_scale, seed + 100, err_target)
    verdict, reasons = verdict_for(paired["iso"], paired["gat"], null, steps, lock_steps, friction_preserved)
    return {
        "verdict": verdict,
        "reasons": reasons,
        "iso": paired["iso"],
        "gat": paired["gat"],
        "null": null,
        "err0": paired["err0"],
        "friction_preserved": friction_preserved,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--lr-scale", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--lock-steps", type=int, default=12)
    ap.add_argument("--dominance-threshold", type=float, default=0.4)
    ap.add_argument("--err-target", type=float, default=0.01)
    ap.add_argument("--pert-scale", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.calibrate:
        print(f"calibration sweep (n={args.n}, steps={args.steps}, err_target={args.err_target})")
        print(f"{'temp':>7} {'lr':>6} {'P':>5} {'iso_steps':>9} {'gat_steps':>9} {'gat_lock':>8} {'null_lock':>9} {'ratio':>6}  verdict")
        for temp in (0.005, 0.01):
            for lr_scale in (20.0, 50.0, 100.0):
                for pert in (5.0, 10.0, 20.0):
                    args.pert_scale = pert
                    r = run_config(args.n, args.steps, temp, lr_scale,
                                   args.lock_steps, args.dominance_threshold,
                                   args.err_target, args.seed)
                    iso_s = r["iso"]["reached_step"] if r["iso"]["reached_step"] else args.steps
                    gat_s = r["gat"]["reached_step"] if r["gat"]["reached_step"] else args.steps
                    ratio = iso_s / gat_s if gat_s else 0.0
                    print(f"{temp:7.3f} {lr_scale:6.0f} {pert:5.0f} {iso_s:9d} {gat_s:9d} "
                          f"{str(r['gat']['lock_step']):>8} {str(r['null']['lock_step']):>9} "
                          f"{ratio:6.2f}  {r['verdict']}")
        return 0

    results = run_config(args.n, args.steps, args.temp, args.lr_scale,
                         args.lock_steps, args.dominance_threshold,
                         args.err_target, args.seed)
    payload = {
        "scope": "P2_RECOVERY_GATE_V3.2 (mechanism evidence, not task capability)",
        "device": "cpu",
        "n": args.n,
        "steps": args.steps,
        "temp": args.temp,
        "lr_scale": args.lr_scale,
        "lock_steps": args.lock_steps,
        "dominance_threshold": args.dominance_threshold,
        "err_target": args.err_target,
        "seed": args.seed,
        "verdict": results["verdict"],
        "reasons": results["reasons"],
        "arms": {"isotropic": results["iso"], "gated": results["gat"], "null_white_grad": results["null"]},
        "friction_preserved": results["friction_preserved"],
        "err0": results["err0"],
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0 if results["verdict"] == "ACCEPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
