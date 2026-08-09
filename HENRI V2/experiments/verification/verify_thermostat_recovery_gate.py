"""Phase 5 P2 — thermostat wavelet wait-for-signal recovery gate (v2).

Mechanism evidence (CPU/CUDA toy-scale), NOT task capability evidence.
Pre-registered acceptance (Phase 5 packet, Task 2.2):
  - ACCEPT iff gated recovery > 4x isotropic control AND lock <= 12 steps
    AND hard Sagnac veto semantics preserved (friction/LR identical for
    identical inputs).
  - KILL if gating does not beat isotropic in the saturation regime.
  - Any NaN/inf arm -> BLOCKED_INFRASTRUCTURE (fail closed), never a
    scientific verdict.

v2 harness fixes (v1 artifacts):
  - Target is ORTHOGONAL (QR of a smooth matrix): the thermostat's
    Newton-Schulz retraction stays in its basin (v1: err0=32 pushed the
    weight outside the basin -> NaN on both arms).
  - Start near-manifold: noise norm 1.0 on ||W*||_F = 8.
  - Gradient is full-band (W - W*)*lr_scale; because W* is smooth
    (coarse-only), the gradient's coarse component is the signal and the
    injected white Langevin noise is the fine component -> wait-for-signal
    dominance is measurable near convergence (v1: gradient became pure
    white noise, lock never fired).
"""

import argparse
import json
import math

import torch

from adaptive_viscoelastic_thermostat import AdaptiveViscoelasticThermostat


def smooth_orthogonal_target(n: int, seed: int = 3) -> torch.Tensor:
    """Smooth (coarse-band-only) ORTHOGONAL target. QR of a smooth matrix
    keeps the operator in the Newton-Schulz basin while its columns are
    low-frequency -> Haar energy concentrated in coarse bands."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(-1, 1, n)
    u = torch.sin(3 * math.pi * t)
    v = torch.cos(2 * math.pi * t) + 0.3 * t
    M = u.unsqueeze(1) @ v.unsqueeze(0) + 0.01 * torch.randn(n, n, generator=g)
    Q, _ = torch.linalg.qr(M)
    return Q


def run_recovery_arm(thermostat, W_star, steps, temp, lr_scale, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    noise = torch.randn_like(W_star, generator=g)
    noise = noise / torch.norm(noise) * 1.0  # near-manifold perturbation
    W = W_star + noise
    err0 = float(torch.norm(W - W_star) / torch.norm(W_star))
    reached = None
    lock_step = None
    last_err = err0
    nan_seen = False
    for step in range(1, steps + 1):
        grad = (W - W_star) * lr_scale
        W, telem = thermostat.step_viscoelastic_creep(
            W, grad, lambda_active=0.3, sagnac_delta=0.4, temperature=temp)
        last_err = float(torch.norm(W - W_star) / torch.norm(W_star))
        if not math.isfinite(last_err):
            nan_seen = True
            break
        if thermostat.use_wavelet_gating and lock_step is None \
                and telem.get("wavelet_locked"):
            lock_step = step
        if last_err < 0.05:
            reached = step
            break
    return {
        "reached_step": reached,
        "recovered": reached is not None,
        "lock_step": lock_step,
        "err0": round(err0, 4),
        "err_final": round(last_err, 4),
        "nan": nan_seen,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--temp", type=float, default=5e-2)
    ap.add_argument("--lr-scale", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--lock-steps", type=int, default=12)
    ap.add_argument("--dominance-threshold", type=float, default=0.4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    th_iso = AdaptiveViscoelasticThermostat(
        d_model=args.n * args.n, device=args.device, use_wavelet_gating=False)
    th_gat = AdaptiveViscoelasticThermostat(
        d_model=args.n * args.n, device=args.device, use_wavelet_gating=True,
        signal_lock_steps=args.lock_steps,
        signal_dominance_threshold=args.dominance_threshold)

    # Veto-semantics preservation: friction and effective LR are pure
    # functions of (lambda_active, sagnac_delta) — must be identical across
    # arms for identical inputs.
    f_iso = th_iso.compute_anisotropic_friction(0.3, 0.4)
    f_gat = th_gat.compute_anisotropic_friction(0.3, 0.4)
    friction_preserved = abs(f_iso - f_gat) < 1e-12

    W_star = smooth_orthogonal_target(args.n)
    iso = run_recovery_arm(th_iso, W_star, args.steps, args.temp, args.lr_scale, seed=10)
    gat = run_recovery_arm(th_gat, W_star, args.steps, args.temp, args.lr_scale, seed=10)

    if iso["nan"] or gat["nan"]:
        verdict = "BLOCKED_INFRASTRUCTURE"
        reasons = [f"NaN/inf arm: iso={iso['nan']} gat={gat['nan']}"]
    else:
        iso_steps = iso["reached_step"] if iso["recovered"] else args.steps
        gat_steps = gat["reached_step"] if gat["recovered"] else args.steps
        ratio = (iso_steps / gat_steps) if gat["recovered"] and gat_steps > 0 else 0.0
        lock_ok = gat["lock_step"] is not None and gat["lock_step"] <= args.lock_steps
        accept = ratio > 4.0 and lock_ok and friction_preserved
        verdict = "ACCEPT" if accept else "KILL"
        reasons = [
            f"recovery ratio iso/gat = {iso_steps}/{gat_steps} = {ratio:.2f} "
            f"(need > 4.0)",
            f"gated lock step {gat['lock_step']} (need <= {args.lock_steps})",
            f"friction preserved: {friction_preserved}",
        ]

    results = {
        "scope": "P2_RECOVERY_GATE_V2 (mechanism evidence, not task capability)",
        "device": args.device,
        "temp": args.temp,
        "steps": args.steps,
        "lock_steps": args.lock_steps,
        "dominance_threshold": args.dominance_threshold,
        "isotropic": iso,
        "gated": gat,
        "verdict": verdict,
        "reasons": reasons,
    }
    text = json.dumps(results, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0 if verdict == "ACCEPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
