#!/usr/bin/env python3
"""OBSERVED: project_stiefel_manifold -- is the NaN REACHABLE from a live caller?

WHY THIS EXISTS
    The threefold receipt recorded the Stiefel result as:
        raw_randn 64x64 (norm 64.77)  -> residual NaN
        scaled x0.1    (norm  6.48)   -> residual 3.219
        scaled x0.01   (norm  0.648)  -> residual 7.464
    and the commit message claimed "callers feeding an unnormalized matrix get
    NaN". That claim asserted REACHABILITY without tracing the actual caller
    inputs. This script completes the trace, because an unreachable defect and
    a reachable one are different findings.

LIVE CALLER CHAIN (grepped, not assumed)
    henri_api_bridge.py:39   THERMOSTAT = AdaptiveViscoelasticThermostat(d_model=4096, device=DEVICE)
    henri_api_bridge.py:93   if os.environ.get("HENRI_SYNTHETIC_EGRESS", "0") == "1":
    henri_api_bridge.py:94       W = torch.eye(128, device=DEVICE)
    henri_api_bridge.py:96       THERMOSTAT.step_viscoelastic_creep(W, grad, ...)
    adaptive_viscoelastic_thermostat.py:221
                             updated_weight = self.project_stiefel_manifold(updated_weight)
    (also the module's own __main__ demo: W = torch.eye(256) + randn*0.01)

So the ACTUAL live inputs are near-identity / orthogonal, NOT raw randn. This
script measures exactly those cases, plus the iteration-count dependence, so the
verdict separates:
    REACHABLE   -- a live caller's own input produces NaN or a non-orthogonal result
    LATENT      -- only inputs no caller produces trigger it
"""
from __future__ import annotations

import inspect
import json
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
import adaptive_viscoelastic_thermostat as avt  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def resid(P):
    if not torch.isfinite(P).all():
        return float("nan")
    n = min(P.shape[0], P.shape[1])
    return float((P.t() @ P - torch.eye(P.shape[1], device=P.device)).norm().item())


def main():
    out = {"device": DEV, "torch": torch.__version__,
           "stiefel_iters_default": None, "cases": {}, "iteration_sweep": {}}
    th = avt.AdaptiveViscoelasticThermostat(d_model=4096, device=DEV)
    out["stiefel_iters_default"] = int(getattr(th, "stiefel_iters", -1))
    print(f"stiefel_iters default = {out['stiefel_iters_default']}")
    print(f"project_stiefel_manifold{inspect.signature(th.project_stiefel_manifold)}")

    g = torch.Generator(device=DEV).manual_seed(7)

    # --- the REAL caller inputs -------------------------------------------------
    cases = {
        "eye128 (henri_api_bridge.py:94)":
            torch.eye(128, device=DEV),
        "eye256 + randn*0.01 (module __main__ demo)":
            torch.eye(256, device=DEV) + torch.randn(256, 256, device=DEV, generator=g) * 0.01,
        "true orthogonal via QR (a genuine Stiefel point)":
            torch.linalg.qr(torch.randn(128, 128, device=DEV, generator=g))[0],
        "eye*0.5 (under-scaled)":
            torch.eye(128, device=DEV) * 0.5,
        "raw randn (NOT produced by any caller)":
            torch.randn(64, 64, device=DEV, generator=g),
    }
    for nm, W in cases.items():
        P = th.project_stiefel_manifold(W.clone())
        r = resid(P)
        sigma = torch.linalg.svdvals(W.detach().float())
        out["cases"][nm] = {
            "input_shape": list(W.shape),
            "input_norm": float(W.norm().item()),
            "sigma_min": float(sigma.min().item()),
            "sigma_max": float(sigma.max().item()),
            "output_finite": bool(torch.isfinite(P).all().item()),
            "orthogonality_residual": r,
            "converged": bool(isinstance(r, float) and r == r and r < 1e-4),
        }

    # --- iteration sweep on the near-identity caller input ----------------------
    W0 = torch.eye(128, device=DEV) + torch.randn(128, 128, device=DEV, generator=g) * 0.01
    for it in (1, 3, 5, 10, 30, 100):
        th.stiefel_iters = it
        r = resid(th.project_stiefel_manifold(W0.clone()))
        out["iteration_sweep"][f"iters={it}"] = {
            "orthogonality_residual": r,
            "converged": bool(isinstance(r, float) and r == r and r < 1e-4),
        }
    th.stiefel_iters = out["stiefel_iters_default"]

    # --- END-TO-END: the live bridge call, exactly as written ------------------
    e2e = {}
    for nm, W in (("eye128 (bridge)", torch.eye(128, device=DEV)),
                  ("eye128 + randn*0.01", W0[:128, :128])):
        grad = torch.randn(W.shape, device=DEV, generator=torch.Generator(device=DEV).manual_seed(11)) * 0.05
        Wn, info = th.step_viscoelastic_creep(W.clone(), grad,
                                              lambda_active=0.08, sagnac_delta=0.04)
        r = resid(Wn)
        e2e[nm] = {
            "output_finite": bool(torch.isfinite(Wn).all().item()),
            "orthogonality_residual": r,
            "stiefel_contract_met": bool(isinstance(r, float) and r == r and r < 1e-4),
            "weight_norm_telem": info.get("weight_norm"),
        }
    out["end_to_end_live_call"] = e2e

    live_inputs_ok = all(v["converged"] for k, v in out["cases"].items()
                         if "NOT produced" not in k)
    out["verdict"] = {
        "live_caller_inputs_converge": bool(live_inputs_ok),
        "reachable_from_live_caller": bool(not live_inputs_ok),
        "nan_only_on_inputs_no_caller_produces":
            bool(not out["cases"]["raw randn (NOT produced by any caller)"]["output_finite"]),
        "docstring_contract_holds_at_default_iters": bool(
            out["end_to_end_live_call"]["eye128 (bridge)"]["stiefel_contract_met"]),
        "reading": ("Stiefel contract holds for the live caller inputs; the NaN is "
                    "LATENT (unreachable from current callers). Correct the earlier "
                    "reachability claim."
                    if live_inputs_ok else
                    "Stiefel contract FAILS on a live caller input -> REACHABLE defect."),
    }
    print(json.dumps(out, indent=1))
    dst = ("/workspace/phase10/HENRI V2/experiments/verification/"
           "stiefel_reachability_observed.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWROTE {dst} ({os.path.getsize(dst)} bytes)")
    print("### DONE")


if __name__ == "__main__":
    main()
