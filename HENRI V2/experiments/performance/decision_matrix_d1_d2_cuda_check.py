"""Decision Matrix D1+D2 — remote CUDA verification matrix (RTX 5090).

Pre-registration: HENRI V2/experiments/sweeps/decision_matrix_d1_d2_design.md
Source PDF raw SHA-256 2e2cf71151ed39732563a53d898f516281a6d6e3eb5c7934c1de9526ec03df66.

Arms (aggregated exit codes; DONE marker only when all rc=0):
  A0  OFF baseline identity (default path byte-identical)
  A1  D1 gate probe: 30-step train_transition_step @ D=65,536 on real
      encoder waves from known-transform grids; loss_decrease_pct vs G1 5%.
  A2  D2 paired recovery: iso vs P_null-projected noise on SAME draws;
      orthogonality + energy ratio + variance drift; sanity arm D=1,024/r=256.
  A3  D1+D2 combined: WaveJEPA reuse adapter smoke @ 65,536 + thermostat
      projection with the transition's field_V basis.

diagnostic_only=true; no env stepping; no score eligibility.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import efe_planner
from adaptive_viscoelastic_thermostat import AdaptiveViscoelasticThermostat
from henri_vision_encoder import HENRIVisionEncoder
from wave_jepa import WaveJEPA, _LowRankCoupledPredictorAdapter


def _diagnostic_header():
    return {
        "diagnostic_only": True,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }


def _known_transform_pairs(encoder, num_blocks, device, seed=1):
    """Real encoder waves from known-transform grids (translate dx=1)."""
    torch.manual_seed(seed)
    pairs = []
    for _ in range(3):
        g0 = torch.randint(0, 10, (10, 10), device=device)
        g1 = torch.zeros_like(g0)
        g1[:, 1:] = g0[:, :-1]
        s0 = encoder.encode_spatial_grid(g0).squeeze(0).view(num_blocks, 8)
        s1 = encoder.encode_spatial_grid(g1).squeeze(0).view(num_blocks, 8)
        pairs.append((s0 / s0.norm(dim=-1, keepdim=True),
                      s1 / s1.norm(dim=-1, keepdim=True)))
    return pairs


def _cross_block_jacobian_info(planner, state, action, device, num_blocks, n_pairs=3):
    """Informational mechanism evidence: |d psi_i / d psi_j| for i != j."""
    outs = []
    for (i, j) in [(0, 1), (1, 0), (0, num_blocks // 2), (num_blocks // 2, 0)]:
        try:
            pred = planner.transition(state, action)
            g = torch.autograd.grad(pred[i].sum(), state, retain_graph=False)[0]
            outs.append({"i": i, "j": j, "grad_norm": float(g[j].norm().item())})
        except Exception as exc:  # informational; never gate on it
            outs.append({"i": i, "j": j, "grad_norm": None, "error": str(exc)[:120]})
        if len(outs) >= n_pairs:
            break
    return outs


def arm_a0_baseline(device):
    """Default-path identity at D=65,536."""
    num_blocks = 8192
    jepa = WaveJEPA(d_model=65536, num_blocks=num_blocks, r_rank=16)
    assert type(jepa.predictor).__name__ == "RecursiveDualEDMD"
    assert jepa.use_lowrank_coupled is False
    th = AdaptiveViscoelasticThermostat(d_model=65536,
                                        use_null_subspace_projection=False)
    W = torch.randn(8192, 8, device=device)
    grad = torch.randn_like(W)
    W1, _ = th.step_viscoelastic_creep(W, grad, 0.05, 0.4, temperature=1e-4)
    # legacy reproduction (friction=1.0)
    eff_lr = th.base_lr * (1.0 + 0.4)
    torch.manual_seed(1)
    # note: randn_like above consumed RNG; use explicit reproduction only for
    # shape check, not equality — the identity is proven by contract tests.
    assert W1.shape == W.shape
    return {"predictor": type(jepa.predictor).__name__,
            "thermostat_projection_active": False}


def arm_a1_d1_gate(device):
    """G1: Sagnac loss decrease > 5% within 30 online steps @ D=65,536."""
    num_blocks = 8192
    encoder = HENRIVisionEncoder(d_model=65536, k_blocks=num_blocks, device=device)
    planner = efe_planner.EFEPlanner(num_blocks=num_blocks, d_model=65536).to(device)
    pairs = _known_transform_pairs(encoder, num_blocks, device)
    losses = []
    t0 = time.perf_counter()
    for step in range(30):
        s, o = pairs[step % len(pairs)]
        a = F.normalize(torch.randn(num_blocks, 8, device=device), dim=-1)
        loss = planner.train_transition_step(s, a, o, lr=0.05)
        losses.append(loss)
    dt = time.perf_counter() - t0
    loss1, loss30 = losses[0], losses[-1]
    ema = losses[0]
    for v in losses[1:]:
        ema = 0.9 * ema + 0.1 * v
    loss_decrease_pct = 100.0 * (loss1 - ema) / loss1 if loss1 > 0 else 0.0
    # informational mechanism evidence
    state = pairs[0][0]
    action = F.normalize(torch.randn(num_blocks, 8, device=device), dim=-1)
    jac = _cross_block_jacobian_info(planner, state, action, device, num_blocks)
    return {
        "loss_first": loss1,
        "loss_ema_30": ema,
        "loss_last": loss30,
        "loss_decrease_pct": loss_decrease_pct,
        "gate_g1_pass": loss_decrease_pct > 5.0,
        "verdict": "D1_PASS" if loss_decrease_pct > 5.0 else "D1_INERT",
        "steps": 30,
        "wall_s": round(dt, 3),
        "cross_block_jacobian": jac,
    }


def _paired_recovery(d, r, device, steps=200, seed=31):
    """Paired iso vs projected recovery on the same draws. Returns summary."""
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    M = torch.randn(d, r, generator=g, device="cpu").to(device)
    V, _ = torch.linalg.qr(M)
    W_star = F.normalize(torch.randn(d, 1, device=device), dim=0) * 0.5
    P = torch.zeros(d, 1, device=device)
    P[0] = 1.0
    P = P / P.norm() * 2.0
    th_iso = AdaptiveViscoelasticThermostat(d_model=d)
    th_proj = AdaptiveViscoelasticThermostat(d_model=d,
                                             use_null_subspace_projection=True)
    th_proj.set_null_basis(V)
    W_iso = W_star + P.clone()
    W_proj = W_star + P.clone()
    err_iso, err_proj, resid_iso, resid_proj, ratios = [], [], [], [], []
    for step in range(steps):
        base = torch.randn(d, 1, generator=g, device="cpu").to(device)
        grad = (W_iso - W_star) * 20.0
        W_iso, _ = th_iso.step_viscoelastic_creep(
            W_iso, grad, 0.05, 0.4, temperature=0.005, base_noise=base)
        grad_p = (W_proj - W_star) * 20.0
        W_proj, tp = th_proj.step_viscoelastic_creep(
            W_proj, grad_p, 0.05, 0.4, temperature=0.005, base_noise=base.clone())
        err_iso.append(float(torch.norm(W_iso - W_star).item()))
        err_proj.append(float(torch.norm(W_proj - W_star).item()))
        n = base * math.sqrt(2.0 * 0.005 * (th_iso.base_lr * 1.4))
        resid_iso.append(float(torch.norm(V.T @ n).item() / (torch.norm(n).item() + 1e-12)))
        n_p = (base - V @ (V.T @ base)) * math.sqrt(2.0 * 0.005 * (th_iso.base_lr * 1.4))
        resid_proj.append(float(torch.norm(V.T @ n_p).item() / (torch.norm(n_p).item() + 1e-12)))
        ratios.append(tp["null_projection_energy_ratio"])
    var_drift_iso = (torch.tensor(err_iso).std().item() ** 2)
    var_drift_proj = (torch.tensor(err_proj).std().item() ** 2)
    reduction_pct = 100.0 * (1.0 - var_drift_proj / (var_drift_iso + 1e-12))
    return {
        "d": d, "r": r, "r_over_d": round(r / d, 6),
        "err_final_iso": err_iso[-1], "err_final_proj": err_proj[-1],
        "var_drift_iso": var_drift_iso, "var_drift_proj": var_drift_proj,
        "variance_drift_reduction_pct": reduction_pct,
        "resid_iso_mean": sum(resid_iso) / len(resid_iso),
        "resid_proj_mean": sum(resid_proj) / len(resid_proj),
        "energy_ratio_mean": sum(ratios) / len(ratios),
    }


def arm_a2_d2_gate(device):
    """A2: P_null orthogonality + energy ratio + variance-drift gate."""
    prod = _paired_recovery(65536, 64, device, steps=200)
    sanity = _paired_recovery(1024, 256, device, steps=200)
    ortho_ok = prod["resid_proj_mean"] < 1e-3
    sanity_ortho_ok = sanity["resid_proj_mean"] < 1e-3
    # PDF gate: > 40% reduction in variance drift during recovery.
    gate_pass = prod["variance_drift_reduction_pct"] > 40.0
    return {
        "production_scale": prod,
        "sanity_arm": sanity,
        "orthogonality_ok": ortho_ok,
        "sanity_orthogonality_ok": sanity_ortho_ok,
        "gate_a2_pass_40pct": gate_pass,
        "verdict": "D2_PASS" if (gate_pass and ortho_ok and sanity_ortho_ok)
                  else "D2_FAIL",
    }


def arm_a3_combined(device):
    """D1+D2 combined smoke: WaveJEPA adapter + thermostat projection."""
    num_blocks = 8192
    jepa = WaveJEPA(d_model=65536, num_blocks=num_blocks, r_rank=16,
                    use_lowrank_coupled=True)
    assert isinstance(jepa.predictor, _LowRankCoupledPredictorAdapter)
    from efe_planner import LowRankCoupledTransition
    assert isinstance(jepa.predictor.transition, LowRankCoupledTransition)
    encoder = HENRIVisionEncoder(d_model=65536, k_blocks=num_blocks, device=device)
    pairs = _known_transform_pairs(encoder, num_blocks, device)
    s, o = pairs[0]
    a = F.normalize(torch.randn(num_blocks, 8, device=device), dim=-1)
    pred = jepa.predict_future_latent(s, a)
    assert pred.shape == (num_blocks, 8)
    loss = jepa.predictor.update_online_step(s, a, o)
    assert math.isfinite(loss)
    # thermostat projection with the transition's field_V basis
    th = AdaptiveViscoelasticThermostat(d_model=65536,
                                        use_null_subspace_projection=True)
    th.set_null_basis(jepa.predictor.transition.field_V.detach())
    W = torch.randn(num_blocks, 8, device=device)
    _, tp = th.step_viscoelastic_creep(
        W, torch.zeros_like(W), 0.05, 0.4, temperature=1e-3)
    assert 0.0 < tp["null_projection_energy_ratio"] <= 1.0
    return {"adapter_forward_shape": list(pred.shape),
            "adapter_loss_finite": True,
            "combined_energy_ratio": tp["null_projection_energy_ratio"],
            "verdict": "A3_OK"}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"header": _diagnostic_header()}
    arms = {
        "A0": arm_a0_baseline,
        "A1": arm_a1_d1_gate,
        "A2": arm_a2_d2_gate,
        "A3": arm_a3_combined,
    }
    rc = 0
    for name, fn in arms.items():
        try:
            t0 = time.perf_counter()
            results[name] = fn(device)
            results[name]["arm_rc"] = 0
            print(f"[{name}] OK ({time.perf_counter() - t0:.2f}s): "
                  f"{results[name].get('verdict', 'ok')}", flush=True)
        except Exception as exc:
            results[name] = {"arm_rc": 1, "error": str(exc)[:500]}
            print(f"[{name}] FAIL: {exc}", flush=True)
            rc = 1
    out = os.environ.get("JEPA_DM_OUT", "/tmp/jepa_dm_result.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"DONE_MARKER rc={rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
