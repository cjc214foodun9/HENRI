"""Phase 8.11 remote CUDA verification matrix (RTX 5090, D=65,536).

Pre-registration: HENRI V2/experiments/sweeps/phase811_native_complex_transition_design.md
Protocol: docs-HENRI_V2_PHASE_8_11_VERIFICATION_AND_SEAL_PRO....pdf
(SHA 7911b094fbd166b283dc13a6afa8138f06aeb69375433bb778c296873c52ec0a).
NOTE: the protocol's generic `gpu_verification_suite.py --mode phase811`
does NOT exist (phantom-CLI family, 4th confirmation); this dedicated
runner is the real artifact, mirroring the 8.10 precedent.

Gates (pre-registered, D=65,536 on RTX 5090):
- G1a identity at scale: forward(s, zero-phase) cos-sim 1.0, per-block unit.
- G1b NATIVE-COMPLEX EXACTNESS (ACCEPT gate): 32 synthetic diagonal-phase
      pairs; <= 3 closed-form fit calls; held-out REAL-egress Sagnac <= 0.05.
- G2 PRODUCTION REAL-WAVE TRANSFER (improvement gate, EXPECTED FAIL):
      live HENRIVisionEncoder real waves (synthetic grids, production
      defaults) lift -> rotate -> egress; held-out pre/post real-metric
      Sagnac; kill = NO improvement (post >= pre - 0.02). Expected FAIL
      records the honest boundary (lossy acos lift on superposition waves);
      NOT a mechanism kill.
- G3 egress contract at scale: real [8192, 8] per-block unit, float32,
      finite, bounded [-1, 1].
- G4 default-OFF at scale: flag OFF -> LowRankCoupledTransition (legacy
      forward finite + per-block unit); flag ON -> NativeComplexWaveTransition.
- G5 latency: forward_complex + egress <= 1.0 ms at D=65,536 (50 iters).
- WIRE: EFEPlanner(use_complex_transition=True) select_action and
      train_transition_step run at scale without device errors.

Multi-arm DONE marker: rc=1 if ANY arm gate fails (aggregated, learned lesson).
Env: JEPA_DM_OUT=/tmp/p811_matrix_d65536.json; HENRI_SMOKE=1 -> bounded seeds.
"""

import json
import os
import time

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("HENRI_SMOKE", "0") == "1"
OUT = os.environ.get("JEPA_DM_OUT", "/tmp/p811_matrix_d65536.json")

NB = 8192
BD = 8
D = NB * BD
NA = 8

results = {}
failures = []


def record(name, ok, **kv):
    results[name] = {"ok": bool(ok), **kv}
    print(f"[{name}] ok={ok} " + " ".join(
        f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in kv.items()), flush=True)
    if not ok:
        failures.append(name)


def unit_wave(seed: int, n: int = NB, d: int = BD) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, d, generator=g, device="cpu").to(DEVICE)
    return w / (torch.norm(w, p=2, dim=-1, keepdim=True) + 1e-9)


def real_sagnac(pred: torch.Tensor, actual: torch.Tensor) -> float:
    p = pred.reshape(-1)
    o = actual.reshape(-1)
    return float(1.0 - torch.dot(p, o) / (torch.norm(p) * torch.norm(o)).clamp(min=1e-12))


def phasor_pairs(n: int, delta_seed: int, alpha_seed: int, span: float = 0.6):
    """NATIVE-DOMAIN trajectory pairs (per pre-registration G1): per-element
    unit-modulus phasors z_{t+1} = z_t * exp(j*delta) with ONE SHARED delta
    (the single action rotation). delta is fixed by delta_seed; the states
    (alpha) vary by alpha_seed. Returns list of (z_t [D] complex,
    z_n [D] complex) on DEVICE."""
    g = torch.Generator(device="cpu").manual_seed(delta_seed)
    delta = ((torch.rand(D, generator=g, device="cpu") - 0.5) * 2.0 * span).to(DEVICE)
    g2 = torch.Generator(device="cpu").manual_seed(alpha_seed)
    pairs = []
    for _ in range(n):
        alpha = ((torch.rand(D, generator=g2, device="cpu") - 0.5) * 2.0 * math.pi).to(DEVICE)
        z_t = torch.polar(torch.ones(D, device=DEVICE), alpha)
        z_n = z_t * torch.polar(torch.ones(D, device=DEVICE), delta)
        pairs.append((z_t, z_n))
    return pairs


def encoder_real_waves(n: int, seed: int = 5):
    """Live production encoder (HENRIVisionEncoder, default basis, no bg_mask)
    over synthetic ARC-like grids; returns [n, 8192, 8] real per-block unit."""
    from henri_vision_encoder import HENRIVisionEncoder
    import numpy as np
    enc = HENRIVisionEncoder(d_model=D, k_blocks=NB, block_dim=BD,
                             device=DEVICE, spatial_basis_kind="default",
                             bg_mask=False)
    rng = np.random.default_rng(seed)
    states, nexts = [], []
    for i in range(n):
        h, w = 10, 10
        grid = np.zeros((h, w), dtype=np.int64)
        r0, c0 = int(rng.integers(0, 5)), int(rng.integers(0, 5))
        color = int(1 + rng.integers(0, 3))
        rh, rw = 3, 3
        grid[r0:r0 + rh, c0:c0 + rw] = color
        s = enc.encode_spatial_grid(grid).squeeze(0).to(DEVICE)
        s = s / (torch.norm(s, p=2, dim=-1, keepdim=True) + 1e-9)
        g2 = np.zeros((h, w), dtype=np.int64)
        dr, dc = 1, -1
        g2[r0 + dr:r0 + dr + rh, c0 + dc:c0 + dc + rw] = color
        n2 = enc.encode_spatial_grid(g2).squeeze(0).to(DEVICE)
        n2 = n2 / (torch.norm(n2, p=2, dim=-1, keepdim=True) + 1e-9)
        states.append(s)
        nexts.append(n2)
    return torch.stack(states), torch.stack(nexts)


def main():
    from complex_phase_transition import NativeComplexWaveTransition

    print(f"[init] device={DEVICE} D={D} smoke={SMOKE}", flush=True)
    a_zero = torch.zeros(NB, BD, device=DEVICE)

    # ---------------- G1a identity at scale ----------------
    ad = NativeComplexWaveTransition(
        dimension=D, num_actions=NA, device=DEVICE, num_blocks=NB, block_dim=BD)
    s = unit_wave(1)
    t0 = time.perf_counter()
    out = ad.forward(s, a_zero)
    lat = (time.perf_counter() - t0) * 1000.0
    cos = float(torch.nn.functional.cosine_similarity(
        out.reshape(1, -1), s.reshape(1, -1), dim=-1))
    norms = torch.norm(out, p=2, dim=-1)
    unit = bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))
    record("G1a_identity", cos > 0.9999 and unit,
           cos=cos, unit=unit, latency_ms=lat)

    # ---------------- G5 latency ----------------
    t0 = time.perf_counter()
    for _ in range(50):
        _ = ad.forward(s, a_zero)
    lat5 = (time.perf_counter() - t0) * 1000.0 / 50.0
    record("G5_latency", lat5 <= 1.0, latency_ms=lat5)

    # ---------------- G1b accept gate (mechanism, D=65,536) ----------------
    n_fit = 16 if SMOKE else 32
    n_eval = 16 if SMOKE else 32
    fit_pairs = phasor_pairs(n_fit, delta_seed=100, alpha_seed=200)
    eval_pairs = phasor_pairs(n_eval, delta_seed=100, alpha_seed=900)
    pre1 = real_sagnac(
        ad.project_to_real_egress(ad.forward_complex(eval_pairs[0][0], 0)),
        ad.project_to_real_egress(eval_pairs[0][1]))
    for _ in range(3):
        for zt, zn in fit_pairs:
            ad.update_phase_complex(zt, zn, 0, lr=1.0)
    sags = [real_sagnac(
        ad.project_to_real_egress(ad.forward_complex(zt, 0)),
        ad.project_to_real_egress(zn)) for zt, zn in eval_pairs]
    post1 = sum(sags) / len(sags)
    # ACCEPT gate: native-complex exactness within <= 3 steps.
    record("G1b_accept_native_complex", post1 <= 0.05,
           pre=pre1, post=post1, n_eval=n_eval)

    # ---------------- G2 production real-wave transfer (EXPECTED FAIL) ------
    ad2 = NativeComplexWaveTransition(
        dimension=D, num_actions=NA, device=DEVICE, num_blocks=NB, block_dim=BD)
    n_tr = 4 if SMOKE else 6
    n_ev = 3 if SMOKE else 4
    st_tr, nx_tr = encoder_real_waves(n_tr, seed=5)
    st_ev, nx_ev = encoder_real_waves(n_ev, seed=55)
    a2 = unit_wave(7)  # stable nonzero action (fingerprint -> index 0)
    pre2 = real_sagnac(ad2.forward(st_ev[0], a2), nx_ev[0])
    for _ in range(3):
        ad2.fit_batch(st_tr, a2.repeat(n_tr, 1, 1), nx_tr,
                      iters=1, lr=1.0)
    sags2 = [real_sagnac(ad2.forward(st_ev[i], a2), nx_ev[i])
             for i in range(n_ev)]
    post2 = sum(sags2) / len(sags2)
    # EXPECTED FAIL: kill fires when NO improvement (post >= pre - 0.02).
    record("G2_transfer_kill", post2 >= pre2 - 0.02,
           pre=pre2, post=post2, n_eval=n_ev)

    # ---------------- G3 egress contract at scale ----------------
    out3 = ad.forward(s, unit_wave(2))
    shape_ok = out3.shape == (NB, BD)
    dtype_ok = out3.dtype == torch.float32
    finite_ok = bool(torch.isfinite(out3).all())
    bound_ok = bool((out3 >= -1.0 - 1e-6).all()) and bool((out3 <= 1.0 + 1e-6).all())
    n3 = torch.norm(out3, p=2, dim=-1)
    unit_ok = bool(torch.allclose(n3, torch.ones_like(n3), atol=1e-4))
    record("G3_egress_contract",
           shape_ok and dtype_ok and finite_ok and bound_ok and unit_ok,
           shape=shape_ok, dtype=dtype_ok, finite=finite_ok,
           bounded=bound_ok, unit=unit_ok)

    # ---------------- G4 default-OFF at scale + WIRE ----------------
    from efe_planner import EFEPlanner, LowRankCoupledTransition
    planner_off = EFEPlanner(num_blocks=NB, d_model=D, num_actions=NA).to(DEVICE)
    legacy_ok = isinstance(planner_off.transition, LowRankCoupledTransition) \
        and not planner_off._use_complex_transition
    s4 = unit_wave(11)
    a4 = unit_wave(12)
    out4 = planner_off.transition(s4, a4)
    n4 = torch.norm(out4, p=2, dim=-1)
    legacy_unit = bool(torch.allclose(n4, torch.ones_like(n4), atol=1e-4))
    record("G4_default_off",
           legacy_ok and legacy_unit and bool(torch.isfinite(out4).all()))

    planner_on = EFEPlanner(
        num_blocks=NB, d_model=D, num_actions=NA,
        use_complex_transition=True).to(DEVICE)
    from complex_phase_transition import NativeComplexWaveTransition
    wired = isinstance(planner_on.transition, NativeComplexWaveTransition) \
        and planner_on._use_complex_transition
    cands = [(i, unit_wave(30 + i)) for i in range(NA)]
    boundary = unit_wave(99).unsqueeze(0)
    try:
        action_id, predicted, table, chosen = planner_on.select_action(
            s4, cands, boundary_axioms=boundary)
        wire_ok = len(table) == NA and bool(torch.isfinite(predicted).all())
    except Exception as exc:  # noqa: BLE001
        wire_ok = False
        results["WIRE_error"] = str(exc)[:300]
    record("WIRE_select_action", wired and wire_ok)

    try:
        pre_step = planner_on.train_transition_step(s4, a4, out4, lr=0.05)
        wire_step_ok = bool(torch.isfinite(torch.tensor(pre_step)))
    except Exception as exc:  # noqa: BLE001
        wire_step_ok = False
        results["WIRE_step_error"] = str(exc)[:300]
    record("WIRE_train_step", wire_step_ok)

    # field-channel roundtrip at scale
    try:
        wave = planner_on.field_channel_wave()
        planner_on.load_field_channel_wave(wave)
        wire_rt = bool(torch.isfinite(planner_on.transition.action_phases).all())
    except Exception as exc:  # noqa: BLE001
        wire_rt = False
        results["WIRE_rt_error"] = str(exc)[:300]
    record("WIRE_field_channel_roundtrip", wire_rt)

    # ---------------- DONE marker (aggregated) ----------------
    rc = 1 if failures else 0
    results["DONE_MARKER"] = {"rc": rc, "failures": failures, "smoke": SMOKE,
                              "device": DEVICE, "D": D}
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"DONE_MARKER rc={rc} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
