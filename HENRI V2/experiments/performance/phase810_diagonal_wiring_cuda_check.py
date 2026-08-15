"""Phase 8.10 remote CUDA verification matrix (RTX 5090, D=65,536).

Pre-registration: HENRI V2/experiments/sweeps/phase810_diagonal_production_wiring_design.md
Sealed verdict: KILL CONFIRMED (arccos real-domain bridge FALSIFIED for the
production L2-normalized real-wave regime). This matrix produces the
production-scale OBSERVED evidence for the kill + the wiring integrity:

- G1  identity: diagonal forward(s, zero) cos-sim 1.0 at D=65,536.
- G2  carrier regime (8.8 CC-OS production real waves): pre/post held-out
      Sagnac; kill expects NO improvement (post >= pre - 0.02).
- G3  learning budget (synthetic, <= 3 fit calls): post >= 0.05 (kill).
- G4  default-OFF: legacy LowRankCoupledTransition path at scale, per-block
      unit, finite; adapter never constructed when flag OFF.
- G5  latency: diagonal forward <= 1.0 ms at D=65,536.
- WIRE: EFEPlanner(use_diagonal_transition=True) constructs the adapter and
      select_action/update paths run at scale without device errors.

Multi-arm DONE marker: rc=1 if ANY arm gate fails (aggregated, learned lesson).
Env: JEPA_DM_OUT=/tmp/p810_result.json; HENRI_SMOKE=1 -> bounded seeds.
"""

import json
import math
import os
import time

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("HENRI_SMOKE", "0") == "1"
OUT = os.environ.get("JEPA_DM_OUT", "/tmp/p810_result.json")

NB = 8192
BD = 8
D = NB * BD
NA = 8

results = {}
failures = []


def record(name, ok, **kv):
    results[name] = {"ok": bool(ok), **kv}
    print(f"[{name}] ok={ok} " + " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items()), flush=True)
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


def main():
    from henri_frequency_domain_transition import FrequencyDomainDiagonalAdapter

    print(f"[init] device={DEVICE} D={D} smoke={SMOKE}", flush=True)

    # ---------------- G1 identity at scale ----------------
    ad = FrequencyDomainDiagonalAdapter(
        num_blocks=NB, block_dim=BD, num_actions=NA, device=DEVICE, d_model=D)
    s = unit_wave(1)
    a_zero = torch.zeros(NB, BD, device=DEVICE)
    t0 = time.perf_counter()
    out = ad.forward(s, a_zero)
    lat = (time.perf_counter() - t0) * 1000.0
    cos = float(torch.nn.functional.cosine_similarity(out.reshape(1, -1), s.reshape(1, -1), dim=-1))
    norms = torch.norm(out, p=2, dim=-1)
    unit = bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))
    record("G1_identity", cos > 0.9999 and unit, cos=cos, unit=unit, latency_ms=lat)

    # ---------------- G5 latency ----------------
    t0 = time.perf_counter()
    for _ in range(50):
        out = ad.forward(s, a_zero)
    lat5 = (time.perf_counter() - t0) * 1000.0 / 50.0
    record("G5_latency", lat5 <= 1.0, latency_ms=lat5)

    # ---------------- G3 learning budget (synthetic, kill evidence) ----------------
    g = torch.Generator(device="cpu").manual_seed(7)
    delta = ((torch.rand(D, generator=g, device="cpu") - 0.5) * 1.2).to(DEVICE)
    s3 = unit_wave(3)
    alpha = torch.acos(s3.reshape(-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    n3 = torch.cos(alpha + delta).reshape(NB, BD)
    n3 = n3 / (torch.norm(n3, p=2, dim=-1, keepdim=True) + 1e-9)
    ad3 = FrequencyDomainDiagonalAdapter(
        num_blocks=NB, block_dim=BD, num_actions=NA, device=DEVICE, d_model=D)
    for _ in range(3):
        ad3.fit_batch(s3.unsqueeze(0), a_zero.unsqueeze(0), n3.unsqueeze(0), iters=1, lr=1.0)
    post3 = real_sagnac(ad3.forward(s3, a_zero), n3)
    # Kill: gate expects post >= 0.05 at production budget
    record("G3_learning_budget_kill", post3 >= 0.05, post_sagnac=post3)

    # ---------------- G2 carrier regime (kill evidence) ----------------
    from henri_spatial_carrier_ingress import VectorizedIncommensurateSpatialIngress
    ingress = VectorizedIncommensurateSpatialIngress(dimension=D, device=DEVICE)
    ad2 = FrequencyDomainDiagonalAdapter(
        num_blocks=NB, block_dim=BD, num_actions=NA, device=DEVICE, d_model=D)
    states, nexts = [], []
    for i in range(6):
        c = 1 + (i % 3)
        r0, c0 = 2.0 + i, 3.0 + i
        st = ingress.encode_single_object(color=c, cx=r0, cy=c0).reshape(NB, BD)
        st = st / (torch.norm(st, p=2, dim=-1, keepdim=True) + 1e-9)
        nx = ingress.apply_translation(
            ingress.encode_single_object(color=c, cx=r0, cy=c0), dx=1.0, dy=-1.0
        ).reshape(NB, BD)
        nx = nx / (torch.norm(nx, p=2, dim=-1, keepdim=True) + 1e-9)
        states.append(st)
        nexts.append(nx)
    pre2 = real_sagnac(ad2.forward(states[0], a_zero), nexts[0])
    ad2.fit_batch(
        torch.stack(states[1:]), a_zero.repeat(5, 1, 1), torch.stack(nexts[1:]),
        iters=1, lr=1.0)
    post2 = real_sagnac(ad2.forward(states[0], a_zero), nexts[0])
    # Kill: expects NO improvement (post >= pre - 0.02)
    record("G2_carrier_kill", post2 >= pre2 - 0.02, pre=pre2, post=post2)

    # ---------------- G4 default-OFF + WIRE at scale ----------------
    from efe_planner import EFEPlanner, LowRankCoupledTransition
    planner_off = EFEPlanner(num_blocks=NB, d_model=D, num_actions=NA)
    legacy_ok = isinstance(planner_off.transition, LowRankCoupledTransition) and not planner_off._use_diagonal_transition
    s4 = unit_wave(11)
    a4 = unit_wave(12)
    out4 = planner_off.transition(s4, a4)
    n4 = torch.norm(out4, p=2, dim=-1)
    legacy_unit = bool(torch.allclose(n4, torch.ones_like(n4), atol=1e-4))
    record("G4_default_off", legacy_ok and legacy_unit and bool(torch.isfinite(out4).all()))

    planner_on = EFEPlanner(
        num_blocks=NB, d_model=D, num_actions=NA, use_diagonal_transition=True)
    from henri_frequency_domain_transition import FrequencyDomainDiagonalAdapter
    wired = isinstance(planner_on.transition, FrequencyDomainDiagonalAdapter)
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

    # single-step diagonal branch
    try:
        pre_step = planner_on.train_transition_step(s4, a4, out4, lr=0.05)
        wire_step_ok = bool(torch.isfinite(torch.tensor(pre_step)))
    except Exception as exc:  # noqa: BLE001
        wire_step_ok = False
        results["WIRE_step_error"] = str(exc)[:300]
    record("WIRE_train_step", wire_step_ok)

    # ---------------- DONE marker (aggregated) ----------------
    rc = 1 if failures else 0
    results["DONE_MARKER"] = {"rc": rc, "failures": failures, "smoke": SMOKE,
                              "device": DEVICE, "D": D}
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"DONE_MARKER rc={rc} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
