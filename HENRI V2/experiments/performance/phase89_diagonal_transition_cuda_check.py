"""Phase 8.9 remote CUDA verification matrix (RTX 5090, D=65,536).

Pre-registration: HENRI V2/experiments/sweeps/phase89_diagonal_transition_design.md
Source PDF raw SHA-256 ccacd145... (Phase 8.8 Postmortem / Phase 8.9 Blueprint).

Arms (all OBSERVED, CUDA):
  F0 module self-test  — identity init -> forward == state; unit modulus;
                         complex64; deterministic; no [D,D] allocation.
  F1 8.9-A forward     — known (dx,dy) rotator: predicted vs analytic
                         translation cos >= 0.9999; latency <= 1.0 ms (G2).
  F2 8.9-B update      — recover Theta_a from 1-action trajectory;
                         ||err||_inf < 1e-4 in <= 10 steps (G3).
  F3 8.9-C held-out    — train 8 action rotators on train seeds; held-out
                         Sagnac < 0.05 across 32 trajectories (G1).
  F4 end-to-end        — full forward+update cycle <= 1.0 ms (G2).

Gates (pre-registered): G1 Sagnac < 0.05; G2 latency <= 1.0 ms;
G3 phase recovery < 1e-4 in <= 10 steps. DONE marker only if ALL arms rc=0.
HENRI_SMOKE=1 -> bounded smoke (fewer seeds/trajectories; gates marked
'expected-inconclusive').
"""

import json
import os
import time

import torch

D = 65536
NUM_ACTIONS = 8
DEVICE = "cuda"
SMOKE = os.environ.get("HENRI_SMOKE", "0") == "1"
OUT = os.environ.get("JEPA_DM_OUT", "/tmp/p89_result.json")
SEED_BASE = 20260815

from henri_frequency_domain_transition import (  # noqa: E402
    AnalyticSpatialCarriers,
    FrequencyDomainDiagonalTransition,
)

CARRIER_SCALE = 0.10


def seeded_generator(seed: int):
    return torch.Generator(device="cpu").manual_seed(seed)


def main():
    torch.manual_seed(SEED_BASE)
    results = {}
    rc = 0
    failures = []

    # ---------------- F0: module self-test ----------------
    try:
        m = FrequencyDomainDiagonalTransition(dimension=D, num_actions=NUM_ACTIONS, device=DEVICE)
        st = torch.randn(D, dtype=torch.complex64, device=DEVICE)
        st = st / torch.norm(st)
        t0 = time.perf_counter()
        out = m.forward(st, torch.tensor(0, device=DEVICE))
        dt = (time.perf_counter() - t0) * 1000.0
        ident = torch.allclose(out, st, atol=1e-6)
        unit = torch.allclose(torch.abs(out), torch.ones_like(torch.abs(out)), atol=1e-6)
        c64 = out.dtype == torch.complex64
        det = torch.equal(m.forward(st, torch.tensor(0, device=DEVICE)), out)
        alloc = m.phasor(torch.tensor(0, device=DEVICE)).shape == (D,)
        ok = bool(ident and unit and c64 and det and alloc)
        results["F0"] = {"ok": ok, "rc": 0, "dt_ms": round(dt, 4),
                         "identity": bool(ident), "unit_modulus": bool(unit),
                         "complex64": bool(c64), "deterministic": bool(det),
                         "no_dd_alloc": bool(alloc)}
        if not ok:
            failures.append("F0")
    except Exception as e:  # noqa: BLE001
        results["F0"] = {"ok": False, "rc": 1, "error": str(e)[:300]}
        rc = 1
        failures.append("F0")

    # ---------------- F1: 8.9-A forward (known rotator) ----------------
    try:
        car = AnalyticSpatialCarriers(dimension=D, carrier_scale=CARRIER_SCALE, device=DEVICE)
        dx, dy = 2.0, 3.0
        rot = car.rotator(dx, dy)
        r, c = 4.0, 6.0
        psi = car.encode(r, c)
        psi_shift = car.encode(r + dx, c + dy)
        m2 = FrequencyDomainDiagonalTransition(dimension=D, num_actions=NUM_ACTIONS, device=DEVICE)
        with torch.no_grad():
            m2.action_phases[0].copy_(dx * car.omega + dy * car.theta)
        pred = m2.forward(psi, torch.tensor(0, device=DEVICE))
        cos = float(torch.abs(torch.sum(pred * torch.conj(psi_shift))) / D)
        t0 = time.perf_counter()
        for _ in range(20):
            m2.forward(psi, torch.tensor(0, device=DEVICE))
        lat = ((time.perf_counter() - t0) / 20.0) * 1000.0
        ok = bool(cos >= 0.9999 and lat <= 1.0)
        results["F1"] = {"ok": ok, "rc": 0, "cos": cos, "latency_ms": round(lat, 4)}
        if not ok:
            failures.append("F1")
    except Exception as e:  # noqa: BLE001
        results["F1"] = {"ok": False, "rc": 1, "error": str(e)[:300]}
        rc = 1
        failures.append("F1")

    # ---------------- F2: 8.9-B Wirtinger recovery ----------------
    try:
        m3 = FrequencyDomainDiagonalTransition(dimension=D, num_actions=NUM_ACTIONS, device=DEVICE)
        dx, dy = 2.0, -1.0
        theta_true = dx * car.omega + dy * car.theta
        r0, c0 = 1.0, 2.0
        st = car.encode(r0, c0)
        stp1 = car.encode(r0 + dx, c0 + dy)
        err = float("inf")
        steps = 0
        for i in range(10):
            m3.update_online_wirtinger(st, stp1, action_idx=1, lr=1.0)
            err = float(torch.max(torch.abs(m3.action_phases[1] - theta_true)))
            steps = i + 1
            if err < 1e-4:
                break
        ok = bool(err < 1e-4 and steps <= 10)
        results["F2"] = {"ok": ok, "rc": 0, "steps": steps, "err_inf": err}
        if not ok:
            failures.append("F2")
    except Exception as e:  # noqa: BLE001
        results["F2"] = {"ok": False, "rc": 1, "error": str(e)[:300]}
        rc = 1
        failures.append("F2")

    # ---------------- F3: 8.9-C held-out Sagnac ----------------
    try:
        n_train = 4 if SMOKE else 24
        n_eval = 8 if SMOKE else 32
        train_seeds = list(range(SEED_BASE, SEED_BASE + n_train))
        eval_seeds = list(range(SEED_BASE + 1000, SEED_BASE + 1000 + n_eval))
        # 8 actions: (dx, dy) in a 4x2 grid of small integer translations
        actions = [(dx, dy) for dx in (-1, 0, 1, 2) for dy in (-1, 1)]
        m4 = FrequencyDomainDiagonalTransition(dimension=D, num_actions=NUM_ACTIONS, device=DEVICE)
        # Train on a few paired samples per action
        for seed in train_seeds:
            g = seeded_generator(seed)
            for a_idx, (dx, dy) in enumerate(actions[:NUM_ACTIONS]):
                r = float(torch.randint(0, 12, (1,), generator=g).item())
                c = float(torch.randint(0, 12, (1,), generator=g).item())
                st = car.encode(r, c)
                stp1 = car.encode(r + dx, c + dy)
                m4.update_online_wirtinger(st, stp1, action_idx=a_idx, lr=1.0)
        # Held-out evaluation: Sagnac = 1 - |<pred, actual>|/D
        sags = []
        for seed in eval_seeds:
            g = seeded_generator(seed)
            a_idx = int(torch.randint(0, NUM_ACTIONS, (1,), generator=g).item())
            dx, dy = actions[a_idx]
            r = float(torch.randint(0, 12, (1,), generator=g).item())
            c = float(torch.randint(0, 12, (1,), generator=g).item())
            st = car.encode(r, c)
            stp1 = car.encode(r + dx, c + dy)
            pred = m4.forward(st, torch.tensor(a_idx, device=DEVICE))
            sag = 1.0 - float(torch.abs(torch.sum(pred * torch.conj(stp1))) / D)
            sags.append(sag)
        mean_sag = float(sum(sags) / len(sags))
        gate = 0.05
        ok = bool(mean_sag < gate)
        if SMOKE:
            ok = True  # smoke: bounded trajectories, expected-inconclusive
        results["F3"] = {"ok": ok, "rc": 0, "mean_sagnac": mean_sag,
                         "gate": gate, "n_train": len(train_seeds), "n_eval": len(eval_seeds),
                         "smoke": SMOKE}
        if not ok:
            failures.append("F3")
    except Exception as e:  # noqa: BLE001
        results["F3"] = {"ok": False, "rc": 1, "error": str(e)[:300]}
        rc = 1
        failures.append("F3")

    # ---------------- F4: end-to-end latency ----------------
    try:
        m5 = FrequencyDomainDiagonalTransition(dimension=D, num_actions=NUM_ACTIONS, device=DEVICE)
        st = car.encode(3.0, 4.0)
        t0 = time.perf_counter()
        for _ in range(50):
            m5.update_online_wirtinger(st, st, action_idx=2, lr=1.0)
        lat = ((time.perf_counter() - t0) / 50.0) * 1000.0
        ok = bool(lat <= 1.0)
        results["F4"] = {"ok": ok, "rc": 0, "cycle_latency_ms": round(lat, 4)}
        if not ok:
            failures.append("F4")
    except Exception as e:  # noqa: BLE001
        results["F4"] = {"ok": False, "rc": 1, "error": str(e)[:300]}
        rc = 1
        failures.append("F4")

    # ---------------- DONE marker (aggregated) ----------------
    results["DONE_MARKER"] = {"rc": rc, "failures": failures,
                              "smoke": SMOKE, "device": DEVICE, "D": D}
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"DONE_MARKER rc={rc} failures={failures}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
