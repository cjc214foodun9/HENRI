"""Phase 8.8 remote CUDA verification matrix (RTX 5090, D=65,536).

Pre-registration: HENRI V2/experiments/sweeps/phase88_spatial_carriers_design.md
Source PDF raw SHA-256 a07eb7d3... (Phase 8.7 Postmortem & Phase 8.8 Blueprint).

Arms:
- S0 carrier self-test: mean cos >= 0.85, determinism, incommensurability.
- S1 8.8-A mechanism: adjacent-translation cos >= 0.85 (gate 1); identity ~1.0;
  distant discrimination; per-block unit norm; fail-closed empty foreground.
- S2 8.8-C transition: train production LowRankCoupledTransition (via EFEPlanner)
  on seeded single-object translation trajectories (carrier ingress);
  held-out Sagnac loss < 0.30 over 32 eval trajectories (gate 2).
- S3 8.8-B latency: full ingress (segment+encode+to_blocks) + transition forward
  <= 45.0 ms at D=65,536 (gate 3).
- S4 Lie structure: apply_translation(w0, dx) vs encode(translated grid) cos >= 0.99.

All arms share seeded deterministic trajectories; DONE marker only if ALL arms rc=0.
HENRI_SMOKE=1 -> bounded arms (proves CUDA path; gates expected inconclusive).
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

D = 65536
NUM_BLOCKS = 8192
OUT = os.environ.get("JEPA_DM_OUT", "/tmp/p88_result.json")
SMOKE = os.environ.get("HENRI_SMOKE", "0") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from henri_spatial_carrier_ingress import VectorizedIncommensurateSpatialIngress  # noqa: E402
from efe_planner import EFEPlanner  # noqa: E402

results: dict = {}


def arm(name, fn):
    t0 = time.time()
    try:
        ok, payload = fn()
        results[name] = {"rc": 0, "ok": bool(ok), "payload": payload, "sec": round(time.time() - t0, 2)}
        print(f"[{name}] rc=0 ok={bool(ok)} {round(time.time() - t0, 2)}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        results[name] = {"rc": 1, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"[{name}] rc=1 {type(exc).__name__}: {exc}", flush=True)
    return results[name]["rc"]


def grid_single(color: int, r: int, c: int, H: int = 16, W: int = 16):
    g = np.zeros((H, W), dtype=int)
    g[r, c] = color
    return g


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())


def main() -> int:
    print(f"phase88 matrix D={D} device={DEVICE} smoke={SMOKE}", flush=True)
    ingress = VectorizedIncommensurateSpatialIngress(dimension=D, carrier_scale=0.10, device=DEVICE)
    assert DEVICE.type == "cuda", "matrix requires CUDA"

    # ---------------- S0 carrier self-test ----------------
    def s0():
        m = ingress.mean_cos_carrier()
        torch.manual_seed(1)
        i2 = VectorizedIncommensurateSpatialIngress(dimension=D, carrier_scale=0.10, device=DEVICE)
        det = bool(torch.equal(ingress.omega_x, i2.omega_x) and torch.equal(ingress.theta_y, i2.theta_y))
        frac = torch.frac((ingress.theta_y + 1e-6) / (ingress.omega_x + 1e-6))
        incomm = float(frac.std().item()) > 1e-3
        ok = m >= 0.85 and det and incomm
        return ok, {"mean_cos": round(m, 4), "deterministic": det, "incommensurate": incomm}

    # ---------------- S1 8.8-A mechanism ----------------
    def s1():
        w0 = ingress.encode_grid(grid_single(color=2, r=8, c=8))
        w1 = ingress.encode_grid(grid_single(color=2, r=8, c=9))
        adj = cos_sim(w0, w1)
        identity = cos_sim(w0, w0)
        wfar = ingress.encode_grid(grid_single(color=2, r=2, c=2))
        disc = cos_sim(w0, wfar)
        blocks = ingress.to_blocks(w0, NUM_BLOCKS)
        norms = blocks.norm(p=2, dim=-1)
        unit = float(norms.max().item() - 1.0) < 1e-4
        fail_closed = False
        try:
            ingress.encode_grid(np.zeros((8, 8), dtype=int))
        except ValueError:
            fail_closed = True
        ok = adj >= 0.85 and identity > 0.9999 and disc < 0.85 and unit and fail_closed
        return ok, {"adjacent_cos": round(adj, 4), "identity_cos": round(identity, 4),
                    "distant_cos": round(disc, 4), "per_block_unit": unit, "fail_closed": fail_closed}

    # ---------------- S2 8.8-C transition training ----------------
    def s2():
        n_train = 2 if SMOKE else 32
        n_eval = 2 if SMOKE else 32
        steps = 2 if SMOKE else 8
        iters = 1 if SMOKE else 3

        def build(seeds, steps_per, start_r=8):
            S, A, N = [], [], []
            for seed in seeds:
                g0 = torch.Generator(device="cpu").manual_seed(seed)
                r = start_r + int(torch.randint(0, 3, (1,), generator=g0).item())
                c = 4 + int(torch.randint(0, 8, (1,), generator=g0).item())
                for t in range(steps_per):
                    s = ingress.encode_grid(grid_single(color=1 + (seed % 5), r=r, c=c + t))
                    S.append(s)
                    if t < steps_per - 1:
                        nxt = ingress.encode_grid(grid_single(color=1 + (seed % 5), r=r, c=c + t + 1))
                        N.append(nxt)
                        A.append(ingress.to_blocks(
                            torch.cat([ingress.translation_wave(1.0, 0.0).real,
                                       ingress.translation_wave(1.0, 0.0).imag], dim=-1), NUM_BLOCKS))
            S = torch.stack(S)
            A = torch.stack(A) if A else torch.empty(0, NUM_BLOCKS, 8, device=DEVICE)
            N = torch.stack(N) if N else torch.empty(0, NUM_BLOCKS, 8, device=DEVICE)
            # align: transitions are (s_t, a_t) -> s_{t+1}; keep the FIRST steps-1 states
            return S[:len(N)] if len(N) else S, A, N

        train_seeds = list(range(n_train))
        eval_seeds = list(range(1000, 1000 + n_eval))
        S_tr, A_tr, N_tr = build(train_seeds, steps)
        planner = EFEPlanner(num_blocks=NUM_BLOCKS, d_model=D).to(DEVICE)
        planner.train_transition_batch(S_tr, A_tr, N_tr, iters=iters, ridge=1e-4, blend=0.5)

        S_ev, A_ev, N_ev = build(eval_seeds, steps)
        preds = torch.stack([planner.transition(S_ev[i], A_ev[i]) for i in range(S_ev.shape[0])])
        p = preds.reshape(preds.shape[0], -1)
        o = N_ev.reshape(N_ev.shape[0], -1)
        held_out = float((1.0 - (p * o).sum(-1) / (p.norm(dim=-1) * o.norm(dim=-1)).clamp(min=1e-12)).mean())
        ok = held_out < 0.30
        return ok, {"held_out_sagnac": round(held_out, 4), "n_train": int(S_tr.shape[0]),
                    "n_eval": int(S_ev.shape[0]), "gate": "<0.30"}

    # ---------------- S3 8.8-B latency ----------------
    def s3():
        g = grid_single(color=3, r=8, c=8)
        waves = [ingress.encode_grid(g) for _ in range(1)]
        S = torch.stack([ingress.to_blocks(w, NUM_BLOCKS) for w in waves])
        act = ingress.to_blocks(torch.cat([ingress.translation_wave(1.0, 0.0).real,
                                           ingress.translation_wave(1.0, 0.0).imag], dim=-1), NUM_BLOCKS)
        planner = EFEPlanner(num_blocks=NUM_BLOCKS, d_model=D).to(DEVICE)
        with torch.no_grad():
            for _ in range(2):  # warmup
                planner.transition(S[0], act)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5):
                w = ingress.encode_grid(g)
                sb = ingress.to_blocks(w, NUM_BLOCKS)
                planner.transition(sb, act)
            torch.cuda.synchronize()
            total_ms = (time.time() - t0) / 5 * 1000.0
        ok = total_ms <= 45.0
        return ok, {"in_situ_ms": round(total_ms, 2), "gate": "<=45.0 ms"}

    # ---------------- S4 Lie structure ----------------
    def s4():
        w0 = ingress.encode_grid(grid_single(color=2, r=8, c=8))
        wt = ingress.encode_grid(grid_single(color=2, r=8, c=9))
        wrot = ingress.apply_translation(w0, 1.0, 0.0)
        c = cos_sim(wrot, wt)
        return c > 0.99, {"operator_vs_encode_cos": round(c, 4)}

    rc = 0
    rc += arm("S0", s0)
    rc += arm("S1", s1)
    rc += arm("S2", s2)
    rc += arm("S3", s3)
    rc += arm("S4", s4)

    rc_total = sum(1 for v in results.values() if v["rc"] != 0)
    results["_meta"] = {
        "phase": "8.8", "commit_env": os.environ.get("HENRI_COMMIT", "unknown"),
        "device": str(DEVICE), "smoke": SMOKE, "d": D, "num_blocks": NUM_BLOCKS,
        "done_marker_rc": 0 if rc_total == 0 else 1,
    }
    print(f"DONE_MARKER rc={0 if rc_total == 0 else 1}", flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    return 0 if rc_total == 0 else 1


if __name__ == "__main__":
    sys_rc = main()
    raise SystemExit(sys_rc)
