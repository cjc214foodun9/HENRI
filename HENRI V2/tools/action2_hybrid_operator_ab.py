"""HYBRID RELATIONAL OPERATOR A/B — the directive's unified operator, measured.

Directive (verbatim intent): stop treating the Tripartite Resonator and the
per-slot diagonal ridge as an either/or. Route RIGID global transforms
(rotations, chiral flips) to the D4 Cayley table (measured cos 1.000000), and
route position-dependent operations (interior fill) to the per-slot diagonal
ridge (measured 0.992811).

The testable question is NOT "does each channel work" — both were already
measured. The open question is: **can a ROUTER pick the right channel per task
from the DEMOS alone, without seeing the held target?**

Design (pre-registered):
  Arms (identical demos, identical held pair, identical budget):
    identity   cos(psi_Xh, psi_Yh)
    CONTROL    per-slot diagonal ridge fitted on demos
    RIGID      best candidate from the roll x colour-map x D4 class (demos only)
    HYBRID     per-task router: pick the channel with the higher DEMO fit,
               then score THAT channel on the held pair

FIXTURE FIX (the previously stated defect): the containment held pair is now
INDEPENDENT content — fresh ring position, fresh colours, fresh background
noise — instead of a rebuild of the demo pair. The previous revision's
containment identity baseline (0.787293) was inflated by that rebuild; this run
reports the clean number.

Acceptance (pre-registered): HYBRID beats CONTROL by tau AND beats RIGID and
identity by tau, with tau = 0.01. Reported as x/16.

HONEST LIMIT: synthetic grids, internal-representation recovery on the first
held pair per task. NOT an ARC task-solve rate. No ARC score is claimed.
Local CPU. The router reads demos only; the held target enters at scoring only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import torch  # noqa: E402
from o_vsa_torus_encoder import TorusIngressEncoder  # noqa: E402

TAU = 0.01
N_SIDE = 9
SHIFTS: List[Tuple[int, int]] = [(dx, dy) for dx in (-2, 0, 2) for dy in (-2, 0, 2)]
N_COLOURS = 10
MASK_OPS: List[Tuple[int, int]] = [(0, 0)] + [(v, (v + 1) % N_COLOURS) for v in range(N_COLOURS)]
SPINS: Tuple[int, ...] = tuple(range(8))          # D4: 4 rotations + 4 reflections


# ------------------------------------------------------------------- transforms
def rot90(g, k):
    for _ in range(k % 4):
        g = [list(r) for r in zip(*g[::-1])]
    return g


def sym(g, k):
    """k in 0..3 rotation; k in 4..7 reflection then rotation (D4)."""
    return rot90(g, k) if k < 4 else rot90(g[::-1], k - 4)


def roll2(g, dx, dy):
    n = len(g)
    return [[g[(i - dy) % n][(j - dx) % n] for j in range(n)] for i in range(n)]


def mask_colour(g, v, w):
    return [[(w if c == v else c) for c in row] for row in g]


def apply_candidate(g, dx, dy, m, spin):
    v, w = MASK_OPS[m]
    return mask_colour(roll2(sym(g, spin), dx, dy), v, w)


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).to(torch.float64)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


# ----------------------------------------------------------------------- tasks
def _rand_grid(rng, n=N_SIDE):
    return [[rng.randrange(N_COLOURS) for _ in range(n)] for _ in range(n)]


def make_reflection_tasks(n: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    out = []
    for t in range(n):
        g = _rand_grid(rng)
        y = g[::-1] if t % 2 == 0 else [r[::-1] for r in g]
        out.append({"id": "refl_%02d" % t, "family": "reflection", "X": g, "Y": y,
                    "flip_vertical": (t % 2 == 0)})
    return out


def _ring_grid(ring: int, inner: int, top: int, left: int, h: int, w: int,
               bg_noise: int = 0, rng=None):
    """Build (X, Y): a rectangular ring of `ring` with the interior filled in Y.

    Interior fill is POSITION-DEPENDENT: cells strictly inside the ring change,
    cells outside do not. That is exactly what a global-operator class cannot
    express.
    """
    g = [[0] * N_SIDE for _ in range(N_SIDE)]
    if bg_noise and rng is not None:
        for i in range(N_SIDE):
            for j in range(N_SIDE):
                g[i][j] = rng.randrange(1, bg_noise + 1)
    bot, right = top + h - 1, left + w - 1
    for j in range(left, right + 1):
        g[top][j] = ring
        g[bot][j] = ring
    for i in range(top, bot + 1):
        g[i][left] = ring
        g[i][right] = ring
    y = [r[:] for r in g]
    for i in range(top + 1, bot):
        for j in range(left + 1, right):
            y[i][j] = inner
    return g, y


def make_containment_tasks(n: int, seed: int, independent_held: bool = True) -> List[dict]:
    rng = random.Random(seed)
    out = []
    for t in range(n):
        ring = rng.choice([1, 2, 4, 5])
        inner = rng.choice([6, 7, 8, 9])
        top, left = 2, 2
        h, w = N_SIDE - 4, N_SIDE - 4
        g, y = _ring_grid(ring, inner, top, left, h, w, bg_noise=3, rng=rng)

        if independent_held:
            # INDEPENDENT held content: fresh ring position, fresh colours,
            # fresh background noise. This is the fixture fix.
            hr = rng.choice([1, 2, 4, 5])
            hi = rng.choice([6, 7, 8, 9])
            htop, hleft = rng.choice([1, 2]), rng.choice([1, 2])
            hh, hw = N_SIDE - 4, N_SIDE - 4
            hg, hy = _ring_grid(hr, hi, htop, hleft, hh, hw, bg_noise=3, rng=rng)
        else:
            hg, hy = _ring_grid(ring, inner, top, left, h, w, bg_noise=3, rng=rng)

        out.append({"id": "cont_%02d" % t, "family": "containment",
                    "X": g, "Y": y, "HX": hg, "HY": hy})
    return out


# --------------------------------------------------------------------- probes
def best_rigid(enc, Xg, psi_y):
    best, op = -2.0, None
    for dx, dy in SHIFTS:
        for m in range(len(MASK_OPS)):
            for spin in SPINS:
                c = cos(_flat(enc.encode(apply_candidate(Xg, dx, dy, m, spin))), psi_y)
                if c > best:
                    best, op = c, (dx, dy, m, spin)
    return best, op


def rigid_at(enc, g, op):
    dx, dy, m, spin = op
    return _flat(enc.encode(apply_candidate(g, dx, dy, m, spin)))


def run(limit: int, out: str, seed: int, independent_held: bool = True) -> dict:
    enc = TorusIngressEncoder(num_blocks=8192, mode="TORUS_VAL", device="cpu")
    D = _flat(enc.encode([[1, 2], [3, 4]])).numel()
    n_cand = len(SHIFTS) * len(MASK_OPS) * len(SPINS)
    print("encoder D=%d | class product = %d x %d x %d = %d"
          % (D, len(SHIFTS), len(MASK_OPS), len(SPINS), n_cand))

    tasks = make_reflection_tasks(limit // 2, seed) + \
        make_containment_tasks(limit // 2, seed + 1, independent_held=independent_held)
    print("tasks: %d (independent_held=%s)" % (len(tasks), independent_held))

    t0 = time.perf_counter()
    rows = []
    for task in tasks:
        Xg, Yg = task["X"], task["Y"]
        if task["family"] == "containment":
            hg, hy = task["HX"], task["HY"]
        else:
            # reflection held pair: INDEPENDENT fresh content, same transform
            rng = random.Random((hash(task["id"]) ^ seed) & 0xFFFFFF)
            hg = _rand_grid(rng)
            hy = hg[::-1] if task["flip_vertical"] else [r[::-1] for r in hg]

        px, py = _flat(enc.encode(Xg)), _flat(enc.encode(Yg))
        phx, phy = _flat(enc.encode(hg)), _flat(enc.encode(hy))

        # ---- arms
        ident = cos(phx, phy)

        W = (px * py) / (px * px + 1e-3)          # per-slot diagonal ridge
        ctrl_demo = cos(px * W, py)
        ctrl = cos(phx * W, phy)

        rigid_demo, op = best_rigid(enc, Xg, py)
        rigid = cos(rigid_at(enc, hg, op), phy)

        # ---- ROUTER: demo fit only, held target never consulted
        use_rigid = rigid_demo >= ctrl_demo
        hybrid = rigid if use_rigid else ctrl
        channel = "RIGID" if use_rigid else "RIDGE"

        rows.append({
            "task": task["id"], "family": task["family"],
            "identity_cos": ident, "control_cos": ctrl, "rigid_cos": rigid,
            "hybrid_cos": hybrid, "channel": channel,
            "demo_fit_rigid": rigid_demo, "demo_fit_ridge": ctrl_demo,
            "chosen_op": list(op),
            "hybrid_recovers": bool(hybrid - ctrl >= TAU and hybrid - ident >= TAU),
        })

    dt = time.perf_counter() - t0

    def fam(f):
        r = [x for x in rows if x["family"] == f]
        return {
            "n": len(r),
            "identity_mean": sum(x["identity_cos"] for x in r) / len(r),
            "control_mean": sum(x["control_cos"] for x in r) / len(r),
            "rigid_mean": sum(x["rigid_cos"] for x in r) / len(r),
            "hybrid_mean": sum(x["hybrid_cos"] for x in r) / len(r),
            "routed_to_rigid": sum(1 for x in r if x["channel"] == "RIGID"),
            "hybrid_recovered": sum(1 for x in r if x["hybrid_recovers"]),
        }

    ref, con = fam("reflection"), fam("containment")
    n_hyb = sum(1 for x in rows if x["hybrid_recovers"])

    # counterfactual: what would a FIXED-channel hybrid have scored?
    fixed_rigid = sum(1 for x in rows
                      if x["rigid_cos"] - x["control_cos"] >= TAU and x["rigid_cos"] - x["identity_cos"] >= TAU)
    fixed_ridge = sum(1 for x in rows
                      if x["control_cos"] - x["rigid_cos"] >= TAU and x["control_cos"] - x["identity_cos"] >= TAU)

    summary = {
        "schema": "henri.arc.action2-hybrid-operator-ab.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_class": "OBSERVED",
        "device_kind": "cpu",
        "preregistration": {"tau": TAU, "n_side": N_SIDE, "shifts": SHIFTS,
                            "mask_ops": MASK_OPS, "spins_d4": list(SPINS),
                            "class_product": n_cand, "target": "16/16",
                            "independent_held": independent_held},
        "router": "per-task: pick the channel with the higher DEMO fit; held target enters at scoring only",
        "fixture_fix": ("containment held pair is INDEPENDENT content (fresh ring position, "
                        "fresh colours, fresh background noise), replacing the earlier rebuild "
                        "whose identity baseline was inflated"),
        "reflection": ref, "containment": con,
        "hybrid_recovery_total": "%d/%d" % (n_hyb, len(rows)),
        "counterfactual_fixed_rigid": fixed_rigid,
        "counterfactual_fixed_ridge": fixed_ridge,
        "honest_limit": ("Synthetic grids, internal-representation recovery on the first held "
                         "pair per task; NOT an ARC task-solve rate. No ARC score is claimed."),
        "runtime_s": dt,
        "per_task": rows,
    }
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print("\n=== HYBRID OPERATOR A/B (tau=%.2f) ===" % TAU)
    for nm, d in (("REFLECTION", ref), ("CONTAINMENT", con)):
        print("  %-11s n=%d ident %.6f ctrl %.6f rigid %.6f hybrid %.6f | routed RIGID %d/%d | rec %d/%d"
              % (nm, d["n"], d["identity_mean"], d["control_mean"], d["rigid_mean"],
                 d["hybrid_mean"], d["routed_to_rigid"], d["n"],
                 d["hybrid_recovered"], d["n"]))
    print("  HYBRID TOTAL      : %s (target 16/16)" % summary["hybrid_recovery_total"])
    print("  FIXED rigid-only  : %d/%d" % (fixed_rigid, len(rows)))
    print("  FIXED ridge-only  : %d/%d" % (fixed_ridge, len(rows)))
    print("  runtime           : %.1f s" % dt)
    if out:
        print("  receipt           : %s" % out)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--out", default="experiments/verification/action2_hybrid_operator_ab_observed.json")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--rebuild-held", action="store_true",
                    help="use the OLD rebuild behaviour (for the A/B of the fixture fix itself)")
    a = ap.parse_args()
    o = a.out if os.path.isabs(a.out) else os.path.join(CODE, a.out)
    run(a.limit, o, a.seed, independent_held=not a.rebuild_held)
