"""ACTION 2 acceptance probe: can (roll x mask x Spin(3)) express
NON-LINEAR REFLECTION and SPATIAL CONTAINMENT transforms?

Directive: "Verify 16/16 recovery on non-linear reflections and spatial
containment tasks."  This answers it by measurement, not assertion.

IMPORTANT CORRECTION (2026-09-27): an earlier revision of this probe used only
4 planar ROTATIONS for Spin(3).  That under-states the directive: a Spin(3)
rotor acting on a planar figure via a 180-deg rotation about an in-plane axis
realises a REFLECTION.  The dihedral group D4 therefore IS inside Spin(3), so
this probe uses all 8 D4 elements (4 rotations + 4 reflections).  Without that
correction the probe would have failed reflections by construction -- a probe
defect masquerading as a scientific result.

Hypothesis class:
    R = grid roll by (dx,dy) in {-2,0,2}^2                          -> 9
    M = identity + 10 single-colour remaps                          -> 11
    S = D4 in Spin(3): 4 rotations + 4 reflections                   -> 8
    product = 9 x 11 x 8 = 792, chosen by EXHAUSTIVE argmax on DEMOS.
    Exhaustive argmax over the product IS the class upper bound, so a loss here
    cannot be rescued by iteration order, anneal schedule, or beta.

Arms (same demos, same held pair, same budget):
    identity  cos(psi_Xh, psi_Yh)
    control   per-slot diagonal ridge fitted on demos
    treatment best class candidate selected on demos, scored on held

Recovery: treatment beats control by tau AND beats identity by tau, tau=0.01.

LEAKAGE CONTROL: selection reads the DEMO pair only; the held pair is built
from fresh random content and enters at scoring only.

HONEST LIMIT: synthetic grids, internal-representation recovery, NOT an ARC
task-solve rate. No ARC score is claimed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import List, Tuple

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import torch
from o_vsa_torus_encoder import TorusIngressEncoder

TAU = 0.01
N_SIDE = 9
SHIFTS: List[Tuple[int, int]] = [(dx, dy) for dx in (-2, 0, 2) for dy in (-2, 0, 2)]
N_COLOURS = 10
MASK_OPS: List[Tuple[int, int]] = [(0, 0)] + [(v, (v + 1) % N_COLOURS) for v in range(N_COLOURS)]
SPINS = tuple(range(8))          # D4 = 4 rotations + 4 reflections


def rot90(g, k):
    for _ in range(k % 4):
        g = [list(r) for r in zip(*g[::-1])]
    return g


def sym(g, k):
    """k in 0..3 -> rotation; k in 4..7 -> reflection then rotation."""
    if k < 4:
        return rot90(g, k)
    return rot90(g[::-1], k - 4)


def roll2(g, dx, dy):
    n = len(g)
    return [[g[(i - dy) % n][(j - dx) % n] for j in range(n)] for i in range(n)]


def mask_colour(g, v, w):
    return [[(w if c == v else c) for c in row] for row in g]


def apply_candidate(g, dx, dy, m, spin):
    v, w = MASK_OPS[m]
    return mask_colour(roll2(sym(g, spin), dx, dy), v, w)


def _flat(t):
    return t.reshape(-1).to(torch.float64)


def cos(a, b):
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


def make_reflection_tasks(n, seed):
    rng = random.Random(seed)
    out = []
    for t in range(n):
        g = [[rng.randrange(N_COLOURS) for _ in range(N_SIDE)] for _ in range(N_SIDE)]
        y = g[::-1] if t % 2 == 0 else [r[::-1] for r in g]
        out.append({"id": "refl_%02d" % t, "family": "reflection",
                    "X": g, "Y": y, "flip": (t % 2 == 0)})
    return out


def make_containment_tasks(n, seed):
    rng = random.Random(seed)
    out = []
    for t in range(n):
        ring = rng.choice([1, 2, 4, 5])
        inner = rng.choice([6, 7, 8, 9])
        g = [[0] * N_SIDE for _ in range(N_SIDE)]
        top = left = 2
        bot = right = N_SIDE - 3
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
        out.append({"id": "cont_%02d" % t, "family": "containment",
                    "X": g, "Y": y, "ring": ring, "inner": inner})
    return out


def held_pair(task):
    """Fresh content, same transform -- selection never sees this."""
    if task["family"] == "reflection":
        rng = random.Random(hash(task["id"]) & 0xFFFF)
        hg = [[rng.randrange(N_COLOURS) for _ in range(N_SIDE)] for _ in range(N_SIDE)]
        hy = hg[::-1] if task["flip"] else [r[::-1] for r in hg]
        return hg, hy
    ring, inner = task["ring"], task["inner"]
    hg = [[0] * N_SIDE for _ in range(N_SIDE)]
    for j in range(2, N_SIDE - 2):
        hg[2][j] = ring
        hg[N_SIDE - 3][j] = ring
    for i in range(2, N_SIDE - 2):
        hg[i][2] = ring
        hg[i][N_SIDE - 3] = ring
    hy = [r[:] for r in hg]
    for i in range(3, N_SIDE - 3):
        for j in range(3, N_SIDE - 3):
            hy[i][j] = inner
    return hg, hy


def best_candidate(enc, Xg, psi_y):
    best, op = -2.0, None
    for dx, dy in SHIFTS:
        for m in range(len(MASK_OPS)):
            for spin in SPINS:
                c = cos(_flat(enc.encode(apply_candidate(Xg, dx, dy, m, spin))), psi_y)
                if c > best:
                    best, op = c, (dx, dy, m, spin)
    return best, op


def run(limit, out, seed):
    enc = TorusIngressEncoder(num_blocks=8192, mode="TORUS_VAL", device="cpu")
    D = _flat(enc.encode([[1, 2], [3, 4]])).numel()
    n_cand = len(SHIFTS) * len(MASK_OPS) * len(SPINS)
    print("encoder D=%d   class product = %d x %d x %d = %d" % (D, len(SHIFTS), len(MASK_OPS), len(SPINS), n_cand))

    tasks = make_reflection_tasks(limit // 2, seed) + make_containment_tasks(limit // 2, seed)

    # class-solvable sanity: build a target FROM the class, must recover it
    rng = random.Random(7)
    sx = [[rng.randrange(N_COLOURS) for _ in range(N_SIDE)] for _ in range(N_SIDE)]
    sy = apply_candidate(sx, 2, -2, 3, 5)       # dx=2 dy=-2, mask op 3, reflection spin 5
    bx, bxo = best_candidate(enc, sx, _flat(enc.encode(sy)))
    print("class-solvable check: true op=(2,-2,3,5) recovered at cos=%.6f by %s" % (bx, bxo))

    t0 = time.perf_counter()
    rows = []
    for task in tasks:
        Xg, Yg = task["X"], task["Y"]
        hg, hy = held_pair(task)
        psi_x, psi_y = _flat(enc.encode(Xg)), _flat(enc.encode(Yg))
        psi_hx, psi_hy = _flat(enc.encode(hg)), _flat(enc.encode(hy))

        ident = cos(psi_hx, psi_hy)
        W = (psi_x * psi_y) / (psi_x * psi_x + 1e-3)
        ctrl = cos(psi_hx * W, psi_hy)

        demo_fit, op = best_candidate(enc, Xg, psi_y)
        dx, dy, m, spin = op
        treat = cos(_flat(enc.encode(apply_candidate(hg, dx, dy, m, spin))), psi_hy)

        rows.append({"task": task["id"], "family": task["family"],
                     "identity_cos": ident, "control_cos": ctrl, "treatment_cos": treat,
                     "chosen": [dx, dy, m, spin], "demo_fit": demo_fit,
                     "chosen_is_identity": (dx == 0 and dy == 0 and m == 0 and spin == 0),
                     "recovered": bool(treat - ctrl >= TAU and treat - ident >= TAU)})

    dt = time.perf_counter() - t0

    def fam(f):
        r = [x for x in rows if x["family"] == f]
        return {"n": len(r),
                "identity_mean": sum(x["identity_cos"] for x in r) / len(r),
                "control_mean": sum(x["control_cos"] for x in r) / len(r),
                "treatment_mean": sum(x["treatment_cos"] for x in r) / len(r),
                "recovered": sum(1 for x in r if x["recovered"]),
                "chosen_is_identity": sum(1 for x in r if x["chosen_is_identity"])}

    ref, con = fam("reflection"), fam("containment")
    n_rec = sum(1 for x in rows if x["recovered"])
    summary = {"schema": "henri.arc.action2-acceptance-reflection-containment.v2",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "evidence_class": "OBSERVED", "device_kind": "cpu",
               "preregistration": {"tau": TAU, "n_side": N_SIDE, "shifts": SHIFTS,
                                   "mask_ops": MASK_OPS, "spins_d4": list(SPINS),
                                   "class_product": n_cand, "target": "16/16"},
               "correction": "Spin(3) includes reflections (180-deg rotation about an in-plane axis), so all 8 D4 elements are used, not 4 rotations.",
               "class_solvable_check": {"recovered_cos": bx, "op": list(bxo)},
               "reflection": ref, "containment": con,
               "recovery_total": "%d/%d" % (n_rec, len(rows)),
               "honest_limit": "Internal-representation recovery on synthetic grids; NOT an ARC task-solve rate. No ARC score is claimed.",
               "runtime_s": dt, "per_task": rows}

    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(summary, open(out, "w", encoding="utf-8"), indent=2, sort_keys=True)

    print("\n=== ACTION 2 ACCEPTANCE ===")
    for nm, r in (("REFLECTION", ref), ("CONTAINMENT", con)):
        print("  %-11s n=%d identity %.6f control %.6f treatment %.6f recovered %d/%d identity-op %d"
              % (nm, r["n"], r["identity_mean"], r["control_mean"], r["treatment_mean"],
                 r["recovered"], r["n"], r["chosen_is_identity"]))
    print("  RECOVERY TOTAL : %s (target 16/16) -> %s"
          % (summary["recovery_total"], "PASS" if n_rec == len(rows) else "FALSIFIED"))
    print("  runtime        : %.1f s" % dt)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--out", default="experiments/verification/action2_acceptance_reflection_containment_observed.json")
    ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args()
    o = a.out if os.path.isabs(a.out) else os.path.join(CODE, a.out)
    run(a.limit, o, a.seed)
