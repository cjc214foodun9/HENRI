"""HYBRID ROUTER CV A/B — >=2 demo pairs, router selects on a HELD-OUT FOLD.

WHY THIS EXISTS
===============
v1 (action2_hybrid_operator_ab_observed.json, sha 940c3962cb19...) exposed two
defects in the hybrid AS SPECIFIED:

  (a) ROUTER BIAS.  v1's router compared IN-SAMPLE demo fits.  The ridge is
      FITTED on the demos, so with 1 demo its in-sample score is the exact
      least-squares optimum (~1.0); the roll x colour x D4 class is SEARCHED and
      caps below 1.0 when the transform is not in D4.  The router therefore
      always preferred the ridge and inherited its failures.
  (b) NO TRANSFER TEST.  The ridge was fitted at ONE ring position, so its
      0.992811 containment number never tested transfer.  Under independent held
      content v1 measured 0.439190 -- BELOW the identity baseline under v1's
      metric.

This probe implements the pre-registered next falsification:

  * K_DEMOS demo pairs per task, each with INDEPENDENT content.
  * Containment demos sit at NON-OVERLAPPING ring positions (zero shared cells),
    so the ridge is fitted ACROSS positions and must transfer to a position it
    never saw.  That is the transfer question v1 raised.
  * Router = LEAVE-ONE-OUT CROSS-VALIDATION: select the operator on K-1 demos,
    score the left-out demo, average over folds.  Selection is on held-out folds.
  * The final held pair is fresh independent content, entering at scoring only.

METRIC — THIS IS A CORRECTION, AND IT IS PRE-REGISTERED
=======================================================
v1's primary metric was cos(Psi_hat, Psi_Y).  On containment that is a BAD
metric and v1's own numbers prove it: under v1's metric a no-op scored 0.849657
because the transform changes only the ring interior, so the wave is dominated
by UNCHANGED background.  A no-op was rewarded for predicting nothing.

PRIMARY metric here:   cos_delta = cos(Psi_hat - Psi_X, Psi_Y - Psi_X)
                       -- cosine of the PREDICTED CHANGE vs the ACTUAL CHANGE.
                       A no-op predicts zero change and scores EXACTLY 0.0.
SECONDARY metric:      cos_full = cos(Psi_hat, Psi_Y)  (v1's metric, for
                       comparability; reported, never used to select).

The router selects on the PRIMARY metric.  Both are reported for every arm.

EVALUATION ORDER AND COST (the hoist)
=====================================
The per-(demo, candidate) encode depends only on (grid, candidate), NOT on the
leave-one-out fold.  So every encode that any fold needs is computed ONCE per
task and reduced to four scalars per candidate:

    N[k] = ||E_k||          A[k] = <E_k, y>        (full-metric numerator)
    C[k] = ||E_k - x||      B[k] = <E_k, y - x>    (delta-metric numerator)

    cos_full(k, j)  = A / (N * ||y||)
    cos_delta(k, j) = B / (C * ||y - x||)

Fold selection and the final search are then pure arithmetic over these arrays.
This changes NO measurement; it removes repeated encoding.  Verified by receipt
fields `candidate_encodes` and `runtime_s`.

ARMS (identical demos, identical held pair, identical budget)
============================================================
  identity    Psi_hat = Psi_X                       (no-op baseline)
  ridge       per-slot diagonal ridge fitted on ALL K demos
  rigid       best roll x colour-map x D4 candidate searched on ALL K demos
  hybrid      the channel the CV router picks, applied to the held pair

PRE-REGISTERED CRITERIA (tau = 0.01), ORDER MATTERS
===================================================
A router IS one of its channels, so it can never EXCEED max(ridge, rigid).
Evaluated in this order, on the PRIMARY metric, for CONTAINMENT:

  (1) if the CV router picks RIDGE on a majority of containment tasks
        -> FALSIFIED_ROUTER_PICKS_RIDGE_ON_CONTAINMENT   (the user's rule)
  (2) elif hybrid_delta - identity_delta >= tau
        -> RECOVERS_WIN_IS_ROUTER
  (3) else
        -> FALSIFIED_NEITHER_CHANNEL_BEATS_IDENTITY_ON_CONTAINMENT

Step (1) is checked first and unconditionally: it is the user's decision rule.
A "win" that arrives via a channel the router should not prefer is still logged
as falsified, with the gain reported beside it.

TIES: a tie (|ridge_cv - rigid_cv| < 1e-6) goes to D4, the LOWER-CAPACITY
channel (792-element finite class vs 65536 continuous parameters -- Occam).
The tie count is reported explicitly, never folded silently.

HONEST LIMITS
=============
  * Synthetic grids, internal-representation recovery on the first held pair per
    task.  NOT an ARC task-solve rate.  No ARC score is claimed.
  * The D4 search is exhaustive over the class -- the class UPPER BOUND, so a
    loss cannot be rescued by iteration order or annealing.
  * The ridge is per-slot DIAGONAL (element-wise).  It cannot select "only the
    interior cells", so a structural failure is expected, not surprising.
  * Local CPU.  Deterministic (fixed seed).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import torch  # noqa: E402
from o_vsa_torus_encoder import TorusIngressEncoder  # noqa: E402

# ------------------------------------------------------------------ config
TAU = 1e-2
LAM = 1e-3
N_SIDE = 12                  # 12x12: holds 5x5... no: holds 3 non-overlapping 4x4 rings
RING_H = 4                   # ring side; interior = 2x2 = 4 cells
STEP = RING_H                # positions spaced by the ring side => ZERO cell overlap
POSITIONS: List[Tuple[int, int]] = [(t, l)
                                    for t in range(0, N_SIDE - RING_H + 1, STEP)
                                    for l in range(0, N_SIDE - RING_H + 1, STEP)]
SHIFTS: List[Tuple[int, int]] = [(dx, dy) for dx in (-2, 0, 2) for dy in (-2, 0, 2)]
N_COLOURS = 10
MASK_OPS: List[Tuple[int, int]] = [(0, 0)] + [(v, (v + 1) % N_COLOURS) for v in range(N_COLOURS)]
SPINS: Tuple[int, ...] = tuple(range(8))      # D4: 4 rotations + 4 reflections
CANDIDATES: List[Tuple[int, int, int, int]] = [(dx, dy, m, spin)
                                               for dx, dy in SHIFTS
                                               for m in range(len(MASK_OPS))
                                               for spin in SPINS]
NC = len(CANDIDATES)

RING_COLOURS = [1, 2, 4, 5, 6, 7, 8, 9]
INNER_COLOURS = [1, 2, 4, 5, 6, 7, 8, 9]


# --------------------------------------------------------------- transforms
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


# ------------------------------------------------------------- wave utils
def _flat(t: torch.Tensor) -> torch.Tensor:
    if t.is_complex():
        t = t.real
    return t.reshape(-1).to(torch.float64)


def _norm(t: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(t))


def cos_full_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = _norm(a), _norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def cos_delta_vec(pred: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    """PRIMARY metric: cos(predicted change, actual change).

    A no-op (pred == x) predicts ZERO change and therefore scores EXACTLY 0.0.
    That is the honest baseline: predicting nothing earns nothing on a
    change-prediction task.  cos_full cannot express this, because on a sparse
    change the unchanged background dominates it (measured: a no-op scored
    0.930874 on containment under cos_full in this layout, and 0.849657 in v1).
    """
    dp = pred - x
    dy = y - x
    np_, ny = _norm(dp), _norm(dy)
    if np_ < 1e-12 or ny < 1e-12:
        return 0.0
    return float(torch.dot(dp, dy) / (np_ * ny))


# ------------------------------------------------- feature hoist (per task)
def candidate_features(enc, grid, x: torch.Tensor, y: torch.Tensor) -> dict:
    """Encode every (grid, candidate) ONCE; reduce to four scalars per candidate.

    Returns arrays N, A, B, C plus the two norms ||y|| and ||y - x||.
      N[k] = ||E_k||          A[k] = <E_k, y>
      C[k] = ||E_k - x||      B[k] = <E_k - x, y - x>

    SELF-CHECK INVARIANT: if E_k == y exactly then s_delta == 1 exactly.  The
    earlier form <E_k, y - x> omitted the -<x, y - x> term, so it was NOT the
    delta metric.  Caught 2026-09-27 by the smoke run: a reflection that IS in
    the class reported delta 0.500000 while reporting full 1.000000 -- an
    impossible pair for a true cos_delta, which is what made the defect visible.
    """
    dY = y - x
    ny, ndy = _norm(y), _norm(dY)
    N = [0.0] * NC
    A = [0.0] * NC
    B = [0.0] * NC
    C = [0.0] * NC
    for k, (dx, dy, m, spin) in enumerate(CANDIDATES):
        E = _flat(enc.encode(apply_candidate(grid, dx, dy, m, spin)))
        dE = E - x                       # PREDICTED CHANGE
        N[k] = _norm(E)
        A[k] = float(torch.dot(E, y))
        B[k] = float(torch.dot(dE, dY))  # FIX: was <E, dY>; missing the -<x, dY> term
        C[k] = _norm(dE)
    return {"N": N, "A": A, "B": B, "C": C, "ny": ny, "ndy": ndy}


def s_full(f: dict, k: int) -> float:
    d = f["N"][k] * f["ny"]
    return 0.0 if d < 1e-12 else f["A"][k] / d


def s_delta(f: dict, k: int) -> float:
    d = f["C"][k] * f["ndy"]
    return 0.0 if d < 1e-12 else f["B"][k] / d


def search_over(feats: Sequence[dict], idx: Sequence[int], metric) -> int:
    """Exhaustive argmax over the whole class, summed over the demo subset."""
    best, best_k = -1e18, 0
    for k in range(NC):
        s = 0.0
        for j in idx:
            s += metric(feats[j], k)
        if s > best:
            best, best_k = s, k
    return best_k


# ----------------------------------------------------------------- fixtures
def _ring_pair(ring: int, inner: int, top: int, left: int, rng) -> Tuple[list, list]:
    """(X, Y): a RING_H x RING_H ring of `ring` on low-count noise background;
    Y additionally FILLS the interior with `inner`.

    Interior fill is POSITION-DEPENDENT: cells strictly inside the ring change,
    cells outside do not.  A global operator class cannot express it; a per-slot
    diagonal operator cannot select interior cells.
    """
    g = [[rng.randrange(0, 3) for _ in range(N_SIDE)] for _ in range(N_SIDE)]
    bot, right = top + RING_H - 1, left + RING_H - 1
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


def make_containment_task(n_demos: int, seed: int) -> dict:
    """n_demos demo pairs at DISTINCT NON-OVERLAPPING positions + 1 held pair at
    a further distinct position.  Distinct (ring, inner) colour pairs."""
    rng = random.Random(seed)
    pos = POSITIONS[:]
    rng.shuffle(pos)
    ring_c = RING_COLOURS[:]
    inner_c = INNER_COLOURS[:]
    rng.shuffle(ring_c)
    rng.shuffle(inner_c)
    demos, grids = [], []
    for i in range(n_demos):
        top, left = pos[i]
        X, Y = _ring_pair(ring_c[i], inner_c[(i + 1 + n_demos) % len(inner_c)],
                          top, left, rng)
        demos.append((X, Y))
        grids.append(X)
    top, left = pos[n_demos]
    HX, HY = _ring_pair(ring_c[n_demos], inner_c[(n_demos + 1 + n_demos) % len(inner_c)],
                        top, left, rng)
    return {"id": "cont", "family": "containment", "demos": demos,
            "grids": grids, "held": (HX, HY),
            "positions": [pos[i] for i in range(n_demos + 1)]}


def make_reflection_task(n_demos: int, seed: int) -> dict:
    """n_demos demo pairs + 1 held pair; transform = vertical flip (D4 spin 4)."""
    rng = random.Random(seed)
    demos, grids = [], []
    for _ in range(n_demos):
        g = [[rng.randrange(N_COLOURS) for _ in range(N_SIDE)] for _ in range(N_SIDE)]
        demos.append((g, g[::-1]))
        grids.append(g)
    g = [[rng.randrange(N_COLOURS) for _ in range(N_SIDE)] for _ in range(N_SIDE)]
    return {"id": "refl", "family": "reflection", "demos": demos, "grids": grids,
            "held": (g, g[::-1]), "positions": None}


# ---------------------------------------------------------------- operators
def fit_ridge(demo_waves: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Per-slot DIAGONAL ridge: W = sum(x*y) / (sum(x^2) + lam)."""
    num = torch.zeros_like(demo_waves[0][0])
    den = torch.zeros_like(demo_waves[0][0])
    for x, y in demo_waves:
        num = num + x * y
        den = den + x * x
    return num / (den + LAM)


# --------------------------------------------------------------------- run
def run(limit: int, out: str, seed: int, n_demos: int, offset: int = 0) -> dict:
    if n_demos < 2:
        raise ValueError("n_demos must be >= 2 for held-out-fold selection")
    if n_demos + 1 > len(POSITIONS):
        raise ValueError(f"need n_demos+1 positions; pool has {len(POSITIONS)}")
    if limit % 2:
        raise ValueError("limit must be even (half containment, half reflection)")

    enc = TorusIngressEncoder(num_blocks=8192, mode="TORUS_VAL", device="cpu")
    zero = _flat(enc.encode([[0] * N_SIDE for _ in range(N_SIDE)]))
    D = zero.numel()
    print(f"encoder D={D} (grid {N_SIDE}x{N_SIDE}) | class = {len(SHIFTS)}x{len(MASK_OPS)}"
          f"x{len(SPINS)} = {NC} | ring {RING_H}x{RING_H} interior {RING_H-2}x{RING_H-2}"
          f" | positions pool {len(POSITIONS)} (step {STEP}, zero overlap)")
    print(f"K_DEMOS={n_demos} | tau={TAU:.2f} | primary=cos_delta (no-op scores 0.0)"
          f" | encodes/task = (K+1)*{NC} = {(n_demos+1)*NC}")

    half = limit // 2
    tasks = ([make_containment_task(n_demos, seed + 100 * (offset + t)) for t in range(half)]
             + [make_reflection_task(n_demos, seed + 5000 + 100 * (offset + half + t))
                for t in range(half)])
    tasks = [{**t, "id": f"{t['id']}_{offset + i:02d}"} for i, t in enumerate(tasks)]
    print(f"tasks: {len(tasks)} ({half} containment + {half} reflection), "
          f"independent content throughout")

    t0 = time.perf_counter()
    rows: List[dict] = []
    encodes = 0
    for task in tasks:
        dgrids = task["grids"]
        dw = [(_flat(enc.encode(X)), _flat(enc.encode(Y))) for X, Y in task["demos"]]
        encodes += 2 * n_demos
        hx, hy = task["held"]
        phx, phy = _flat(enc.encode(hx)), _flat(enc.encode(hy))
        encodes += 2

        # ---- HOIST: one encode per (demo|held, candidate), shared by all folds
        feats = [candidate_features(enc, dgrids[j], dw[j][0], dw[j][1])
                 for j in range(n_demos)]
        feats.append(candidate_features(enc, hx, phx, phy))
        encodes += (n_demos + 1) * NC
        H = n_demos                                   # held feature index

        # ---- arms on the held pair
        W_all = fit_ridge(dw)
        ridge_pred = W_all * phx
        ridge_delta = cos_delta_vec(ridge_pred, phx, phy)
        ridge_full = cos_full_vec(ridge_pred, phy)

        op_all = search_over(feats, list(range(n_demos)), s_delta)
        rigid_delta = s_delta(feats[H], op_all)
        rigid_full = s_full(feats[H], op_all)

        arms = {
            "identity": (0.0, cos_full_vec(phx, phy)),
            "ridge": (ridge_delta, ridge_full),
            "rigid": (rigid_delta, rigid_full),
        }

        # ---- CV router: select on K-1 demos, score the left-out demo
        ridge_folds, rigid_folds = [], []
        for i in range(n_demos):
            tr = [j for j in range(n_demos) if j != i]
            W_i = fit_ridge([dw[j] for j in tr])
            ridge_folds.append(cos_delta_vec(W_i * dw[i][0], dw[i][0], dw[i][1]))
            op_i = search_over(feats, tr, s_delta)
            rigid_folds.append(s_delta(feats[i], op_i))
        cv_r = sum(ridge_folds) / n_demos
        cv_g = sum(rigid_folds) / n_demos
        n_ties = 1 if abs(cv_r - cv_g) < 1e-6 else 0
        channel = "D4" if n_ties else ("RIDGE" if cv_r > cv_g else "D4")

        hd, hf = arms["ridge" if channel == "RIDGE" else "rigid"]

        # ---- transfer diagnostic: ridge fitted on ONE demo vs ALL demos
        W1 = fit_ridge(dw[:1])
        ridge_1 = cos_delta_vec(W1 * phx, phx, phy)

        rd, rg = ridge_delta, rigid_delta
        # CANONICAL NAMESPACE.  channel is "RIDGE"/"D4"; held_winner must use the
        # SAME keys.  Defect fixed 2026-09-27: the earlier form stored
        # "ridge"/"rigid" and then compared channel.lower() == w.lower(), so
        # "d4" != "rigid" and EVERY D4 routing scored router_correct=False.
        # Caught by the K=3 smoke: reflection routed D4 1/1 but reported
        # correct 0/1 while rigid_delta == 1.000000 exactly.
        w = "TIE" if abs(rd - rg) < 1e-6 else ("RIDGE" if rd > rg else "D4")

        rows.append({
            "task": task["id"], "family": task["family"],
            "identity_delta": arms["identity"][0], "identity_full": arms["identity"][1],
            "ridge_delta": rd, "ridge_full": ridge_full,
            "rigid_delta": rg, "rigid_full": rigid_full,
            "hybrid_delta": hd, "hybrid_full": hf,
            "channel": channel, "held_winner": w,
            "router_correct": (channel == w) if w != "TIE" else True,
            "n_ties_cv": n_ties, "cv_ridge": cv_r, "cv_rigid": cv_g,
            "routing_gap": max(rd, rg) - hd,
            "ridge_1_demo_delta": ridge_1, "ridge_k_demo_delta": rd,
            "chosen_op": list(CANDIDATES[op_all]),
        })

    dt = time.perf_counter() - t0

    def fam(f: str) -> dict:
        r = [x for x in rows if x["family"] == f]
        n = len(r)

        def mean(k):
            return sum(x[k] for x in r) / n
        return {
            "n": n,
            "identity_delta": mean("identity_delta"), "identity_full": mean("identity_full"),
            "ridge_delta": mean("ridge_delta"), "ridge_full": mean("ridge_full"),
            "rigid_delta": mean("rigid_delta"), "rigid_full": mean("rigid_full"),
            "hybrid_delta": mean("hybrid_delta"), "hybrid_full": mean("hybrid_full"),
            "routed_to_ridge": sum(1 for x in r if x["channel"] == "RIDGE"),
            "routed_to_rigid": sum(1 for x in r if x["channel"] == "D4"),
            "router_correct": sum(1 for x in r if x["router_correct"]),
            "n_ties_cv": sum(x["n_ties_cv"] for x in r),
            "routing_gap_mean": mean("routing_gap"),
            "ridge_1_demo_delta": mean("ridge_1_demo_delta"),
            "ridge_k_demo_delta": mean("ridge_k_demo_delta"),
        }

    con, ref = fam("containment"), fam("reflection")
    n_hyb_d = sum(1 for x in rows if x["hybrid_delta"] - x["identity_delta"] >= TAU)
    n_corr = sum(1 for x in rows if x["router_correct"])

    # ---- PRE-REGISTERED VERDICT (containment, PRIMARY metric), ordered
    frac_ridge = con["routed_to_ridge"] / con["n"]
    con_gain = con["hybrid_delta"] - con["identity_delta"]
    if frac_ridge > 0.5:
        verdict = "FALSIFIED_ROUTER_PICKS_RIDGE_ON_CONTAINMENT"
    elif con_gain >= TAU:
        verdict = "RECOVERS_WIN_IS_ROUTER"
    else:
        verdict = "FALSIFIED_NEITHER_CHANNEL_BEATS_IDENTITY_ON_CONTAINMENT"

    def fixed_total(key: str) -> int:
        return sum(1 for x in rows if x[key] - x["identity_delta"] >= TAU)

    summary = {
        "schema": "henri.arc.action2-hybrid-router-cv.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_class": "OBSERVED",
        "device_kind": "cpu",
        "preregistration": {
            "tau": TAU, "lam": LAM, "n_side": N_SIDE, "ring_h": RING_H, "step": STEP,
            "shifts": SHIFTS, "mask_ops": MASK_OPS, "spins_d4": list(SPINS),
            "class_product": NC, "n_demos": n_demos,
            "positions_pool": POSITIONS, "positions_non_overlapping": True,
            "primary_metric": "cos_delta = cos(pred-X, Y-X); no-op scores 0.0",
            "secondary_metric": "cos_full = cos(pred, Y)  [v1 metric]",
            "router": "leave-one-out CV over K demos on the PRIMARY metric; tie -> D4",
            "verdict_rule": ("ordered: containment router picks RIDGE (majority) -> "
                             "FALSIFIED; elif hybrid - identity >= tau -> RECOVERS "
                             "(win is the router); else -> FALSIFIED (neither channel "
                             "beats identity)"),
            "label_namespace": "channel and held_winner both use {RIDGE, D4, TIE}", "search_objective": "primary metric (cos_delta), exhaustive argmax over the class", "evaluation_hoist": ("per-(item,candidate) encodes are fold-independent and "
                                 "computed once; folds are arithmetic over 4 scalars"),
        },
        "fixture": (f"containment demos at DISTINCT NON-OVERLAPPING {RING_H}x{RING_H} ring "
                    f"positions (zero shared cells on the grid); held pair at a further "
                    f"distinct position; distinct (ring,inner) colours per pair"),
        "containment": con, "reflection": ref,
        "router_correct_total": f"{n_corr}/{len(rows)}",
        "hybrid_recovery_total_delta": f"{n_hyb_d}/{len(rows)}",
        "suite_totals_delta": {
            "hybrid": n_hyb_d,
            "fixed_ridge": fixed_total("ridge_delta"),
            "fixed_rigid": fixed_total("rigid_delta"),
        },
        "containment_best_channel_gain": max(con["ridge_delta"], con["rigid_delta"])
                                        - con["identity_delta"],
        "verdict": verdict,
        "candidate_encodes": encodes,
        "honest_limit": ("Synthetic grids, internal-representation recovery on the first "
                         "held pair per task; NOT an ARC task-solve rate. Non-overlapping "
                         "positions isolate the transfer question. The D4 search is "
                         "exhaustive (class upper bound). No ARC score is claimed."),
        "runtime_s": dt,
        "per_task": rows,
    }
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(f"\n=== HYBRID ROUTER CV A/B  (K={n_demos} demos, tau={TAU}, hoisted) ===")
    for nm, d in (("REFLECTION", ref), ("CONTAINMENT", con)):
        print(f"  {nm:<11} n={d['n']} | delta: ident {d['identity_delta']:.6f} "
              f"ridge {d['ridge_delta']:.6f} rigid {d['rigid_delta']:.6f} "
              f"hybrid {d['hybrid_delta']:.6f}")
        print(f"  {'':<11}     | full : ident {d['identity_full']:.6f} "
              f"ridge {d['ridge_full']:.6f} rigid {d['rigid_full']:.6f} "
              f"hybrid {d['hybrid_full']:.6f}")
        print(f"  {'':<11}     | routed RIDGE {d['routed_to_ridge']}/{d['n']}  "
              f"D4 {d['routed_to_rigid']}/{d['n']} | correct {d['router_correct']}/{d['n']} "
              f"| gap {d['routing_gap_mean']:.6f} | ties {d['n_ties_cv']}")
        print(f"  {'':<11}     | ridge transfer: 1-demo {d['ridge_1_demo_delta']:.6f} -> "
              f"{n_demos}-demo {d['ridge_k_demo_delta']:.6f}")
    print(f"  ROUTER CORRECT : {summary['router_correct_total']}")
    print(f"  HYBRID RECOVERY (delta): {summary['hybrid_recovery_total_delta']}")
    print(f"  SUITE (delta)  : hybrid {summary['suite_totals_delta']['hybrid']} | "
          f"fixed-ridge {summary['suite_totals_delta']['fixed_ridge']} | "
          f"fixed-rigid {summary['suite_totals_delta']['fixed_rigid']}")
    print(f"  VERDICT        : {verdict}")
    print(f"  encodes        : {encodes} | runtime {dt:.1f} s")
    if out:
        print(f"  receipt        : {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--n-demos", type=int, default=3)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out",
                    default="experiments/verification/action2_hybrid_router_cv_observed.json")
    ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args()
    o = a.out if os.path.isabs(a.out) else os.path.join(CODE, a.out)
    run(a.limit, o, a.seed, a.n_demos, a.offset)
