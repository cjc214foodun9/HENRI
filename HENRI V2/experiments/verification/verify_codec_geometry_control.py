"""Phase 5 P3 — codec-geometry control gate: functor goal vs identity goal.

Mechanism evidence (CPU toy-scale), NOT task capability evidence.

Question: does a functor-compiled goal (Psi_goal = bind(W_task, Psi_X_test),
W_task = sum_i Psi_Y_i (X) Psi_X_i^dag over in-context demo pairs) give
candidate-DISCRIMINATIVE goal distances, where the identity/prompt goal
(production_arc_run fallback layer 3) is quasi-orthogonal and flat?

Exercises the REAL production chain end-to-end:
  HENRIVisionEncoder.encode_grid (real S^{D-1} waves)
  -> ring quantization (production path, sagnac_mcts_planner.py:191-201)
  -> HolographicTaskFunctorCompiler.compile_functor (ring, mod-256, O(D))
  -> single_pass_associative_retrieval
  -> dequantize -> efe_planner.goal_distance semantics (1 - cos, both
     normalized) — the exact consumer invariant.

Pre-registered gate (Phase 5 packet, Task 3 / FM4):
  ACCEPT iff
    (a) rank of the true target == 1 under the FUNCTOR goal,
    (b) d_functor(true target) < d_identity(true target),
    (c) functor monotone spearman(goal_distance, grid_similarity) >= 0.7,
    (d) identity spearman < functor spearman - 0.3 (control is FLAT).
  KILL if the functor ranking is flat or no better than identity.
  BLOCKED_INFRASTRUCTURE on NaN/inf. No lambda/weight tuning anywhere.

Zero-pretraining invariant: W_task is compiled ONLY from in-context demo
pairs at test time; no task solutions are stored or pre-ingested.
"""

import argparse
import json
import math

import torch

from henri_vision_encoder import HENRIVisionEncoder
from zone_c_epistemic_axiom_harness import (
    HolographicTaskFunctorCompiler,
    qFHRREpistemicCodec,
)


OBJECT_SHAPES = {
    "L": __import__("numpy").array(
        [[1, 0], [1, 0], [1, 1]], dtype=int),
    "block": __import__("numpy").array(
        [[2, 2], [2, 2]], dtype=int),
    "bar": __import__("numpy").array(
        [[3, 3, 3]], dtype=int),
}


def make_object_task(seed: int = 7, grid: int = 12, n_demos: int = 3,
                     shift: int = 1):
    """ARC-faithful object-centric task: the SAME objects (reused across
    demos AND test, per the ARC object-reuse prior and HENRI CC-OS factoring)
    are placed at different positions; the rule is 'translate every object
    down by `shift` rows'. W_task compiled from demo pairs must therefore
    capture an object-level operator, not memorize cell patterns.

    Returns (demos, (X_t, Y_t)) as numpy grids."""
    np = __import__("numpy")
    rng = np.random.default_rng(seed)
    shapes = list(OBJECT_SHAPES.values())
    grid_size = grid

    def place_objects(rng):
        """Place the 3 fixed objects at random non-overlapping positions."""
        placed = []
        canvas = np.zeros((grid_size, grid_size), dtype=int)
        for shape in shapes:
            H, W = shape.shape
            for _ in range(200):
                r = int(rng.integers(0, grid_size - H - shift))
                c = int(rng.integers(0, grid_size - W))
                patch = canvas[r:r + H, c:c + W]
                if np.all(patch == 0):
                    canvas[r:r + H, c:c + W] = shape
                    placed.append((r, c, H, W, shape))
                    break
            else:
                raise RuntimeError("could not place objects without overlap")
        return canvas, placed

    def shift_canvas(canvas, placed, shift):
        out = np.zeros_like(canvas)
        for (r, c, H, W, shape) in placed:
            out[r + shift:r + shift + H, c:c + W] = shape
        return out

    demos = []
    for _ in range(n_demos):
        X, placed = place_objects(rng)
        Y = shift_canvas(X, placed, shift)
        demos.append((X, Y))
    X_t, placed_t = place_objects(rng)
    Y_t = shift_canvas(X_t, placed_t, shift)
    return demos, (X_t, Y_t)


def ring_of(real_wave: torch.Tensor, k_bins: int = 256) -> torch.Tensor:
    """Production ring quantization (sagnac_mcts_planner.py:191-192)."""
    return ((torch.clamp(real_wave, -1.0, 1.0) + 1.0) / 2.0 * (k_bins - 1)).to(torch.uint8)


def real_of(ring_wave: torch.Tensor, k_bins: int = 256) -> torch.Tensor:
    """Inverse of ring_of (production dequantize, sagnac_mcts_planner.py:201)."""
    return ring_wave.to(torch.float32) / (k_bins - 1) * 2.0 - 1.0


def goal_distance(pred: torch.Tensor, goal: torch.Tensor) -> float:
    """efE planner goal_distance invariant: 1 - Re<.,.>/(||.|| ||.||) in [0,2]."""
    p = pred.reshape(-1).float()
    p = p / (torch.norm(p) + 1e-12)
    g = goal.reshape(-1).float()
    g = g / (torch.norm(g) + 1e-12)
    return float((1.0 - torch.dot(p, g)).item())


def spearman(x: list, y: list) -> float:
    """Spearman rank correlation with AVERAGE ranks for ties (no scipy)."""
    n = len(x)
    if n < 3:
        return 0.0

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def grid_sim(a, b) -> float:
    a = __import__("numpy").asarray(a)
    b = __import__("numpy").asarray(b)
    return 1.0 - float((a != b).mean())


def flip_cells(grid, k: int, seed: int = 1) -> "np.ndarray":
    """Flip k cells of grid to a different color (randomized, seeded)."""
    rng = __import__("numpy").random.default_rng(seed)
    out = grid.copy()
    H, W = out.shape
    idx = rng.choice(H * W, size=min(k, H * W), replace=False)
    for i in idx:
        r, c = divmod(int(i), W)
        out[r, c] = (out[r, c] + 1 + rng.integers(0, 2)) % 3
    return out


def shift_grid(grid, dr: int) -> "np.ndarray":
    """Translate every non-zero cell down by `dr` rows (clipped at edges).
    dr < 0 shifts up. Inverse-direction candidate for the translation rule."""
    np = __import__("numpy")
    H, W = grid.shape
    out = np.zeros_like(grid)
    rows, cols = np.nonzero(grid)
    for r, c in zip(rows, cols):
        nr = r + dr
        if 0 <= nr < H:
            out[nr, c] = grid[r, c]
    return out


def run_config(D: int, n_demos: int, seed: int = 7):
    enc = HENRIVisionEncoder(d_model=D, k_blocks=D // 8, device="cpu")
    codec = qFHRREpistemicCodec(d_model=D, k_bins=256, device="cpu")
    compiler = HolographicTaskFunctorCompiler(codec)

    demos, (X_t, Y_t) = make_object_task(seed=seed, n_demos=n_demos, shift=2)

    # Encode + ring-quantize demos (production chain).
    demo_pairs_ring = []
    for x, y in demos:
        wx, wy = enc.encode_grid(x), enc.encode_grid(y)
        demo_pairs_ring.append((ring_of(wx), ring_of(wy)))
    w_task = compiler.compile_functor(demo_pairs_ring)

    # Test input wave + functor goal.
    w_x_t = enc.encode_grid(X_t)
    ring_x_t = ring_of(w_x_t)
    ring_goal_functor = compiler.single_pass_associative_retrieval(w_task, ring_x_t)
    goal_functor = real_of(ring_goal_functor)

    # Identity goal (production fallback layer 3 semantics: the initial
    # state / prompt wave itself).
    goal_identity = w_x_t

    w_y_t = enc.encode_grid(Y_t)

    # Memorization probe: is the functor algebra SOUND for an exact
    # in-demo pair (X_demo1 -> Y_demo1)? If d_fun(mem_target) ~ 0, the
    # algebra is exact and the failure is GENERALIZATION to novel object
    # placements (ARC-relevant). If d_fun(mem_target) is large, the
    # ring-quantized binding itself is lossy (algebra broken).
    mem_x, mem_y = demos[0]
    w_mem_x = enc.encode_grid(mem_x)
    ring_mem_x = ring_of(w_mem_x)
    ring_goal_mem = compiler.single_pass_associative_retrieval(w_task, ring_mem_x)
    goal_mem = real_of(ring_goal_mem)
    w_mem_y = enc.encode_grid(mem_y)
    d_fun_mem = goal_distance(w_mem_y, goal_mem)
    d_id_mem = goal_distance(w_mem_y, w_mem_x)

    # Transformation-family candidates of the SAME input: the rule is
    # translate-down-2. A task-aware goal must rank the true target first
    # and order wrong-direction / wrong-orientation / recolor / random below.
    np = __import__("numpy")
    Y_wrong_dir = shift_grid(X_t, -2)   # translate UP (opposite rule)
    Y_rot = np.rot90(X_t, k=-1)         # rotate 90 cw (wrong transformation)
    Y_recolor = X_t.copy()
    Y_recolor[Y_recolor == 1] = 4
    Y_recolor[Y_recolor == 2] = 1
    Y_recolor[Y_recolor == 4] = 2
    Y_random = np.random.default_rng(42).integers(0, 3, size=X_t.shape)
    cands = [
        ("target", Y_t, None),
        ("wrong_direction", Y_wrong_dir, None),
        ("rot90", Y_rot, None),
        ("recolor", Y_recolor, None),
        ("random", Y_random, None),
    ]
    sims = [grid_sim(c[1], Y_t) for c in cands]

    # Score every candidate under BOTH goals with the consumer invariant.
    d_functor, d_ident = {}, {}
    for name, grid, _ in cands:
        w_c = enc.encode_grid(grid)
        d_functor[name] = goal_distance(w_c, goal_functor)
        d_ident[name] = goal_distance(w_c, goal_identity)

    rank_functor = 1 + sum(1 for c in cands[1:] if d_functor[c[0]] < d_functor["target"])
    rank_ident = 1 + sum(1 for c in cands[1:] if d_ident[c[0]] < d_ident["target"])

    # Monotone descent: goal_distance vs grid_similarity over candidates.
    sp_functor = spearman([d_functor[c[0]] for c in cands], sims)
    sp_ident = spearman([d_ident[c[0]] for c in cands], sims)

    return {
        "D": D,
        "n_demos": n_demos,
        "d_functor_target": round(d_functor["target"], 4),
        "d_identity_target": round(d_ident["target"], 4),
        "d_functor_memorized": round(d_fun_mem, 4),
        "d_identity_memorized": round(d_id_mem, 4),
        "rank_functor": rank_functor,
        "rank_identity": rank_ident,
        "spearman_functor": round(sp_functor, 3),
        "spearman_identity": round(sp_ident, 3),
        "w_task_dtype": str(w_task.dtype),
        "w_task_shape": list(w_task.shape),
    }


def verdict_for(r: dict) -> str:
    for k in ("d_functor_target", "d_identity_target", "spearman_functor", "spearman_identity"):
        if not math.isfinite(r[k]):
            return "BLOCKED_INFRASTRUCTURE"
    rank_ok = r["rank_functor"] == 1
    better = r["d_functor_target"] < r["d_identity_target"]
    mono = r["spearman_functor"] >= 0.7
    discrim = r["spearman_functor"] > r["spearman_identity"] + 0.3
    if rank_ok and better and mono and discrim:
        return "ACCEPT"
    return "KILL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--n-demos", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.calibrate:
        print(f"{'D':>7} {'demos':>5} {'d_fun':>8} {'d_id':>8} {'memF':>7} "
              f"{'memI':>7} {'rankF':>6} {'rankI':>6} {'spF':>6} {'spI':>6}  verdict")
        for D in (1024, 4096, 16384):
            for n_demos in (1, 2, 3, 4):
                r = run_config(D, n_demos, seed=args.seed)
                print(f"{D:7d} {n_demos:5d} {r['d_functor_target']:8.3f} "
                      f"{r['d_identity_target']:8.3f} {r['d_functor_memorized']:7.3f} "
                      f"{r['d_identity_memorized']:7.3f} {r['rank_functor']:6d} "
                      f"{r['rank_identity']:6d} {r['spearman_functor']:6.2f} "
                      f"{r['spearman_identity']:6.2f}  {verdict_for(r)}")
        return 0

    r = run_config(args.d, args.n_demos, seed=args.seed)
    verdict = verdict_for(r)
    payload = {
        "scope": "P3_CODEC_GEOMETRY_CONTROL (mechanism evidence, not task capability)",
        "device": "cpu",
        "config": {"D": args.d, "n_demos": args.n_demos, "seed": args.seed},
        "verdict": verdict,
        "metrics": r,
        "accept_rule": {
            "rank_functor==1": True,
            "d_functor < d_identity": True,
            "spearman_functor>=0.7": True,
            "spearman_functor > spearman_identity+0.3": True,
        },
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0 if verdict == "ACCEPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
