"""Phase 8.26 — CEGIS Codebook Snap & Pixel Cleanliness.

HENRI-ANALYSIS-2026-08-SOLVING-FRONTIER (sha 8c508808...), Phase 8.26:
"Pass continuous wave output through Counterexample-Guided Inductive Snap.
Rejects single-pixel noise and enforces discrete grid conservation
invariants."

Design:
- Continuous predicted grids (per-cell color scores) are quantized to a
  discrete color codebook.
- A bounded CEGIS loop (max 3 iterations) finds counterexample cells:
  isolated single-pixel artifacts (a cell whose color differs from all 4
  neighbors and belongs to no >=2-cell same-color run). Each is reverted to
  its dominant neighbor color. The refined grid is the new candidate; the
  loop re-checks until no counterexample remains or the budget is spent.
- Conservation invariants are enforced after snapping: total non-background
  cell count and per-color counts must match the reference grid within
  tolerance. A violation sets `conservation_ok = False` (fail-closed: the
  caller may veto the candidate).

Zero-pretraining invariant: this module consumes only live observation
grids + predicted fields; no task solutions are stored or pre-ingested.
"""

from __future__ import annotations

import torch

import numpy as np


def _neighbors_mask(grid: np.ndarray, color: int) -> np.ndarray:
    """4-neighborhood equality mask per cell (True where a neighbor has
    the same color)."""
    h, w = grid.shape
    same = np.zeros_like(grid, dtype=bool)
    same[:-1, :] |= grid[:-1, :] == grid[1:, :]
    same[1:, :] |= grid[1:, :] == grid[:-1, :]
    same[:, :-1] |= grid[:, :-1] == grid[:, 1:]
    same[:, 1:] |= grid[:, 1:] == grid[:, :-1]
    return same


def _dominant_neighbor_color(grid: np.ndarray, r: int, c: int) -> int:
    """Most common color among the 4-neighborhood (fallback: self)."""
    colors = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            colors.append(int(grid[rr, cc]))
    if not colors:
        return int(grid[r, c])
    return max(set(colors), key=colors.count)


def cegis_grid_snap(
    pred_grid: np.ndarray,
    ref_grid: np.ndarray | None = None,
    max_iters: int = 3,
    conservation_tol: float = 0.05,
    min_count_tol: int = 3,
) -> dict:
    """Quantize a continuous predicted grid and clean it via CEGIS.

    pred_grid: [H, W] float per-cell color scores (continuous; round to int
        color indices). Ref_grid: [H, W] int current observation used for
        conservation invariants (optional).
    Returns dict with 'grid' (cleaned int grid), 'isolated_pixels_removed',
    'conservation_ok', 'iterations_used'.
    """
    cand = np.round(np.asarray(pred_grid)).astype(int)
    removed = 0
    used = 0
    for it in range(max_iters):
        used = it + 1
        # Counterexample scan: isolated pixels (no same-color neighbor).
        same = _neighbors_mask(cand, 0)
        h, w = cand.shape
        isolated = np.zeros_like(cand, dtype=bool)
        for r in range(h):
            for c in range(w):
                color = int(cand[r, c])
                nbrs = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        nbrs.append(int(cand[rr, cc]))
                if nbrs and all(n != color for n in nbrs):
                    isolated[r, c] = True
        if not isolated.any():
            break
        for r, c in zip(*np.where(isolated)):
            cand[r, c] = _dominant_neighbor_color(cand, r, c)
            removed += 1
    # Conservation invariants vs reference.
    conservation_ok = True
    if ref_grid is not None:
        ref = np.asarray(ref_grid).astype(int)
        if ref.shape == cand.shape:
            n_cand = int((cand != 0).sum())
            n_ref = int((ref != 0).sum())
            if abs(n_cand - n_ref) > max(min_count_tol,
                                         conservation_tol * n_ref):
                conservation_ok = False
            for color in np.unique(ref):
                if color == 0:
                    continue
                d_cand = int((cand == color).sum())
                d_ref = int((ref == color).sum())
                if abs(d_cand - d_ref) > max(min_count_tol,
                                             conservation_tol * d_ref):
                    conservation_ok = False
    return {
        "grid": cand,
        "isolated_pixels_removed": removed,
        "conservation_ok": conservation_ok,
        "iterations_used": used,
    }


def _verify_cegis_snap() -> int:
    """Gate G8.26: snap recovers the clean grid on synthetic noise.

    Pre-registered:
    - G8.26a: pixel accuracy >= 0.95 after snap on a grid with 5% injected
      isolated-pixel noise (baseline unsnapped accuracy < 0.98).
    - G8.26b: conservation_ok True on the recovered grid.
    - Falsification: snap does not improve accuracy or breaks conservation
      => default-OFF, unpromoted.
    """
    rng = np.random.default_rng(826)
    # Clean synthetic grid: 4 solid quadrants of colors 1..4 on 16x16.
    clean = np.zeros((16, 16), dtype=int)
    clean[:8, :8] = 1
    clean[:8, 8:] = 2
    clean[8:, :8] = 3
    clean[8:, 8:] = 4
    # Inject 5% isolated single-pixel noise.
    noisy = clean.copy()
    mask = rng.random(clean.shape) < 0.05
    noisy[mask] = rng.integers(1, 10, size=int(mask.sum()))
    # Baseline: raw quantized grid accuracy.
    base_acc = float((noisy == clean).mean())
    res = cegis_grid_snap(noisy.astype(float), ref_grid=clean)
    snap_acc = float((res["grid"] == clean).mean())
    print(f"[verify_cegis_snap] baseline pixel acc: {base_acc:.4f}")
    print(f"[verify_cegis_snap] snapped pixel acc:  {snap_acc:.4f} "
          f"(gate >= 0.95), isolated removed {res['isolated_pixels_removed']}, "
          f"conservation_ok {res['conservation_ok']}")
    assert snap_acc >= 0.95, f"G8.26a FAIL: snapped acc {snap_acc:.4f} < 0.95"
    assert res["conservation_ok"], "G8.26b FAIL: conservation violated"
    assert snap_acc > base_acc, "G8.26 FAIL: snap did not improve accuracy"
    print("[verify_cegis_snap] G8.26 PASS")
    return 0


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--mode", default=None)
    _args = _ap.parse_args()
    if _args.mode == "verify_cegis_snap":
        raise SystemExit(_verify_cegis_snap())
    raise SystemExit(f"unknown --mode {_args.mode!r} "
                     f"(expected verify_cegis_snap)")
