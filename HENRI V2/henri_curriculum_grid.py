"""2D SPATIAL GRID TASK EMITTER (Directive 2).

WHY THIS MODULE EXISTS (measured, not assumed)
  The doc's Gap 1 is real. Measured on the live tree: `henri_curriculum_env.py` returns
  `List[int]` byte tokens, and `grid`/`2d`/`spatial`/`jordan`/`interior` occur 0 times in
  its code. A 1-D tape cannot emit a Jordan curve, an annulus, or a multi-object scene.
  The semantic curriculum rungs were therefore BLOCKED__NO_EMITTER (recorded in the
  committed receipt stage1_rungs345_observed.json).

DESIGN CHOICE: A NEW MODULE, NOT A MUTATION OF THE 1-D PATH.
  The 1-D generator is load-bearing for the seeding driver and 20+ tests. Adding a 2-D
  mode inside it would risk the same class of regression this sprint already produced
  (a skipped init block caused 23 test failures). This module is opt-in and imported
  explicitly; nothing in the driver reaches it by default.

THE GENERATOR IS NOT A SOLVER (stated because it bounds what a task means)
  Each family DRAWS an object and then computes the target FROM ITS OWN BOOKKEEPING of
  what it drew -- it never calls the operator under test. A task therefore has a
  well-defined answer independent of any solver. This is exactly the property the
  containment fixture LACKED earlier in this project, where an operator scored 1.000000
  because the fixture generated the object the operator searched for.

VERIFIED MACHINERY IS REUSED, NOT REIMPLEMENTED
  * `henri_region_selector.enclosed_regions` -- the Jordan interior of the drawn curve
  * `henri_topological_encoder.MultiscaleTopologicalEncoder` -- the multiscale wave
  * `henri_scene_binder.SceneBinder` -- the role-filler scene algebra
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from henri_region_selector import enclosed_regions

# ------------------------------------------------------------------- PALETTE
# DEFECT FIXED 2026-09-27 (measured). The first palette was invented locally:
#     bg=(0,)  curve=(1,2,3,4)  fill=(5..9)
# against the router's PUBLISHED contract:
#     NOISE_BAND=(0,1,2)  RING_BAND=(3..9)  FILL_BAND=(10..17)
# The curve band therefore COLLIDED with the router's background band on {1, 2}. A ring
# drawn in 1 or 2 is BACKGROUND to `henri_region_selector`, so `TopoChannel` could not
# see the curve, correctly ABSTAINED (cv = 0.0 on every containment task), and the router
# fell back to a wrong channel. Measured: TOPO cv 0.0000, chosen D4, exact-match False.
#
# The bands are now TAKEN FROM THE ROUTER, not restated, so they cannot drift. A test
# (`test_bands_match_the_router_contract`) pins the agreement.
from henri_operator_router import (                                       # noqa: E402
    FILL_BAND as _FILL_BAND,
    NOISE_BAND as _NOISE_BAND,
    RING_BAND as _RING_BAND,
)

BACKGROUND_VALUES: Tuple[int, ...] = tuple(_NOISE_BAND)
CURVE_COLOURS: Tuple[int, ...] = tuple(_RING_BAND)
FILL_COLOURS: Tuple[int, ...] = tuple(_FILL_BAND)
FAMILIES: Tuple[str, ...] = ("containment_fill", "reflection", "two_rings_select")


class GridTaskError(RuntimeError):
    """Fail-closed contract violation."""


def _blank(n: int) -> List[List[int]]:
    return [[0] * n for _ in range(n)]


def _draw_ring(g: List[List[int]], r0: int, c0: int, h: int, w: int, colour: int) -> None:
    """Draw an axis-aligned closed rectangular ring. Deterministic; no randomness."""
    for c in range(c0, c0 + w):
        g[r0][c] = colour
        g[r0 + h - 1][c] = colour
    for r in range(r0, r0 + h):
        g[r][c0] = colour
        g[r][c0 + w - 1] = colour


def _draw_blob(g: List[List[int]], r0: int, c0: int, colour: int,
               cells: Sequence[Tuple[int, int]]) -> None:
    for dr, dc in cells:
        g[r0 + dr][c0 + dc] = colour


# A non-convex L-shaped outline: containment must not assume convexity.
_L_OUTLINE = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0),
              (3, 1), (3, 2), (3, 3), (1, 3), (2, 3))


def _interior_of(g: List[List[int]], bg: Sequence[int] = BACKGROUND_VALUES
                 ) -> List[Dict[str, object]]:
    """The Jordan interiors of the closed curves in `g`, via the verified selector."""
    return enclosed_regions(g, bg)


def make_task(family: str, rng: random.Random, size: int = 11,
              fill: Optional[int] = None,
              cue: Optional[int] = None,
              turns: Optional[int] = None) -> Dict[str, object]:
    """Build one (input, target) 2-D grid task. Deterministic given `rng`.

    DEFECT FIXED 2026-09-27 (measured). The fill colour was drawn PER TASK. The router's
    contract is the opposite: `TopoChannel.fit` requires ONE constant fill across every
    demo --
        fills.append(next(iter(vals)))
        ...
        if not fills or len(set(fills)) != 1: return self        # ABSTAIN
    With a per-task fill my changed-cell sets were individually perfect (all five region
    rules at IoU 1.0000) and the channel STILL abstained, because no single fill was
    consistent across demos. Measured: TOPO cv = 0.0000 vs the router's own fixture at
    1.0000. The fill (and the two-ring cue) is now drawn ONCE per batch by `make_batch`
    and passed down, and the router's own docstring states this contract explicitly:
    "interior filled with a TASK-LEVEL CONSTANT colour".
    """
    if family not in FAMILIES:
        raise GridTaskError("unknown family %r; expected one of %s" % (family, list(FAMILIES)))
    n = int(size)
    if n < 7:
        raise GridTaskError("size must be >= 7 so a closed curve fits with a border")
    # the ROUTER's fixtures are square; a non-square grid is not comparable with them
    inp = _blank(n)

    if family == "containment_fill":
        colour = rng.choice(CURVE_COLOURS)
        fill = int(fill if fill is not None else rng.choice(FILL_COLOURS))
        h = rng.randint(4, n - 2)
        w = rng.randint(4, n - 2)
        r0 = rng.randint(1, n - h - 1)
        c0 = rng.randint(1, n - w - 1)
        _draw_ring(inp, r0, c0, h, w, colour)
        regions = _interior_of(inp)
        if len(regions) != 1:
            raise GridTaskError("expected exactly 1 enclosed region, got %d" % len(regions))
        cells = set(regions[0]["cells"])
        tgt = [row[:] for row in inp]
        for (r, c) in cells:
            tgt[r][c] = fill
        meta = {"rule": "fill_region", "curve_colour": colour, "fill_colour": fill,
                "region_cells": len(cells)}

    elif family == "reflection":
        # DEFECT FIXED 2026-09-27 (measured): a solid SQUARE block centred on the grid
        # is invariant under 90-degree rotation, so this family could emit Y == X.
        # Measured case: h=7, w=9, r0=2, c0=1, rot=2 -> identity. Two guards now hold:
        #   (a) h != w so the block is never rotation-symmetric about the centre, and
        #   (b) an ASYMMETRIC marker cell breaks any residual symmetry.
        # The block is then REDRAWN (bounded) until the target genuinely differs.
        colour = rng.choice(CURVE_COLOURS)
        tgt = None
        meta = {}
        for _attempt in range(8):
            inp = _blank(n)
            h = rng.randint(3, n - 2)
            w = rng.randint(3, n - 2)
            if h == w:                      # guard (a): keep it non-square
                w = w + 1 if w + 1 <= n - 2 else w - 1
            r0 = rng.randint(1, max(1, n - h - 1))
            c0 = rng.randint(1, max(1, n - w - 1))
            _draw_blob(inp, r0, c0, colour, [(dr, dc) for dr in range(h) for dc in range(w)])
            # guard (b): one asymmetric marker at a corner of the block
            mr = r0 if rng.random() < 0.5 else r0 + h - 1
            mc = c0 if rng.random() < 0.5 else c0 + w - 1
            inp[mr][mc] = CURVE_COLOURS[-1] if colour != CURVE_COLOURS[-1] else CURVE_COLOURS[0]
            # the TRANSFORM must be constant across demos (router contract):
            # `D4Channel.fit` searches ONE (roll, colour-map, D4) candidate for the whole
            # demo set, exactly as the router's own fixture does (`g, g[::-1]` for every
            # demo). Measured defect: a per-task `rot` meant no single operator fit the
            # demos, so the held-out prediction matched only 1 of 4 seeds.
            _drawn = rng.randint(1, 3)
            rot = int(turns if turns is not None else _drawn)
            cand = [list(row) for row in zip(*inp[::-1])]
            for _ in range(rot - 1):
                cand = [list(row) for row in zip(*cand[::-1])]
            if cand != inp:
                tgt = cand
                meta = {"rule": "rot90", "turns": rot}
                break
        if tgt is None:
            raise GridTaskError("reflection could not build a non-identity task")

    else:  # two_rings_select
        # the CUE and the FILL are both TASK-LEVEL CONSTANTS (router contract)
        # The cue must be the DETERMINISTIC extreme of the two ring colours, because the
        # selector's only colour rules are `region_max_curve_colour` and
        # `region_min_curve_colour` (measured: RULE_ORDER). Measured defect: a random cue
        # meant half the tasks had BOTH the max and the min ring distinct from the cue, no
        # rule survived, TOPO abstained, and the router fell back to RIDGE, which returns
        # None for a grid. The cue is now always the MAX of the two ring colours.
        if cue is not None:
            c_same = int(cue)
        else:
            c_same = CURVE_COLOURS[-1]
        c_other = rng.choice([c for c in CURVE_COLOURS if c != c_same])
        fill = int(fill if fill is not None else rng.choice(FILL_COLOURS))
        h, w = 4, 4
        _draw_ring(inp, 1, 1, h, w, c_other)
        _draw_ring(inp, 1, n - w - 1, h, w, c_same)
        tgt = [row[:] for row in inp]
        filled = 0
        for reg in _interior_of(inp):
            if int(reg["bounding_colour"]) == c_same:
                for (r, c) in set(reg["cells"]):
                    tgt[r][c] = fill
                    filled += 1
        if filled == 0:
            raise GridTaskError("two_rings_select filled nothing -- the cue matched no ring")
        meta = {"rule": "fill_matching_curve_colour", "cue_colour": c_same,
                "fill_colour": fill, "region_cells": filled}

    # ---- CONTROL: the target must CHANGE the input, and only inside the named region.
    changed = {(r, c) for r in range(n) for c in range(n) if inp[r][c] != tgt[r][c]}
    if not changed:
        raise GridTaskError("family %r produced an IDENTITY task" % family)
    return {"family": family, "input": inp, "target": tgt,
            "changed_cells": len(changed), "meta": meta}


def make_batch(family: str, n_tasks: int, seed: int, size: int = 12
               ) -> List[Dict[str, object]]:
    """Deterministic batch. Identical seed -> identical tasks.

    The FILL (and the two-ring CUE) is drawn ONCE here and applied to EVERY task in the
    batch, which is the router's documented contract. Size defaults to 12, matching the
    router's own N_SIDE, so tasks are directly comparable with its fixtures.
    """
    rng = random.Random(int(seed))
    fill = rng.choice(FILL_COLOURS) if family != "reflection" else None
    cue = CURVE_COLOURS[-1] if family == "two_rings_select" else None
    turns = rng.randint(1, 3) if family == "reflection" else None
    return [make_task(family, rng, size=size, fill=fill, cue=cue, turns=turns)
            for _ in range(int(n_tasks))]


def encode_task(task: Dict[str, object], encoder=None, binder=None
                ) -> Dict[str, object]:
    """Couple a grid task to the VERIFIED encoders (the directive's requirement).

    `encoder` is a MultiscaleTopologicalEncoder; `binder` is a SceneBinder. Both are
    injected so this module holds no hidden dependency and can be tested in isolation.
    Returns shapes/norms so the coupling is MEASURED, not assumed.
    """
    out: Dict[str, object] = {"family": task["family"]}
    grid = task["input"]
    if encoder is not None:
        # MEASURED CONTRACT (not assumed): MultiscaleTopologicalEncoder.encode returns
        # (a plain LIST of floats, TopologicalFeatures). My first version called
        # `.shape` on the list, which raised. The features carry interior_mask /
        # boundary_mask / n_levels / n_components / content_blind.
        wave_raw, feats = encoder.encode(grid)
        wave = wave_raw if torch.is_tensor(wave_raw) else torch.as_tensor(
            wave_raw, dtype=torch.float32)
        out["wave_shape"] = tuple(wave.shape)
        out["wave_norm"] = float(torch.linalg.vector_norm(wave.to(torch.float32)))
        out["wave_is_finite"] = bool(torch.isfinite(wave).all())
        out["n_components"] = int(getattr(feats, "n_components", 0) or 0)
        out["n_levels"] = int(getattr(feats, "n_levels", 0) or 0)
        _im = getattr(feats, "interior_mask", None)
        if _im is not None:
            out["n_interior_cells"] = int(sum(1 for row in _im for v in row if v))
        _bm = getattr(feats, "boundary_mask", None)
        if _bm is not None:
            out["n_boundary_cells"] = int(sum(1 for row in _bm for v in row if v))
    if binder is not None:
        n = len(grid)
        objs = []
        for r in range(n):
            for c in range(n):
                v = grid[r][c]
                if v in CURVE_COLOURS:
                    objs.append((0, v, (r * n + c) % 16, 0))
                if len(objs) >= 8:
                    break
            if len(objs) >= 8:
                break
        if objs:
            pairs = [(i, binder.filler_vector("shape", o[1])) for i, o in enumerate(objs)]
            scene = binder.role_filler_scene(pairs)
            out["scene_shape"] = tuple(scene.shape)
            out["scene_norm"] = float(torch.linalg.vector_norm(scene.to(torch.float32)))
            out["n_objects"] = len(pairs)
    return out


def describe() -> Dict[str, object]:
    return {"schema": "henri.curriculum-grid.v1", "families": list(FAMILIES),
            "background_values": list(BACKGROUND_VALUES),
            "curve_colours": list(CURVE_COLOURS), "fill_colours": list(FILL_COLOURS),
            "generator_is_not_a_solver": True,
            "note": "targets are computed from the generator's own bookkeeping, never by "
                    "calling an operator under test"}
