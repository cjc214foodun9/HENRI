"""Connected-component REGION SELECTION for the topological channel (Directive 1).

WHY THIS MODULE EXISTS — the measured failure it must repair
============================================================
The suite receipt (experiments/verification/operator_router_family_suite.json)
records that the topological channel ABSTAINS on all three out-of-family
families (`topo_own_delta = 0.0` on each) and the router therefore falls back to
D4 at delta 0.211808 / 0.114703 / 0.117104. The cause is stated in the channel
itself: it marks ALL background that the border cannot reach, so on two nested
curves it marks the annulus AND the innermost core together and mismatches the
target. Shape is NOT the limit (the non-convex control is exact at
0.9999999999999792); TOPOLOGICAL REGION SELECTION is the limit.

THE RULE CLASS
==============
Instead of one mask, enumerate candidate REGIONS from the input's own geometry:

  interior_all              union of every enclosed background region (legacy)
  region_largest            the single largest enclosed region by area
  region_smallest           the single smallest enclosed region by area
  region_max_curve_colour   the enclosed region bounded by the HIGHEST-coloured curve
  region_min_curve_colour   the enclosed region bounded by the LOWEST-coloured curve

An enclosed region is a 4-connected component of BACKGROUND cells that does NOT
touch the grid border. Its bounding curve is the set of curve cells 4-adjacent to
it, and its colour is the mode of those cells' values.

AMBIGUITY IS DECLINED, NOT GUESSED
==================================
`region_largest` and `region_smallest` are only offered when the extreme area is
STRICTLY extreme. When the two largest areas tie (two same-sized rings), the rule
is OMITTED from the candidate set rather than resolved by an arbitrary tie-break.
That is what forces a relational fixture (`two_rings_select`) onto the colour
rule, and it is honest: "I cannot pick among equal-area regions" is a real answer.

THE PRE-REGISTERED BAR (unchanged, fixed before this code existed)
=================================================================
  out-of-family: pass with mask IoU > 0.5
  in-family:     still >= 0.99
  controls:      non-convex >= 0.99 ; open-curve TOPO-own delta == 0.0 exactly
Selection inside the channel is by DEMO fit with CROSS-DEMO CONSISTENCY, and the
held-out pair validates — so a rule that fits by luck fails the suite.

NO CLAIM
========
Synthetic grids only. Nothing here claims an ARC or SciCode score.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Dict, List, Sequence, Set, Tuple

Cell = Tuple[int, int]

# Candidate rules in PREFERENCE order (simplest / most general first).
RULE_ORDER: Tuple[str, ...] = (
    "interior_all",
    "region_largest",
    "region_smallest",
    "region_max_curve_colour",
    "region_min_curve_colour",
)


def _neighbors(i: int, j: int):
    return ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))


def _components(grid: Sequence[Sequence[int]], want_background: bool,
                bg: Sequence[int]) -> List[Dict[str, object]]:
    """4-connected components of background (or curve) cells."""
    n, m = len(grid), len(grid[0])
    seen = [[False] * m for _ in range(n)]
    out: List[Dict[str, object]] = []
    for i in range(n):
        for j in range(m):
            if seen[i][j]:
                continue
            is_bg = grid[i][j] in bg
            if is_bg != want_background:
                continue
            q = deque([(i, j)])
            seen[i][j] = True
            cells: Set[Cell] = set()
            touches = False
            while q:
                a, b = q.popleft()
                cells.add((a, b))
                if a in (0, n - 1) or b in (0, m - 1):
                    touches = True
                for x, y in _neighbors(a, b):
                    if 0 <= x < n and 0 <= y < m and not seen[x][y] \
                            and (grid[x][y] in bg) == want_background:
                        seen[x][y] = True
                        q.append((x, y))
            out.append({"cells": cells, "area": len(cells), "touches_border": touches})
    return out


def curve_components(grid: Sequence[Sequence[int]], bg: Sequence[int]) -> List[Set[Cell]]:
    return [c["cells"] for c in _components(grid, False, bg)]


def enclosed_regions(grid: Sequence[Sequence[int]], bg: Sequence[int]) -> List[Dict[str, object]]:
    """Background regions the border cannot reach, each with its bounding-curve colour.

    `bounding_colour` is the MODE of the values of curve cells 4-adjacent to the
    region — i.e. the colour of the wall that encloses it. It is None when the
    region has no adjacent curve cell (which would make it unreachable-but-open,
    a malformed fixture).
    """
    n, m = len(grid), len(grid[0])
    regions: List[Dict[str, object]] = []
    for comp in _components(grid, True, bg):
        if comp["touches_border"]:
            continue
        vals: List[int] = []
        for (i, j) in comp["cells"]:                       # type: ignore[union-attr]
            for a, b in _neighbors(i, j):
                if 0 <= a < n and 0 <= b < m and grid[a][b] not in bg:
                    vals.append(grid[a][b])
        comp["bounding_colour"] = Counter(vals).most_common(1)[0][0] if vals else None
        comp["boundary_cells"] = len(vals)
        regions.append(comp)
    regions.sort(key=lambda r: (-int(r["area"]), sorted(r["cells"])[0]))  # type: ignore[arg-type]
    return regions


def region_candidates(grid: Sequence[Sequence[int]],
                      bg: Sequence[int]) -> Dict[str, Set[Cell]]:
    """Every WELL-DEFINED candidate region, keyed by rule name.

    A rule is OMITTED (never guessed) when it is ambiguous:
      * region_largest / region_smallest  -> omitted on an area TIE at the extreme
      * colour rules                      -> omitted with fewer than 1 bounded region,
                                             or when the extremal colour ties
    `interior_all` is present whenever any enclosed region exists.
    """
    regs = enclosed_regions(grid, bg)
    out: Dict[str, Set[Cell]] = {}
    if not regs:
        return out

    all_cells: Set[Cell] = set()
    for r in regs:
        all_cells |= r["cells"]                                   # type: ignore[operator]
    out["interior_all"] = all_cells

    areas = [int(r["area"]) for r in regs]
    desc = sorted(range(len(regs)), key=lambda i: (-areas[i], sorted(regs[i]["cells"])[0]))  # type: ignore[arg-type]
    asc = sorted(range(len(regs)), key=lambda i: (areas[i], sorted(regs[i]["cells"])[0]))    # type: ignore[arg-type]

    if len(areas) == 1 or areas[desc[0]] > areas[desc[1]]:
        out["region_largest"] = set(regs[desc[0]]["cells"])       # type: ignore[arg-type]
    if len(areas) == 1 or areas[asc[0]] < areas[asc[1]]:
        out["region_smallest"] = set(regs[asc[0]]["cells"])       # type: ignore[arg-type]

    bounded = [r for r in regs if r["bounding_colour"] is not None]
    if bounded:
        hi = max(int(r["bounding_colour"]) for r in bounded)      # type: ignore[arg-type]
        lo = min(int(r["bounding_colour"]) for r in bounded)      # type: ignore[arg-type]
        hi_r = [r for r in bounded if int(r["bounding_colour"]) == hi]   # type: ignore[arg-type]
        lo_r = [r for r in bounded if int(r["bounding_colour"]) == lo]   # type: ignore[arg-type]
        if len(hi_r) == 1:
            out["region_max_curve_colour"] = set(hi_r[0]["cells"])       # type: ignore[arg-type]
        if len(lo_r) == 1:
            out["region_min_curve_colour"] = set(lo_r[0]["cells"])       # type: ignore[arg-type]
    return out


def iou(a: Set[Cell], b: Set[Cell]) -> float:
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def describe(grid: Sequence[Sequence[int]], bg: Sequence[int]) -> Dict[str, object]:
    """Compact JSON-safe description, for telemetry and tests."""
    regs = enclosed_regions(grid, bg)
    return {
        "n_enclosed": len(regs),
        "regions": [{"area": int(r["area"]), "bounding_colour": r["bounding_colour"],
                     "touches_border": bool(r["touches_border"])} for r in regs],
        "candidates": sorted(region_candidates(grid, bg)),
        "n_curve_components": len(curve_components(grid, bg)),
    }
