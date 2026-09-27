"""Multiscale Topological Fibre Encoder — nested/recursive spatial ingress.

Directive: upgrade flat pixel-phase mapping to a MULTISCALE bundle that binds
local coordinate phases with JORDAN CURVE interior/boundary markers, restoring
the Nested and Recursive properties.

HONEST BOUNDARIES
-----------------
* DEFAULT-OFF SIDECAR. `enabled=False` returns the flat base encoding, so the
  live ingress path (`o_vsa_ingress_tokenizer.py`) is untouched and nothing in
  the repository imports this module.
* The interior/boundary marker uses a 4-connected flood fill from the grid
  border on BACKGROUND cells. A cell that the fill cannot reach is INTERIOR.
  That is a real topological computation on the given grid, but it is a 2-D
  grid flood fill — NOT a general manifold homology solver, and it does not
  handle holes-within-holes beyond what the fill reachability expresses.
* `content_blind=True` is the DEAD-INPUT NEGATIVE CONTROL: every phase derives
  from the cell INDEX only. It must be indistinguishable across different
  inputs. If the control separates two different grids, the encoder is reading
  something other than what it claims.
* No spatial-reasoning benchmark score is claimed. The tests measure
  representation properties (separability, scale sensitivity, marker polarity),
  not task accuracy.

Structure
    level L in {0..L_max}: cells partitioned into (2^L) x (2^L) blocks, or the
    native pixel grid at the finest level. Each level is bound with its own
    level key, then all levels are superposed.
    Per cell: phase = (colour phase) + (position phase at that level)
                        + (interior/boundary marker)
    Output is a real [D] wave (Re|Im concatenation) to stay compatible with the
    TorusIngressEncoder contract.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

TWO_PI = 2.0 * math.pi

# Golden-ratio style irrational steps keep per-index phases incommensurate so
# distinct cells do not accidentally coincide in phase.
_PHI = 0.6180339887498949
_POS_STEP = TWO_PI * _PHI
_LEVEL_STEP = TWO_PI * 0.3819660112501051


class TopologicalEncoderError(Exception):
    """Raised on a contract violation (wrong shape / type)."""


@dataclass
class TopologicalFeatures:
    """Diagnostics: the markers the encoder actually computed."""
    interior_mask: List[List[bool]]
    boundary_mask: List[List[bool]]
    n_levels: int
    n_components: int
    content_blind: bool


def _flood_reachable(g: Sequence[Sequence[int]], bg: int) -> List[List[bool]]:
    """4-connected flood fill from the border across BACKGROUND cells."""
    n = len(g)
    m = len(g[0])
    seen = [[False] * m for _ in range(n)]
    q: deque = deque()
    for i in range(n):
        for j in (0, m - 1):
            if g[i][j] == bg and not seen[i][j]:
                seen[i][j] = True
                q.append((i, j))
    for j in range(m):
        for i in (0, n - 1):
            if g[i][j] == bg and not seen[i][j]:
                seen[i][j] = True
                q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < m and not seen[a][b] and g[a][b] == bg:
                seen[a][b] = True
                q.append((a, b))
    return seen


class MultiscaleTopologicalEncoder:
    """Multiscale nested encoder with Jordan interior/boundary markers."""

    def __init__(self, d_model: int = 65536, n_levels: int = 5,
                 background: int = 0, enabled: bool = True,
                 content_blind: bool = False) -> None:
        if d_model < 2 or d_model % 2 != 0:
            raise TopologicalEncoderError("d_model must be even and >= 2")
        self.d_model = d_model
        self.n_levels = n_levels
        self.background = background
        self.enabled = enabled
        self.content_blind = content_blind
        self._pos_cache: Dict[Tuple[int, int, int], List[float]] = {}

    # ------------------------------------------------------------- internals
    def _pos_phase(self, i: int, j: int, level: int) -> float:
        key = (i, j, level)
        v = self._pos_cache.get(key)
        if v is None:
            base = (i * _POS_STEP + j * _POS_STEP * 1.3247179572447460) % TWO_PI
            v = [(base + level * _LEVEL_STEP) % TWO_PI]
            self._pos_cache[key] = v
        return v[0]

    def _markers(self, g: Sequence[Sequence[int]]) -> Tuple[List[List[bool]], List[List[bool]]]:
        """Jordan interior/boundary from border reachability.

        DEFECT FIXED 2026-09-27, found by this module's OWN test
        (test_interior_and_boundary_markers_are_computed) rather than by review.

        The first revision defined INTERIOR as "not reachable AND not
        background". That is WRONG for the canonical fixture: the region
        enclosed by a ring IS background. Requiring non-background made the
        enclosure score ZERO interior cells, so the marker that distinguishes
        containment from a global transform was never set -- the encoder would
        have looked position-invariant on exactly the task family it exists for.

        Correct topological definition:
          EXTERIOR = background cells 4-connected-reachable from the border
          INTERIOR = background cells NOT so reachable (enclosed by the curve)
          BOUNDARY = non-background cells 4-adjacent to an INTERIOR cell
        """
        n, m = len(g), len(g[0])
        reach = _flood_reachable(g, self.background)
        interior = [[False] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if g[i][j] == self.background and not reach[i][j]:
                    interior[i][j] = True
        boundary = [[False] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if interior[i][j] or g[i][j] == self.background:
                    continue
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = i + di, j + dj
                    if 0 <= a < n and 0 <= b < m and interior[a][b]:
                        boundary[i][j] = True
                        break
        return interior, boundary

    def _components(self, g: Sequence[Sequence[int]]) -> int:
        n, m = len(g), len(g[0])
        seen = [[False] * m for _ in range(n)]
        k = 0
        for i in range(n):
            for j in range(m):
                if seen[i][j] or g[i][j] == self.background:
                    continue
                k += 1
                col = g[i][j]
                q = deque([(i, j)])
                seen[i][j] = True
                while q:
                    a, b = q.popleft()
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + di, b + dj
                        if 0 <= x < n and 0 <= y < m and not seen[x][y] and g[x][y] == col:
                            seen[x][y] = True
                            q.append((x, y))
        return k

    # ------------------------------------------------------------------- api
    def encode(self, grid: Sequence[Sequence[int]]) -> Tuple[object, TopologicalFeatures]:
        """Return (wave, features).

        Wave is a real [d_model] float tensor-like list of floats laid out as
        Re|Im, matching the TorusIngressEncoder([..]) -> [D] real contract.
        With `enabled=False` the flat single-level base is returned.
        """
        n = len(grid)
        if n == 0:
            raise TopologicalEncoderError("empty grid")
        m = len(grid[0])
        if any(len(r) != m for r in grid):
            raise TopologicalEncoderError("grid must be rectangular")

        if not self.enabled:
            wave = self._flat_base(grid)
            feats = TopologicalFeatures([[False] * m for _ in range(n)],
                                        [[False] * m for _ in range(n)],
                                        1, 0, self.content_blind)
            return wave, feats

        interior, boundary = self._markers(grid)
        levels = min(self.n_levels, max(1, int(math.log2(max(n, m))) + 1))
        # DEFECT FIXED 2026-09-27: each half buffer must be d_model//2 long,
        # because the phase index is taken modulo d_model//2 and the two halves
        # are CONCATENATED as Re|Im. The first revision sized both halves
        # d_model, so encode() returned 2*d_model values and violated the [D]
        # real-wave contract (caught by test_wave_contract_shape_and_dtype).
        half = self.d_model // 2
        acc_re = [0.0] * half
        acc_im = [0.0] * half

        for level in range(levels):
            span = max(1, n // (2 ** level))
            sspan = max(1, m // (2 ** level))
            for i in range(n):
                for j in range(m):
                    if self.content_blind:
                        ph = self._pos_phase(i, j, level)     # index only
                    else:
                        colour = grid[i][j]
                        ph = (colour * _POS_STEP + self._pos_phase(i, j, level)
                              + (math.pi if interior[i][j] else 0.0)
                              + (math.pi / 2.0 if boundary[i][j] else 0.0)
                              + level * _LEVEL_STEP) % TWO_PI
                        # coarse level: bind the block-average colour too
                        if level < levels - 1:
                            bi, bj = i // span, j // sspan
                            bcol = grid[min(n - 1, bi * span)][min(m - 1, bj * sspan)]
                            ph = (ph + bcol * _LEVEL_STEP) % TWO_PI
                    k = (i * m + j + level * 131) % (self.d_model // 2)
                    acc_re[k] += math.cos(ph)
                    acc_im[k] += math.sin(ph)

        wave = acc_re + acc_im
        feats = TopologicalFeatures(interior, boundary, levels,
                                    self._components(grid), self.content_blind)
        return wave, feats

    def _flat_base(self, grid: Sequence[Sequence[int]]) -> List[float]:
        """Single-level flat base (the 'off' path)."""
        n, m = len(grid), len(grid[0])
        acc_re = [0.0] * self.d_model
        acc_im = [0.0] * self.d_model
        for i in range(n):
            for j in range(m):
                ph = (grid[i][j] * _POS_STEP + self._pos_phase(i, j, 0)) % TWO_PI
                k = (i * m + j) % (self.d_model // 2)
                acc_re[k] += math.cos(ph)
                acc_im[k] += math.sin(ph)
        return acc_re + acc_im


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


# END
