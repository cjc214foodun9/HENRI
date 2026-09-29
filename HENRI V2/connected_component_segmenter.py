"""
Connected-Component Object Segmenter (CC-OS) & Topological Masking for Project HENRI V2.

Partitions 2D ARC-AGI-3 grid frames into discrete, typed object records o_i = (kappa_i, tau_i, v_i)
using 8-connected topological adjacency, parity contour masking (winding numbers), and 
Clifford geometric binding operators.
"""

import math
import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional


class ObjectRecord:
    """Discrete, typed object record o_i = (kappa_i, tau_i, v_i)."""

    def __init__(
        self,
        object_id: int,
        tracking_key: Tuple[float, float],
        mech_type: str,
        color: int,
        bbox: Tuple[int, int, int, int],
        area: int,
        pixels: List[Tuple[int, int]],
        interior_pixels: Optional[List[Tuple[int, int]]] = None,
        exterior_pixels: Optional[List[Tuple[int, int]]] = None,
    ):
        self.object_id = object_id
        self.tracking_key = tracking_key  # (centroid_y, centroid_x)
        self.mech_type = mech_type        # 'single_pixel', 'line_segment', 'solid_block', 'frame_border', 'enclosed_contour'
        self.color = color                # 0..9
        self.bbox = bbox                  # (min_r, min_c, max_r, max_c)
        self.area = area
        self.pixels = pixels              # List of (r, c)
        self.interior_pixels = interior_pixels if interior_pixels is not None else []
        self.exterior_pixels = exterior_pixels if exterior_pixels is not None else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.object_id,
            "tracking_key": self.tracking_key,
            "type": self.mech_type,
            "color": self.color,
            "bbox": self.bbox,
            "area": self.area,
            "num_interior": len(self.interior_pixels),
            "num_exterior": len(self.exterior_pixels),
        }


class ParityContourMask:
    """
    Topological Masking Primitive:
    Computes winding numbers and parity flood-fill reflections over 2D object boundaries,
    categorizing pixels into IN (interior) and OUT (exterior) spatial regions.
    """

    @staticmethod
    def compute_parity_contour(grid_shape: Tuple[int, int], contour_pixels: List[Tuple[int, int]], fast: bool = False, want_exterior: bool = True) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Calculates parity contour mask via flood-fill interior/exterior classification.
        Returns: (interior_pixels, exterior_pixels)

        fast=True selects `_parity_contour_fast`, which is algorithmically identical
        but removes the measured dominant cost (numpy scalar indexing).
        want_exterior=False skips building the exterior list, which
        `henri_vision_encoder.encode_grid` never reads. Defaults preserve the
        production path byte-for-byte.
        """
        if fast:
            return ParityContourMask._parity_contour_fast(
                grid_shape[0], grid_shape[1], contour_pixels,
                want_exterior=want_exterior)
        rows, cols = grid_shape
        mask = np.zeros((rows, cols), dtype=int)
        
        # Mark contour boundary as 1
        for r, c in contour_pixels:
            if 0 <= r < rows and 0 <= c < cols:
                mask[r, c] = 1

        # Flood-fill exterior background from (0,0) padded border with 2
        padded = np.zeros((rows + 2, cols + 2), dtype=int)
        padded[1:rows+1, 1:cols+1] = mask
        
        queue = [(0, 0)]
        padded[0, 0] = 2
        while queue:
            curr_r, curr_c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows + 2 and 0 <= nc < cols + 2:
                    if padded[nr, nc] == 0:
                        padded[nr, nc] = 2
                        queue.append((nr, nc))

        # Extract interior (0) and exterior (2)
        interior_pixels = []
        exterior_pixels = []

        for r in range(rows):
            for c in range(cols):
                val = padded[r + 1, c + 1]
                if val == 0:
                    interior_pixels.append((r, c))
                elif want_exterior and val == 2 and (r, c) not in contour_pixels:
                    exterior_pixels.append((r, c))

        return interior_pixels, exterior_pixels

    @staticmethod
    def _parity_contour_fast(rows: int, cols: int, contour_pixels: List[Tuple[int, int]], want_exterior: bool = True) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Identical output to the legacy path; the measured cost removed.

        MEASURED (experiments/verification/seg_cost_probe.py, CPU):
            16x16, 230 components:
              legacy (numpy scalar indexing)  46182.4 us   1.00x
              deque  (my first attempt)       66528.7 us   0.69x  SLOWER
              pylist (this implementation)    26867.6 us   1.72x
              scipy  (ndimage.label floor)    25492.6 us   1.81x
              one 1-px call: legacy 198.83 us -> pylist 113.22 us

        The dominant cost is NUMPY SCALAR INDEXING (`padded[nr, nc]` costs
        ~150-350 ns per access), not `list.pop(0)`. A deque does not help -- it
        was measured 0.69x, i.e. slower, and is not used here.

        `want_exterior=False` skips allocating the exterior list. At 16x16 that
        list holds up to 256 tuples per component and is discarded unread by
        encode_grid, which consumes only `interior_pixels`. Interior output is
        unaffected, so identity holds for the value that is actually consumed.

        Visit order is irrelevant to the result: with the barrier fixed, the set
        of cells reachable from the pad corner is determined by reachability
        alone, so LIFO (this) and FIFO (legacy) mark the same cells. Identity is
        asserted element-by-element in
        experiments/verification/test_fast_segmenter_equiv.py (200/200 fuzz cases).
        """
        H, W = rows + 2, cols + 2
        padded = [[0] * W for _ in range(H)]
        cs = set()
        for r, c in contour_pixels:
            if 0 <= r < rows and 0 <= c < cols:
                padded[r + 1][c + 1] = 1
                cs.add((r, c))

        stack = [(0, 0)]
        padded[0][0] = 2
        while stack:
            cr, cc = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and padded[nr][nc] == 0:
                    padded[nr][nc] = 2
                    stack.append((nr, nc))

        interior_pixels = []
        exterior_pixels = []
        for r in range(rows):
            row = padded[r + 1]
            for c in range(cols):
                v = row[c + 1]
                if v == 0:
                    interior_pixels.append((r, c))
                elif want_exterior and v == 2 and (r, c) not in cs:
                    exterior_pixels.append((r, c))
        return interior_pixels, exterior_pixels

    @staticmethod
    def bind_topological_roles(
        interior_wave: torch.Tensor,
        exterior_wave: torch.Tensor,
        spatial_role_interior: torch.Tensor,
        spatial_role_exterior: torch.Tensor
    ) -> torch.Tensor:
        """
        Binds interior and exterior region waves to spatial role hypervectors
        using FHRR complex frequency-domain phase binding.
        """
        int_bound = torch.fft.ifft(torch.fft.fft(interior_wave, dim=-1) * torch.fft.fft(spatial_role_interior, dim=-1), dim=-1).real
        ext_bound = torch.fft.ifft(torch.fft.fft(exterior_wave, dim=-1) * torch.fft.fft(spatial_role_exterior, dim=-1), dim=-1).real
        
        topological_wave = int_bound + ext_bound
        # Renormalize per block
        norms = torch.norm(topological_wave, dim=-1, keepdim=True).clamp_min(1e-6)
        return topological_wave / norms


class ConnectedComponentSegmenter:
    """
    Partitions raw 2D grid arrays into topological object records using 8-connectivity
    and parity contour masks.
    """

    def __init__(self, background_color: int = 0):
        self.background_color = background_color

    def segment_grid(self, grid: List[List[int]], want_exterior: bool = True, fast: bool = False) -> List[ObjectRecord]:
        """
        Segments a 2D grid into a list of ObjectRecords with parity contour classification.

        want_exterior=False skips building `exterior_pixels` for every record.
        fast=True selects the list-of-lists flood fill (measured 1.72-1.80x, output
        element-for-element identical).

        DEFECT FIXED 2026-09-29: BOTH parameters were previously accepted and then
        DROPPED on the floor -- the call to compute_parity_contour below did not
        forward them, so `want_exterior=False` measured 1.00x (a no-op) and the
        `fast` argument did not exist at all. `_parity_contour_fast` was dead code:
        grep for `fast=True` returned only its own docstring. Both are now
        forwarded, and `experiments/verification/flag_wiring_audit.py` asserts that
        every flag on this class reaches a consumer.

        Defaults preserve the production path byte-for-byte; mech_type
        classification does not read the exterior list (it branches on
        len(interior_px) > 0).
        """
        arr = np.array(grid, dtype=int)
        rows, cols = arr.shape
        visited = np.zeros((rows, cols), dtype=bool)
        objects = []
        obj_id = 0

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for r in range(rows):
            for c in range(cols):
                color = int(arr[r, c])
                if color != self.background_color and not visited[r, c]:
                    component = []
                    queue = [(r, c)]
                    visited[r, c] = True

                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        component.append((curr_r, curr_c))

                        for dr, dc in neighbors:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if not visited[nr, nc] and arr[nr, nc] == color:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))

                    area = len(component)
                    r_coords = [p[0] for p in component]
                    c_coords = [p[1] for p in component]

                    min_r, max_r = min(r_coords), max(r_coords)
                    min_c, max_c = min(c_coords), max(c_coords)
                    height = max_r - min_r + 1
                    width = max_c - min_c + 1

                    centroid_r = float(np.mean(r_coords))
                    centroid_c = float(np.mean(c_coords))

                    # Compute Parity Contour Mask (IN/OUT)
                    interior_px, exterior_px = ParityContourMask.compute_parity_contour(
                        (rows, cols), component,
                        fast=fast, want_exterior=want_exterior)

                    if area == 1:
                        mech_type = "single_pixel"
                    elif height == 1 or width == 1:
                        mech_type = "line_segment"
                    elif len(interior_px) > 0:
                        mech_type = "enclosed_contour"
                    elif (min_r == 0 or max_r == rows - 1) and (min_c == 0 or max_c == cols - 1) and area > 10:
                        mech_type = "frame_border"
                    else:
                        mech_type = "solid_block"

                    rec = ObjectRecord(
                        object_id=obj_id,
                        tracking_key=(centroid_r, centroid_c),
                        mech_type=mech_type,
                        color=color,
                        bbox=(min_r, min_c, max_r, max_c),
                        area=area,
                        pixels=component,
                        interior_pixels=interior_px,
                        exterior_pixels=exterior_px,
                    )
                    objects.append(rec)
                    obj_id += 1

        return objects
