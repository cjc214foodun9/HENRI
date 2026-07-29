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
    def compute_parity_contour(grid_shape: Tuple[int, int], contour_pixels: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Calculates parity contour mask via flood-fill interior/exterior classification.
        Returns: (interior_pixels, exterior_pixels)
        """
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
                elif val == 2 and (r, c) not in contour_pixels:
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

    def segment_grid(self, grid: List[List[int]]) -> List[ObjectRecord]:
        """
        Segments a 2D grid into a list of ObjectRecords with parity contour classification.
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
                    interior_px, exterior_px = ParityContourMask.compute_parity_contour((rows, cols), component)

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
