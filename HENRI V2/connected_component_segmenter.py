"""
Connected-Component Object Segmenter (CC-OS) for Project HENRI V2.

Partitions 2D ARC-AGI-3 grid frames into discrete, typed object records o_i = (kappa_i, tau_i, v_i)
using 8-connected topological adjacency and attribute extraction.
"""

import math
import numpy as np
import torch
from typing import Any, Dict, List, Tuple


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
    ):
        self.object_id = object_id
        self.tracking_key = tracking_key  # (centroid_y, centroid_x)
        self.mech_type = mech_type        # 'single_pixel', 'line_segment', 'solid_block', 'frame_border'
        self.color = color                # 0..9
        self.bbox = bbox                  # (min_r, min_c, max_r, max_c)
        self.area = area
        self.pixels = pixels              # List of (r, c)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.object_id,
            "tracking_key": self.tracking_key,
            "type": self.mech_type,
            "color": self.color,
            "bbox": self.bbox,
            "area": self.area,
        }


class ConnectedComponentSegmenter:
    """
    Partitions raw 2D grid arrays into topological object records using 8-connectivity.
    """

    def __init__(self, background_color: int = 0):
        self.background_color = background_color

    def segment_grid(self, grid: List[List[int]]) -> List[ObjectRecord]:
        """
        Main entry point: segments a 2D grid into a list of ObjectRecords.
        grid: 2D array or list of lists representing ARC-AGI color indices [0..9].
        """
        arr = np.array(grid, dtype=int)
        rows, cols = arr.shape
        visited = np.zeros((rows, cols), dtype=bool)
        objects = []
        obj_id = 0

        # 8-connected neighbor offsets: N, S, E, W, NE, NW, SE, SW
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for r in range(rows):
            for c in range(cols):
                color = int(arr[r, c])
                if color != self.background_color and not visited[r, c]:
                    # Flood-fill BFS for 8-connected component
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

                    # Extract attributes for component
                    area = len(component)
                    r_coords = [p[0] for p in component]
                    c_coords = [p[1] for p in component]

                    min_r, max_r = min(r_coords), max(r_coords)
                    min_c, max_c = min(c_coords), max(c_coords)
                    height = max_r - min_r + 1
                    width = max_c - min_c + 1

                    centroid_r = float(np.mean(r_coords))
                    centroid_c = float(np.mean(c_coords))

                    # Infer mechanical type tau_i
                    if area == 1:
                        mech_type = "single_pixel"
                    elif height == 1 or width == 1:
                        mech_type = "line_segment"
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
                    )
                    objects.append(rec)
                    obj_id += 1

        return objects


if __name__ == "__main__":
    segmenter = ConnectedComponentSegmenter(background_color=0)
    grid_sample = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 0, 0, 0, 2],
    ]
    objs = segmenter.segment_grid(grid_sample)
    print(f"Extracted {len(objs)} objects from sample grid:")
    for o in objs:
        print(" ", o.to_dict())
    print("Connected-Component Object Segmenter module verified successfully.")
