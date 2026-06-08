import json
import numpy as np

with open("ARC-AGI-2/data/evaluation/0934a4d8.json", "r") as f:
    task = json.load(f)

def find_block_of_8(grid):
    grid = np.array(grid)
    coords = np.argwhere(grid == 8)
    if len(coords) == 0:
        return None
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    return r_min, c_min, r_max - r_min + 1, c_max - c_min + 1

for idx, pair in enumerate(task["train"]):
    inp = np.array(pair["input"])
    out = np.array(pair["output"])
    r8, c8, h, w = find_block_of_8(inp)
    
    # Search for EXACT MATCH of output in input (excluding the block of 8 itself)
    for r in range(inp.shape[0] - h + 1):
        for c in range(inp.shape[1] - w + 1):
            if r == r8 and c == c8:
                continue
            sub = inp[r:r+h, c:c+w]
            symmetries = [
                ("identity", sub),
                ("flipud", np.flipud(sub)),
                ("fliplr", np.fliplr(sub)),
                ("rot180", np.rot90(sub, 2)),
            ]
            if sub.shape == out.T.shape:
                symmetries.append(("transpose", sub.T))
                symmetries.append(("rot90", np.rot90(sub)))
                symmetries.append(("rot270", np.rot90(sub, 3)))
            
            for name, sym in symmetries:
                if sym.shape == out.shape and np.array_equal(sym, out):
                    print(f"Pair {idx+1}:")
                    print(f"  Block 8 at: r={r8}, c={c8}, h={h}, w={w}")
                    print(f"  Match at:   r={r}, c={c}, symmetry={name}")
                    print(f"  Row offset: r - r8 = {r - r8}")
                    print(f"  Col offset: c - c8 = {c - c8}")
                    # Let's see mirroring formulas:
                    # If mirrored horizontally:
                    print(f"  c_mirrored (30-c8-w): {30 - c8 - w}")
                    print(f"  r_mirrored (30-r8-h): {30 - r8 - h}")
