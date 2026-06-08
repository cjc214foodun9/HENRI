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
    res = find_block_of_8(pair["input"])
    if not res:
        continue
    r8, c8, h, w = res
    print(f"\nTrain {idx+1}: Block of 8 of shape {h}x{w}")
    
    # Let's search all subgrids of shape h x w.
    # We can try all 8 symmetries of the subgrid (rotations and flips) to see if they match the output.
    for r in range(inp.shape[0] - h + 1):
        for c in range(inp.shape[1] - w + 1):
            if r == r8 and c == c8:
                continue
            sub = inp[r:r+h, c:c+w]
            
            # Check all 8 symmetries if sub and out are square, or if shape matches after rotation
            # If shape is not square, only flips and 180-degree rotations are possible if we keep the same shape,
            # or if transposed shape matches output shape.
            symmetries = []
            # Flips
            symmetries.append(("identity", sub))
            symmetries.append(("flipud", np.flipud(sub)))
            symmetries.append(("fliplr", np.fliplr(sub)))
            symmetries.append(("rot180", np.rot90(sub, 2)))
            
            if sub.shape == out.T.shape:
                symmetries.append(("transpose", sub.T))
                symmetries.append(("rot90", np.rot90(sub)))
                symmetries.append(("rot270", np.rot90(sub, 3)))
                symmetries.append(("transpose_fliplr", np.fliplr(sub.T)))
                
            for name, sym in symmetries:
                if sym.shape == out.shape and np.array_equal(sym, out):
                    print(f"  EXACT MATCH found at ({r}, {c}) with symmetry: {name}")
