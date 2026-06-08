import json
import numpy as np
from pathlib import Path

def find_block_of_8(grid):
    grid = np.array(grid)
    coords = np.argwhere(grid == 8)
    if len(coords) == 0:
        return None
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    return r_min, c_min, r_max - r_min + 1, c_max - c_min + 1

def analyze_train_pairs(task_dict):
    analysis_text = ""
    for idx, pair in enumerate(task_dict["train"]):
        inp = np.array(pair["input"])
        out = np.array(pair["output"])
        H_in, W_in = inp.shape
        H_out, W_out = out.shape
        
        analysis_text += f"--- Train Pair {idx+1} Analysis ---\n"
        analysis_text += f"- Input Shape: {H_in}x{W_in}, Output Shape: {H_out}x{W_out}\n"
        
        # Look for contiguous block of any color matching the output shape
        found_blocks = []
        for color in range(10):
            coords = np.argwhere(inp == color)
            if len(coords) > 0:
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0)
                h_b, w_b = r_max - r_min + 1, c_max - c_min + 1
                if h_b == H_out and w_b == W_out:
                    found_blocks.append((color, r_min, c_min))
        for color, r_b, c_b in found_blocks:
            analysis_text += f"- Found block/region of color {color} at row {r_b}, col {c_b} matching output shape {H_out}x{W_out}.\n"
            
        # Check subgrid matches
        matches = []
        for r in range(H_in - H_out + 1):
            for c in range(W_in - W_out + 1):
                sub = inp[r:r+H_out, c:c+W_out]
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
                        matches.append((r, c, name))
        for r, c, name in matches:
            analysis_text += f"- Output matches input subgrid at row {r}, col {c} under symmetry '{name}'.\n"
            
        # Pixel differences
        if H_in == H_out and W_in == W_out:
            diffs = []
            for r in range(H_in):
                for c in range(W_in):
                    if inp[r, c] != out[r, c]:
                        diffs.append((r, c, inp[r, c], out[r, c]))
            if len(diffs) > 0 and len(diffs) < 30:
                analysis_text += "- Pixel differences detected:\n"
                for r, c, c_from, c_to in diffs:
                    analysis_text += f"  * Cell at ({r}, {c}) changed color from {c_from} to {c_to}\n"
            elif len(diffs) == 0:
                analysis_text += "- Input and Output grids are identical.\n"
                
    return analysis_text

task_files = sorted(list(Path("ARC-AGI-2/data/evaluation").glob("*.json")))[:3]
for tf in task_files:
    print(f"\n================= TASK: {tf.name} =================")
    with open(tf, "r") as f:
        task_dict = json.load(f)
    print(analyze_train_pairs(task_dict))
