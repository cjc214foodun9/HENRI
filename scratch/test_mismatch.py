import json
import numpy as np
from collections import Counter

with open("ARC-AGI-2/data/evaluation/135a2760.json", "r") as f:
    task = json.load(f)

def correct_sequence(seq):
    best_reconstruction = None
    min_errors = len(seq)
    
    for P in [2, 3, 4, 5]:
        if len(seq) < P * 2:
            continue
        blocks = []
        for i in range(0, len(seq) - P + 1, P):
            blocks.append(tuple(seq[i : i+P]))
            
        if not blocks:
            continue
            
        most_common_block = Counter(blocks).most_common(1)[0][0]
        reconstructed = []
        for i in range(len(seq)):
            reconstructed.append(most_common_block[i % P])
            
        errors = sum(1 for a, b in zip(seq, reconstructed) if a != b)
        if errors < min_errors:
            min_errors = errors
            best_reconstruction = reconstructed
            
    return best_reconstruction

def solve_task_2(grid):
    grid_np = np.array(grid)
    H, W = grid_np.shape
    
    row_colors = []
    for r in range(H):
        counts = Counter(grid_np[r]).most_common(2)
        if counts[0][1] >= W - 2:
            row_colors.append(counts[0][0])
            
    if not row_colors:
        return grid
        
    C = Counter(row_colors).most_common(1)[0][0]
    
    repaired = np.copy(grid_np)
    for r in range(H):
        c_indices = np.where(grid_np[r] == C)[0]
        if len(c_indices) >= 2:
            for i in range(len(c_indices) - 1):
                c_start = c_indices[i] + 1
                c_end = c_indices[i+1]
                if c_end - c_start >= 4:
                    seq = grid_np[r, c_start:c_end].tolist()
                    corrected = correct_sequence(seq)
                    if corrected:
                        repaired[r, c_start:c_end] = corrected
                        
    return repaired.tolist()

pair = task["train"][0]
res = solve_task_2(pair["input"])
out = pair["output"]
print("Correct:", res == out)
if res != out:
    res_np = np.array(res)
    out_np = np.array(out)
    diff = np.where(res_np != out_np)
    print("Diff locations:", diff)
    print("Res values:", res_np[diff])
    print("Out values:", out_np[diff])
