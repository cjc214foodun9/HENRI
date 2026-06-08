import json
import numpy as np

with open("ARC-AGI-2/data/evaluation/0934a4d8.json", "r") as f:
    task = json.load(f)

def solve_task_1(grid):
    grid_np = np.array(grid)
    H, W = grid_np.shape
    
    # 1. Find block of 8
    coords = np.argwhere(grid_np == 8)
    if len(coords) == 0:
        return grid
    r8, c8 = coords.min(axis=0)
    r8_max, c8_max = coords.max(axis=0)
    h = r8_max - r8 + 1
    w = c8_max - c8 + 1
    
    # We want to find a candidate of shape (h, w) to replace grid_np[r8:r8+h, c8:c8+w]
    # We will test all possible subgrids in the original grid
    best_candidate = None
    min_mismatches = 999999
    
    # Generate candidates with shape (h, w)
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            # Skip regions that overlap with the block of 8
            if not (r + h <= r8 or r >= r8 + h or c + w <= c8 or c >= c8 + w):
                continue
            sub = grid_np[r : r+h, c : c+w]
            # Symmetries
            symmetries = [
                sub,
                np.flipud(sub),
                np.fliplr(sub),
                np.rot90(sub, 2)
            ]
            for cand in symmetries:
                # Test candidate
                filled = np.copy(grid_np)
                filled[r8 : r8+h, c8 : c8+w] = cand
                
                # Calculate mismatches under horizontal (31-c) and vertical (31-r) reflections
                h_mismatches = sum(filled[r_idx, c_idx] != filled[r_idx, 31 - c_idx] for r_idx in range(H) for c_idx in range(W) if 0 <= 31 - c_idx < W)
                v_mismatches = sum(filled[r_idx, c_idx] != filled[31 - r_idx, c_idx] for r_idx in range(H) for c_idx in range(W) if 0 <= 31 - r_idx < H)
                total = h_mismatches + v_mismatches
                if total < min_mismatches:
                    min_mismatches = total
                    best_candidate = cand
                    
    # Generate candidates with shape (w, h) and transpose
    for r in range(H - w + 1):
        for c in range(W - h + 1):
            if not (r + w <= r8 or r >= r8 + h or c + h <= c8 or c >= c8 + w):
                continue
            sub = grid_np[r : r+w, c : c+h]
            symmetries = [
                sub.T,
                np.flipud(sub).T,
                np.fliplr(sub).T,
                np.rot90(sub, 2).T
            ]
            for cand in symmetries:
                filled = np.copy(grid_np)
                filled[r8 : r8+h, c8 : c8+w] = cand
                
                h_mismatches = sum(filled[r_idx, c_idx] != filled[r_idx, 31 - c_idx] for r_idx in range(H) for c_idx in range(W) if 0 <= 31 - c_idx < W)
                v_mismatches = sum(filled[r_idx, c_idx] != filled[31 - r_idx, c_idx] for r_idx in range(H) for c_idx in range(W) if 0 <= 31 - r_idx < H)
                total = h_mismatches + v_mismatches
                if total < min_mismatches:
                    min_mismatches = total
                    best_candidate = cand
                    
    print(f"Solved with {min_mismatches} mismatches.")
    return best_candidate.tolist()

# Test on training pairs
for idx, pair in enumerate(task["train"]):
    res = solve_task_1(pair["input"])
    print(f"Train {idx+1} Correct:", res == pair["output"])

# Test on test pair
res_test = solve_task_1(task["test"][0]["input"])
print("Test Correct:", res_test == task["test"][0]["output"])
