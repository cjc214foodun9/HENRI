import json
import numpy as np
from collections import Counter

with open("ARC-AGI-2/data/evaluation/135a2760.json", "r") as f:
    task = json.load(f)

def correct_sequence(seq, pat_color, bg_color):
    if len(seq) < 4:
        return seq
    best_reconstruction = None
    min_errors = len(seq)
    best_P = -1
    
    # Try periods 2 to 8
    for P in range(2, 9):
        if len(seq) < P * 2:
            continue
        # Extract overlapping blocks and normalize them
        normalized_blocks = []
        for i in range(len(seq) - P + 1):
            block = seq[i : i+P]
            shift = (P - (i % P)) % P
            normalized = block[shift:] + block[:shift]
            normalized_blocks.append(tuple(normalized))
            
        if not normalized_blocks:
            continue
            
        most_common_block = Counter(normalized_blocks).most_common(1)[0][0]
        reconstructed = [most_common_block[i % P] for i in range(len(seq))]
        errors = sum(1 for a, b in zip(seq, reconstructed) if a != b)
        
        # We want the reconstruction to consist of only pattern color and bg color,
        # or at least not introduce random other colors.
        # But since the block is voted from the sequence, it should naturally be clean.
        if errors < min_errors:
            min_errors = errors
            best_reconstruction = reconstructed
            best_P = P
            
    if best_P != -1 and min_errors < len(seq) / 2:
        return best_reconstruction
    return seq

def solve_task_2(grid):
    grid_np = np.array(grid)
    H, W = grid_np.shape
    
    # 1. Find the separator color (most common color in the grid)
    counts = Counter(grid_np.flatten())
    C_sep = counts.most_common(1)[0][0]
    
    # 2. Find separator rows (mostly C_sep)
    sep_rows = []
    for r in range(H):
        if np.sum(grid_np[r] == C_sep) >= W - 1:
            sep_rows.append(r)
    # Ensure boundaries are included
    r_splits = [-1] + sep_rows + [H]
    
    # 3. Find separator columns (mostly C_sep)
    sep_cols = []
    for c in range(W):
        if np.sum(grid_np[:, c] == C_sep) >= H - 1:
            sep_cols.append(c)
    c_splits = [-1] + sep_cols + [W]
    
    repaired = np.copy(grid_np)
    
    # 4. Process each cell in the grid
    for i in range(len(r_splits) - 1):
        r_start = r_splits[i] + 1
        r_end = r_splits[i+1]
        if r_end - r_start < 3: # Cell must have at least inner border top/bottom and 1 interior row
            continue
            
        for j in range(len(c_splits) - 1):
            c_start = c_splits[j] + 1
            c_end = c_splits[j+1]
            if c_end - c_start < 3:
                continue
                
            cell = grid_np[r_start:r_end, c_start:c_end]
            
            # Find inner border color (most common color on cell boundary)
            boundary_elements = []
            boundary_elements.extend(cell[0, :].tolist())
            boundary_elements.extend(cell[-1, :].tolist())
            boundary_elements.extend(cell[:, 0].tolist())
            boundary_elements.extend(cell[:, -1].tolist())
            # Exclude C_sep if it's on the boundary (though boundary should be C_ib)
            boundary_counts = Counter(boundary_elements)
            if C_sep in boundary_counts and len(boundary_counts) > 1:
                del boundary_counts[C_sep]
            C_ib = boundary_counts.most_common(1)[0][0]
            
            # Interior coordinates
            int_r_start = r_start + 1
            int_r_end = r_end - 1
            int_c_start = c_start + 1
            int_c_end = c_end - 1
            
            interior = grid_np[int_r_start:int_r_end, int_c_start:int_c_end]
            H_int, W_int = interior.shape
            
            # Find pattern color (most common color in interior that is not C_sep)
            int_counts = Counter(interior.flatten())
            if C_sep in int_counts:
                del int_counts[C_sep]
            if not int_counts:
                continue # No pattern inside
            C_pat = int_counts.most_common(1)[0][0]
            
            # Determine direction of pattern (horizontal if W_int > H_int, else vertical)
            if W_int > H_int:
                # Correct horizontally
                for r_idx in range(int_r_start, int_r_end):
                    seq = repaired[r_idx, int_c_start:int_c_end].tolist()
                    corrected = correct_sequence(seq, C_pat, C_sep)
                    repaired[r_idx, int_c_start:int_c_end] = corrected
            else:
                # Correct vertically
                for c_idx in range(int_c_start, int_c_end):
                    seq = repaired[int_r_start:int_r_end, c_idx].tolist()
                    corrected = correct_sequence(seq, C_pat, C_sep)
                    repaired[int_r_start:int_r_end, c_idx] = corrected
                    
    return repaired.tolist()

# Test on training pairs
for idx, pair in enumerate(task["train"]):
    res = solve_task_2(pair["input"])
    print(f"Train {idx+1} Correct:", res == pair["output"])

# Test on test pair
res_test = solve_task_2(task["test"][0]["input"])
print("Test Correct:", res_test == task["test"][0]["output"])
