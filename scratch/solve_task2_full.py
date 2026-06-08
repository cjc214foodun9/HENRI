import json
import numpy as np
from collections import Counter

with open("ARC-AGI-2/data/evaluation/135a2760.json", "r") as f:
    task = json.load(f)

def correct_sequence(seq):
    if len(seq) < 4:
        return seq
    best_reconstruction = None
    min_errors = len(seq)
    best_P = -1
    
    for P in [2, 3, 4, 5, 6, 7, 8]:
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
        if errors <= 2 and errors < min_errors:
            min_errors = errors
            best_reconstruction = reconstructed
            best_P = P
            
    if best_P != -1:
        return best_reconstruction
    return seq

def solve_task_2(grid):
    grid_np = np.array(grid)
    H, W = grid_np.shape
    
    # Exclude the most frequent color in the entire grid
    most_freq_color = Counter(grid_np.flatten()).most_common(1)[0][0]
    
    # Evaluate partition score and bounding box area for each color
    best_B = None
    max_area = -1
    
    for color in range(10):
        if color == most_freq_color:
            continue
        coords = np.argwhere(grid_np == color)
        if len(coords) == 0:
            continue
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        active_H = r_max - r_min + 1
        active_W = c_max - c_min + 1
        area = active_H * active_W
        
        r_lines_count = 0
        for r in range(r_min, r_max + 1):
            if grid_np[r, c_min] == color and grid_np[r, c_max] == color:
                if np.sum(grid_np[r, c_min:c_max+1] == color) >= 0.8 * active_W:
                    r_lines_count += 1
                    
        c_lines_count = 0
        for c in range(c_min, c_max + 1):
            if grid_np[r_min, c] == color and grid_np[r_max, c] == color:
                if np.sum(grid_np[r_min:r_max+1, c] == color) >= 0.8 * active_H:
                    c_lines_count += 1
                    
        score = r_lines_count + c_lines_count
        if score >= 3:
            if area > max_area:
                max_area = area
                best_B = color
                
    if best_B is None:
        return grid
        
    B = best_B
    coords = np.argwhere(grid_np == B)
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    active_H = r_max - r_min + 1
    active_W = c_max - c_min + 1
    
    r_lines = [r for r in range(r_min, r_max + 1) if grid_np[r, c_min] == B and grid_np[r, c_max] == B and np.sum(grid_np[r, c_min:c_max+1] == B) >= 0.8 * active_W]
    c_lines = [c for c in range(c_min, c_max + 1) if grid_np[r_min, c] == B and grid_np[r_max, c] == B and np.sum(grid_np[r_min:r_max+1, c] == B) >= 0.8 * active_H]
    
    r_partitions = [r_min - 1] + r_lines + [r_max + 1]
    c_partitions = [c_min - 1] + c_lines + [c_max + 1]
    
    repaired = np.copy(grid_np)
    
    # Loop over cell regions
    for i in range(len(r_partitions) - 1):
        r_start = r_partitions[i] + 1
        r_end = r_partitions[i+1]
        if r_end - r_start < 3:
            continue
        for j in range(len(c_partitions) - 1):
            c_start = c_partitions[j] + 1
            c_end = c_partitions[j+1]
            if c_end - c_start < 3:
                continue
                
            cell = grid_np[r_start:r_end, c_start:c_end]
            IB = Counter(cell.flatten()).most_common(1)[0][0]
            
            # Check if IB is a border color
            has_ib_row = any(np.all(cell[r] == IB) for r in range(cell.shape[0]))
            has_ib_col = any(np.all(cell[:, c] == IB) for c in range(cell.shape[1]))
            is_ib_border = has_ib_row or has_ib_col
            
            # Correct horizontal patterns
            for r_idx in range(r_start, r_end):
                row = repaired[r_idx, c_start:c_end]
                if is_ib_border:
                    ib_cols = np.where(row == IB)[0]
                    if len(ib_cols) >= 2:
                        for k in range(len(ib_cols) - 1):
                            p_start = c_start + ib_cols[k] + 1
                            p_end = c_start + ib_cols[k+1]
                            if p_end - p_start >= 4:
                                seq = repaired[r_idx, p_start:p_end].tolist()
                                corrected = correct_sequence(seq)
                                repaired[r_idx, p_start:p_end] = corrected
                else:
                    seq = row.tolist()
                    corrected = correct_sequence(seq)
                    repaired[r_idx, c_start:c_end] = corrected
                        
            # Correct vertical patterns
            for c_idx in range(c_start, c_end):
                col = repaired[r_start:r_end, c_idx]
                if is_ib_border:
                    ib_rows = np.where(col == IB)[0]
                    if len(ib_rows) >= 2:
                        for k in range(len(ib_rows) - 1):
                            p_start = r_start + ib_rows[k] + 1
                            p_end = r_start + ib_rows[k+1]
                            if p_end - p_start >= 4:
                                seq = repaired[p_start:p_end, c_idx].tolist()
                                corrected = correct_sequence(seq)
                                repaired[p_start:p_end, c_idx] = corrected
                else:
                    seq = col.tolist()
                    corrected = correct_sequence(seq)
                    repaired[r_start:r_end, c_idx] = corrected
                        
    return repaired.tolist()

# Test on training pairs
for idx, pair in enumerate(task["train"]):
    res = solve_task_2(pair["input"])
    print(f"Train {idx+1} Correct:", res == pair["output"])

# Test on test pair
res_test = solve_task_2(task["test"][0]["input"])
print("Test Correct:", res_test == task["test"][0]["output"])
