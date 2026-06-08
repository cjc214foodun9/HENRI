import json
import numpy as np

with open("ARC-AGI-2/data/evaluation/136b0064.json", "r") as f:
    task = json.load(f)

# Helper to classify shape and return its length
def get_shape_length(block):
    # block is 3x3
    count = np.sum(block > 0)
    if count == 7:
        return 2
    elif count == 4:
        return 2
    elif count == 5:
        return 3
    elif count == 6:
        return 4
    else:
        return 0

def solve(pair):
    inp = np.array(pair["input"])
    H, W = inp.shape
    
    # 1. Find columns and rows of separators
    # Separator is 4. Usually col 7 is 4.
    sep_col = 7
    left = inp[:, :sep_col]
    right = inp[:, sep_col+1:]
    
    # Find position of 5 in the right half
    r5, c5 = np.where(right == 5)
    start_col = c5[0]
    
    # Find blocks in left half
    left_row_sums = np.sum(left, axis=1)
    non_zero_rows = np.where(left_row_sums > 0)[0]
    
    blocks = []
    if len(non_zero_rows) > 0:
        curr_block = [non_zero_rows[0]]
        for r in non_zero_rows[1:]:
            if r == curr_block[-1] + 1:
                curr_block.append(r)
            else:
                blocks.append(curr_block)
                curr_block = [r]
        blocks.append(curr_block)
        
    # Extract segments
    left_segments = []
    right_segments = []
    
    for b_rows in blocks:
        block_data = left[b_rows, :]
        col_0_2 = block_data[:, :3]
        col_4_6 = block_data[:, 4:7]
        
        c1 = np.unique(col_0_2)
        c1 = c1[c1 != 0]
        if len(c1) > 0:
            left_segments.append((c1[0], get_shape_length(col_0_2)))
            
        c2 = np.unique(col_4_6)
        c2 = c2[c2 != 0]
        if len(c2) > 0:
            right_segments.append((c2[0], get_shape_length(col_4_6)))
            
    segments = left_segments + right_segments
    print("Segments:", segments)
    
    # Backtracking search to find all valid paths
    paths = []
    
    def search(seg_idx, r_last, c_last, current_path):
        if seg_idx == len(segments):
            paths.append(current_path)
            return
            
        color, length = segments[seg_idx]
        r_start = r_last + 1
        c_start = c_last
        
        if r_start >= H:
            return
            
        # Try Horizontal Right
        if c_start + length - 1 < 7:
            coords = [(r_start, c_start + k) for k in range(length)]
            search(seg_idx + 1, r_start, c_start + length - 1, current_path + [(color, coords)])
            
        # Try Horizontal Left
        if c_start - length + 1 >= 0:
            coords = [(r_start, c_start - k) for k in range(length)]
            search(seg_idx + 1, r_start, c_start - length + 1, current_path + [(color, coords)])
            
        # Try Vertical Down
        coords = [(r_start + k, c_start) for k in range(length)]
        # Ensure it doesn't exceed grid height
        if r_start + length - 1 < H:
            search(seg_idx + 1, r_start + length - 1, c_start, current_path + [(color, coords)])

    search(0, 0, start_col, [])
    print(f"Number of valid paths: {len(paths)}")
    
    # Let's check if the correct output is in the paths
    out = np.array(pair["output"])
    correct_path = None
    for p in paths:
        # Reconstruct grid from path
        grid = np.zeros_like(out)
        grid[0, start_col] = 5
        for color, coords in p:
            for r, c in coords:
                grid[r, c] = color
        if np.array_equal(grid, out):
            correct_path = p
            break
    print("Correct path found in search:", correct_path is not None)

for idx, pair in enumerate(task["train"]):
    print(f"\n--- Train {idx+1} ---")
    solve(pair)
