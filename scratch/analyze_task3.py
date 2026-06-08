import json
import numpy as np

with open("ARC-AGI-2/data/evaluation/136b0064.json", "r") as f:
    task = json.load(f)

for idx, pair in enumerate(task["train"]):
    print(f"\n=== Train {idx+1} ===")
    inp = np.array(pair["input"])
    out = np.array(pair["output"])
    
    # Left shapes: columns 0..6
    # Right shapes: columns 8..14
    # Column 7 is separator 4
    left = inp[:, :7]
    right = inp[:, 8:]
    
    print("Left colors:", np.unique(left))
    print("Right colors:", np.unique(right))
    print("Output colors:", np.unique(out))
    
    # Let's extract the non-zero coordinates of each color in output
    out_colors = np.unique(out)
    for c in out_colors:
        if c == 0 or c == 5:
            continue
        coords = np.argwhere(out == c)
        print(f"Output Color {c}: coords={coords.tolist()}")
        
    # Let's find the 3x3 blocks in left
    # Blocks are separated by rows of all 0s (excluding separator column)
    # Let's find rows where left has only 0s
    left_row_sums = np.sum(left, axis=1)
    non_zero_rows = np.where(left_row_sums > 0)[0]
    print("Non-zero rows in left:", non_zero_rows.tolist())
    
    # Group consecutive non-zero rows into blocks
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
        
    for b_idx, b_rows in enumerate(blocks):
        print(f"Block {b_idx}: rows {b_rows}")
        block_data = left[b_rows, :]
        # Check col 0..2 and col 4..6
        col_0_2 = block_data[:, :3]
        col_4_6 = block_data[:, 4:7]
        
        c1 = np.unique(col_0_2)
        c1 = c1[c1 != 0]
        if len(c1) > 0:
            print(f"  Col 0..2 Color {c1[0]}:")
            print(col_0_2)
        c2 = np.unique(col_4_6)
        c2 = c2[c2 != 0]
        if len(c2) > 0:
            print(f"  Col 4..6 Color {c2[0]}:")
            print(col_4_6)
