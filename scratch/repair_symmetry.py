import json
import numpy as np
from collections import Counter

with open("ARC-AGI-2/data/evaluation/0934a4d8.json", "r") as f:
    task = json.load(f)

inp = np.array(task["test"][0]["input"])
out = np.array(task["test"][0]["output"])
H, W = inp.shape

# Repair grid symmetry
repaired = np.copy(inp)
for r in range(H):
    for c in range(W):
        # The 4 symmetric coordinates
        coords = [
            (r, c),
            (r, 31 - c),
            (31 - r, c),
            (31 - r, 31 - c)
        ]
        colors = []
        for rr, cc in coords:
            if 0 <= rr < H and 0 <= cc < W:
                if inp[rr, cc] != 8:
                    colors.append(inp[rr, cc])
        if len(colors) > 0:
            # Majority vote
            majority_color = Counter(colors).most_common(1)[0][0]
            repaired[r, c] = majority_color

print("Repairs completed.")

# Check the possible mirrored regions in the repaired grid
r8, c8, h, w = 14, 0, 9, 3

# If we mirror left-to-right, the columns are W - c8 - w - 2 = 25..27
c_start = 28 - c8 - w
sub_repaired = repaired[r8 : r8+h, c_start : c_start+w]
sub_repaired_flipped = np.fliplr(sub_repaired)
print("Using c_start = 25:")
print("Matches expected:", np.array_equal(sub_repaired_flipped, out))
print("Repaired flipped sub:\n", sub_repaired_flipped)

# What if we use c_start = 27 (i.e. c = 30 - c8 - w = 27)?
c_start_27 = 30 - c8 - w
sub_repaired_27 = repaired[r8 : r8+h, c_start_27 : c_start_27+w]
sub_repaired_flipped_27 = np.fliplr(sub_repaired_27)
print("\nUsing c_start = 27:")
print("Matches expected:", np.array_equal(sub_repaired_flipped_27, out))
print("Repaired flipped sub:\n", sub_repaired_flipped_27)
