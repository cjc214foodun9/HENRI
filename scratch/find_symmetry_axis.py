import json
import numpy as np

with open("ARC-AGI-2/data/evaluation/0934a4d8.json", "r") as f:
    task = json.load(f)

inp = np.array(task["train"][0]["input"])
H, W = inp.shape
mask = (inp == 8)

# Let's search for horizontal reflection symmetry around column index 'c_sym' (can be integer or half-integer)
# x' = 2 * c_sym - x
# We test all c_sym from 0 to 29 (in steps of 0.5)
best_c, min_c_mismatches = -1, H * W
for c_sym in np.arange(0, W, 0.5):
    mismatches = 0
    valid_count = 0
    for r in range(H):
        for c in range(W):
            if mask[r, c]:
                continue
            c_mir = int(2 * c_sym - c)
            if 0 <= c_mir < W:
                if mask[r, c_mir]:
                    continue
                valid_count += 1
                if inp[r, c] != inp[r, c_mir]:
                    mismatches += 1
    if valid_count > 100 and mismatches < min_c_mismatches:
        min_c_mismatches = mismatches
        best_c = c_sym

print(f"Best horizontal symmetry axis: c={best_c} with {min_c_mismatches} mismatches")

# Vertical reflection symmetry around row index 'r_sym'
best_r, min_r_mismatches = -1, H * W
for r_sym in np.arange(0, H, 0.5):
    mismatches = 0
    valid_count = 0
    for r in range(H):
        for c in range(W):
            if mask[r, c]:
                continue
            r_mir = int(2 * r_sym - r)
            if 0 <= r_mir < H:
                if mask[r_mir, c]:
                    continue
                valid_count += 1
                if inp[r, c] != inp[r_mir, c]:
                    mismatches += 1
    if valid_count > 100 and mismatches < min_r_mismatches:
        min_r_mismatches = mismatches
        best_r = r_sym

print(f"Best vertical symmetry axis: r={best_r} with {min_r_mismatches} mismatches")

# Let's also check if the output block itself is symmetric.
# The output block has size 9x4.
out = np.array(task["train"][0]["output"])
print("Output symmetries:")
for name, sym in [("id", out), ("flipud", np.flipud(out)), ("fliplr", np.fliplr(out)), ("rot180", np.rot90(out, 2))]:
    print(f"  {name} matches output: {np.array_equal(sym, out)}")
