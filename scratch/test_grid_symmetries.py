import json
import numpy as np

with open("ARC-AGI-2/data/evaluation/0934a4d8.json", "r") as f:
    task = json.load(f)

symmetries = {
    "horizontal": lambda r, c: (r, 31 - c),
    "vertical": lambda r, c: (31 - r, c),
    "diagonal": lambda r, c: (c, r),
    "counter_diagonal": lambda r, c: (31 - c, 31 - r)
}

for idx, pair in enumerate(task["train"]):
    inp = np.array(pair["input"])
    mask = (inp == 8)
    H, W = inp.shape
    print(f"\nTrain Pair {idx+1}:")
    for name, sym_fn in symmetries.items():
        mismatches = 0
        valid_count = 0
        for r in range(H):
            for c in range(W):
                if mask[r, c]:
                    continue
                r_mir, c_mir = sym_fn(r, c)
                if 0 <= r_mir < H and 0 <= c_mir < W:
                    if mask[r_mir, c_mir]:
                        continue
                    valid_count += 1
                    if inp[r, c] != inp[r_mir, c_mir]:
                        mismatches += 1
        print(f"  {name}: {mismatches} mismatches (out of {valid_count} valid comparisons)")
