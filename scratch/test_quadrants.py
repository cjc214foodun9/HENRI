import json
import numpy as np

with open("ARC-AGI-2/data/evaluation/0934a4d8.json", "r") as f:
    task = json.load(f)

for idx, pair in enumerate(task["train"]):
    inp = np.array(pair["input"])
    print(f"\nTrain Pair {idx+1}:")
    # Let's split into 15x15 quadrants:
    # Q1: top-left (0..14, 0..14)
    # Q2: top-right (0..14, 15..29)
    # Q3: bottom-left (15..29, 0..14)
    # Q4: bottom-right (15..29, 15..29)
    quads = {
        "Q1": inp[0:15, 0:15],
        "Q2": inp[0:15, 15:30],
        "Q3": inp[15:30, 0:15],
        "Q4": inp[15:30, 15:30]
    }
    for q_name, q in quads.items():
        # Let's check transpose symmetry within each quadrant:
        # Since it's 15x15, transpose is 15x15.
        mask_q = (q == 8)
        valid = ~(mask_q | mask_q.T)
        mismatches = np.sum(q[valid] != q.T[valid])
        print(f"  {q_name}: {mismatches} transpose mismatches")
