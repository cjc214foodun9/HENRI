import json
import numpy as np

with open("ARC-AGI-2/data/evaluation/0934a4d8.json", "r") as f:
    task = json.load(f)

for idx, pair in enumerate(task["train"]):
    inp = np.array(pair["input"])
    out = np.array(pair["output"])
    H_in, W_in = inp.shape
    H_out, W_out = out.shape
    print(f"\nTrain Pair {idx+1}:")
    
    # Let's search for the output grid subgrid by matching values, but allowing translation
    # What if the values are directly copied from some region?
    # Let's search for any region in input that has the same colors as output, or similar.
    # We can measure correlation or match percentage.
    for r in range(H_in - H_out + 1):
        for c in range(W_in - W_out + 1):
            sub = inp[r:r+H_out, c:c+W_out]
            match_pixels = np.sum(sub == out)
            if match_pixels > H_out * W_out * 0.5: # More than 50% match
                print(f"  Match at ({r}, {c}): {match_pixels}/{H_out * W_out} pixels match.")
                # Show differences
                diff_mask = (sub != out)
                print("  Subgrid in input:")
                print(sub)
                print("  Expected output:")
                print(out)
