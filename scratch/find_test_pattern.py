import json
import numpy as np

with open("ARC-AGI-2/data/evaluation/0934a4d8.json", "r") as f:
    task = json.load(f)

inp = np.array(task["test"][0]["input"])
out = np.array(task["test"][0]["output"])
H_in, W_in = inp.shape
H_out, W_out = out.shape

best_r, best_c, best_score, best_sym_name, best_sym = -1, -1, -1, "", None

for r in range(H_in - H_out + 1):
    for c in range(W_in - W_out + 1):
        sub = inp[r:r+H_out, c:c+W_out]
        
        symmetries = [
            ("identity", sub),
            ("flipud", np.flipud(sub)),
            ("fliplr", np.fliplr(sub)),
            ("rot180", np.rot90(sub, 2)),
        ]
        
        for name, sym in symmetries:
            score = np.sum(sym == out)
            if score > best_score:
                best_score = score
                best_r = r
                best_c = c
                best_sym_name = name
                best_sym = sym

print(f"Best match at ({best_r}, {best_c}) with symmetry {best_sym_name}: score {best_score}/{H_out*W_out}")
print("Subgrid sym:\n", best_sym)
print("Output:\n", out)
