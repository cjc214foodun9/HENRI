import json
import numpy as np
from collections import Counter

with open("ARC-AGI-2/data/evaluation/135a2760.json", "r") as f:
    task = json.load(f)

def analyze(grid, name):
    grid_np = np.array(grid)
    H, W = grid_np.shape
    print(f"\n--- Analyzing {name} ({H}x{W}) ---")
    
    # Let's count colors
    counts = Counter(grid_np.flatten())
    print("Color counts:", counts.most_common())
    
    # For each color, check if it forms a rectangular grid or borders
    # Let's look at the structure of rows and cols containing each color
    for color, count in counts.items():
        coords = np.argwhere(grid_np == color)
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        print(f"Color {color}: count={count}, bbox=[{r_min}:{r_max}, {c_min}:{c_max}] ({(r_max-r_min+1)}x{(c_max-c_min+1)})")

print("TRAIN 1:")
analyze(task["train"][0]["input"], "Train 1")
print("\nTEST:")
analyze(task["test"][0]["input"], "Test")
