import json

with open("ARC-AGI-2/data/evaluation/136b0064.json", "r") as f:
    task = json.load(f)

def print_grid(grid, title):
    print(f"--- {title} ---")
    for row in grid:
        print(" ".join(str(cell) if cell != 0 else "." for cell in row))

for idx in range(3):
    print(f"\n================= PAIR {idx+1} =================")
    print_grid(task["train"][idx]["input"], "INPUT")
    print_grid(task["train"][idx]["output"], "OUTPUT")
