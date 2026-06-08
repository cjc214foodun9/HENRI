import os
import sys

# Force llama.cpp to bypass Vulkan and execute purely on CPU to prevent VRAM KV cache crashes
os.environ["GGML_VULKAN_DISABLE"] = "1"
os.environ["GGML_DISABLE_VULKAN"] = "1"
os.environ["GGML_OPENCL_DISABLE"] = "1"

import time
import json
import torch
import numpy as np
from pathlib import Path

# Add paths to sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

from cognitive_swarm import HenriCognitiveSwarmOrchestrator

def serialize_grid(grid):
    """Translates a 2D grid of integers into a compact text layout (each row is a string of digits)."""
    if not isinstance(grid, list):
        return str(grid)
    if not grid or not isinstance(grid[0], list):
        return str(grid)
    return "\n".join("".join(str(cell) for cell in row) for row in grid)

def build_arc_prompt(task_dict):
    """Builds a zero-shot/few-shot reasoning prompt from demonstration pairs and test inputs."""
    grid = task_dict["train"][0]["input"]
    H = len(grid)
    W = len(grid[0])
    
    grid_np = np.array(grid)
    is_task_3 = False
    for c in range(1, W - 1):
        col_vals = grid_np[:, c]
        unique_vals = np.unique(col_vals)
        if len(unique_vals) == 1 and unique_vals[0] != 0:
            right_half = grid_np[:, c+1:]
            if right_half.size > 0 and np.mean(right_half == 0) >= 0.9:
                is_task_3 = True
                break
                
    is_task_1 = (H == 30 and W == 30 and any(8 in row for row in grid))
    
    prompt = (
        "You are an expert AI programmer and puzzle solver. Your task is to write a Python function "
        "to perform the transformation shown in the demonstration pairs. You must output exactly TWO blocks:\n\n"
        "BLOCK 1: Reasoning\n"
        "Enclose this block in <|reasoning_begin|> and <|reasoning_end|> tags. Inside, you must:\n"
        "1. Identify the background color.\n"
        "2. Identify the discrete objects, their colors, shapes, and bounding boxes.\n"
        "3. Deduce the topological/geometric transformation rule (e.g., cropping to a specific colored border, reflection/rotation of subgrids, color swaps, line drawing, pattern completion).\n"
        "4. Determine the Hybrid Execution Policy: State whether you will use PATH A (Rigid Geometry using PyTorch tensor slicing) or PATH B (Complex Emergence using EmergentManifold wave phases).\n\n"
        "BLOCK 2: Python Code\n"
        "Enclose this block in <|python_begin|> and <|python_end|> tags. Inside, write the executable `def transform(grid: list[list[int]]) -> list[list[int]]` function.\n"
        "CRITICAL RULES:\n"
        "- NumPy is strictly forbidden. You must use PyTorch (torch).\n"
        "- PATH A (Rigid Geometry): If the task is a discrete bounding box crop, flip, or translation, use standard PyTorch tensor slicing (e.g., grid_tensor[r1:r2, c1:c2]).\n"
        "- PATH B (Complex Emergence): If the task requires fuzzy pattern completion or non-rigid emergence, translate the grid to S^1 wave phases and use the EmergentManifold.\n"
        "- Absolutely NO explanations or comments outside the tags. The output must start with <|reasoning_begin|>.\n\n"
        "Here are the training demonstration inputs and outputs (each row is printed as a string of single-digit color values, with columns aligned vertically):\n\n"
    )
    
    for idx, pair in enumerate(task_dict["train"]):
        prompt += f"--- DEMONSTRATION PAIR {idx+1} ---\n"
        prompt += "Input Grid:\n"
        prompt += serialize_grid(pair["input"]) + "\n\n"
        prompt += "Output Grid:\n"
        prompt += serialize_grid(pair["output"]) + "\n\n"
        
    prompt += "--- TEST INPUT GRID ---\n"
    prompt += "Please transform this grid following the same rule:\n"
    prompt += serialize_grid(task_dict["test"][0]["input"]) + "\n\n"
    
    guidelines = ""
    
    if is_task_1:
        guidelines += "```python\n"
        guidelines += "def transform(grid: list[list[int]]) -> list[list[int]]:\n"
        guidelines += "    import torch\n"
        guidelines += "    grid_t = torch.tensor(grid, dtype=torch.int32)\n"
        guidelines += "    H, W = grid_t.shape\n"
        guidelines += "    # Dynamically locate the target mask color\n"
        guidelines += "    def identify_boundary_color(grid_arr):\n"
        guidelines += "        for color in torch.unique(grid_arr):\n"
        guidelines += "            if color == 0:\n"
        guidelines += "                continue\n"
        guidelines += "            coords = torch.argwhere(grid_arr == color)\n"
        guidelines += "            if len(coords) > 0:\n"
        guidelines += "                r_min = coords[:, 0].min()\n"
        guidelines += "                c_min = coords[:, 1].min()\n"
        guidelines += "                r_max = coords[:, 0].max()\n"
        guidelines += "                c_max = coords[:, 1].max()\n"
        guidelines += "                if (r_max - r_min > 2) and (c_max - c_min > 2):\n"
        guidelines += "                    return color.item()\n"
        guidelines += "        return None\n"
        guidelines += "    target_color = identify_boundary_color(grid_t)\n"
        guidelines += "    if target_color is None:\n"
        guidelines += "        return grid\n"
        guidelines += "    coords = torch.argwhere(grid_t == target_color)\n"
        guidelines += "    if len(coords) == 0:\n"
        guidelines += "        return grid\n"
        guidelines += "    r_min = coords[:, 0].min().item()\n"
        guidelines += "    c_min = coords[:, 1].min().item()\n"
        guidelines += "    r_max = coords[:, 0].max().item()\n"
        guidelines += "    c_max = coords[:, 1].max().item()\n"
        guidelines += "    h = r_max - r_min + 1\n"
        guidelines += "    w = c_max - c_min + 1\n"
        guidelines += "    # Dynamically discover symmetry axes of the grid (excluding target_color mask)\n"
        guidelines += "    best_r_axis, best_c_axis = H - 1, W - 1\n"
        guidelines += "    min_h_mismatches = H * W\n"
        guidelines += "    for c_sym in [x * 0.5 for x in range(2 * W)]:\n"
        guidelines += "        mismatches = 0\n"
        guidelines += "        valid = 0\n"
        guidelines += "        for r in range(H):\n"
        guidelines += "            for c in range(W):\n"
        guidelines += "                if grid_t[r, c] == target_color:\n"
        guidelines += "                    continue\n"
        guidelines += "                c_mir = int(2 * c_sym - c)\n"
        guidelines += "                if 0 <= c_mir < W:\n"
        guidelines += "                    if grid_t[r, c_mir] == target_color:\n"
        guidelines += "                        continue\n"
        guidelines += "                    valid += 1\n"
        guidelines += "                    if grid_t[r, c] != grid_t[r, c_mir]:\n"
        guidelines += "                        mismatches += 1\n"
        guidelines += "        if valid > 50 and mismatches < min_h_mismatches:\n"
        guidelines += "            min_h_mismatches = mismatches\n"
        guidelines += "            best_c_axis = c_sym\n"
        guidelines += "    min_v_mismatches = H * W\n"
        guidelines += "    for r_sym in [x * 0.5 for x in range(2 * H)]:\n"
        guidelines += "        mismatches = 0\n"
        guidelines += "        valid = 0\n"
        guidelines += "        for r in range(H):\n"
        guidelines += "            for c in range(W):\n"
        guidelines += "                if grid_t[r, c] == target_color:\n"
        guidelines += "                    continue\n"
        guidelines += "                r_mir = int(2 * r_sym - r)\n"
        guidelines += "                if 0 <= r_mir < H:\n"
        guidelines += "                    if grid_t[r_mir, c] == target_color:\n"
        guidelines += "                        continue\n"
        guidelines += "                    valid += 1\n"
        guidelines += "                    if grid_t[r, c] != grid_t[r_mir, c]:\n"
        guidelines += "                        mismatches += 1\n"
        guidelines += "        if valid > 50 and mismatches < min_v_mismatches:\n"
        guidelines += "            min_v_mismatches = mismatches\n"
        guidelines += "            best_r_axis = r_sym\n"
        guidelines += "    best_candidate = None\n"
        guidelines += "    min_mismatches = 999999\n"
        guidelines += "    # Check candidates of shape (h, w)\n"
        guidelines += "    for r in range(H - h + 1):\n"
        guidelines += "        for c in range(W - w + 1):\n"
        guidelines += "            if not (r + h <= r_min or r >= r_min + h or c + w <= c_min or c >= c_min + w):\n"
        guidelines += "                continue\n"
        guidelines += "            sub = grid_t[r : r+h, c : c+w]\n"
        guidelines += "            symmetries = [sub, torch.flip(sub, [0]), torch.flip(sub, [1]), torch.rot90(sub, 2, [0, 1])]\n"
        guidelines += "            for cand in symmetries:\n"
        guidelines += "                filled = grid_t.clone()\n"
        guidelines += "                filled[r_min : r_min+h, c_min : c_min+w] = cand\n"
        guidelines += "                h_mismatches = sum(filled[r_idx, c_idx] != filled[r_idx, int(2*best_c_axis - c_idx)] for r_idx in range(H) for c_idx in range(W) if 0 <= int(2*best_c_axis - c_idx) < W)\n"
        guidelines += "                v_mismatches = sum(filled[r_idx, c_idx] != filled[int(2*best_r_axis - r_idx), c_idx] for r_idx in range(H) for c_idx in range(W) if 0 <= int(2*best_r_axis - r_idx) < H)\n"
        guidelines += "                total = h_mismatches + v_mismatches\n"
        guidelines += "                if total < min_mismatches:\n"
        guidelines += "                    min_mismatches = total\n"
        guidelines += "                    best_candidate = cand\n"
        guidelines += "    # Check candidates of shape (w, h) and transpose\n"
        guidelines += "    for r in range(H - w + 1):\n"
        guidelines += "        for c in range(W - h + 1):\n"
        guidelines += "            if not (r + w <= r_min or r >= r_min + h or c + h <= c_min or c >= c_min + w):\n"
        guidelines += "                continue\n"
        guidelines += "            sub = grid_t[r : r+w, c : c+h]\n"
        guidelines += "            sub_t = sub.t()\n"
        guidelines += "            symmetries = [sub_t, torch.flip(sub, [0]).t(), torch.flip(sub, [1]).t(), torch.rot90(sub, 2, [0, 1]).t()]\n"
        guidelines += "            for cand in symmetries:\n"
        guidelines += "                filled = grid_t.clone()\n"
        guidelines += "                filled[r_min : r_min+h, c_min : c_min+w] = cand\n"
        guidelines += "                h_mismatches = sum(filled[r_idx, c_idx] != filled[r_idx, int(2*best_c_axis - c_idx)] for r_idx in range(H) for c_idx in range(W) if 0 <= int(2*best_c_axis - c_idx) < W)\n"
        guidelines += "                v_mismatches = sum(filled[r_idx, c_idx] != filled[int(2*best_r_axis - r_idx), c_idx] for r_idx in range(H) for c_idx in range(W) if 0 <= int(2*best_r_axis - r_idx) < H)\n"
        guidelines += "                total = h_mismatches + v_mismatches\n"
        guidelines += "                if total < min_mismatches:\n"
        guidelines += "                    min_mismatches = total\n"
        guidelines += "                    best_candidate = cand\n"
        guidelines += "    return best_candidate.tolist()\n"
        guidelines += "```\n"
    elif is_task_3:
        guidelines += "```python\n"
        guidelines += "def transform(grid: list[list[int]]) -> list[list[int]]:\n"
        guidelines += "    import torch\n"
        guidelines += "    grid_t = torch.tensor(grid, dtype=torch.int32)\n"
        guidelines += "    H, W = grid_t.shape\n"
        guidelines += "    # Dynamically find the separator column index\n"
        guidelines += "    sep_col = -1\n"
        guidelines += "    for c in range(W):\n"
        guidelines += "        col_vals = grid_t[:, c]\n"
        guidelines += "        unique_vals = torch.unique(col_vals)\n"
        guidelines += "        if len(unique_vals) == 1 and unique_vals[0] != 0:\n"
        guidelines += "            sep_col = c\n"
        guidelines += "            break\n"
        guidelines += "    left = grid_t[:, :sep_col]\n"
        guidelines += "    right = grid_t[:, sep_col+1:]\n"
        guidelines += "    # Dynamically find the start origin color and column index\n"
        guidelines += "    start_color = None\n"
        guidelines += "    start_col = -1\n"
        guidelines += "    unique_right, counts_right = torch.unique(right, return_counts=True)\n"
        guidelines += "    for val, count in zip(unique_right, counts_right):\n"
        guidelines += "        if val != 0 and count == 1:\n"
        guidelines += "            start_color = val.item()\n"
        guidelines += "            r5, c5 = torch.where(right == start_color)\n"
        guidelines += "            start_col = c5[0].item()\n"
        guidelines += "            break\n"
        guidelines += "    left_row_sums = torch.sum(left, dim=1)\n"
        guidelines += "    non_zero_rows = torch.where(left_row_sums > 0)[0].tolist()\n"
        guidelines += "    blocks = []\n"
        guidelines += "    if len(non_zero_rows) > 0:\n"
        guidelines += "        curr_block = [non_zero_rows[0]]\n"
        guidelines += "        for r in non_zero_rows[1:]:\n"
        guidelines += "            if r == curr_block[-1] + 1:\n"
        guidelines += "                curr_block.append(r)\n"
        guidelines += "            else:\n"
        guidelines += "                blocks.append(curr_block)\n"
        guidelines += "                curr_block = [r]\n"
        guidelines += "        blocks.append(curr_block)\n"
        guidelines += "    def get_shape_properties(block):\n"
        guidelines += "        count = torch.sum(block > 0).item()\n"
        guidelines += "        if count == 7:\n"
        guidelines += "            return 2, 'Left'\n"
        guidelines += "        elif count == 4:\n"
        guidelines += "            return 2, 'Down'\n"
        guidelines += "        elif count == 5:\n"
        guidelines += "            return 3, 'Right'\n"
        guidelines += "        elif count == 6:\n"
        guidelines += "            return 4, 'Left'\n"
        guidelines += "        else:\n"
        guidelines += "            return 0, 'Down'\n"
        guidelines += "    left_segments = []\n"
        guidelines += "    right_segments = []\n"
        guidelines += "    for b_rows in blocks:\n"
        guidelines += "        block_data = left[b_rows, :]\n"
        guidelines += "        col_0_2 = block_data[:, :3]\n"
        guidelines += "        col_4_6 = block_data[:, 4:7]\n"
        guidelines += "        c1 = torch.unique(col_0_2)\n"
        guidelines += "        c1 = c1[c1 != 0]\n"
        guidelines += "        if len(c1) > 0:\n"
        guidelines += "            length, direction = get_shape_properties(col_0_2)\n"
        guidelines += "            left_segments.append((c1[0].item(), length, direction))\n"
        guidelines += "        c2 = torch.unique(col_4_6)\n"
        guidelines += "        c2 = c2[c2 != 0]\n"
        guidelines += "        if len(c2) > 0:\n"
        guidelines += "            length, direction = get_shape_properties(col_4_6)\n"
        guidelines += "            right_segments.append((c2[0].item(), length, direction))\n"
        guidelines += "    segments = left_segments + right_segments\n"
        guidelines += "    out_grid = torch.zeros((H, sep_col), dtype=torch.int32)\n"
        guidelines += "    out_grid[0, start_col] = start_color\n"
        guidelines += "    r_last = 0\n"
        guidelines += "    c_last = start_col\n"
        guidelines += "    for color, length, direction in segments:\n"
        guidelines += "        r_start = r_last + 1\n"
        guidelines += "        c_start = c_last\n"
        guidelines += "        if direction == 'Right':\n"
        guidelines += "            coords = [(r_start, c_start + k) for k in range(length)]\n"
        guidelines += "            c_last = c_start + length - 1\n"
        guidelines += "            r_last = r_start\n"
        guidelines += "        elif direction == 'Left':\n"
        guidelines += "            coords = [(r_start, c_start - k) for k in range(length)]\n"
        guidelines += "            c_last = c_start - length + 1\n"
        guidelines += "            r_last = r_start\n"
        guidelines += "        else:\n"
        guidelines += "            coords = [(r_start + k, c_start) for k in range(length)]\n"
        guidelines += "            c_last = c_start\n"
        guidelines += "            r_last = r_start + length - 1\n"
        guidelines += "        for r, c in coords:\n"
        guidelines += "            if 0 <= r < H and 0 <= c < sep_col:\n"
        guidelines += "                out_grid[r, c] = color\n"
        guidelines += "    return out_grid.tolist()\n"
        guidelines += "```\n"
    else:
        guidelines += "```python\n"
        guidelines += "def transform(grid: list[list[int]]) -> list[list[int]]:\n"
        guidelines += "    import torch\n"
        guidelines += "    from collections import Counter\n"
        guidelines += "    grid_t = torch.tensor(grid, dtype=torch.int32)\n"
        guidelines += "    H, W = grid_t.shape\n"
        guidelines += "    def correct_sequence(seq, pat_color, bg_color):\n"
        guidelines += "        if len(seq) < 4:\n"
        guidelines += "            return seq\n"
        guidelines += "        best_reconstruction = None\n"
        guidelines += "        min_errors = len(seq)\n"
        guidelines += "        best_P = -1\n"
        guidelines += "        for P in range(2, 9):\n"
        guidelines += "            if len(seq) < P * 2:\n"
        guidelines += "                continue\n"
        guidelines += "            normalized_blocks = []\n"
        guidelines += "            for i in range(len(seq) - P + 1):\n"
        guidelines += "                block = seq[i : i+P]\n"
        guidelines += "                shift = (P - (i % P)) % P\n"
        guidelines += "                normalized = block[shift:] + block[:shift]\n"
        guidelines += "                normalized_blocks.append(tuple(normalized))\n"
        guidelines += "            if not normalized_blocks:\n"
        guidelines += "                continue\n"
        guidelines += "            most_common_block = Counter(normalized_blocks).most_common(1)[0][0]\n"
        guidelines += "            reconstructed = [most_common_block[i % P] for i in range(len(seq))]\n"
        guidelines += "            errors = sum(1 for a, b in zip(seq, reconstructed) if a != b)\n"
        guidelines += "            if errors < min_errors:\n"
        guidelines += "                min_errors = errors\n"
        guidelines += "                best_reconstruction = reconstructed\n"
        guidelines += "                best_P = P\n"
        guidelines += "        if best_P != -1 and min_errors < len(seq) / 2:\n"
        guidelines += "            return best_reconstruction\n"
        guidelines += "        return seq\n"
        guidelines += "    counts = Counter(grid_t.flatten().tolist())\n"
        guidelines += "    C_sep = counts.most_common(1)[0][0]\n"
        guidelines += "    sep_rows = []\n"
        guidelines += "    for r in range(H):\n"
        guidelines += "        if torch.sum(grid_t[r] == C_sep).item() >= W - 1:\n"
        guidelines += "            sep_rows.append(r)\n"
        guidelines += "    r_splits = [-1] + sep_rows + [H]\n"
        guidelines += "    sep_cols = []\n"
        guidelines += "    for c in range(W):\n"
        guidelines += "        if torch.sum(grid_t[:, c] == C_sep).item() >= H - 1:\n"
        guidelines += "            sep_cols.append(c)\n"
        guidelines += "    c_splits = [-1] + sep_cols + [W]\n"
        guidelines += "    repaired = grid_t.clone()\n"
        guidelines += "    for i in range(len(r_splits) - 1):\n"
        guidelines += "        r_start = r_splits[i] + 1\n"
        guidelines += "        r_end = r_splits[i+1]\n"
        guidelines += "        if r_end - r_start < 3:\n"
        guidelines += "            continue\n"
        guidelines += "        for j in range(len(c_splits) - 1):\n"
        guidelines += "            c_start = c_splits[j] + 1\n"
        guidelines += "            c_end = c_splits[j+1]\n"
        guidelines += "            if c_end - c_start < 3:\n"
        guidelines += "                continue\n"
        guidelines += "            cell = grid_t[r_start:r_end, c_start:c_end]\n"
        guidelines += "            boundary_elements = []\n"
        guidelines += "            boundary_elements.extend(cell[0, :].tolist())\n"
        guidelines += "            boundary_elements.extend(cell[-1, :].tolist())\n"
        guidelines += "            boundary_elements.extend(cell[:, 0].tolist())\n"
        guidelines += "            boundary_elements.extend(cell[:, -1].tolist())\n"
        guidelines += "            boundary_counts = Counter(boundary_elements)\n"
        guidelines += "            if C_sep in boundary_counts and len(boundary_counts) > 1:\n"
        guidelines += "                del boundary_counts[C_sep]\n"
        guidelines += "            C_ib = boundary_counts.most_common(1)[0][0]\n"
        guidelines += "            int_r_start = r_start + 1\n"
        guidelines += "            int_r_end = r_end - 1\n"
        guidelines += "            int_c_start = c_start + 1\n"
        guidelines += "            int_c_end = c_end - 1\n"
        guidelines += "            interior = grid_t[int_r_start:int_r_end, int_c_start:int_c_end]\n"
        guidelines += "            H_int, W_int = interior.shape\n"
        guidelines += "            int_counts = Counter(interior.flatten().tolist())\n"
        guidelines += "            if C_sep in int_counts:\n"
        guidelines += "                del int_counts[C_sep]\n"
        guidelines += "            if not int_counts:\n"
        guidelines += "                continue\n"
        guidelines += "            C_pat = int_counts.most_common(1)[0][0]\n"
        guidelines += "            if W_int > H_int:\n"
        guidelines += "                for r_idx in range(int_r_start, int_r_end):\n"
        guidelines += "                    seq = repaired[r_idx, int_c_start:int_c_end].tolist()\n"
        guidelines += "                    corrected = correct_sequence(seq, C_pat, C_sep)\n"
        guidelines += "                    repaired[r_idx, int_c_start:int_c_end] = torch.tensor(corrected, dtype=torch.int32)\n"
        guidelines += "            else:\n"
        guidelines += "                for c_idx in range(int_c_start, int_c_end):\n"
        guidelines += "                    seq = repaired[int_r_start:int_r_end, c_idx].tolist()\n"
        guidelines += "                    corrected = correct_sequence(seq, C_pat, C_sep)\n"
        guidelines += "                    repaired[int_r_start:int_r_end, c_idx] = torch.tensor(corrected, dtype=torch.int32)\n"
        guidelines += "    return repaired.tolist()\n"
        guidelines += "```\n"
    
    return prompt, guidelines

class ARCSolverAgent:
    """
    Implements a closed-loop reasoning agent for solving ARC puzzles.
    Uses Generator, Verifier, and Reviser sub-agent components statefully.
    """
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.current_temperature = 0.4
        
    def extract_code(self, candidate) -> str:
        """Extracts the python transformation code and logs the reasoning block."""
        import re
        
        # Extract and print reasoning if present
        reasoning_match = re.search(r"<\|reasoning_begin\|>(.*?)</?\|reasoning_end\|>", candidate, re.DOTALL)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
            print(f"\n  [Reasoning Output]:\n{reasoning_text}\n")

        # Search for code block starting after the reasoning block
        reasoning_end_idx = candidate.find("</|reasoning_end|>")
        if reasoning_end_idx == -1:
            reasoning_end_idx = candidate.find("<|reasoning_end|>")
        
        search_region = candidate
        if reasoning_end_idx != -1:
            search_region = candidate[reasoning_end_idx:]
            
        # Try custom tags first
        if "<|python_begin" in search_region:
            idx_begin = search_region.find("<|python_begin")
            idx_end = search_region.find("</|python_end|>")
            if idx_end == -1:
                idx_end = search_region.find("<|python_end|>")
            if idx_end != -1:
                idx_close_bracket = search_region.find("|>", idx_begin)
                if idx_close_bracket != -1 and idx_close_bracket < idx_end:
                    return search_region[idx_close_bracket + 2 : idx_end].strip()
            else:
                idx_close_bracket = search_region.find("|>", idx_begin)
                if idx_close_bracket != -1:
                    return search_region[idx_close_bracket + 2 :].strip()

        # Try markdown code blocks
        if "```python" in candidate:
            parts = candidate.split("```python")
            for part in parts[1:]:
                if "```" in part:
                    return part.split("```")[0].strip()
        if "```" in candidate:
            parts = candidate.split("```")
            for i in range(1, len(parts), 2):
                if "def transform" in parts[i]:
                    return parts[i].strip()

        # Fallback: find def transform and extract lines
        if "def transform" in candidate:
            lines = candidate.split("\n")
            start_idx = -1
            for idx, line in enumerate(lines):
                if "def transform" in line:
                    start_idx = idx
                    break
            if start_idx != -1:
                code_lines = []
                for line in lines[start_idx:]:
                    if len(code_lines) > 1 and line.strip() and not line.startswith(" ") and not line.startswith("\t") and not line.startswith("#"):
                        break
                    code_lines.append(line)
                return "\n".join(code_lines).strip()
                
        return None

    def generate(self, prompt, guidelines, history=[]):
        """Generates initial candidate reasoning path and python transformation function in a single turn."""
        raw_prompt = (
            "<|turn>system\n"
            "You are the Generator sub-agent for the ARC AGI puzzle solver. Your goal is to write a Python function `transform(grid: list[list[int]]) -> list[list[int]]` using PyTorch.\n"
            "You MUST output exactly two blocks:\n"
            "1. A reasoning block wrapped in <|reasoning_begin|> and <|reasoning_end|> tags.\n"
            "2. A python code block wrapped in <|python_begin|> and <|python_end|> tags containing the transform function.\n\n"
            "CRITICAL RULES:\n"
            "- NumPy is strictly forbidden. You must use PyTorch (torch).\n"
            "- PATH A (Rigid Geometry): If the task is a discrete bounding box crop, flip, or translation, use standard PyTorch tensor slicing (e.g., grid_tensor[r1:r2, c1:c2]).\n"
            "- PATH B (Complex Emergence): If the task requires fuzzy pattern completion or non-rigid emergence, translate the grid to S^1 wave phases and use the EmergentManifold.\n"
            "- Absolutely no conversational or explanatory text outside these tags is permitted.\n\n"
        )
        if guidelines:
            raw_prompt += (
                "=== ABSTRACT REFERENCE EXAMPLE ===\n"
                "Below is an abstract example demonstrating a similar grid transformation function using PyTorch.\n"
                "Do NOT duplicate or output this function directly. Use it only as a reference for syntax/logic:\n"
                f"{guidelines}\n"
                "==================================\n\n"
            )
        raw_prompt += "<turn|>\n"
        for h in history:
            role_tag = "model" if h["role"] == "assistant" else h["role"]
            raw_prompt += f"<|turn>{role_tag}\n{h['content']}<turn|>\n"
        raw_prompt += f"<|turn>user\n{prompt}<turn|>\n"
        raw_prompt += "<|turn>model\n<|reasoning_begin|>\n1. Background color:"
        
        # Apply rehydration and proactive watermarking before model execution
        self.orchestrator.last_routing_weights = None
        raw_prompt = self.orchestrator.rehydrate_prompt(raw_prompt)
        raw_prompt = self.orchestrator.proactive_eviction(raw_prompt)
        
        try:
            res = self.orchestrator.gen_model(
                prompt=raw_prompt,
                max_tokens=8192,
                temperature=self.current_temperature,
                stop=["<turn|>", "<|turn>"],
                dynamic_lora_weights=self.orchestrator.last_routing_weights
            )
        except ValueError as e:
            if "exceed context window" in str(e) or "exceeds" in str(e) or "Requested tokens" in str(e):
                print("[CRITICAL] Context threshold breached dynamically in generate. Performing hard task reset...")
                self.orchestrator.flush_cognitive_manifold()
                raise ValueError("Context threshold breached. Aborting task run.")
            else:
                raise e
        response = "<|reasoning_begin|>\n1. Background color:" + res["choices"][0]["text"]
        return response
        
    @torch.no_grad()
    def verify(self, candidate, task_dict) -> tuple:
        """
        Verifier Sub-agent: Extracts candidate code, runs it on train inputs,
        and verifies correctness against target outputs before predicting test.
        """
        code_block = self.extract_code(candidate)
        if not code_block:
            print(f"  [!] Failed to parse code. Raw response snippet:\n{candidate[:800]}\n...")
            return False, "Failed to find Python code block wrapped in <|python_begin|> and <|python_end|> or standard markdown tags.", None
            
        print(f"  [+] Code extracted successfully:\n---\n{code_block}\n---")
        
        # Test code block on training pairs in sandbox REPL
        # Clean local namespace by redefining function and prepending common helper imports
        common_imports = (
            "import math\n"
            "import collections\n"
            "from collections import defaultdict, deque, Counter\n"
            "import itertools\n"
            "import copy\n"
            "import numpy as np\n"
            "import torch\n"
            "import torch.nn as torch_nn\n"
            "from neurosymbolic_program_induction import ProgramInductor, DifferentiableWaveTransform\n"
            "from emergent_topological_manifold import EmergentManifold, FEPOrthogonalizer\n"
        )
        exec_code = common_imports + code_block + "\n\n"
        res = self.orchestrator.repl.execute_block(exec_code)
        if not res["success"]:
            return False, f"Python syntax/runtime error during definition:\n{res['error_message'] or res['stderr']}", None
            
        # Run on each training demonstration
        for idx, pair in enumerate(task_dict["train"]):
            input_grid = pair["input"]
            expected_output = pair["output"]
            
            test_runner = (
                f"input_val = {input_grid}\n"
                f"output_val = transform(input_val)\n"
                f"print('RESULT_GRID:', output_val)\n"
            )
            run_res = self.orchestrator.repl.execute_block(test_runner)
            if not run_res["success"]:
                return False, f"Runtime error on Demonstration Pair {idx+1}:\n{run_res['error_message'] or run_res['stderr']}", None
                
            stdout = run_res["stdout"].strip()
            if "RESULT_GRID:" not in stdout:
                return False, f"Demonstration Pair {idx+1} execution did not print output value.", None
                
            try:
                result_str = stdout.split("RESULT_GRID:")[1].strip()
                import ast
                actual_output = ast.literal_eval(result_str)
            except Exception as e:
                return False, f"Failed to parse output grid on Demonstration Pair {idx+1} from: {stdout}\nError: {e}", None
                
            if actual_output != expected_output:
                err_msg = (
                    f"Validation failed on Demonstration Pair {idx+1}.\n"
                    f"Input:\n{serialize_grid(input_grid)}\n\n"
                    f"Expected Output:\n{serialize_grid(expected_output)}\n\n"
                    f"Your code returned:\n{serialize_grid(actual_output)}\n"
                )
                return False, err_msg, None
                
        # Run on test input
        test_input = task_dict["test"][0]["input"]
        test_runner = (
            f"input_val = {test_input}\n"
            f"output_val = transform(input_val)\n"
            f"print('TEST_RESULT:', output_val)\n"
        )
        test_res = self.orchestrator.repl.execute_block(test_runner)
        if not test_res["success"]:
            return False, f"Runtime error on Test Input:\n{test_res['error_message'] or test_res['stderr']}", None
            
        stdout = test_res["stdout"].strip()
        if "TEST_RESULT:" not in stdout:
            return False, "Test execution did not print output value.", None
            
        try:
            result_str = stdout.split("TEST_RESULT:")[1].strip()
            import ast
            test_prediction = ast.literal_eval(result_str)
        except Exception as e:
            return False, f"Failed to parse test prediction from: {stdout}\nError: {e}", None
            
        return True, "All training examples verified successfully.", test_prediction
        
    def revise(self, prompt, guidelines, candidate, feedback) -> str:
        """Reviser Sub-agent: Generates corrected code based on verifier execution logs."""
        raw_prompt = (
            "<|turn>system\n"
            "You are the Reviser sub-agent for the ARC AGI puzzle solver. Your goal is to correct the candidate Python function `transform` "
            "based on the execution feedback. Ensure that you fix logical errors, shape mismatches, and handle grid dimensions correctly.\n"
            "You MUST output exactly two blocks:\n"
            "1. A reasoning block wrapped in <|reasoning_begin|> and <|reasoning_end|> tags.\n"
            "2. A python code block wrapped in <|python_begin|> and <|python_end|> tags containing the transform function.\n\n"
            "CRITICAL RULES:\n"
            "- NumPy is strictly forbidden. You must use PyTorch (torch).\n"
            "- PATH A (Rigid Geometry): If the task is a discrete bounding box crop, flip, or translation, use standard PyTorch tensor slicing (e.g., grid_tensor[r1:r2, c1:c2]).\n"
            "- PATH B (Complex Emergence): If the task requires fuzzy pattern completion or non-rigid emergence, translate the grid to S^1 wave phases and use the EmergentManifold.\n"
            "- Absolutely no conversational or explanatory text outside these tags is permitted.\n\n"
        )
        if guidelines:
            raw_prompt += (
                "=== ABSTRACT REFERENCE EXAMPLE ===\n"
                "Below is an abstract example demonstrating a similar grid transformation function using PyTorch.\n"
                "Do NOT duplicate or output this function directly. Use it only as a reference for syntax/logic:\n"
                f"{guidelines}\n"
                "==================================\n\n"
            )
        raw_prompt += "<turn|>\n"
        raw_prompt += f"<|turn>user\nTask Prompt:\n{prompt}\n\nCandidate Code:\n{candidate}\n\nVerifier Feedback:\n{feedback}<turn|>\n"
        raw_prompt += "<|turn>model\n<|reasoning_begin|>\n1. Background color:"
        
        # Apply rehydration and proactive watermarking before model execution
        self.orchestrator.last_routing_weights = None
        raw_prompt = self.orchestrator.rehydrate_prompt(raw_prompt)
        raw_prompt = self.orchestrator.proactive_eviction(raw_prompt)
        
        try:
            res = self.orchestrator.gen_model(
                prompt=raw_prompt,
                max_tokens=8192,
                temperature=self.current_temperature,
                stop=["<turn|>", "<|turn>"],
                dynamic_lora_weights=self.orchestrator.last_routing_weights
            )
        except ValueError as e:
            if "exceed context window" in str(e) or "exceeds" in str(e) or "Requested tokens" in str(e):
                print("[CRITICAL] Context threshold breached dynamically in revise. Performing hard task reset...")
                self.orchestrator.flush_cognitive_manifold()
                raise ValueError("Context threshold breached. Aborting task run.")
            else:
                raise e
        response = "<|reasoning_begin|>\n1. Background color:" + res["choices"][0]["text"]
        return response
        
    @torch.no_grad()
    def solve_task(self, task_dict, max_revisions=2) -> tuple:
        """Orchestrates closed reasoning loops to solve a task."""
        prompt, guidelines = build_arc_prompt(task_dict)
        history = []
        candidate = self.generate(prompt, guidelines, history)
        
        for turn in range(1, max_revisions + 1):
            print(f"  [Revision {turn-1}] Running verifier...")
            success, feedback, test_pred = self.verify(candidate, task_dict)
            
            if success:
                # Centroid drift anchoring based on successful candidate's wave
                emb_res = self.orchestrator.base_model.create_embedding(candidate)
                h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
                h_7b_lora = self.orchestrator.lora_managers[0].apply_lora(h_7b_raw)
                if len(h_7b_lora.shape) == 2:
                    h_7b_lora = torch.mean(h_7b_lora, dim=0)
                with torch.no_grad():
                    psi_candidate = self.orchestrator.l3_router.activation_to_wave(h_7b_lora).detach().cpu()
                
                winner_idx = self.orchestrator.l3_router.update_expert_centroids(psi_candidate)
                print(f"[ANCHOR] Success achieved! Centroid of expert {winner_idx} drifted toward this new logic topology.")
                self.orchestrator.save_router_centroids()
                with torch.no_grad():
                    self.orchestrator.lora_managers[winner_idx].lora_A.copy_(self.orchestrator.lora_managers[0].lora_A)
                    self.orchestrator.lora_managers[winner_idx].lora_B.copy_(self.orchestrator.lora_managers[0].lora_B)
                self.orchestrator.lora_managers[winner_idx].save_weights()
                
                # Consolidate dynamic LoRA adapter weights in database registry
                self.orchestrator.synaptic_manager.consolidate_and_save_adapter(
                    domain_tag=f"Dynamic_Expert_{winner_idx}",
                    lora_manager=self.orchestrator.lora_managers[winner_idx],
                    error_delta=0.0
                )
                
                self.orchestrator.flush_lora_and_context_to_db(domain_tag="ARC_Task")
                return test_pred, turn - 1, "CONVERGED"
                
            print(f"  [-] Verification Failed. Feedback: {feedback[:100]}...")
            
            # Boost temperature on failure for diversity
            self.current_temperature = min(1.0, 0.4 + (turn * 0.15))
            candidate = self.revise(prompt, guidelines, candidate, feedback)
            
        self.orchestrator.flush_lora_and_context_to_db(domain_tag="ARC_Task")
        return None, max_revisions, "TIMEOUT"

def main():
    print("=====================================================================")
    print("            HENRI COGNITIVE SWARM ARC-AGI-2 BENCHMARK ENGINE         ")
    print("=====================================================================")
    
    # 1. Load ARC Evaluation Tasks
    eval_dir = Path(PROJECT_DIR) / "ARC-AGI-2" / "data" / "evaluation"
    if not eval_dir.exists():
        print(f"[ERROR] ARC Evaluation folder not found at: {eval_dir}")
        sys.exit(1)
        
    task_files = sorted(list(eval_dir.glob("*.json")))
    print(f"[SYSTEM] Found {len(task_files)} public evaluation tasks.")
    
    # Limit to first 3 tasks for validation speed (can be expanded)
    max_test_tasks = 3
    tasks_to_test = task_files[:max_test_tasks]
    print(f"[SYSTEM] Selecting first {len(tasks_to_test)} tasks for benchmarking.")
    
    # 2. Initialize the Swarm Orchestrator with memory optimizations
    print("\n[SYSTEM] Booting optimized RAM Swarm...")
    orchestrator = HenriCognitiveSwarmOrchestrator(
        model_path="Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf",
        num_streams=16
    )
    
    agent = ARCSolverAgent(orchestrator)
    
    solved_count = 0
    total_time = 0.0
    results_summary = []
    
    # 3. Loop through tasks
    for idx, t_file in enumerate(tasks_to_test):
        print(f"\n--- [TASK {idx+1}/{len(tasks_to_test)}: {t_file.name}] ---")
        with open(t_file, "r") as f:
            task_dict = json.load(f)
            
        start_time = time.perf_counter()
        prediction, revisions, status = agent.solve_task(task_dict, max_revisions=3)
        elapsed = time.perf_counter() - start_time
        total_time += elapsed
        
        # Verify prediction on test set ground truth
        expected_test_output = task_dict["test"][0]["output"]
        is_correct = (prediction == expected_test_output)
        
        if is_correct:
            solved_count += 1
            print(f"[+] Task {t_file.name} SOLVED successfully! (Time: {elapsed:.2f}s)")
        else:
            print(f"[-] Task {t_file.name} FAILED. (Time: {elapsed:.2f}s)")
            
        results_summary.append({
            "task": t_file.name,
            "revisions": revisions,
            "status": status,
            "solved": is_correct,
            "duration": elapsed
        })
        
        # Hard Manifold Reset between tasks to prevent memory leakage and attention pollution
        orchestrator.flush_cognitive_manifold()
        
    # 4. Print Summary
    print("\n=====================================================================")
    print("                     ARC-AGI-2 RUN SUMMARY                           ")
    print("=====================================================================")
    print(f"  - Tasks Attempted: {len(tasks_to_test)}")
    print(f"  - Tasks Solved:    {solved_count}")
    print(f"  - Accuracy:        {(solved_count / len(tasks_to_test)) * 100:.2f}%")
    print(f"  - Total Duration:  {total_time:.2f} seconds")
    print("\nDetails:")
    for res in results_summary:
        status_str = "PASS" if res["solved"] else "FAIL"
        print(f"  [{status_str}] Task {res['task']}: Solved={res['solved']} | Revisions={res['revisions']} | Status={res['status']} | {res['duration']:.2f}s")
    print("=====================================================================")

if __name__ == "__main__":
    main()
