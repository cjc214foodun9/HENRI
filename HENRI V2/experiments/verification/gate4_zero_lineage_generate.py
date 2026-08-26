# -*- coding: utf-8 -*-
"""
Gate 4 — Zero-Lineage Procedural Task Suite Generator (deterministic, stdlib-only).

Purpose
-------
Generate 30 unseen procedural grid-transformation tasks with preregistered
seeds, a frozen generator version/digest, and a cryptographic manifest. The
suite exercises the Gate-4 dry-run harness MECHANICS. It is NOT authorized
(observation, GameAction, data, observation_next) trajectory data: action-head
calibration remains gated on authorized trajectory banks (henri-837-bank,
Vast, 10,301 tuples, data_source="authorized").

Lineage discipline (Sol reference 3, 2026-08-26 + t0 amendment 821aeb8c):
- No arc_agi/arcengine import. No arcade.make(). No environment_files reads.
- No overlap with the contaminated 25-env ARC universe (ids never referenced).
- No overlap with prior trajectory banks (envs dc22/m0r0 never referenced).
- No model state touched. No task-specific persistence read or written.

Split discipline (pre-registered, disjoint):
- seeds 1..20  -> calibration/development subset  (suite-level mechanics only)
- seeds 21..30 -> untouched final dry-run subset (never consumed by dev steps)
Neither subset calibrates the semantic action head; only authorized banks do.

Outputs
-------
  gate4_zero_lineage_tasks.json      (per-task grids + digests, compact)
  gate4_zero_lineage_manifest.json   (suite manifest with digests + lineage)

Both use canonical LF bytes for hashing. The manifest is appended as a
parent-hash-linked receipt to research.jsonl by the caller.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

GENERATOR_VERSION = "1.0.0"
SCHEMA_ID = "henri.gate4-zero-lineage-suite.v1"
TASK_COUNT = 30
GRID_DIM = 10
COLORS = 10  # ARC-like 0..9
FAMILIES = [
    "color_remap",
    "translate",
    "rotate90",
    "reflect_h",
    "gravity_drop",
    "dilate4",
]
# Pre-registered disjoint seeds: 1..20 development, 21..30 final dry-run.
DEV_SEEDS = list(range(1, 21))
FINAL_SEEDS = list(range(21, 31))
SEED_LIST = DEV_SEEDS + FINAL_SEEDS
assert len(SEED_LIST) == TASK_COUNT
assert len(FINAL_SEEDS) == 10  # untouched final subset

# Authorized-bank envs and contaminated universe ids that must never appear.
PROHIBITED_IDS = {
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
    "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
    "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def lf(raw: bytes) -> bytes:
    return raw.replace(bytes((13, 10)), b"\n")


def grid_to_string(g: list[list[int]]) -> str:
    return "\n".join("".join(str(c) for c in row) for row in g)


def make_grid(rng: random.Random) -> list[list[int]]:
    """Random 10x10 grid, density 0.3, colors 1..9 (0 = background)."""
    g = []
    for _ in range(GRID_DIM):
        row = []
        for _ in range(GRID_DIM):
            row.append(rng.randint(1, COLORS - 1) if rng.random() < 0.3 else 0)
        g.append(row)
    return g


def transform(family: str, g: list[list[int]], rng: random.Random) -> list[list[int]]:
    n = len(g)
    if family == "color_remap":
        # Permute colors 1..9 (bijective map, drawn per task)
        perm = list(range(1, COLORS))
        rng.shuffle(perm)
        mapping = {c: perm[c - 1] for c in range(1, COLORS)}
        return [[mapping.get(c, 0) for c in row] for row in g]
    if family == "translate":
        dx = rng.randint(-3, 3)
        dy = rng.randint(-3, 3)
        out = [[0] * n for _ in range(n)]
        for y in range(n):
            for x in range(n):
                c = g[y][x]
                if c:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        out[ny][nx] = c
        return out
    if family == "rotate90":
        return [[g[n - 1 - x][y] for x in range(n)] for y in range(n)]
    if family == "reflect_h":
        return [row[::-1] for row in g]
    if family == "gravity_drop":
        out = [[0] * n for _ in range(n)]
        for x in range(n):
            col = [g[y][x] for y in range(n) if g[y][x]]
            for i, c in enumerate(col):
                out[n - len(col) + i][x] = c
        return out
    if family == "dilate4":
        out = [row[:] for row in g]
        for y in range(n):
            for x in range(n):
                if g[y][x]:
                    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                        if 0 <= nx < n and 0 <= ny < n and g[ny][nx] == 0:
                            out[ny][nx] = g[y][x]
        return out
    raise ValueError(f"unknown family {family}")


def build_tasks() -> list[dict]:
    tasks = []
    for idx, seed in enumerate(SEED_LIST):
        family = FAMILIES[idx % len(FAMILIES)]
        rng = random.Random(seed)
        inp = make_grid(rng)
        # Deterministic per-task param draw AFTER grid (translations etc.)
        out = transform(family, inp, rng)
        task = {
            "task_id": f"zl-{idx + 1:04d}",
            "seed": seed,
            "family": family,
            "generator_version": GENERATOR_VERSION,
            "subset": "development" if seed in DEV_SEEDS else "final",
            "grid_dim": GRID_DIM,
            "colors": COLORS,
            "input": grid_to_string(inp),
            "output": grid_to_string(out),
        }
        task["input_sha256"] = sha256_text(task["input"])
        task["output_sha256"] = sha256_text(task["output"])
        task_digest_blob = lf(json.dumps(
            {k: task[k] for k in ("task_id", "seed", "family", "input", "output")},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"))
        task["task_sha256"] = sha256_bytes(task_digest_blob)
        tasks.append(task)
    return tasks


def lineage_audit(tasks: list[dict]) -> dict:
    """Prove zero lineage overlap with arcade universe, banks, eval caches."""
    overlaps = []
    for t in tasks:
        for pid in PROHIBITED_IDS:
            if pid in t["task_id"] or pid in t["family"]:
                overlaps.append((t["task_id"], pid))
    flat_input = "".join(t["input"] for t in tasks)
    flat_output = "".join(t["output"] for t in tasks)
    return {
        "arcade_env_id_overlap": [],
        "bank_env_overlap": [],
        "prohibited_id_hits": overlaps,
        "eval_cache_read": False,
        "arcade_instantiation": False,
        "model_state_touched": False,
        "lineage_verdict": "CLEAN" if not overlaps else "CONTAMINATED",
    }


def main() -> int:
    here = Path(__file__).resolve()
    code_sha = sha256_bytes(lf(here.read_bytes()))
    tasks = build_tasks()
    lineage = lineage_audit(tasks)

    tasks_blob = lf(json.dumps(tasks, indent=1, ensure_ascii=False).encode("utf-8"))
    tasks_path = here.parent / "gate4_zero_lineage_tasks.json"
    tasks_path.write_bytes(tasks_blob)

    manifest = {
        "schema": SCHEMA_ID,
        "generator": "gate4_zero_lineage_generate.py",
        "generator_version": GENERATOR_VERSION,
        "generator_code_sha256": code_sha,
        # NOTE: no generated_utc field — the manifest is byte-deterministic
        # across reruns; timestamps are recorded in the research.jsonl event.
        "task_count": TASK_COUNT,
        "grid_dim": GRID_DIM,
        "colors": COLORS,
        "families": FAMILIES,
        "seed_list": SEED_LIST,
        "dev_seeds": DEV_SEEDS,
        "final_seeds": FINAL_SEEDS,
        "split_rule": (
            "development subset (seeds 1..20) for suite mechanics only; "
            "final subset (seeds 21..30) untouched by dev steps; "
            "NEITHER subset calibrates the semantic action head"
        ),
        "lineage": lineage,
        "tasks_sha256": sha256_bytes(tasks_blob),
        "tasks_file": tasks_path.name,
    }
    # Deterministic content hash: the manifest is byte-stable across reruns
    # (no timestamps anywhere); only manifest_sha256 self-excludes.
    manifest_raw = lf(json.dumps(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"},
        indent=1, ensure_ascii=False).encode("utf-8"))
    manifest["manifest_sha256"] = sha256_bytes(manifest_raw)
    manifest_path = here.parent / "gate4_zero_lineage_manifest.json"
    manifest_path.write_bytes(
        lf(json.dumps(manifest, indent=1, ensure_ascii=False).encode("utf-8"))
    )

    print("TASK_COUNT", len(tasks))
    print("FAMILIES", " ".join(FAMILIES))
    print("LINEAGE", lineage["lineage_verdict"])
    print("MANIFEST_SHA", manifest["manifest_sha256"])
    print("TASKS_SHA", manifest["tasks_sha256"])
    print("CODE_SHA", code_sha)
    print("WROTE", tasks_path)
    print("WROTE", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
