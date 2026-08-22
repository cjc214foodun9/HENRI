#!/usr/bin/env python
"""Diagnose the P3 pilot 0/30 anomaly: regenerate first-3 items with the
exact pilot prompts and print raw output tails + lengths (no unit tests).
Also compares the P2-style minimal prompt for the same items.
"""
import gzip
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, "/root/class51_p3")

from henri_backbone_adapter import QwenBackboneAdapter  # noqa: E402
from henri_backbone_retrieval import build_arm_a_prompt  # noqa: E402

GZ = "/root/class51_p3/HumanEval.jsonl.gz"
MODEL_DIR = "/root/models/qwen3vl-8b-0c351dd0"
MANIFEST = "/root/models/qwen3vl-8b-0c351dd0/qwen3vl8b_tree_manifest.json"

raw_gz = pathlib.Path(GZ).read_bytes()
assert hashlib.sha256(raw_gz).hexdigest().startswith("b796127e")
raw = gzip.decompress(raw_gz)
items = [json.loads(line) for line in raw.decode().splitlines()][:3]

adapter = QwenBackboneAdapter(
    model_dir=MODEL_DIR, manifest_path=MANIFEST,
    revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    verify_shards=True, max_new_tokens=384,
).load()
print("frozen:", adapter.telemetry.trainable_params == 0)

for it in items:
    tid = it["task_id"]
    p3_prompt = build_arm_a_prompt(it["prompt"])[0]
    p2_prompt = "Complete the following Python function. Return only the code, no explanation.\n\n" + it["prompt"]
    for label, prompt in (("P3", p3_prompt), ("P2", p2_prompt)):
        t0 = time.time()
        resp, _ = adapter.generate_text(prompt)
        dt = time.time() - t0
        print(f"=== {tid} [{label}] len={len(resp)} gen_s={dt:.2f} ===")
        print("HEAD:", repr(resp[:120]))
        print("TAIL:", repr(resp[-200:]))
        print()
