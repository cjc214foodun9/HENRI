#!/usr/bin/env python
"""Kill experiment: is the 0/30 anomaly caused by the loader's test-field
join? Tests BOTH loader paths on real generations for HumanEval/0..2."""
import gzip
import hashlib
import json
import pathlib
import subprocess
import sys

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

# --- decisive type check ---
for it in items:
    print(it["task_id"], "test_type=", type(it["test"]).__name__,
          "imports_type=", type(it.get("imports", [])).__name__,
          "test_head=", repr((it["test"] if isinstance(it["test"], str) else it["test"][0])[:40]))

adapter = QwenBackboneAdapter(
    model_dir=MODEL_DIR, manifest_path=MANIFEST,
    revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    verify_shards=True, max_new_tokens=384,
).load()


def normalize_answer(code: str) -> str:
    code = code.strip()
    for fence in ("```python", "```"):
        if fence in code:
            code = code.split(fence, 1)[-1]
            code = code.rsplit("```", 1)[0] if "```" in code else code
    return code.strip()


def run_tests(code, test_str, timeout=15):
    body = f"{code}\n\n{test_str}"
    try:
        proc = subprocess.run([sys.executable, "-c", body],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "")[-300:]
    return True, ""


for it in items:
    resp, _ = adapter.generate_text(build_arm_a_prompt(it["prompt"])[0])
    code = normalize_answer(resp)
    buggy_test = "\n".join(it["test"])          # current pilot loader (char-join if str)
    fixed_test = it["test"] if isinstance(it["test"], str) else "\n".join(it["test"])
    ok_buggy, err_buggy = run_tests(code, buggy_test)
    ok_fixed, err_fixed = run_tests(code, fixed_test)
    print(f"{it['task_id']}: buggy_join_pass={ok_buggy} err={err_buggy[:80]!r} | "
          f"fixed_pass={ok_fixed} err={err_fixed[:80]!r}")
