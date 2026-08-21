"""Path A demo-coverage preflight (pre-registered, 2026-08-20).

Counts items with 0/1/>=2 valid prompt-docstring pairs on the SAME
official HumanEval slice the runner uses. Reads ONLY the prompt text and
the entry point; NEVER the `test` field or reference answer. Parses both
sides under the production ASTDiscriminativeEncoder (full D). Writes a
SHA-256 hashed receipt.

Gate (design doc): >=2-example coverage inadequate
-> BLOCKED_INSUFFICIENT_DEMONSTRATIONS (do not force Path A).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (script lives at <root>/experiments/verification/)
HENRI_DIR = os.path.join(REPO_ROOT, "HENRI V2")
sys.path.insert(0, HENRI_DIR)
sys.path.insert(0, REPO_ROOT)

from henri_task_operator import extract_docstring_examples  # noqa: E402
from qfhrr_ast_discriminative_kernel import ASTDiscriminativeEncoder  # noqa: E402

CACHE = os.path.join(REPO_ROOT, "data", "HumanEval.jsonl.gz")
URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
LIMIT = 50
D_MODEL = 65536


def load_items() -> list[dict]:
    if not os.path.exists(CACHE):
        import urllib.request
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        print("[DATASET] downloading pinned HumanEval.jsonl.gz")
        urllib.request.urlretrieve(URL, CACHE)
    raw = open(CACHE, "rb").read()
    ds_sha = hashlib.sha256(raw).hexdigest()
    with gzip.open(CACHE, "rt", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    print(f"[DATASET] items={len(items)} sha256={ds_sha}")
    return items[:LIMIT], ds_sha


def main() -> int:
    t0 = time.perf_counter()
    items, ds_sha = load_items()
    enc = ASTDiscriminativeEncoder(d_model=D_MODEL, device="cpu")

    counts = {0: 0, 1: 0, 2: 0, "ge2": 0}
    entry_mismatch = 0
    unparsable = 0
    detail = []

    for it in items:
        task_id = it["task_id"]
        prompt = it["prompt"]
        entry = it["entry_point"]
        pairs = extract_docstring_examples(prompt, entry)
        n = len(pairs)
        bucket = n if n < 2 else "ge2"
        counts[bucket] = counts.get(bucket, 0) + 1

        item_rec = {
            "task_id": task_id,
            "pairs": n,
            "calls_reference_entry": True,
            "parsable_in": True,
            "parsable_out": True,
        }
        for in_src, out_src in pairs:
            # call must reference entry (extraction already filters; double-check)
            if not in_src.strip().startswith(entry + "("):
                entry_mismatch += 1
                item_rec["calls_reference_entry"] = False
            # parse both sides under the production encoder
            if enc.encode_code_string(in_src) is None:
                unparsable += 1
                item_rec["parsable_in"] = False
            if enc.encode_code_string(out_src) is None:
                unparsable += 1
                item_rec["parsable_out"] = False
        detail.append(item_rec)

    zero = counts[0]
    one = counts[1]
    ge2 = counts["ge2"]
    adequate = ge2 >= 1  # G2 accept needs >=1 item with demos; >=2 pairs needed to compile

    receipt = {
        "preflight": "path_a_demo_coverage",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "HumanEval",
        "dataset_sha256": ds_sha,
        "slice": LIMIT,
        "items_total": len(items),
        "items_0_pairs": zero,
        "items_1_pair": one,
        "items_ge2_pairs": ge2,
        "entry_mismatches": entry_mismatch,
        "unparsable_sides": unparsable,
        "d_model": D_MODEL,
        "encoder": "ASTDiscriminativeEncoder",
        "adequate": adequate,
        "verdict": "CONTINUE" if adequate else "BLOCKED_INSUFFICIENT_DEMONSTRATIONS",
        "detail": detail,
    }
    out_dir = os.path.join(REPO_ROOT, "experiments", "verification")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"path_a_coverage_preflight_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode("utf-8")).hexdigest()
    receipt["receipt_sha256"] = digest
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(json.dumps({k: v for k, v in receipt.items() if k != "detail"}, indent=2))
    print(f"[RECEIPT] {out_path}")
    print(f"[RECEIPT_SHA256] {digest}")
    print(f"[WALL] {time.perf_counter() - t0:.1f}s")
    return 0 if adequate else 2


if __name__ == "__main__":
    sys.exit(main())
