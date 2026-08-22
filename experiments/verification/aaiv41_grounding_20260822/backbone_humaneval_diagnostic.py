#!/usr/bin/env python
"""CLASS51 P2 diagnostic: HumanEval code synthesis on the frozen Qwen3-VL-8B-Instruct backbone.

NOT an official AAII v4.1 score. This is a convenience diagnostic baseline:
canonical public HumanEval items (openai/human-eval, GitHub raw), greedy
generation, deterministic grading by the canonical unit tests in a
subprocess sandbox (15s timeout, no network). Held-out status is CONDITIONAL
(Qwen training-lineage overlap cannot be excluded).

Receipt: henri.run-evidence.v1 with item-level outcomes and digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from henri_backbone_adapter import (  # noqa: E402
    QwenBackboneAdapter,
    backbone_enabled,
)

CANONICAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl"
FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_bytes(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "henri-class51-diagnostic/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def extract_code(response: str) -> str:
    match = FENCE_RE.search(response)
    return match.group(1).strip() if match else response.strip()


def run_unit_test(code: str, test_code: str, timeout: int = 15) -> bool:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "solution.py"
        path.write_text(code + "\n\n" + test_code, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                timeout=timeout,
                cwd=directory,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt", default="/tmp/class51_p2_humaneval_receipt.json")
    parser.add_argument("--data-url", default=CANONICAL_URL)
    args = parser.parse_args()

    if not backbone_enabled():
        print("FATAL: HENRI_BACKBONE not set; fail-closed.")
        return 2

    started = time.time()
    outcome = {
        "schema_id": "henri.run-evidence.v1",
        "kind": "diagnostic-baseline",
        "not_official_aaii": True,
        "held_out_status": "CONDITIONAL",
        "status": "PENDING",
    }

    try:
        raw = fetch_bytes(args.data_url)
        outcome["sources"] = {
            "humaneval": {"url": args.data_url, "sha256": sha256_bytes(raw), "bytes": len(raw)},
        }
        items = []
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            items.append(record)
            if len(items) >= args.n:
                break
        if len(items) == 0:
            raise RuntimeError("HumanEval fetch returned zero items")

        adapter = QwenBackboneAdapter(
            model_dir=args.model_dir,
            manifest_path=args.manifest,
            max_new_tokens=args.max_new_tokens,
        )
        adapter.load()

        item_results = []
        passed = 0
        for item in items:
            prompt = item.get("prompt", "")
            test_code = item.get("test", "")
            task_id = item.get("task_id", "unknown")
            message = (
                "Complete the following Python function. Return only the code, no explanation.\n\n"
                + prompt
            )
            try:
                response, _telemetry = adapter.generate_text(message)
                completion = extract_code(response)
                if completion.startswith(prompt[-60:]):
                    completion = completion[len(prompt[-60:]):]
                code = prompt + "\n" + completion
                is_pass = run_unit_test(code, test_code)
            except Exception as exc:  # noqa: BLE001 - fail-closed item
                is_pass = False
                item_results.append({
                    "task_id": task_id,
                    "is_pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            if is_pass:
                passed += 1
            item_results.append({"task_id": task_id, "is_pass": is_pass})

        attempted = len(item_results)
        outcome["metrics"] = {
            "passed": passed,
            "attempted": attempted,
            "execution_errors": sum(1 for r in item_results if r.get("error")),
            "accuracy": round(passed / attempted, 4) if attempted else 0.0,
            "subset": f"humaneval_first_{attempted}",
        }
        outcome["items"] = item_results
        outcome["telemetry"] = adapter.telemetry.to_dict()
        outcome["memory"] = adapter.memory_report()
        outcome["elapsed_seconds"] = round(time.time() - started, 2)
        outcome["status"] = "COMPLETE"
    except Exception as exc:  # noqa: BLE001 - fail-closed receipt
        outcome["status"] = "BLOCKED"
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        outcome["elapsed_seconds"] = round(time.time() - started, 2)

    Path(args.receipt).write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2))
    return 0 if outcome["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
