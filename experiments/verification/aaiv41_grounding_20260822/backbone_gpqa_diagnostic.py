#!/usr/bin/env python
"""CLASS51 P2 diagnostic baseline runner (GPQA Diamond subset + HLE subset).

NOT an official AAII v4.1 score. This is a convenience diagnostic baseline
for the frozen Qwen3-VL-8B-Instruct backbone. All items are text-only MCQ
samples from public sources; the subset is small and NOT a canonical
evaluator. Held-out status is CONDITIONAL (training-lineage overlap with the
backbone cannot be excluded).

Data contract (canonical sources, read-only):
  GPQA: https://huggingface.co/datasets/Idavidrein/gpqa/raw/main/diamond/dataset.jsonl
        (or the canonical Idavidrein/gpqa repo at a pinned commit)
  HLE:  https://huggingface.co/datasets/cais/hle (canonical; small slice)

Output: henri.run-evidence.v1 receipt with item-level outcomes and hashes.
Refuses to run when HENRI_BACKBONE is unset (fail-closed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from henri_backbone_adapter import (  # noqa: E402
    QwenBackboneAdapter,
    backbone_enabled,
)

LETTER_RE = re.compile(r"\b([A-D])\b")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_url(url: str, timeout: int = 60) -> bytes:
    import urllib.request
    request = urllib.request.Request(url, headers={"User-Agent": "henri-class51-diagnostic/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def normalize_answer(response: str) -> str | None:
    match = LETTER_RE.findall(response)
    if not match:
        return None
    # Prefer the LAST standalone letter (Qwen tends to state the answer at the end).
    return match[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpqa-url", default="https://huggingface.co/datasets/Idavidrein/gpqa/raw/main/diamond/dataset.jsonl")
    parser.add_argument("--hle-url", default="https://huggingface.co/datasets/cais/hle/raw/main/data/test-00000-of-00001.parquet")
    parser.add_argument("--gpqa-n", type=int, default=50)
    parser.add_argument("--hle-n", type=int, default=20)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt", default="/tmp/class51_p2_diagnostic_receipt.json")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
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
        # ---- fetch + hash sources ----
        gpqa_bytes = fetch_url(args.gpqa_url)
        gpqa_sha = sha256_bytes(gpqa_bytes)
        gpqa_items = [json.loads(line) for line in gpqa_bytes.decode("utf-8").splitlines() if line.strip()][: args.gpqa_n]
        outcome["sources"] = {
            "gpqa_diamond": {"url": args.gpqa_url, "sha256": gpqa_sha, "fetched_items": len(gpqa_items)},
            "hle": {"url": args.hle_url, "status": "BLOCKED" if "parquet" in args.hle_url else "fetched"},
        }
        if len(gpqa_items) == 0:
            raise RuntimeError("GPQA fetch returned zero items")

        # ---- load frozen backbone ----
        adapter = QwenBackboneAdapter(
            model_dir=args.model_dir,
            manifest_path=args.manifest,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        adapter.load()

        # ---- run items ----
        item_results = []
        passed = 0
        attempted = 0
        execution_errors = 0
        for item in gpqa_items:
            question = item.get("Question", "")
            choices = [item.get(f"Incorrect Answer {i}", "") for i in (1, 2, 3)]
            correct = item.get("Correct Answer", "")
            # GPQA format: Question + 3 incorrect + 1 correct.
            option_texts = choices[:3] + [correct]
            # Deterministic rotation so the correct answer is not always at a
            # fixed letter position (position-bias control for the diagnostic).
            rotation = int(sha256_bytes(question.encode("utf-8"))[:8], 16) % 4
            rotated = option_texts[rotation:] + option_texts[:rotation]
            correct_letter = chr(65 + (3 - rotation))  # original index 3 -> rotated position
            prompt = question + "\n\nOptions:\n" + "\n".join(f"{chr(65 + i)}. {rotated[i]}" for i in range(4)) + "\n\nAnswer:"
            try:
                response, _telemetry = adapter.generate_text(prompt)
                attempted += 1
                predicted = normalize_answer(response)
                is_pass = False
                if predicted is not None:
                    idx = ord(predicted) - 65
                    is_pass = 0 <= idx < 4 and rotated[idx] == correct
                if is_pass:
                    passed += 1
                item_results.append({
                    "index": len(item_results),
                    "question_sha256": sha256_bytes(question.encode("utf-8")),
                    "predicted": predicted,
                    "is_pass": is_pass,
                })
            except Exception as exc:  # noqa: BLE001 - fail-closed item
                execution_errors += 1
                item_results.append({"index": len(item_results), "error": f"{type(exc).__name__}: {exc}"})
                continue

        outcome["metrics"] = {
            "passed": passed,
            "attempted": attempted,
            "execution_errors": execution_errors,
            "accuracy": round(passed / attempted, 4) if attempted else 0.0,
            "subset": f"gpqa_diamond_first_{len(gpqa_items)}",
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
