#!/usr/bin/env python
"""CLASS51 P1 remote smoke: real Qwen3-VL-8B-Instruct on CUDA.

Exercises: provenance (revision + optional shard SHA-256), load, frozen
invariant, text generation, image-text generation, memory report.
Writes a JSON receipt to --receipt path. Fail-closed on any error.

Run on Vast (GPU exclusive):
  HENRI_BACKBONE=1 /venv/main/bin/python backbone_smoke.py \
    --model-dir /root/models/qwen3vl-8b-0c351dd0 \
    --manifest /root/models/qwen3vl-8b-0c351dd0/qwen3vl8b_tree_manifest.json \
    --verify --receipt /tmp/backbone_smoke_receipt.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from henri_backbone_adapter import (  # noqa: E402
    BackboneError,
    QwenBackboneAdapter,
)


def make_test_image(path: Path) -> None:
    """Create a tiny deterministic RGB image for the multimodal smoke."""
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (64, 64), (200, 30, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 24, 24), fill=(30, 200, 30))
    image.save(path, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--receipt", default="/tmp/backbone_smoke_receipt.json")
    parser.add_argument("--text", default="Describe a red square with a green inner square in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    started = time.time()
    try:
        adapter = QwenBackboneAdapter(
            model_dir=args.model_dir,
            manifest_path=args.manifest,
            verify_shards=args.verify,
            max_new_tokens=args.max_new_tokens,
        )
        adapter.load()

        image_path = Path("/tmp/backbone_smoke_image.png")
        make_test_image(image_path)
        text_out, text_telemetry = adapter.generate_text(
            "What is 84 * 3 / 2? Answer with the number only."
        )
        image_out, image_telemetry = adapter.generate_image_text(args.text, image_path)

        receipt = {
            "schema_id": "henri.class51-smoke.v1",
            "status": "PASS",
            "elapsed_seconds": round(time.time() - started, 2),
            "model_id": adapter.model_id,
            "revision": adapter.revision,
            "text_prompt": "What is 84 * 3 / 2? Answer with the number only.",
            "text_output": text_out,
            "image_prompt": args.text,
            "image_output": image_out,
            "telemetry": text_telemetry.to_dict(),
            "memory": adapter.memory_report(),
            "trainable_params": text_telemetry.trainable_params,
            "frozen": text_telemetry.trainable_params == 0,
        }
        Path(args.receipt).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 0
    except (BackboneError, Exception) as exc:  # noqa: BLE001 - fail-closed receipt
        failure = {
            "schema_id": "henri.class51-smoke.v1",
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.time() - started, 2),
        }
        Path(args.receipt).write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
