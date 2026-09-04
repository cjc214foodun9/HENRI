"""Carrier K5 (TZCSM) — frozen encoder pin (Path-B re-scope, CLASS51 precedent).

Decision APPROVE_USER_20260904_K5_TZCSM, option 1 (2026-09-04):
  frozen, revision-pinned backbone = CLASS51 artifact
    model:  Qwen/Qwen3-VL-8B-Instruct
    revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
    artifact dir on CUDA target: /root/models/qwen3vl-8b-0c351dd0/
    shard SHA-256 (OBSERVED 2026-09-04, sha256sum on target):
      model-00001-of-00004.safetensors  d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa
      model-00002-of-00004.safetensors  8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5
      model-00003-of-00004.safetensors  83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192
      model-00004-of-00004.safetensors  0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5

Contract:
  - DEFAULT-OFF. No artifact path is touched unless K5_INGEST_ENCODER=1.
  - The C1-C6 disposable fixture suite keeps the deterministic FIXTURE codec
    (zone_c_world_knowledge_fixtures.py); the pin below governs corpus ingest.
  - verify_encoder_artifact(): sha256 every shard, compare to the pin, fail
    closed (typed error) on any mismatch. Cross-process hash identity = C1.
  - Zero trainable parameters. Eval-only by policy (CLASS51 amendment).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DEFAULT_ARTIFACT_DIR = "/root/models/qwen3vl-8b-0c351dd0"

# OBSERVED 2026-09-04 on the CUDA target (sha256sum). Keyed by file name.
SHARD_SHA256 = {
    "model-00001-of-00004.safetensors": "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
    "model-00002-of-00004.safetensors": "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
    "model-00003-of-00004.safetensors": "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
    "model-00004-of-00004.safetensors": "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
}


class EncoderPinError(RuntimeError):
    """Base class for encoder-pin contract failures."""


class EncoderDisabledError(EncoderPinError):
    """Raised when encoder use is attempted without K5_INGEST_ENCODER=1."""


class EncoderArtifactMismatchError(EncoderPinError):
    """Raised when artifact bytes do not match the sealed pin."""


def encoder_enabled() -> bool:
    return os.environ.get("K5_INGEST_ENCODER", "0").strip() in {"1", "true", "True", "yes"}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def pin_record() -> dict:
    """Immutable pin record; used by receipts and telemetry."""
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "shards": dict(SHARD_SHA256),
        "policy": "frozen-eval-only-zero-trainable",
        "default_off_flag": "K5_INGEST_ENCODER",
    }


def verify_encoder_artifact(artifact_dir: str | os.PathLike | None = None) -> dict:
    """Assert artifact bytes match the sealed pin. Fail closed on mismatch.

    Returns per-shard {name: sha256} on success. C1: identical hashes across
    processes by construction (byte comparison, no random component).
    """
    if not encoder_enabled():
        raise EncoderDisabledError(
            "encoder use requires K5_INGEST_ENCODER=1 (default-OFF pin)"
        )
    base = Path(artifact_dir or DEFAULT_ARTIFACT_DIR)
    if not base.is_dir():
        raise EncoderPinError(f"encoder artifact dir missing: {base}")
    result = {}
    for name, expected in SHARD_SHA256.items():
        p = base / name
        if not p.is_file():
            raise EncoderArtifactMismatchError(f"missing shard: {p}")
        actual = _sha256_file(p)
        if actual != expected:
            raise EncoderArtifactMismatchError(
                f"shard mismatch {name}: expected {expected[:16]}... got {actual[:16]}..."
            )
        result[name] = actual
    return result


if __name__ == "__main__":
    # Deterministic receipt emission.
    record = pin_record()
    print(json.dumps(record, indent=2, sort_keys=True))
