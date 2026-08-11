"""ARC Public Corpus Ingress Channel - Phase 7.1 (bounded, default-off).

Read-only parser for official public ARC-AGI task JSON (grid-pair corpus).
Requires an explicit provenance manifest with exact environment->task-ID
mapping. Exact match only: no fuzzy matching, aliases, reconstruction, or
fallback to cached environment files (those are evaluation machinery and
FORBIDDEN_LEAKAGE_SOURCE).

The public corpus contains (input_grid, output_grid) transformation pairs.
It does NOT contain interactive (observation, GameAction, data) trajectories;
action-head calibration from grid pairs alone remains BLOCKED_NO_ACTION_TRAJECTORIES.

Typed statuses:
- LOADED_PUBLIC_DEMOS
- BLOCKED_DATASET_ID_MISMATCH   (env id absent from manifest / task id absent from corpus)
- BLOCKED_NO_DEMONSTRATIONS    (task present but zero train pairs)
- BLOCKED_DIGEST_MISMATCH      (corpus sha256 != manifest pin)
- BLOCKED_SCHEMA_INVALID       (JSON shape violation)
- BLOCKED_MANIFEST_MISSING     (manifest path absent or unreadable)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_STATUS_LOADED = "LOADED_PUBLIC_DEMOS"
_STATUS_ID_MISMATCH = "BLOCKED_DATASET_ID_MISMATCH"
_STATUS_NO_DEMOS = "BLOCKED_NO_DEMONSTRATIONS"
_STATUS_DIGEST = "BLOCKED_DIGEST_MISMATCH"
_STATUS_SCHEMA = "BLOCKED_SCHEMA_INVALID"
_STATUS_MANIFEST = "BLOCKED_MANIFEST_MISSING"
_STATUS_CORPUS = "BLOCKED_CORPUS_UNAVAILABLE"


class PublicIngressError(Exception):
    """Typed failure for the public corpus ingress channel."""


@dataclass
class PublicIngressResult:
    """Deterministic outcome of a corpus lookup for one environment."""

    status: str
    reason: str = ""
    task_id: Optional[str] = None
    demo_pairs: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == _STATUS_LOADED


def sha256_file(path: str) -> str:
    """SHA-256 of raw file bytes (canonical raw bytes; no line-ending normalization)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _grid_to_np(grid) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2:
        raise PublicIngressError(f"grid is not 2D: shape={arr.shape}")
    return arr


def load_task_json(path: str, expected_sha256: Optional[str] = None) -> Dict:
    """Load and validate one public ARC-AGI task JSON file (fail closed)."""
    p = Path(path)
    if not p.is_file():
        raise PublicIngressError(f"corpus file not found: {path}")
    raw = p.read_bytes()
    if expected_sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise PublicIngressError(
                f"sha256 mismatch: expected {expected_sha256} got {actual}"
            )
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PublicIngressError(f"corpus JSON invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise PublicIngressError("corpus root is not a JSON object")
    if "train" not in data or "test" not in data:
        raise PublicIngressError("corpus missing 'train'/'test' splits")
    if not isinstance(data["train"], list) or not isinstance(data["test"], list):
        raise PublicIngressError("corpus splits are not lists")
    for pair in data["train"] + data["test"]:
        if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
            raise PublicIngressError("corpus pair missing 'input'/'output'")
        _grid_to_np(pair["input"])
        _grid_to_np(pair["output"])
    return data


def load_manifest(path: str) -> Dict:
    """Load the provenance manifest.

    Schema:
      {"envs": {"<environment_id>": {"task_id": "<id>",
                                     "corpus_path": "<path>",
                                     "sha256": "<hex>"}}}
    """
    p = Path(path)
    if not p.is_file():
        raise PublicIngressError(f"manifest file not found: {path}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PublicIngressError(f"manifest JSON invalid: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("envs"), dict):
        raise PublicIngressError("manifest schema invalid: missing 'envs' object")
    for env_id, entry in data["envs"].items():
        if not isinstance(entry, dict):
            raise PublicIngressError(f"manifest entry for {env_id} is not an object")
        for key in ("task_id", "corpus_path", "sha256"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise PublicIngressError(
                    f"manifest entry for {env_id} missing '{key}'"
                )
    return data


def resolve_demos(
    manifest_path: str, env_id: str
) -> PublicIngressResult:
    """Resolve demonstration pairs for one environment via the manifest.

    Exact-match only. Absent mapping -> BLOCKED_DATASET_ID_MISMATCH.
    """
    try:
        manifest = load_manifest(manifest_path)
    except PublicIngressError as exc:
        return PublicIngressResult(status=_STATUS_MANIFEST, reason=str(exc))

    entry = manifest["envs"].get(env_id)
    if entry is None:
        return PublicIngressResult(
            status=_STATUS_ID_MISMATCH,
            reason=f"environment '{env_id}' has no manifest mapping",
        )
    task_id = entry["task_id"]
    corpus_path = entry["corpus_path"]
    expected_sha = entry["sha256"]

    try:
        task = load_task_json(corpus_path, expected_sha)
    except PublicIngressError as exc:
        # Distinguish digest failure from corpus-availability and schema
        # failures.
        reason = str(exc)
        if "sha256 mismatch" in reason:
            return PublicIngressResult(
                status=_STATUS_DIGEST,
                reason=f"{task_id}@{corpus_path}: {reason}",
                task_id=task_id,
            )
        if "not found" in reason or "unreadable" in reason:
            return PublicIngressResult(
                status=_STATUS_CORPUS,
                reason=f"{task_id}@{corpus_path}: {reason}",
                task_id=task_id,
            )
        return PublicIngressResult(
            status=_STATUS_SCHEMA,
            reason=f"{task_id}@{corpus_path}: {reason}",
            task_id=task_id,
        )

    pairs = []
    for pair in task["train"]:
        pairs.append((_grid_to_np(pair["input"]), _grid_to_np(pair["output"])))
    if not pairs:
        return PublicIngressResult(
            status=_STATUS_NO_DEMOS,
            reason=f"task '{task_id}' has zero train pairs",
            task_id=task_id,
            provenance={
                "task_id": task_id,
                "corpus_path": corpus_path,
                "corpus_sha256": expected_sha,
            },
        )
    return PublicIngressResult(
        status=_STATUS_LOADED,
        reason="exact manifest mapping resolved",
        task_id=task_id,
        demo_pairs=pairs,
        provenance={
            "task_id": task_id,
            "corpus_path": corpus_path,
            "corpus_sha256": expected_sha,
            "source": "public_arc_corpus",
        },
    )
