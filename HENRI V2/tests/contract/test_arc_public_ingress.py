"""Contract tests for arc_public_ingress (Phase 7.1).

CPU-only, no production artifact, no network. Covers every typed status
and the fail-closed guards.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from arc_public_ingress import (
    PublicIngressError,
    load_manifest,
    load_task_json,
    resolve_demos,
)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _task_bytes(train_pairs=3, test_pairs=1) -> bytes:
    data = {
        "train": [
            {"input": [[0, 0], [0, 1]], "output": [[1, 1], [1, 0]]}
            for _ in range(train_pairs)
        ],
        "test": [
            {"input": [[2, 2], [2, 3]], "output": [[3, 3], [3, 2]]}
            for _ in range(test_pairs)
        ],
    }
    return json.dumps(data).encode("utf-8")


@pytest.fixture()
def corpus(tmp_path: Path):
    raw = _task_bytes()
    p = tmp_path / "task.json"
    p.write_bytes(raw)
    return p, hashlib.sha256(raw).hexdigest()


@pytest.fixture()
def manifest(tmp_path: Path, corpus) -> Path:
    task_path, sha = corpus
    m = tmp_path / "manifest.json"
    _write(
        m,
        json.dumps(
            {
                "envs": {
                    "lf52": {
                        "task_id": "00d62c1b",
                        "corpus_path": str(task_path),
                        "sha256": sha,
                    }
                }
            }
        ),
    )
    return m


def test_loaded_public_demos(manifest: Path):
    res = resolve_demos(str(manifest), "lf52")
    assert res.ok
    assert res.status == "LOADED_PUBLIC_DEMOS"
    assert len(res.demo_pairs) == 3
    assert res.provenance["task_id"] == "00d62c1b"
    assert res.provenance["source"] == "public_arc_corpus"
    for x, y in res.demo_pairs:
        assert isinstance(x, np.ndarray) and x.ndim == 2
        assert isinstance(y, np.ndarray) and y.ndim == 2


def test_id_mismatch(tmp_path: Path):
    m = tmp_path / "manifest.json"
    _write(m, json.dumps({"envs": {}}))
    res = resolve_demos(str(m), "lf52")
    assert not res.ok
    assert res.status == "BLOCKED_DATASET_ID_MISMATCH"


def test_manifest_missing(tmp_path: Path):
    res = resolve_demos(str(tmp_path / "nope.json"), "lf52")
    assert not res.ok
    assert res.status == "BLOCKED_MANIFEST_MISSING"


def test_digest_mismatch(tmp_path: Path, corpus):
    task_path, _ = corpus
    m = tmp_path / "manifest.json"
    _write(
        m,
        json.dumps(
            {
                "envs": {
                    "lf52": {
                        "task_id": "00d62c1b",
                        "corpus_path": str(task_path),
                        "sha256": "0" * 64,
                    }
                }
            }
        ),
    )
    res = resolve_demos(str(m), "lf52")
    assert res.status == "BLOCKED_DIGEST_MISMATCH"


def test_corpus_missing(tmp_path: Path):
    m = tmp_path / "manifest.json"
    _write(
        m,
        json.dumps(
            {
                "envs": {
                    "lf52": {
                        "task_id": "00d62c1b",
                        "corpus_path": str(tmp_path / "absent.json"),
                        "sha256": "0" * 64,
                    }
                }
            }
        ),
    )
    res = resolve_demos(str(m), "lf52")
    assert res.status == "BLOCKED_CORPUS_UNAVAILABLE"


def test_schema_invalid(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"train": "notalist", "test": []}', encoding="utf-8")
    sha = hashlib.sha256(bad.read_bytes()).hexdigest()
    m = tmp_path / "manifest.json"
    _write(
        m,
        json.dumps(
            {
                "envs": {
                    "lf52": {
                        "task_id": "00d62c1b",
                        "corpus_path": str(bad),
                        "sha256": sha,
                    }
                }
            }
        ),
    )
    res = resolve_demos(str(m), "lf52")
    assert res.status == "BLOCKED_SCHEMA_INVALID"


def test_no_demos(tmp_path: Path):
    data = {"train": [], "test": [{"input": [[0]], "output": [[1]]}]}
    p = tmp_path / "nodemos.json"
    p.write_bytes(json.dumps(data).encode("utf-8"))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    m = tmp_path / "manifest.json"
    _write(
        m,
        json.dumps(
            {
                "envs": {
                    "lf52": {
                        "task_id": "00d62c1b",
                        "corpus_path": str(p),
                        "sha256": sha,
                    }
                }
            }
        ),
    )
    res = resolve_demos(str(m), "lf52")
    assert res.status == "BLOCKED_NO_DEMONSTRATIONS"


def test_load_task_json_rejects_non_2d(tmp_path: Path):
    data = {
        "train": [{"input": [0, 1, 2], "output": [[1], [1]]}],  # 1-D input grid
        "test": [],
    }
    p = tmp_path / "badgrid.json"
    p.write_bytes(json.dumps(data).encode("utf-8"))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(PublicIngressError):
        load_task_json(str(p), sha)


def test_load_task_json_rejects_missing_splits(tmp_path: Path):
    p = tmp_path / "nosplits.json"
    p.write_bytes(json.dumps({"train": []}).encode("utf-8"))
    with pytest.raises(PublicIngressError):
        load_task_json(str(p))
