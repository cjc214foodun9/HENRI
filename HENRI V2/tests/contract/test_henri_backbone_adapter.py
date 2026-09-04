"""Contract tests for the CLASS51 backbone adapter (henri_backbone_adapter.py).

CPU-only by design: these tests exercise the default-OFF gate, provenance
validation, freeze-for-baseline, SHA-256 verification, and fail-closed CLI
contract without loading a real foundation model. Real-model verification
runs on the remote CUDA target via scripts/backbone_smoke.py.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
import torch.nn as nn

from henri_backbone_adapter import (
    BackboneDisabledError,
    BackboneProvenanceError,
    QwenBackboneAdapter,
    backbone_enabled,
    freeze_for_baseline,
    sha256_file,
    verify_shard_hashes,
    DEFAULT_REVISION,
    ENV_ENABLE_FLAG,
)


def make_model_dir(tmp_path, revision=None):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = {}
    if revision is not None:
        config["revision"] = revision
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model_dir


def make_manifest(tmp_path, revision, shards):
    manifest = {"revision": revision, "files": {}}
    for name, content in shards.items():
        digest = hashlib.sha256(content).hexdigest()
        manifest["files"][name] = {"size": len(content), "lfs_sha256": digest}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_backbone_enabled_flag(monkeypatch):
    monkeypatch.delenv(ENV_ENABLE_FLAG, raising=False)
    assert backbone_enabled() is False
    monkeypatch.setenv(ENV_ENABLE_FLAG, "1")
    assert backbone_enabled() is True


def test_disabled_by_default_raises(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_ENABLE_FLAG, raising=False)
    with pytest.raises(BackboneDisabledError):
        QwenBackboneAdapter(model_dir=str(tmp_path))


def test_missing_model_dir_raises_after_gate(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_ENABLE_FLAG, "1")
    with pytest.raises(BackboneProvenanceError):
        QwenBackboneAdapter(model_dir=str(tmp_path / "nope"))


def test_config_revision_mismatch_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_ENABLE_FLAG, "1")
    model_dir = make_model_dir(tmp_path, revision="deadbeef")
    with pytest.raises(BackboneProvenanceError):
        QwenBackboneAdapter(model_dir=str(model_dir))._check_config_revision()


def test_config_without_revision_passes(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_ENABLE_FLAG, "1")
    model_dir = make_model_dir(tmp_path)  # no explicit revision field
    QwenBackboneAdapter(model_dir=str(model_dir))._check_config_revision()


def test_verify_shards_requires_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_ENABLE_FLAG, "1")
    model_dir = make_model_dir(tmp_path)
    adapter = QwenBackboneAdapter(model_dir=str(model_dir), verify_shards=True)
    with pytest.raises(BackboneProvenanceError):
        adapter._check_config_revision()


def test_verify_shards_match(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_ENABLE_FLAG, "1")
    model_dir = make_model_dir(tmp_path)
    shard = b"fake-shard-bytes-0123456789"
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(shard)
    manifest = make_manifest(
        tmp_path, DEFAULT_REVISION, {"model-00001-of-00001.safetensors": shard}
    )
    adapter = QwenBackboneAdapter(
        model_dir=str(model_dir), manifest_path=str(manifest), verify_shards=True
    )
    adapter._check_config_revision()
    assert adapter.telemetry.manifest_sha256 is not None


def test_verify_shards_mismatch_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_ENABLE_FLAG, "1")
    model_dir = make_model_dir(tmp_path)
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"original-bytes")
    manifest = make_manifest(
        tmp_path, DEFAULT_REVISION, {"model-00001-of-00001.safetensors": b"different-bytes"}
    )
    with pytest.raises(BackboneProvenanceError):
        verify_shard_hashes(model_dir, manifest, expected_revision=DEFAULT_REVISION)


def test_verify_revision_mismatch_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_ENABLE_FLAG, "1")
    model_dir = make_model_dir(tmp_path)
    manifest = make_manifest(tmp_path, "other-revision", {})
    with pytest.raises(BackboneProvenanceError):
        verify_shard_hashes(model_dir, manifest, expected_revision=DEFAULT_REVISION)


def test_sha256_file_known_digest(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"hello")
    assert sha256_file(path) == hashlib.sha256(b"hello").hexdigest()


def test_freeze_for_baseline_zero_trainable():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    total, trainable_before = freeze_for_baseline(model)
    assert total == sum(p.numel() for p in model.parameters())
    assert trainable_before == total
    assert all(p.requires_grad is False for p in model.parameters())
    assert not model.training


def test_main_cli_disabled_fails_closed(monkeypatch):
    monkeypatch.delenv(ENV_ENABLE_FLAG, raising=False)
    from henri_backbone_adapter import main

    with pytest.raises(BackboneDisabledError):
        main(["--text", "hi", "--model-dir", "."])


def test_main_cli_provenance_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_ENABLE_FLAG, "1")
    from henri_backbone_adapter import main

    with pytest.raises(BackboneProvenanceError):
        main(["--text", "hi", "--model-dir", str(tmp_path / "missing")])
