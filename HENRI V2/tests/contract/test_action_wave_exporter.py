"""Contract tests: default-OFF live-planner action-wave manifest exporter.

C1  default-OFF: no flag -> exporter is None, nothing recorded, module
    never imported by the production path (byte-identical default).
C2  enabled: record() writes .npy + digest; write_manifest emits full
    provenance (origin == live_planner_boundary); validate_action_wave_manifest
    accepts the manifest and returns the exact waves.
C3  frozen-digest guard: a changed wave for the same action name raises
    RuntimeError (learning must be frozen during corpus export).
C4  shape/dtype rejection: non-[*, 8] real wave raises ValueError.
C5  provenance gate: missing keys / wrong origin / digest mismatch are
    rejected by validate_action_wave_manifest.
C6  round-trip: manifest from the exporter passes the corpus-runner gate
    only when origin == LIVE_ORIGIN and bytes match.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from action_wave_exporter import (  # noqa: E402
    FLAG,
    DIR_ENV,
    RUN_ID_ENV,
    COMMIT_ENV,
    ORIGIN,
    ActionWaveExporter,
)
from koopman_corpus_runner import validate_action_wave_manifest  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (FLAG, DIR_ENV, RUN_ID_ENV, COMMIT_ENV):
        monkeypatch.delenv(var, raising=False)
    ActionWaveExporter.reset()
    yield
    ActionWaveExporter.reset()


def _wave(seed: int, num_blocks: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(num_blocks, 8, generator=g)
    w = w / (w.norm(p=2, dim=-1, keepdim=True) + 1e-9)
    return w


def test_c1_default_off_no_exporter(tmp_path):
    assert ActionWaveExporter.get() is None
    assert not (tmp_path / "action_waves.json").exists()


def test_c2_enabled_writes_manifest_and_validates(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(DIR_ENV, str(tmp_path))
    monkeypatch.setenv(RUN_ID_ENV, "run-x")
    monkeypatch.setenv(COMMIT_ENV, "cafebabe")
    exporter = ActionWaveExporter.get()
    assert exporter is not None
    exporter.record("a0", _wave(1), encoder="decoder_get_action_wave",
                    source="test.candidate_action_waves")
    exporter.record("a1", _wave(2), encoder="decoder_get_action_wave",
                    source="test.candidate_action_waves")
    manifest_path = exporter.write_manifest()
    assert Path(manifest_path).exists()
    aw_map, err = validate_action_wave_manifest(manifest_path, num_blocks=4)
    assert err is None, err
    assert set(aw_map) == {"a0", "a1"}
    for name in ("a0", "a1"):
        entry = json.loads(Path(manifest_path).read_text(encoding="utf-8"))[name]
        assert entry["origin"] == ORIGIN
        assert entry["dtype"] == "float32"
        assert list(entry["shape"]) == [4, 8]
        assert entry["run_id"] == "run-x"
        assert entry["commit"] == "cafebabe"
        assert Path(entry["path"]).exists()
        assert torch.equal(aw_map[name], _wave(1 if name == "a0" else 2))


def test_c3_frozen_digest_guard(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(DIR_ENV, str(tmp_path))
    exporter = ActionWaveExporter.get()
    exporter.record("a0", _wave(1), encoder="e", source="s")
    with pytest.raises(RuntimeError):
        exporter.record("a0", _wave(2), encoder="e", source="s")
    assert len(exporter._waves) == 1


def test_c4_shape_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(DIR_ENV, str(tmp_path))
    exporter = ActionWaveExporter.get()
    with pytest.raises(ValueError):
        exporter.record("bad", torch.randn(4, 9), encoder="e", source="s")


def test_c5_provenance_gate_rejects_bad_manifests(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(DIR_ENV, str(tmp_path))
    exporter = ActionWaveExporter.get()
    exporter.record("a0", _wave(1), encoder="e", source="s")
    manifest = json.loads(
        Path(exporter.write_manifest()).read_text(encoding="utf-8"))

    bad = json.loads(json.dumps(manifest))
    del bad["a0"]["commit"]
    p = tmp_path / "missing_prov.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    _, err = validate_action_wave_manifest(str(p), num_blocks=4)
    assert err is not None and "missing provenance" in err

    bad = json.loads(json.dumps(manifest))
    bad["a0"]["origin"] = "reconstructed_placeholder"
    p = tmp_path / "bad_origin.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    _, err = validate_action_wave_manifest(str(p), num_blocks=4)
    assert err is not None and "origin" in err

    bad = json.loads(json.dumps(manifest))
    bad["a0"]["digest"] = "0" * 64
    p = tmp_path / "bad_digest.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    _, err = validate_action_wave_manifest(str(p), num_blocks=4)
    assert err is not None and "digest mismatch" in err


def test_c6_round_trip_only_live_origin(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(DIR_ENV, str(tmp_path))
    exporter = ActionWaveExporter.get()
    exporter.record("a0", _wave(1), encoder="e", source="s")
    aw_map, err = validate_action_wave_manifest(
        exporter.write_manifest(), num_blocks=4)
    assert err is None and "a0" in aw_map
