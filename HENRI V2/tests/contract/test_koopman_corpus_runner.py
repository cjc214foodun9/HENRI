"""Contract tests: K2 production action-wave provenance gate (default-OFF).

The runner must FAIL CLOSED without live-planner action waves. Placeholder
rings are never verdict-capable: absent/invalid provenance ->
BLOCKED_MISSING_PRODUCTION_ACTION_WAVES and no fit is constructed.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from koopman_corpus_runner import LIVE_ORIGIN, validate_action_wave_manifest  # noqa: E402


def _wave_file(path, num_blocks=8, dtype=np.float32):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((num_blocks, 8)).astype(dtype)
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    np.save(path, w)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return str(path), digest


def _entry(tmp_path, **over):
    path, digest = _wave_file(str(tmp_path / "a0.npy"))
    d = {"path": path, "source": "efe_planner.get_learnable_action_wave",
         "commit": "7d1f7363", "run_id": "run-1", "episode": "e0", "step": 0,
         "shape": [8, 8], "dtype": "float32", "normalization": "unit_rows",
         "encoder": "HENRIVisionEncoder.v2", "basis": "production",
         "digest": digest, "origin": LIVE_ORIGIN}
    d.update(over)
    return d


def _write(tmp_path, entries):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return str(p)


def test_c1_rejects_placeholder_origin(tmp_path):
    e = _entry(tmp_path, origin="placeholder-ring")
    err = validate_action_wave_manifest(_write(tmp_path, {"a0": e}))[1]
    assert err is not None and "origin" in err


def test_c2_rejects_missing_provenance(tmp_path):
    e = _entry(tmp_path)
    del e["commit"]
    err = validate_action_wave_manifest(_write(tmp_path, {"a0": e}))[1]
    assert err is not None and "missing provenance" in err


def test_c3_rejects_digest_mismatch(tmp_path):
    e = _entry(tmp_path, digest="0" * 64)
    err = validate_action_wave_manifest(_write(tmp_path, {"a0": e}))[1]
    assert err is not None and "digest mismatch" in err


def test_c4_rejects_shape_mismatch(tmp_path):
    path, digest = _wave_file(str(tmp_path / "a0.npy"), num_blocks=4)
    e = _entry(tmp_path, path=path, digest=digest, shape=[8, 8])
    err = validate_action_wave_manifest(_write(tmp_path, {"a0": e}))[1]
    assert err is not None and "shape" in err


def test_c5_rejects_dtype_mismatch(tmp_path):
    path, digest = _wave_file(str(tmp_path / "a0.npy"), dtype=np.float64)
    e = _entry(tmp_path, path=path, digest=digest)
    err = validate_action_wave_manifest(_write(tmp_path, {"a0": e}))[1]
    assert err is not None and "dtype" in err


def test_c6_accepts_full_provenance(tmp_path):
    out, err = validate_action_wave_manifest(
        _write(tmp_path, {"a0": _entry(tmp_path)}))
    assert err is None
    assert set(out) == {"a0"}
    assert tuple(out["a0"].shape) == (8, 8)
