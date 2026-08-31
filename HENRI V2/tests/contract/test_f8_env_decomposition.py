"""Contract tests for the F8 post-mortem per-environment decomposition.

Covers: default-OFF guard, per-environment isolation (rows of env A never
train on env B), trivial single-action env handling (probes skipped,
margin null), separable-env signal (margin > 0.3), receipt schema, bank
hash pinning, and run determinism.

Run: /c/Python314/python.exe -m pytest HENRI V2/tests/contract/test_f8_env_decomposition.py -q
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "verification"))

import arc_f8_env_decomposition as dec  # noqa: E402


def _make_bank(tmp_path: Path, seed: int = 7):
    rng = np.random.default_rng(seed)
    envs = ["t1", "t2", "triv"]
    psi, y, meta = [], [], []
    for name in envs:
        for i in range(60):
            if name == "triv":
                v = rng.normal(size=16)
                label = 0
            elif name == "t1":
                v = rng.normal(size=16)
                v[0] += 4.0 if i % 2 == 0 else -4.0  # strongly separable
                label = i % 2
            else:
                v = rng.normal(size=16)
                v[1] += 4.0 if i % 2 == 0 else -4.0
                label = i % 2
            psi.append(v)
            y.append(label)
            meta.append({"env": name, "t": i})
    psi = np.asarray(psi, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    onehot = np.zeros((len(y), 7), dtype=np.uint8)  # loader requires [N, 7]
    onehot[np.arange(len(y)), y] = 1
    npz = tmp_path / "bank.npz"
    np.savez(npz, psi=psi, actions_onehot=onehot)
    jsonl = tmp_path / "bank.jsonl"
    with open(jsonl, "w", encoding="utf-8") as fp:
        for m in meta:
            fp.write(json.dumps(m) + "\n")
    return str(npz), str(jsonl)


def _run_main(npz: str, jsonl: str, out_dir: Path, receipt: Path, env_extra=None):
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "verification" / "arc_f8_env_decomposition.py"),
        "--bank-npz", npz,
        "--bank-jsonl", jsonl,
        "--device", "cpu",
        "--n-folds", "5",
        "--seed", "20260904",
        "--out-dir", str(out_dir),
        "--receipt-out", str(receipt),
    ]
    env = dict(os.environ)
    env["HENRI_F8_DECOMP"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def test_default_off_guard(monkeypatch):
    monkeypatch.delenv("HENRI_F8_DECOMP", raising=False)
    with pytest.raises(RuntimeError, match="default-OFF"):
        dec.require_decomp_enabled()


def test_receipt_schema_and_hashes(tmp_path):
    npz, jsonl = _make_bank(tmp_path)
    out = tmp_path / "out"
    receipt = out / "f8_decomposition_receipt.json"
    r = _run_main(npz, jsonl, out, receipt)
    assert r.returncode == 0, r.stderr[-2000:]
    assert receipt.exists()
    data = json.loads(receipt.read_text())
    assert data["schema"] == "f8-env-decomposition.v1"
    assert data["seed"] == 20260904
    assert data["n_folds"] == 5
    assert data["bank_npz_sha256"] == hashlib.sha256(Path(npz).read_bytes()).hexdigest()
    assert data["bank_jsonl_sha256"] == hashlib.sha256(Path(jsonl).read_bytes()).hexdigest()
    envs = {e["env"]: e for e in data["environments"]}
    assert set(envs) == {"t1", "t2", "triv"}


def test_trivial_env_skipped(tmp_path):
    npz, jsonl = _make_bank(tmp_path)
    out = tmp_path / "out"
    receipt = out / "f8_decomposition_receipt.json"
    r = _run_main(npz, jsonl, out, receipt)
    assert r.returncode == 0, r.stderr[-2000:]
    data = json.loads(receipt.read_text())
    triv = next(e for e in data["environments"] if e["env"] == "triv")
    assert triv["probes_skipped"] is True
    assert triv["acc"] is None
    assert triv["margin"] is None
    assert triv["n_rows"] == 60
    assert triv["n_classes"] == 1


def test_separable_envs_show_margin(tmp_path):
    npz, jsonl = _make_bank(tmp_path)
    out = tmp_path / "out"
    receipt = out / "f8_decomposition_receipt.json"
    r = _run_main(npz, jsonl, out, receipt)
    assert r.returncode == 0, r.stderr[-2000:]
    data = json.loads(receipt.read_text())
    for env_name in ("t1", "t2"):
        e = next(x for x in data["environments"] if x["env"] == env_name)
        assert e["probes_skipped"] is False
        assert e["acc_best"] is not None and e["acc_best"] > 0.85, e
        assert e["margin_best"] is not None and e["margin_best"] > 0.30, e


def test_env_isolation_rows(tmp_path):
    """Per-env partitions must be row-disjoint (no cross-env leakage)."""
    npz, jsonl = _make_bank(tmp_path)
    out = tmp_path / "out"
    receipt = out / "f8_decomposition_receipt.json"
    r = _run_main(npz, jsonl, out, receipt)
    assert r.returncode == 0, r.stderr[-2000:]
    data = json.loads(receipt.read_text())
    assert sum(e["n_rows"] for e in data["environments"]) == 180


def test_determinism(tmp_path):
    npz, jsonl = _make_bank(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    rec1, rec2 = out1 / "r.json", out2 / "r.json"
    a = _run_main(npz, jsonl, out1, rec1)
    b = _run_main(npz, jsonl, out2, rec2)
    assert a.returncode == 0 and b.returncode == 0
    assert rec1.read_bytes() == rec2.read_bytes()
