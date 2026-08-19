# -*- coding: utf-8 -*-
"""Phase 8.32 tests — calibrator ingest (authorized bank -> sealed artifact).

Software verification only (reduced dims); never capability evidence.
"""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from henri_calibrated_action_head import production_activation_eligible
from henri_calibrator_ingest import (
    CalibratorIngestError,
    ingest_bank_to_artifact,
)
from henri_trajectory_bank import TrajectoryBank

D = 2048  # reduced dim (production D = 65536)
LATENT = 64


@pytest.fixture()
def workdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _make_bank(workdir, n=64, run_id="ingest-run", actions=None,
               action_names=None):
    b = TrajectoryBank(workdir, run_id=run_id,
                       provenance="unit-test-authorized",
                       action_names=action_names)
    g = torch.Generator().manual_seed(11)
    for i in range(n):
        w = torch.randn(D, generator=g)
        w = w / w.norm()
        a = actions[i % len(actions)] if actions else f"ACTION{1 + (i % 6)}"
        b.record(w, a, meta={"env": "sp80", "step": i})
    r = b.flush()
    return r


def test_ingest_vocab_mismatch_raises(workdir):
    # Bank vocab = [ACTION3, ACTION4] ONLY (custom vocab): the canonical
    # target [ACTION1..6] is not fully covered -> filter raises.
    r = _make_bank(workdir, actions=["ACTION3"] * 30 + ["ACTION4"] * 30,
                   action_names=["ACTION3", "ACTION4"])
    with pytest.raises(CalibratorIngestError):
        ingest_bank_to_artifact(
            r["npz_path"], r["manifest_path"],
            os.path.join(workdir, "a.json"),
            wave_dim=D, latent_dim=LATENT, action_dim=6)


def test_ingest_too_few_records_after_filter_raises(workdir):
    # Bank vocab extends to ACTION8; exactly ONE canonical ACTION2 row +
    # 40 out-of-vocab ACTION7 rows (n=41 = len(actions), so no cycling).
    # Canonical filter keeps exactly 1 -> M<2 -> record-count check fires.
    r = _make_bank(
        workdir, n=41,
        actions=["ACTION2"] * 1 + ["ACTION7"] * 40,
        action_names=[f"ACTION{i}" for i in range(1, 9)])
    with pytest.raises(CalibratorIngestError) as ei:
        ingest_bank_to_artifact(
            r["npz_path"], r["manifest_path"],
            os.path.join(workdir, "a.json"),
            wave_dim=D, latent_dim=LATENT, action_dim=6)
    assert "records" in str(ei.value).lower()


def test_ingest_end_to_end_writes_sealed_artifact(workdir):
    r = _make_bank(workdir)
    art_path = os.path.join(workdir, "artifact.json")
    art = ingest_bank_to_artifact(
        r["npz_path"], r["manifest_path"], art_path,
        wave_dim=D, latent_dim=LATENT, action_dim=6,
    )
    assert art["schema_id"].startswith("henri.calibrated-action-head")
    assert art["data_source"] == "authorized"
    assert art["action_dim"] == 6
    assert art["train_count"] + art["held_out_count"] == 64
    assert art["weight_sha256"]
    # Artifact file exists and re-loads with self-hash intact
    assert os.path.isfile(art_path)
    with open(art_path) as f:
        raw = json.load(f)
    assert raw["artifact_sha256"] == art["artifact_sha256"]
    # Random waves -> honest gate: not qualified; authorized data_source
    ok, reason = production_activation_eligible(art)
    assert ok is False and reason == "ACTION_HEAD_NOT_QUALIFIED"


def test_ingest_digest_corruption_raises(workdir):
    r = _make_bank(workdir)
    z = np.load(r["npz_path"])
    psi = z["psi"].copy()
    nxt = z["next_wave"].copy()
    onehot = z["actions_onehot"].copy()
    names = z["action_names"].copy()
    z.close()  # release handle BEFORE overwriting (Windows file-lock)
    psi[0, 0] += 1.0
    np.savez(r["npz_path"], psi=psi, next_wave=nxt,
             actions_onehot=onehot, action_names=names)
    with pytest.raises(Exception) as ei:
        ingest_bank_to_artifact(
            r["npz_path"], r["manifest_path"],
            os.path.join(workdir, "a.json"),
            wave_dim=D, latent_dim=LATENT, action_dim=6)
    assert "digest" in str(ei.value).lower()


def test_ingest_max_records_cap(workdir):
    r = _make_bank(workdir, n=100)
    art = ingest_bank_to_artifact(
        r["npz_path"], r["manifest_path"],
        os.path.join(workdir, "a.json"),
        wave_dim=D, latent_dim=LATENT, action_dim=6,
        max_records=40)
    assert art["train_count"] + art["held_out_count"] <= 40


def test_cli_smoke(workdir):
    r = _make_bank(workdir, n=32)
    out = os.path.join(workdir, "cli_artifact.json")
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, "HENRI V2/henri_calibrator_ingest.py",
         "--bank", r["npz_path"], "--manifest", r["manifest_path"],
         "--artifact", out, "--wave-dim", str(D), "--latent-dim", str(LATENT),
         "--action-dim", "6", "--max-records", "24"],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert os.path.isfile(out)
    assert "activation: False" in proc.stdout  # honest gate in CLI output
