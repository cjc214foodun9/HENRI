# -*- coding: utf-8 -*-
"""Gate-4 calibrated-egress contract tests (CPU, default-OFF, fail-closed).

Covers the Sol reference-3 verification ladder items 1-6:
  - generator determinism + lineage (zero overlap with arcade/banks/eval caches)
  - no-dense-allocation guard (no [65536, 65536])
  - artifact tamper/compatibility rejection
  - action ordering / OOV validation
  - generic egress ON + calibrated action head OFF -> diagnostic_only=true,
    score_eligible=false (ACTION_HEAD_NOT_CALIBRATED)
  - negative guard probe reaches the intended action-head boundary
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HENRI = REPO / "HENRI V2"
sys.path.insert(0, str(HENRI))

from henri_calibrated_egress import (  # noqa: E402
    CANONICAL_ARC_ACTIONS,
    CalibratedEgressError,
    egress_state,
    load_calibrated_artifact,
    validate_action_schema,
)
from henri_calibrated_egress import dry_run_cli  # noqa: E402

GEN = HENRI / "experiments" / "verification" / "gate4_zero_lineage_generate.py"
MANIFEST = HENRI / "experiments" / "verification" / "gate4_zero_lineage_manifest.json"
TASKS = HENRI / "experiments" / "verification" / "gate4_zero_lineage_tasks.json"

PROHIBITED = {
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
    "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
    "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
}


def _lf(raw: bytes) -> bytes:
    return raw.replace(bytes((13, 10)), b"\n")


# --------------------------------------------------------------------------
# 1. Generator determinism + lineage
# --------------------------------------------------------------------------

def test_generator_outputs_exist_and_manifest_hashes_match():
    assert GEN.is_file() and MANIFEST.is_file() and TASKS.is_file()
    manifest = json.loads(_lf(MANIFEST.read_bytes()))
    tasks = json.loads(_lf(TASKS.read_bytes()))
    assert manifest["task_count"] == 30
    assert len(tasks) == 30
    # Manifest self-hash consistent with the file bytes (excludes
    # manifest_sha256 and the non-deterministic generated_utc).
    m2 = json.loads(_lf(MANIFEST.read_bytes()))
    import hashlib
    blob = json.dumps(
        {k: v for k, v in m2.items() if k not in ("manifest_sha256", "generated_utc")},
        indent=1, ensure_ascii=False).encode("utf-8")
    assert hashlib.sha256(blob).hexdigest() == manifest["manifest_sha256"]
    # Tasks file hash matches manifest.
    import hashlib
    assert hashlib.sha256(_lf(TASKS.read_bytes())).hexdigest() == manifest["tasks_sha256"]


def test_generator_seed_split_disjoint_and_deterministic():
    manifest = json.loads(_lf(MANIFEST.read_bytes()))
    dev, final = set(manifest["dev_seeds"]), set(manifest["final_seeds"])
    assert dev.isdisjoint(final)
    assert len(dev) == 20 and len(final) == 10
    assert sorted(manifest["seed_list"]) == sorted(dev | final)
    # Determinism: rerun the generator, compare manifest hash.
    r = subprocess.run(
        [sys.executable, str(GEN)],
        capture_output=True, text=True, cwd=str(REPO), timeout=120,
    )
    assert r.returncode == 0, r.stderr
    import hashlib
    m2 = json.loads(_lf(MANIFEST.read_bytes()))
    assert m2["manifest_sha256"] == manifest["manifest_sha256"], (
        "generator rerun changed manifest (nondeterminism)")


def test_generator_zero_lineage():
    manifest = json.loads(_lf(MANIFEST.read_bytes()))
    lineage = manifest["lineage"]
    assert lineage["lineage_verdict"] == "CLEAN"
    assert lineage["arcade_env_id_overlap"] == []
    assert lineage["bank_env_overlap"] == []
    assert not lineage["eval_cache_read"]
    assert not lineage["arcade_instantiation"]
    tasks = json.loads(_lf(TASKS.read_bytes()))
    flat = " ".join(t["task_id"] for t in tasks)
    for pid in PROHIBITED:
        assert pid not in flat, f"prohibited id {pid} leaked into suite"
    # No overlap with bank envs (dc22, m0r0) either (they are also prohibited).
    assert "dc22" not in flat and "m0r0" not in flat


# --------------------------------------------------------------------------
# 2. No-dense-allocation guard
# --------------------------------------------------------------------------

def test_module_has_no_dense_wave_allocation():
    src = (HENRI / "henri_calibrated_egress.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = (
                f.attr if isinstance(f, ast.Attribute)
                else f.id if isinstance(f, ast.Name) else ""
            )
            if name in ("eye", "outer", "zeros", "ones", "empty"):
                args = [a for a in node.args if isinstance(a, ast.Constant)]
                if any(isinstance(a.value, (list, tuple)) and len(a.value) >= 2
                       for a in args):
                    forbidden.append((name, getattr(node, "lineno", 0)))
    assert not forbidden, f"potential dense allocation sites: {forbidden}"
    # Calibration delegates to the SVD-form ingest pipeline (no [D,D]).
    assert "ingest_bank_to_artifact" in src


# --------------------------------------------------------------------------
# 3. Artifact tamper / compatibility rejection
# --------------------------------------------------------------------------

def _qualified_artifact(tmp_path, **overrides) -> Path:
    art = {
        "schema_id": "henri.calibrated-action-head.v1",
        "version": "1",
        "status": "ON",
        "is_qualified": True,
        "data_source": "authorized",
        "calibration_mse_heldout": 0.01,
        "sagnac_stress_proxy_action_l2": 0.05,
        "wave_dim": 65536,
        "latent_dim": 2048,
        "action_dim": 6,
        "action_ordering": list(CANONICAL_ARC_ACTIONS),
        "artifact_sha256": "",
    }
    art.update(overrides)
    blob = json.dumps({k: v for k, v in art.items() if k != "artifact_sha256"},
                      sort_keys=True).encode("utf-8")
    import hashlib
    art["artifact_sha256"] = hashlib.sha256(blob).hexdigest()
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(art, indent=1), encoding="utf-8")
    return p


def test_artifact_tamper_rejected(tmp_path):
    p = _qualified_artifact(tmp_path)
    art = json.loads(p.read_text(encoding="utf-8"))
    art["is_qualified"] = False
    p.write_text(json.dumps(art, indent=1), encoding="utf-8")
    with pytest.raises(CalibratedEgressError, match="self-hash mismatch"):
        load_calibrated_artifact(str(p))


def test_artifact_live_ingest_post_seal_fields_accepted(tmp_path):
    # The live calibrator appends bank_* fields AFTER sealing artifact_sha256.
    # The loader must hash the sealed payload only; tampering with sealed
    # fields still fails.
    art = {
        "schema_id": "henri.calibrated-action-head.v1",
        "version": "1",
        "status": "OFF",
        "is_qualified": False,
        "data_source": "authorized",
        "calibration_mse_heldout": 25.1,
        "sagnac_stress_proxy_action_l2": 12.3,
        "wave_dim": 65536,
        "latent_dim": 2048,
        "action_dim": 6,
        "action_ordering": list(CANONICAL_ARC_ACTIONS),
        "artifact_sha256": "",
    }
    import hashlib
    blob = json.dumps({k: v for k, v in art.items() if k != "artifact_sha256"},
                      sort_keys=True).encode("utf-8")
    art["artifact_sha256"] = hashlib.sha256(blob).hexdigest()
    # Post-seal fields appended by ingest_bank_to_artifact.
    art["bank_npz_sha256"] = "b" * 64
    art["bank_dataset_digest"] = "d" * 64
    p = tmp_path / "live_artifact.json"
    p.write_text(json.dumps(art, indent=1), encoding="utf-8")
    loaded = load_calibrated_artifact(str(p))
    assert loaded["is_qualified"] is False
    assert loaded["bank_npz_sha256"] == "b" * 64
    # Tampering with a SEALED field still fails despite the bank fields.
    art["is_qualified"] = True
    p.write_text(json.dumps(art, indent=1), encoding="utf-8")
    with pytest.raises(CalibratedEgressError, match="self-hash mismatch"):
        load_calibrated_artifact(str(p))


def test_artifact_action_ordering_validation(tmp_path):
    p = _qualified_artifact(tmp_path, action_ordering=[
        "ACTION2", "ACTION1", "ACTION3", "ACTION4", "ACTION5", "ACTION6"])
    with pytest.raises(CalibratedEgressError, match="action_ordering"):
        load_calibrated_artifact(str(p))


def test_artifact_oov_action_dim_rejected(tmp_path):
    # Canonical ordering retained, but action_dim != 6 -> must fail on action_dim.
    p = _qualified_artifact(tmp_path, action_dim=8)
    with pytest.raises(CalibratedEgressError, match="action_dim"):
        load_calibrated_artifact(str(p))


# --------------------------------------------------------------------------
# 4. Egress state machine (fail-closed)
# --------------------------------------------------------------------------

def test_egress_state_negative_no_artifact():
    s = egress_state(None)
    assert s["score_eligible"] is False
    assert s["diagnostic_only"] is True
    assert s["score_block_reason"] == "ACTION_HEAD_NOT_CALIBRATED"


def test_egress_state_qualified_but_head_inactive(tmp_path):
    art = json.loads(_qualified_artifact(tmp_path).read_text(encoding="utf-8"))
    s = egress_state(art, trained_head_active=False)
    assert s["score_eligible"] is False
    assert s["diagnostic_only"] is True
    assert s["score_block_reason"] == "ACTION_HEAD_NOT_CALIBRATED"


def test_egress_state_synthetic_never_activates(tmp_path):
    p = _qualified_artifact(tmp_path, data_source="synthetic_fixture")
    art = json.loads(p.read_text(encoding="utf-8"))
    s = egress_state(art, trained_head_active=True, task_validated=True)
    assert s["score_eligible"] is False
    assert s["score_block_reason"] == "ACTION_HEAD_SYNTHETIC_ONLY"


def test_egress_state_needs_task_validation(tmp_path):
    art = json.loads(_qualified_artifact(tmp_path).read_text(encoding="utf-8"))
    s = egress_state(art, trained_head_active=True, task_validated=False)
    assert s["score_eligible"] is False
    assert s["score_block_reason"] == "ACTION_HEAD_NOT_TASK_VALIDATED"


def test_egress_state_full_positive_only_with_task_validation(tmp_path):
    art = json.loads(_qualified_artifact(tmp_path).read_text(encoding="utf-8"))
    s = egress_state(art, trained_head_active=True, task_validated=True)
    assert s["score_eligible"] is True
    assert s["egress_mode"] == "CALIBRATED"


# --------------------------------------------------------------------------
# 5. Dry-run CLI negative probe (generic egress ON + head OFF)
# --------------------------------------------------------------------------

def test_dry_run_negative_probe_no_artifact(capsys):
    rc = dry_run_cli(["--wave-dim", "65536"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2  # BLOCKED
    assert out["verdict"] == "BLOCKED"
    assert out["score_eligible"] is False
    assert out["diagnostic_only"] is True
    assert out["score_block_reason"] == "ACTION_HEAD_NOT_CALIBRATED"


def test_dry_run_negative_probe_unqualified_artifact(tmp_path, capsys):
    p = _qualified_artifact(tmp_path, is_qualified=False, status="OFF",
                            calibration_mse_heldout=24.2,
                            sagnac_stress_proxy_action_l2=12.0)
    rc = dry_run_cli(["--artifact", str(p), "--wave-dim", "65536"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert out["verdict"] == "BLOCKED"
    assert out["score_eligible"] is False
    # The FIRST violated condition determines the reason: not qualified.
    assert out["score_block_reason"] == "ACTION_HEAD_NOT_QUALIFIED"
    assert out["artifact_qualified"] is False


def test_dry_run_negative_probe_tampered_artifact(tmp_path, capsys):
    p = _qualified_artifact(tmp_path)
    art = json.loads(p.read_text(encoding="utf-8"))
    art["is_qualified"] = False
    p.write_text(json.dumps(art, indent=1), encoding="utf-8")
    rc = dry_run_cli(["--artifact", str(p), "--wave-dim", "65536"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert out["verdict"] == "BLOCKED"
    assert out["artifact_loaded"] is False
    assert "self-hash mismatch" in out["artifact_error"]


def test_dry_run_positive_only_with_task_validation_flag(tmp_path, capsys):
    # Simulation-only: qualified artifact + trained-active head + the
    # --task-validated flag. The flag is never used for real promotion.
    import torch
    from arc_action_head import ActionHead
    p = _qualified_artifact(tmp_path)
    head = ActionHead(d_hidden=2048, n_actions=6)
    sd = {
        "head.weight": head.head.weight.detach().clone(),
        "head.bias": head.head.bias.detach().clone(),
    }
    ckpt = tmp_path / "henri_action_head.pt"
    torch.save({
        "state_dict": sd,
        "calibration_dataset_digest": "test-synthetic-fixture",
        "d_model": 65536,
    }, ckpt)
    rc = dry_run_cli(["--artifact", str(p), "--checkpoint", str(ckpt),
                      "--wave-dim", "65536", "--latent-dim", "2048",
                      "--task-validated"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["verdict"] == "PASS"
    assert out["score_eligible"] is True
    assert out["trained_action_head_active"] is True
    # Without the flag, the SAME artifact + head stays blocked.
    rc2 = dry_run_cli(["--artifact", str(p), "--checkpoint", str(ckpt),
                       "--wave-dim", "65536", "--latent-dim", "2048"])
    out2 = json.loads(capsys.readouterr().out)
    assert rc2 == 2
    assert out2["score_block_reason"] == "ACTION_HEAD_NOT_TASK_VALIDATED"


# --------------------------------------------------------------------------
# 6. export-head refusal (unqualified / synthetic never export a trained head)
# --------------------------------------------------------------------------

def test_export_head_refuses_unqualified_artifact(tmp_path):
    from henri_calibrated_egress import CalibratedEgressError, export_head_cli
    p = _qualified_artifact(tmp_path, is_qualified=False, status="OFF",
                            calibration_mse_heldout=24.2)
    with pytest.raises(CalibratedEgressError, match="unqualified"):
        export_head_cli(["--artifact", str(p), "--bank", "x.npz",
                         "--checkpoint-out", str(tmp_path / "h.pt")])


def test_export_head_refuses_synthetic_artifact(tmp_path):
    from henri_calibrated_egress import CalibratedEgressError, export_head_cli
    p = _qualified_artifact(tmp_path, data_source="synthetic_fixture")
    with pytest.raises(CalibratedEgressError, match="synthetic"):
        export_head_cli(["--artifact", str(p), "--bank", "x.npz",
                         "--checkpoint-out", str(tmp_path / "h.pt")])


# --------------------------------------------------------------------------
# 7. CLI entry regression (main() with no argv must not crash)
# --------------------------------------------------------------------------

def test_main_entry_no_argv_does_not_typeerror():
    from henri_calibrated_egress import main
    with pytest.raises(SystemExit) as ei:
        main()
    assert ei.value.code == 2  # argparse usage/help path


def test_main_dispatch_dry_run_subcommand(capsys):
    from henri_calibrated_egress import main
    rc = main(["dry-run", "--wave-dim", "65536"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert out["verdict"] == "BLOCKED"
    assert out["score_block_reason"] == "ACTION_HEAD_NOT_CALIBRATED"
