"""Contract test: the field-channel artifact overlay must be SOUND.

The 41 `field_channel_checkpoints/*.pt` files are gitignored external overlays, so
no commit check ever covered them. Measured 2026-09-16: 40 files are byte-identical
in size (104,859,636 B) and ONE is truncated (55,287,808 B) and unreadable. Nothing
detected it for an unknown period.

SKIP-WHEN-ABSENT IS LOAD-BEARING. The overlay is NOT in Git, so a clean CI checkout
legitimately has no such directory. Failing there would convert a correct clean run
into a false red -- the bucket-4 failure class. So absence => SKIP with a named
reason, never FAIL.

The test asserts the PREFLIGHT CONTRACT, not the artifacts' scientific worth:
  - every file present loads (ZIP intact + torch.load succeeds when torch exists)
  - byte count matches the measured invariant
  - the preflight reports failures instead of raising on them
  - the write-once manifest guard refuses to clobber non-trivial evidence
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]                  # ...\HENRI V2
VERIF = REPO / "experiments" / "verification"
sys.path.insert(0, str(VERIF))

CKPT_DIR = REPO / "field_channel_checkpoints"
EXPECTED_BYTES = 104_859_636

pytestmark = pytest.mark.contract


def _preflight_module():
    import importlib
    import field_channel_preflight as fcp
    importlib.reload(fcp)
    return fcp


def test_preflight_skips_when_overlay_absent(tmp_path, monkeypatch):
    """Absence must SKIP (exit 2), not FAIL -- a clean checkout has no overlay."""
    fcp = _preflight_module()
    monkeypatch.setattr(fcp, "CKPT_DIR", tmp_path / "does_not_exist")
    rc = fcp.main(["--verify", "--quiet"])
    assert rc == fcp.EXIT_SKIP, f"absent overlay must SKIP (2), got rc={rc}"


def test_preflight_reports_but_does_not_raise_on_corrupt_file(tmp_path, monkeypatch):
    """A corrupt artifact is REPORTED with a typed status, never an exception."""
    fcp = _preflight_module()
    d = tmp_path / "fc"
    d.mkdir()
    bad = d / "field_channel_broken.pt"
    bad.write_bytes(b"not a zip archive at all")
    monkeypatch.setattr(fcp, "CKPT_DIR", d)
    rc = fcp.main(["--verify", "--quiet"])
    assert rc == fcp.EXIT_FAIL, "a corrupt artifact must FAIL the verify gate"
    rec = fcp.probe(bad)
    assert rec["status"] != "OK", "corrupt file must not report OK"
    assert "detail" in rec and rec["detail"], "failure must be named, not silent"


def test_preflight_flags_byte_count_mismatch(tmp_path, monkeypatch):
    """Byte count is an invariant; a mismatch is a typed failure, not a pass."""
    fcp = _preflight_module()
    d = tmp_path / "fc"
    d.mkdir()
    import zipfile
    f = d / "field_channel_short.pt"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("x", b"tiny")
    rec = fcp.probe(f)
    assert rec["bytes"] != EXPECTED_BYTES
    assert rec["status"] in ("BYTE_COUNT_MISMATCH", "ZIP_UNREADABLE", "LOAD_FAILED")


def test_write_once_guard_refuses_to_clobber(tmp_path):
    """CLASS52: an artifact recording state must not be silently replaced."""
    fcp = _preflight_module()
    man = tmp_path / "m.json"
    recs = [{"name": "a.pt", "bytes": 1, "status": "OK"},
            {"name": "b.pt", "bytes": 1, "status": "OK"}]
    first = fcp.write_manifest(recs, man)
    before = first.read_bytes()
    second = fcp.write_manifest(recs, man)
    assert second != man, "writer must divert to a timestamped sibling"
    assert man.read_bytes() == before, "existing manifest must stay byte-identical"


@pytest.mark.skipif(not CKPT_DIR.is_dir(), reason="field-channel overlay absent (gitignored external state)")
def test_live_overlay_matches_the_measured_invariant():
    """Every present file carries the measured byte count, and failures are named.

    This documents the KNOWN truncated artifact rather than hiding it: the test
    asserts the preflight DETECTS it. When the artifact is legitimately repaired or
    retired, this test must be updated with the new evidence -- do not delete it.
    """
    fcp = _preflight_module()
    files = sorted(CKPT_DIR.glob("*.pt"))
    assert files, "overlay directory exists but holds no *.pt files"
    records = [fcp.probe(p) for p in files]
    bad = [r for r in records if r["status"] != "OK"]
    ok = [r for r in records if r["status"] == "OK"]
    assert ok, "no readable artifact in the overlay"
    for r in ok:
        assert r["bytes"] == EXPECTED_BYTES, f"{r['name']} deviates from the invariant"
    # The known truncated artifact: assert it is NAMED if present, never silently ignored.
    names = {r["name"] for r in bad}
    if names:
        assert all(r.get("detail") for r in bad), "every failure must carry a detail"
        print(f"KNOWN BAD field-channel artifacts ({len(bad)}): {sorted(names)}")
