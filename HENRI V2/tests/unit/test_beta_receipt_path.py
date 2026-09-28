"""Guard the committed-receipt path against ad-hoc and background overwrite.

INCIDENT THIS TESTS (measured 2026-09-27): a backgrounded run of the calibration
tool targeted the COMMITTED, ledger-cited receipt path simply because the module
default IS that path. It missed clobbering committed evidence only because its
crash sat BEFORE its write -- accidental protection. These tests make the
protection structural: the default stays byte-identical for normal reproduction,
but any override redirects, and a MALFORMED override RAISES rather than falling
back to the committed default.
"""
import importlib.util
import os
import sys
import pytest

C = None
for d in (os.path.dirname(os.path.abspath(__file__)),):
    p = d
    for _ in range(6):
        if os.path.exists(os.path.join(p, "tools", "hopfield_beta_calibration.py")):
            C = p
            break
        p = os.path.dirname(p)
    if C:
        break
assert C, "HENRI V2 root not found"
if C not in sys.path:
    sys.path.insert(0, C)

_spec = importlib.util.spec_from_file_location(
    "hbcal_under_test", os.path.join(C, "tools", "hopfield_beta_calibration.py"))
hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hb)

TMP = os.path.join(os.path.expandvars(r"%LOCALAPPDATA%"), "Temp")


def test_default_resolves_to_the_module_constant(monkeypatch):
    monkeypatch.delenv("HENRI_RECEIPT_DIR", raising=False)
    assert os.path.abspath(hb.resolve_receipt_path(None)) == os.path.abspath(hb.REC)


def test_out_override_wins(monkeypatch):
    monkeypatch.delenv("HENRI_RECEIPT_DIR", raising=False)
    t = os.path.join(TMP, "hb_override.json")
    assert os.path.abspath(hb.resolve_receipt_path(t)) == os.path.abspath(t)


def test_env_override_honoured(monkeypatch):
    t = os.path.join(TMP, "hb_env.json")
    monkeypatch.setenv("HENRI_RECEIPT_DIR", t)
    assert os.path.abspath(hb.resolve_receipt_path(None)) == os.path.abspath(t)


def test_out_beats_env(monkeypatch):
    monkeypatch.setenv("HENRI_RECEIPT_DIR", os.path.join(TMP, "hb_env2.json"))
    t = os.path.join(TMP, "hb_out2.json")
    assert os.path.abspath(hb.resolve_receipt_path(t)) == os.path.abspath(t)


def test_directory_override_keeps_the_basename(monkeypatch):
    monkeypatch.delenv("HENRI_RECEIPT_DIR", raising=False)
    got = hb.resolve_receipt_path(TMP + os.sep)
    assert os.path.basename(got) == os.path.basename(hb.REC)
    assert os.path.dirname(got) == os.path.abspath(TMP)


@pytest.mark.parametrize("bad", ["", "   "])
def test_malformed_override_RAISES_never_falls_back(bad, monkeypatch):
    """The exact failure mode: a bad override must NOT silently write committed evidence."""
    monkeypatch.delenv("HENRI_RECEIPT_DIR", raising=False)
    with pytest.raises(ValueError):
        hb.resolve_receipt_path(bad)


def test_verdict_block_does_not_crash_before_writing_the_receipt():
    """BEHAVIOURAL guard, not a text fragment.

    The original form asserted `'sel["M%d" % m]' not in src`. That is wrong twice:
    (a) it fires on a COMMENT that quotes the bug, and (b) a literal left in DEAD
    code would satisfy the letter while the live verdict path still crashed before
    its write. The regression this guards is observable: the tool must RUN to
    completion and produce a receipt.
    """
    import subprocess
    import sys as _sys
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "beta_verdict_regression.json")
    if os.path.exists(tmp):
        os.remove(tmp)
    r = subprocess.run([_sys.executable, os.path.join("tools", "hopfield_beta_calibration.py"),
                        "--out", tmp], cwd=C, capture_output=True, text=True, timeout=560)
    blob = (r.stdout or "") + (r.stderr or "")
    assert "Traceback" not in blob, "the tool crashed:\n" + blob[-900:]
    assert r.returncode == 0, "tool rc=%d" % r.returncode
    assert os.path.exists(tmp), "no receipt was written -> the write did not run"
    import json as _json
    d = _json.loads(open(tmp, encoding="utf-8").read())
    assert "verdict" in d and "pairwise_counts" in d
    print("  verdict:", d["verdict"])
