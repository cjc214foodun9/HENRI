"""Contract tests: receipt-path override for the verification runners.

WHY THESE EXIST
    The runners under experiments/verification/ hardcoded their output path and wrote
    it UNCONDITIONALLY. Those files are committed and cited in commit-message digest
    ledgers, so an experimental or background invocation overwrites a ledger-cited
    artifact. That happened twice in one session (a stale --with-demo job clobbered
    trilevel_loop_observed.json after it was committed), and each occurrence had to be
    found by re-deriving digests and repaired with `git checkout --`.

    These tests pin the contract that prevents it:
        priority  --out  >  HENRI_RECEIPT_DIR  >  default (byte-identical)
        a bad override RAISES -- it must NEVER fall back to the ledger-cited path,
        because silently writing to the committed location is the exact failure
        this module exists to prevent.

NOTHING HERE WRITES TO A COMMITTED PATH. Every redirect targets tmp_path, and the
integration test hashes the committed receipt before and after to prove it is
untouched.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

import pytest

R = pathlib.Path(__file__).resolve().parents[2]
V = R / "experiments" / "verification"
sys.path.insert(0, str(V))

from receipt_path import (  # noqa: E402
    ENV_VAR,
    FLAG,
    ReceiptPathError,
    is_redirected,
    resolve_receipt_path,
)

DEFAULT = R / "experiments" / "verification" / "trilevel_loop_observed.json"
HW_RUNNER = V / "run_hardware_blocked_register.py"
HW_RECEIPT = V / "hardware_substrate_blocked.json"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "ABSENT"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No override leaks between tests."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    return monkeypatch


# --------------------------------------------------------------- default path
def test_no_override_returns_the_committed_default():
    """Default behaviour must be UNCHANGED, or clean-clone reproduction breaks."""
    assert resolve_receipt_path(DEFAULT, argv=["prog.py"]) == DEFAULT


def test_default_is_returned_even_when_env_is_blank(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "   ")
    assert resolve_receipt_path(DEFAULT, argv=["prog.py"]) == DEFAULT


# ------------------------------------------------------------------- --out
def test_out_file_path_is_used_verbatim(tmp_path):
    target = tmp_path / "custom.json"
    assert resolve_receipt_path(DEFAULT, argv=["p", FLAG, str(target)]) == target


def test_out_directory_joins_the_default_file_name(tmp_path):
    """A directory redirect keeps each runner's identity (no name collision)."""
    got = resolve_receipt_path(DEFAULT, argv=["p", FLAG, str(tmp_path)])
    assert got == tmp_path / DEFAULT.name


def test_out_creates_missing_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "r.json"
    assert resolve_receipt_path(DEFAULT, argv=["p", FLAG, str(nested)]) == nested
    assert nested.parent.is_dir()


def test_out_equals_form_is_accepted(tmp_path):
    target = tmp_path / "eq.json"
    assert resolve_receipt_path(DEFAULT, argv=["p", f"{FLAG}={target}"]) == target


def test_out_wins_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "envdir"))
    target = tmp_path / "argv.json"
    assert resolve_receipt_path(DEFAULT, argv=["p", FLAG, str(target)]) == target


def test_other_arguments_are_left_alone(tmp_path):
    """The tri-level harness has --with-demo; consuming it would break that runner."""
    got = resolve_receipt_path(DEFAULT, argv=["p", "--with-demo", FLAG, str(tmp_path)])
    assert got == tmp_path / DEFAULT.name


# --------------------------------------------------------------- HENRI_RECEIPT_DIR
def test_env_dir_is_used_and_created(tmp_path, monkeypatch):
    d = tmp_path / "receipts"
    monkeypatch.setenv(ENV_VAR, str(d))
    assert resolve_receipt_path(DEFAULT, argv=["p"]) == d / DEFAULT.name
    assert d.is_dir()


def test_env_expands_user(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "~")
    got = resolve_receipt_path(DEFAULT, argv=["p"])
    assert not str(got).startswith("~")
    assert got.name == DEFAULT.name


# ------------------------------------------------------------ bad override raises
@pytest.mark.parametrize("argv", [
    ["p", "--out="],          # empty value
    ["p", "--out"],           # dangling flag
    ["p", "--out", "--other"],  # next token is another flag
])
def test_unusable_out_raises(argv):
    with pytest.raises(ReceiptPathError):
        resolve_receipt_path(DEFAULT, argv=argv)


def test_env_pointing_at_a_file_raises(tmp_path, monkeypatch):
    """A file where a directory belongs must not be silently reinterpreted."""
    f = tmp_path / "not_a_dir.txt"
    f.write_text("x")
    monkeypatch.setenv(ENV_VAR, str(f))
    with pytest.raises(ReceiptPathError) as ei:
        resolve_receipt_path(DEFAULT, argv=["p"])
    assert "not a directory" in str(ei.value)


def test_a_bad_override_never_returns_the_committed_default(tmp_path, monkeypatch):
    """The whole point: a typo must NOT silently write the ledger-cited artifact."""
    f = tmp_path / "file.txt"
    f.write_text("x")
    monkeypatch.setenv(ENV_VAR, str(f))
    with pytest.raises(ReceiptPathError):
        resolve_receipt_path(DEFAULT, argv=["p"])
    # and with a dangling flag
    with pytest.raises(ReceiptPathError):
        resolve_receipt_path(DEFAULT, argv=["p", "--out"])


# ----------------------------------------------------------------- is_redirected
def test_is_redirected_flags_only_real_moves(tmp_path):
    assert is_redirected(DEFAULT, DEFAULT) is False
    assert is_redirected(DEFAULT, tmp_path / "x.json") is True


def test_is_redirected_is_path_normalised(tmp_path):
    same_via_dotdot = DEFAULT.parent / ".." / "verification" / DEFAULT.name
    assert is_redirected(DEFAULT, same_via_dotdot) is False


# --------------------------------------------------- INTEGRATION: committed files safe
def test_runner_redirected_to_tmp_leaves_the_committed_receipt_untouched(tmp_path):
    """End-to-end for one runner: the committed receipt must be byte-identical.

    Uses the hardware register because it is the fastest of the three. The redirect
    target is tmp_path, NEVER the committed location.
    """
    assert HW_RECEIPT.exists(), "expected committed receipt to exist"
    before = (sha(HW_RECEIPT), HW_RECEIPT.stat().st_mtime_ns)

    r = subprocess.run(
        [sys.executable, str(HW_RUNNER), "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=1200, cwd=str(V),
        env={**os.environ, ENV_VAR: ""},
    )
    out = r.stdout + r.stderr

    assert r.returncode == 0, f"runner failed:\n{out[-2000:]}"
    assert "REDIRECTED away from the committed receipt" in out, \
        "a redirected run must say so in its own log"
    written = tmp_path / HW_RECEIPT.name
    assert written.exists(), "redirected receipt was not written to the temp dir"

    after = (sha(HW_RECEIPT), HW_RECEIPT.stat().st_mtime_ns)
    assert after == before, (
        "the committed receipt CHANGED during a redirected run -- the override does "
        "not actually divert the write"
    )


def test_env_redirect_also_protects_the_committed_receipt(tmp_path):
    """The env var path must protect the ledger exactly as --out does."""
    before = sha(HW_RECEIPT)
    r = subprocess.run(
        [sys.executable, str(HW_RUNNER)],
        capture_output=True, text=True, timeout=1200, cwd=str(V),
        env={**os.environ, ENV_VAR: str(tmp_path)},
    )
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert (tmp_path / HW_RECEIPT.name).exists()
    assert sha(HW_RECEIPT) == before


def test_runners_no_longer_hardcode_their_output_path():
    """Guards the regression at the source level: no runner may write OUT blindly.

    A hardcoded `OUT = ...` assignment is exactly the defect this change removes, so
    its return is a failure.
    """
    import re

    for name in ["run_egress_snap_verification.py", "run_trilevel_loop.py",
                 "run_hardware_blocked_register.py"]:
        src = (V / name).read_text(encoding="utf-8", errors="replace")
        assert "resolve_receipt_path(DEFAULT_OUT)" in src, f"{name} not wired"
        assert "DEFAULT_OUT =" in src, f"{name} has no DEFAULT_OUT"
        # the OLD shape: OUT assigned directly from the committed path
        assert not re.search(r"^OUT = R / \"experiments\"", src, re.M), \
            f"{name} still hardcodes OUT"
