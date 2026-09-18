"""Receipt-path resolution for the verification runners (one override point).

WHY THIS EXISTS
    The runners under ``experiments/verification/`` hardcoded their output path as
    ``R / "experiments" / "verification" / "<name>_observed.json"`` and wrote it
    UNCONDITIONALLY. Those files are committed and cited in commit-message digest
    ledgers, so an experimental or background invocation overwrites a ledger-cited
    artifact. That happened twice in one session: a stale ``--with-demo`` job
    overwrote ``trilevel_loop_observed.json`` after it had been committed, and each
    occurrence had to be found by re-deriving digests and repaired with
    ``git checkout --``.

    This module gives every runner a single override point, so an experimental run
    can be aimed at a scratch directory and cannot silently clobber the ledger.

DEFAULT IS UNCHANGED
    With no override set, the path is EXACTLY what it was before: the runner's
    committed location. Normal reproduction from a clean clone therefore stays
    byte-identical, and the committed receipts still regenerate in place.

CONTRACT
    Priority:  ``--out <path>`` on argv  >  ``HENRI_RECEIPT_DIR`` <dir>  >  default

    * ``--out`` pointing at a DIRECTORY joins the runner's default file name, so a
      redirect keeps each runner's identity rather than creating a name collision.
    * ``--out`` pointing at a non-existent path is taken as a FILE path; its parent
      directory is created if missing.
    * ``HENRI_RECEIPT_DIR`` is treated as a DIRECTORY (created if missing) and the
      runner's default file name is joined to it.
    * ``--out=`` with an empty value, ``--out`` with no following argument, or a
      ``HENRI_RECEIPT_DIR`` that exists but is not a directory all RAISE. A bad
      override must never fall back to the committed default, because silently
      writing to the ledger-cited path is the precise failure this module prevents.
    * ``--out`` wins over the environment, and both win over the default.

    Only ``--out`` and ``HENRI_RECEIPT_DIR`` are consumed from argv; every other
    argument (e.g. the tri-level harness's ``--with-demo``) is left untouched.
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Optional

ENV_VAR = "HENRI_RECEIPT_DIR"
FLAG = "--out"


class ReceiptPathError(RuntimeError):
    """Raised when a receipt-path override is present but unusable."""


def _argv_out(argv: Optional[list] = None) -> Optional[str]:
    """Return the ``--out`` value from argv, or None when the flag is absent."""
    args = list(sys.argv if argv is None else argv)
    for i, a in enumerate(args):
        if a == FLAG:
            if i + 1 >= len(args):
                raise ReceiptPathError(f"{FLAG} requires a path argument")
            value = args[i + 1]
            if not value or value.startswith("-"):
                raise ReceiptPathError(
                    f"{FLAG} requires a path argument, got {value!r}"
                )
            return value
        if a.startswith(FLAG + "="):
            value = a[len(FLAG) + 1:]
            if not value:
                raise ReceiptPathError(f"{FLAG}= requires a path argument")
            return value
    return None


def resolve_receipt_path(default: pathlib.Path,
                         argv: Optional[list] = None) -> pathlib.Path:
    """Return the receipt path for this run.

    ``default`` is the runner's committed location and is returned UNCHANGED when no
    override is set, so default behaviour is byte-identical.

    Raises ``ReceiptPathError`` on an unusable override (see module docstring).
    """
    default = pathlib.Path(default)
    out = _argv_out(argv)
    env_dir = (os.environ.get(ENV_VAR) or "").strip()

    if out:
        p = pathlib.Path(out).expanduser()
        # An existing directory, or a trailing separator, means "put the file here
        # under its own default name". Otherwise it is a literal file path.
        if p.is_dir() or out.endswith(("/", os.sep)):
            p = p / default.name
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    if env_dir:
        d = pathlib.Path(env_dir).expanduser()
        if d.exists() and not d.is_dir():
            raise ReceiptPathError(
                f"{ENV_VAR}={env_dir!r} exists but is not a directory; refusing to "
                "guess a receipt path"
            )
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
        return d / default.name

    return default


def is_redirected(default: pathlib.Path, resolved: pathlib.Path) -> bool:
    """True when ``resolved`` writes somewhere other than the committed default.

    Runners use this to print WHERE they wrote, so a redirected run is visible in
    its own log rather than inferred from a missing ledger update.
    """
    try:
        return pathlib.Path(resolved).resolve() != pathlib.Path(default).resolve()
    except OSError:  # pragma: no cover - resolve() on an odd path
        return True
