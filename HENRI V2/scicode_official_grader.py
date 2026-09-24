#!/usr/bin/env python3
"""Official SciCode grader target resolver.

WHY THIS MODULE EXISTS (measured 2026-09-24, own tool calls)
============================================================
The SciCode scaffold arm excludes most items as `BLOCKED_EXTERNAL_CONSTANT`, because
`target` -- the expected return value each published test asserts -- is assigned ZERO
times in both corpus splits:

    problems_dev.jsonl   15 rows    219 `target` uses    0 assignments
    problems_test.jsonl  65 rows   1172 `target` uses    0 assignments

That exclusion is CORRECT protocol. The runner header states it:
    "NOT a substitute grader. The published tests are used verbatim. Where an
     [external constant is needed] ... no local tolerance, no re-derived expected value,
     no local grader injection."

`target` is supplied by the BENCHMARK'S OWN grader, which injects it at grade time:

    eval/inspect_ai/scicode.py:237   targets = process_hdf5_to_tuple(step_id, n, h5py_file)
                           :239   target  = targets[i]

So the legitimate resolution is to use the OFFICIAL grader and its OFFICIAL data file.
This module does exactly that, and it is NOT the forbidden thing:

    forbidden  = inventing a tolerance, re-deriving an expected value, injecting a
                 local grader of our own construction
    what this  = executing the upstream accessor `process_hdf5_to_tuple` against the
                 upstream `test_data.h5`, both obtained from the published source

MEASURED SUBSTRATE (own calls)
    repo        https://github.com/scicode-bench/SciCode  (clone rc=0, 304 KB)
    install     `pip install -e .` -> `import scicode` OK
    data        test_data.h5  1,049,345,865 B, 338 groups keyed by sub-step id
                  dev  overlap 50/50   test overlap 288/291
    accessor    src/scicode/parse/parse.py:126 process_hdf5_to_tuple
                  dev ids are the corpus `step_number` verbatim ('78.1'),
                  NOT f"{problem_id}.{step_number}"
    test count  per sub-step: 33 sub-steps ship 3 tests, 17 ship 4
                  => passing a fixed test_num=4 raises KeyError on 33 of them.
                     Pass the ACTUAL count. (This was my own first bug.)
    RESULT      50/50 dev sub-steps resolve official targets, errors NONE
    values      tuple of bool arrays, e.g. ('78.1', n=3) -> True

DEPENDENCY NOTE
    `parse.py:11` imports `datasets` at module level for functions this accessor does
    not use. If absent, this module injects a stub for that ONE name so the OFFICIAL
    accessor executes UNMODIFIED. The stub supplies no numbers and is recorded in the
    report. When `datasets` is genuinely installed, no stub is used.

EVIDENCE DISCIPLINE
    A target resolved here is `OFFICIAL_GRADER_TARGETS`: the value is the benchmark's,
    not ours. It is NOT evidence of HENRI capability, and a pass rate computed with it
    is a number about the candidate source, never about the wave egress.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Evidence class for any value this module returns.
EVIDENCE_CLASS = "OFFICIAL_GRADER_TARGETS"

#: Env override for the grader checkout root (must contain src/scicode and the h5).
ENV_ROOT = "HENRI_SCICODE_GRADER_ROOT"

#: Env override for the h5 path alone.
ENV_H5 = "HENRI_SCICODE_TEST_DATA_H5"

#: Config field inside the dataset pin naming the expected h5 byte count.
PIN_H5_BYTES = "official_grader_bytes"

#: Config field naming the expected h5 sha256 (optional; recommended).
PIN_H5_SHA = "official_grader_sha256"


class OfficialGraderUnavailable(RuntimeError):
    """Raised when the official grader cannot supply a target. Always fail closed."""


def _candidate_roots() -> List[pathlib.Path]:
    """Grader checkout roots to probe, most explicit first."""
    out: List[pathlib.Path] = []
    env = os.environ.get(ENV_ROOT)
    if env:
        out.append(pathlib.Path(env))
    # Local staging used by the UHR-05 probe; not a repo path by design (1.05 GB).
    out.append(pathlib.Path(os.path.expandvars(r"%LOCALAPPDATA%\Temp\scicode_official")))
    out.append(pathlib.Path("C:/Users/chan/AppData/Local/Temp/scicode_official"))
    out.append(pathlib.Path.home() / "scicode_official")
    # Repo-adjacent optional checkout (gitignored; the h5 must never enter git).
    here = pathlib.Path(__file__).resolve().parent
    out.append(here / "_external" / "scicode_official")
    return out


def _h5_path(root: pathlib.Path) -> pathlib.Path:
    env = os.environ.get(ENV_H5)
    if env:
        return pathlib.Path(env)
    return root / "eval" / "data" / "test_data.h5"


def _import_accessor(root: pathlib.Path):
    """Import the OFFICIAL `process_hdf5_to_tuple`, unmodified.

    Stubs the single unused `datasets` name if that package is absent, and records
    the fact so the receipt can state whether the environment was pristine.
    """
    src = root / "src"
    if not (src / "scicode" / "parse" / "parse.py").exists():
        raise OfficialGraderUnavailable(f"official grader source absent under {root}")
    stubbed = False
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import datasets  # noqa: F401
    except ImportError:
        stub = types.ModuleType("datasets")

        def _unused(*_a, **_k):  # pragma: no cover - must never be called
            raise OfficialGraderUnavailable(
                "stub `datasets.load_dataset` was called; it is not used by "
                "process_hdf5_to_tuple and must not supply any value"
            )

        stub.load_dataset = _unused
        sys.modules["datasets"] = stub
        stubbed = True
    from scicode.parse.parse import process_hdf5_to_tuple  # official, unmodified

    return process_hdf5_to_tuple, stubbed


@dataclass
class OfficialTargetResolver:
    """Resolve SciCode targets from the benchmark's own grader + data file.

    Construct with ``OfficialTargetResolver.discover()``. ``available()`` is False when
    the grader or the h5 is missing; ``resolve()`` then raises
    ``OfficialGraderUnavailable`` so callers fail closed instead of scoring.
    """

    root: pathlib.Path
    h5: pathlib.Path
    accessor: Any = None
    datasets_stubbed: bool = False
    reason: str = ""
    _counts: Dict[str, int] = field(default_factory=dict, repr=False)
    calls: int = field(default=0, repr=False)

    # -------------------------------------------------------------- discovery
    @classmethod
    def discover(cls, root: Optional[pathlib.Path] = None) -> "OfficialTargetResolver":
        roots = [root] if root else _candidate_roots()
        first_err = ""
        for r in roots:
            if r is None:
                continue
            h5 = _h5_path(r)
            if not h5.exists():
                first_err = first_err or f"h5 absent: {h5}"
                continue
            try:
                acc, stubbed = _import_accessor(r)
            except Exception as exc:  # noqa: BLE001
                first_err = first_err or f"{type(exc).__name__}: {exc}"
                continue
            return cls(root=r, h5=h5, accessor=acc, datasets_stubbed=stubbed, reason="ok")
        return cls(root=pathlib.Path("."), h5=pathlib.Path("."),
                   reason=first_err or "official grader not found")

    def available(self) -> bool:
        return self.accessor is not None and self.h5.exists()

    # ---------------------------------------------------------------- reading
    def test_count(self, step_id: str) -> int:
        """Number of published tests for this sub-step, from the official h5.

        MEASURED: 33 dev sub-steps ship 3 tests and 17 ship 4. A fixed count raises
        `KeyError` on the short ones, which is how my first probe mis-reported
        "17/50 resolved".
        """
        if step_id in self._counts:
            return self._counts[step_id]
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise OfficialGraderUnavailable("h5py required to read the official h5") from exc
        with h5py.File(str(self.h5), "r") as f:
            if step_id not in f:
                raise OfficialGraderUnavailable(f"{step_id} absent from the official h5")
            n = len([k for k in f[step_id].keys() if k.startswith("test")])
        if n <= 0:
            raise OfficialGraderUnavailable(f"{step_id} has no tests in the official h5")
        self._counts[step_id] = n
        return n

    def resolve(self, step_id: str, test_num: Optional[int] = None) -> Tuple[Any, ...]:
        """Return the OFFICIAL target tuple for `step_id` (all published tests)."""
        if not self.available():
            raise OfficialGraderUnavailable(self.reason or "official grader unavailable")
        n = self.test_count(step_id) if test_num is None else int(test_num)
        self.calls += 1
        return tuple(self.accessor(step_id, n, str(self.h5)))

    def resolve_or_none(self, step_id: str) -> Optional[Tuple[Any, ...]]:
        """Best-effort variant for probing. Records nothing; never fabricates."""
        try:
            return self.resolve(step_id)
        except Exception:  # noqa: BLE001
            return None

    # --------------------------------------------------------------- reporting
    def pin(self) -> Dict[str, Any]:
        """Provenance for the receipt. sha256 is computed once and cached."""
        out: Dict[str, Any] = {
            "evidence_class": EVIDENCE_CLASS,
            "root": str(self.root),
            "h5_path": str(self.h5),
            "h5_bytes": self.h5.stat().st_size if self.h5.exists() else 0,
            "datasets_stubbed": bool(self.datasets_stubbed),
            "reason": self.reason,
        }
        if self.h5.exists():
            h = hashlib.sha256()
            with open(self.h5, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            out["h5_sha256"] = h.hexdigest()
        return out

    def can_target(self, step_id: str, n_tests: int) -> Tuple[bool, str]:
        """Can the OFFICIAL grader supply targets for this item, with the mapping proven?

        Returns ``(ok, reason)``. Used by the runner's attemptability filter so that an
        item becomes attemptable ONLY when a real target exists -- never as a guess.
        """
        if not self.available():
            return False, self.reason or "official grader unavailable"
        try:
            n = self.test_count(step_id)
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        if n != int(n_tests):
            return False, f"count mismatch: corpus={n_tests} h5={n}"
        return True, "ok"

    def report(self) -> Dict[str, Any]:
        return {
            "name": "OfficialTargetResolver",
            "available": self.available(),
            "calls": self.calls,
            "distinct_step_ids": len(self._counts),
            "pin": self.pin(),
        }


#: Preamble that makes the OFFICIAL accessor available inside the executing sandbox.
#: It is the same mechanism the upstream grader uses (eval/scripts/test_generated_code.py
#: writes `targets = process_hdf5_to_tuple(step_id, n)` then `target = targets[i]` before
#: each test). Nothing here derives or invents a value.
_TARGETS_PREAMBLE = """\
import sys as _henri_official_sys
if {src!r} not in _henri_official_sys.path:
    _henri_official_sys.path.insert(0, {src!r})
from scicode.parse.parse import process_hdf5_to_tuple as _henri_official_pht
targets = _henri_official_pht({step_id!r}, {n}, {h5!r})
"""


def _scalar_literal(v: Any) -> str:
    """Serialise ONE scalar as a Python literal that is VALID under `eval`.

    DEFECT THIS REPAIRS (measured, own call, 2026-09-24): `repr(float('nan'))` is the
    bare token `nan`, and `eval('nan')` raises NameError. A nan/inf in any official
    target would therefore have produced a NameError in the sandbox and silently
    mis-classified EVERY affected test as ERROR_OTHER. Same for `inf`.
    Complex parts are emitted recursively so `complex(nan, 0j)` survives too.

    NUMPY SCALAR TYPE IS PRESERVED (measured, own call): the official accessor returns
    numpy scalars (`process_hdf5_to_tuple` yields `subgroup[()]`), and calling `.item()`
    turned them into Python scalars. Measured as 64/167 target entries whose type changed
    `float64 -> float`. Numerically identical, but the injected value should BE the
    official value, so numpy scalars are emitted as numpy scalars (`_np.float64(0.5)`).
    """
    import math

    import numpy as np

    if isinstance(v, np.generic):
        name = type(v).__name__
        # numpy scalar constructors accept the corresponding Python scalar.
        if isinstance(v, np.complexfloating):
            inner = f"complex({_scalar_literal(complex(v).real)}, " \
                    f"{_scalar_literal(complex(v).imag)})"
        elif isinstance(v, np.floating):
            f = float(v)
            if math.isnan(f):
                inner = "float('nan')"
            elif math.isinf(f):
                inner = "float('inf')" if f > 0 else "float('-inf')"
            else:
                inner = repr(f)
        elif isinstance(v, np.bool_):
            inner = "True" if bool(v) else "False"
        elif isinstance(v, np.integer):
            inner = repr(int(v))
        else:  # np.str_, np.bytes_, etc. -- repr() is a valid literal for these
            inner = repr(v.item())
        return f"_np.{name}({inner})"
    if isinstance(v, complex):
        return f"complex({_scalar_literal(v.real)}, {_scalar_literal(v.imag)})"
    if isinstance(v, float):
        if math.isnan(v):
            return "float('nan')"
        if math.isinf(v):
            return "float('inf')" if v > 0 else "float('-inf')"
        return repr(v)
    return repr(v)


def _py_literal(v: Any) -> str:
    """Serialise ONE official target value as an exact Python literal.

    FIDELITY ARGUMENT (why this is not a re-derived value)
      The value the sandbox sees is the value the official h5 holds: array payloads are
      emitted as exact JSON-style literals plus an explicit dtype and an explicit
      `.reshape(...)`, so shape AND dtype round-trip. `round_trip_ok()` proves this per
      value before any payload is built.

    SHAPE IS CARRIED EXPLICITLY -- measured defect this repairs (own call, 2026-09-24):
      sub-step '10.3' target[0].var0 has shape (0, 3). `[[...]].tolist()` -> `[]`, and
      `_np.array([])` rebuilds with shape (0,), NOT (0, 3). An EMPTY array's `tolist()`
      carries no dimensions at all, so shape was silently lost. `round_trip_ok` caught it
      and refused the item -- exactly its purpose.
      Fix: emit `_np.array(<flat ravel>, dtype=<dtype>).reshape(<shape>)`.

    NESTED CONTAINER TYPE IS PRESERVED: the official accessor returns `list` for some
    entries (via `process_hdf5_list`) and `tuple` for others. Emitting a tuple for a list
    changes the object the test compares against, so the container type is reproduced.
    """
    import numpy as np

    if isinstance(v, np.ndarray):
        dtype = "object" if v.dtype.kind == "O" else str(v.dtype)
        flat = ", ".join(_scalar_literal(x) for x in v.ravel().tolist())
        return (f"_np.array([{flat}], dtype={dtype!r})"
                f".reshape({tuple(v.shape)!r})")
    if isinstance(v, tuple):
        inner = ", ".join(_py_literal(x) for x in v)
        if len(v) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(v, list):
        return "[" + ", ".join(_py_literal(x) for x in v) + "]"
    return _scalar_literal(v)


def serialize_targets(targets: Any) -> str:
    """Serialise the official target as a Python literal, PRESERVING container type."""
    return _py_literal(targets)


def round_trip_ok(value: Any) -> bool:
    """Prove `serialize_targets` is lossless for this value. Used as a per-item control.

    Compares container TYPE as well as contents, so a list->tuple change is rejected.
    """
    import numpy as np

    try:
        back = eval(serialize_targets(value), {"_np": np})  # noqa: S307 - our own literal
    except Exception:  # noqa: BLE001
        return False
    if isinstance(value, np.ndarray):
        if not isinstance(back, np.ndarray):
            return False
        if value.shape != back.shape or value.dtype != back.dtype:
            return False
        if value.dtype.kind == "O":
            return value.ravel().tolist() == back.ravel().tolist()
        if value.dtype.kind in "fc":
            return bool(np.array_equal(value, back, equal_nan=True))
        return bool(np.array_equal(value, back))
    if isinstance(value, tuple):
        if not isinstance(back, tuple) or len(back) != len(value):
            return False
        return all(round_trip_ok(a) and round_trip_ok(b) for a, b in zip(value, back))
    if isinstance(value, list):
        if not isinstance(back, list) or len(back) != len(value):
            return False
        return all(round_trip_ok(a) and round_trip_ok(b) for a, b in zip(value, back))
    if isinstance(value, float) and value != value:  # NaN
        return isinstance(back, float) and back != back
    import numpy as _np

    if isinstance(value, _np.generic):
        # EXACT type match for numpy scalars: `_scalar_literal` emits `_np.<type>(...)`
        # precisely so the injected value has the official accessor's own type.
        if type(value) is not type(back):
            return False
        if isinstance(value, _np.complexfloating):
            return round_trip_ok(complex(value).real) and round_trip_ok(complex(value).imag)
        if isinstance(value, _np.floating):
            f, g = float(value), float(back)
            return (f != f and g != g) or (f == g)
        return bool(value == back)
    return bool(value == back)


def build_tests_with_targets(tests: List[str], step_id: str,
                             resolver: "OfficialTargetResolver", *,
                             in_sandbox: bool = False) -> str:
    """Return the published tests with the OFFICIAL `target` supplied before each one.

    CONTRACT (fail-closed, no substitution)
      * `tests` are the corpus `test_cases` strings, used VERBATIM. They are not
        edited, wrapped, or relaxed.
      * `target` is bound from the official accessor's own output. No tolerance, no
        re-derived expected value, no locally constructed grader.
      * If the official h5 does not cover this sub-step, or if its test count does not
        equal `len(tests)` (the mapping `test_cases[i] -> targets[i]` unproven), this
        RAISES. The caller must then keep the item excluded rather than score it.

    MEASURED BASIS for the count check (own calls, 2026-09-24):
      corpus `test_cases` counts vs official h5 test counts -- 50/50 sub-steps AGREE,
      0 disagree; distribution identical at {3: 33, 4: 17}. So index alignment is
      established, and a mismatch is a genuine surprise that must stop the run.
    """
    if not resolver.available():
        raise OfficialGraderUnavailable(resolver.reason or "official grader unavailable")
    n = resolver.test_count(step_id)
    if n != len(tests):
        raise OfficialGraderUnavailable(
            f"{step_id}: corpus ships {len(tests)} test_cases but the official h5 has "
            f"{n} tests; the i-th-test mapping is unproven for this item"
        )
    targets = resolver.resolve(step_id, n)
    if len(targets) != len(tests):
        raise OfficialGraderUnavailable(
            f"{step_id}: official accessor returned {len(targets)} targets for "
            f"{len(tests)} tests"
        )
    # Per-item losslessness control. A value that cannot round-trip must NOT be scored.
    for i, tv in enumerate(targets):
        if not round_trip_ok(tv):
            raise OfficialGraderUnavailable(
                f"{step_id}: official target[{i}] does not round-trip through the "
                f"literal serialiser; refusing to inject a lossy value"
            )

    if in_sandbox:
        # Fidelity-maximal path: mirror upstream and resolve INSIDE the sandbox.
        # NOT the default: measured (own call) that the interpreter the sandbox may
        # select lacks h5py, and the sandbox runs under rlimits -- neither is a good
        # place to open a 1.05 GB file. Kept for cross-checking only.
        parts = [_TARGETS_PREAMBLE.format(src=str(resolver.root / "src"), step_id=step_id,
                                          n=n, h5=str(resolver.h5))]
        for i, test in enumerate(tests):
            parts.append(f"target = targets[{i}]\n")
            parts.append(test)
            parts.append("\n")
        return "\n".join(parts)

    # DEFAULT: resolve in the RUNNER process and inject the exact literal values.
    # The sandbox then needs neither h5py nor the `scicode` package, and never opens
    # the official h5. Values are the official ones, serialised losslessly.
    parts = ["import numpy as _np\n"]
    for i, test in enumerate(tests):
        parts.append(f"target = {serialize_targets(targets[i])}\n")
        parts.append(test)
        parts.append("\n")
    return "\n".join(parts)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Probe/resolve official SciCode targets.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--corpus", default=None, help="problems_dev.jsonl to enumerate ids")
    ap.add_argument("--probe", action="store_true", help="resolve every corpus sub-step")
    ap.add_argument("--out", default=None, help="write the provenance pin JSON here")
    args = ap.parse_args()

    r = OfficialTargetResolver.discover(pathlib.Path(args.root) if args.root else None)
    print(f"available = {r.available()}   reason = {r.reason}")
    if not r.available():
        print("OFFICIAL_TARGETS_UNAVAILABLE")
        return 2

    print(f"root      = {r.root}")
    print(f"h5        = {r.h5}  bytes = {r.h5.stat().st_size}")
    print(f"datasets  = {'STUBBED (unused by the accessor)' if r.datasets_stubbed else 'real'}")

    if args.corpus:
        rows = [json.loads(l) for l in pathlib.Path(args.corpus).read_text(
            encoding="utf-8").splitlines() if l.strip()]
        ids = [st["step_number"] for row in rows for st in row.get("sub_steps", [])]
        ok, errs = 0, {}
        for sid in ids:
            try:
                r.resolve(sid)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                errs[type(exc).__name__] = errs.get(type(exc).__name__, 0) + 1
        print(f"RESOLVED = {ok} / {len(ids)}   errors = {errs or 'NONE'}")

    pin = r.pin()
    print(f"pin       = {json.dumps({k: v for k, v in pin.items() if k != 'h5_sha256'})}")
    print(f"h5 sha256 = {pin.get('h5_sha256', '')[:16]}...")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    print("OFFICIAL_TARGETS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
