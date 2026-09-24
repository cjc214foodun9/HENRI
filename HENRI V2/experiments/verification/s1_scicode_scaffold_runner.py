"""S1 SciCode scaffold runner — CPU-only instrument validation and baseline.

WHAT THIS IS
    A deterministic, offline, CPU-only runner that executes real SciCode dev-split
    subproblems through the repository's existing sandboxed execution path
    (``henri_sandbox_harness.SandboxHarness``), classifies every non-pass
    outcome, and writes a machine-readable receipt branching its verdict on
    MEASUREMENT.

WHAT THIS IS NOT  (explicit non-claims, also recorded in the receipt)
    1. NOT evidence of HENRI capability. There is no working code generator
       wired to SciCode: the project's text EGRESS is under repair (open-answer
       channel down; defect A2, the egress id->string unbinder, only now being
       fixed). The only candidate source registered here is
       ``NullCandidateSource``, which emits the empty string and is labelled as
       such. Any pass@1 produced here is a floor for an ABSENT generator.
    2. NOT a claim about the AAII index, its weights, or any constituent score.
    3. NOT a substitute grader. The published tests are used verbatim. Where an
       item's published test references a constant that the dev split does not
       ship (``target``), the item is BLOCKED_EXTERNAL_CONSTANT and is excluded
       from the score numerator and denominator. No local tolerance, no
       re-derived expected value, no local grader injection.
    4. NOT a namespace-isolated run. On Windows the harness reports
       ``mode=container-rlimit isolated=False surrogate=True``: resource limits
       and a wall-clock timeout, NOT network/pid/mount isolation.

DEFECTS FROM THE PRIOR M3 CONTROL THAT THIS RUNNER MUST NOT REPEAT
    D16  SciCode sub_steps are SEQUENTIAL. The reference arm therefore uses the
         accumulated protocol (all prior sub_steps of the same problem), which
         is SciCode's own contract. No isolated arm is run, so the isolation
         artifact cannot recur.
    D17  A failure classifier that searched only ``stderr[:180]`` fired zero
         times because the traceback header fills that window. Every branch
         here searches the FULL stderr text.
    D19  ``SandboxResult.passed`` is a PROPERTY, not a field. Passing/failing is
         derived exclusively from ``henri_sandbox_harness.STATUS_PASSED``
         compared against ``result.status``. The property value is recorded
         alongside for calibration, never used for counting.
    D20  ``python -I`` implies ``-s``, so user site-packages are invisible to
         the sandbox child; and the harness additionally forces
         ``PYTHONNOUSERSITE=1``. The runner therefore (a) uses a DEDICATED venv
         as the sandbox interpreter and (b) proves in-sandbox importability
         with a probe that runs INSIDE the sandbox before any item executes,
         and records the interpreter path and the resolved module file paths.
    D21  Eight FAILED rows were persisted without stderr and were
         unattributable. Here the FULL stderr is written into the row and
         fsynced to disk BEFORE any aggregation happens; aggregation reads the
         file back.

CONTAMINATION GUARD (falsifiable; two independent guards, both self-tested)
    G1  Dataset reference text (``ground_truth_code`` / ``general_solution``) may
        be read ONLY while the explicitly-named reference-control arm is
        active. ``_dataset_field`` raises ``ContaminationError`` otherwise.
    G2  No scored-arm payload may share a >= 60-character normalised substring
        with the item's reference code. ``_assert_no_reference_overlap`` raises
        ``ContaminationError`` if it does.
    Both guards are exercised at run time with a known-positive input and a
    known-negative input, and the four self-test outcomes are recorded in the
    receipt. A guard that cannot fail is not a guard.

VERDICT BRANCHING (never print a score when the harness is unvalidated)
    VOID     a control misbehaved, or the guards did not self-test as designed
    BLOCKED  the dataset is missing/pin-mismatched, or no interpreter with the
             required modules exists, or the selected window is empty
    SCORED   controls all behaved AND the dataset pin matched AND the window is
             non-empty -> pass@1 = candidate passes / attemptable items
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import os
import pathlib
import re
import sys
import time

# UHR-05 (DEAD HARDCODED PATH): absolute path to a worktree that need not exist;
# the same class as the `parents[1]` break fixed in the M1 gate. Resolve from
# __file__ (parents[2] == HENRI V2), with an explicit override for other checkouts.
V2 = pathlib.Path(os.environ.get("HENRI_S1_V2") or pathlib.Path(__file__).resolve().parents[2])
REPO = pathlib.Path(r"C:/Users/chan/henri-worktrees/aaii-v43")
sys.path.insert(0, str(V2))

import henri_eval_infra as ei  # noqa: E402
import henri_sandbox_harness as sh  # noqa: E402
from henri_eval_infra import ItemLedger  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

SCHEMA_ID = "henri.run-evidence.v1"
DATASET = V2 / "data/official_benchmarks/scicode/problems_dev.jsonl"
PIN = V2 / "experiments/verification/SCICODE_DATASET_PIN_20260916.json"
OUT_ROOT = V2 / "experiments/verification"
BENCHMARK_ID = "s1-scicode-scaffold"

MAX_ITEMS = 16
ITEM_TIMEOUT_S = 120.0
MEM_LIMIT_MB = 4096
# M3's resource fix: thread-pool over-allocation in the sandbox child produced
# spurious OpenBLAS allocation failures. Pin every numeric thread pool to 1.
CHILD_ENV_EXTRA = {
    "OPENBLAS_NUM_THREADS": "1",
    "OPENBLAS_DEFAULT_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

TAXONOMY = (
    "STATUS_PASSED",
    "FAILED_ASSERT",
    "TIMEOUT",
    "BLOCKED_EXTERNAL_CONSTANT",
    "BLOCKED_DEPENDENCY",
    "BLOCKED_INFRA_RESOURCE",
    "ERROR_OTHER",
)

#: taxonomy -> the repo's existing per-item status vocabulary
#: (henri_eval_infra.STATUS_VALUES). The precise class is kept in row["taxonomy"],
#: so this mapping loses no information.
TAXONOMY_TO_RUN_STATUS = {
    "STATUS_PASSED": "PASSED",
    "FAILED_ASSERT": "FAILED",
    "TIMEOUT": "TIMEOUT",
    "BLOCKED_EXTERNAL_CONSTANT": "SKIPPED",
    "BLOCKED_DEPENDENCY": "SKIPPED",
    "BLOCKED_INFRA_RESOURCE": "SKIPPED",
    "ERROR_OTHER": "EXECUTION_ERROR",
}

#: Constants that the published tests reference but that the dev split does not
#: ship; SciCode's official grader supplies them. A failure naming one of these
#: is a DATASET limitation, never a candidate failure.
EXTERNAL_CONSTANT_NAMES = ("target",)

#: Dataset fields the runner is allowed to read outside the reference-control
#: arm. Enforced by ``_dataset_field``. ``ground_truth_code`` and
#: ``general_solution`` are deliberately absent -> guard G1.
READABLE_FIELDS = ("problem_id", "required_dependencies", "sub_steps", "step_number",
                   "function_header", "step_description_prompt", "step_background",
                   "test_cases", "general_tests")
REFERENCE_ONLY_FIELDS = ("ground_truth_code", "general_solution")

REFERENCE_ARM_ACTIVE = False  # flipped only by the reference-control arm

MAX_OUTPUT_CHARS = 200_000


class ContaminationError(RuntimeError):
    """Raised when dataset reference text would reach the scored arm."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


@contextlib.contextmanager
def reference_arm_active():
    global REFERENCE_ARM_ACTIVE
    prev = REFERENCE_ARM_ACTIVE
    REFERENCE_ARM_ACTIVE = True
    try:
        yield
    finally:
        REFERENCE_ARM_ACTIVE = prev


def _dataset_field(obj: dict, name: str):
    """Read a dataset field, enforcing contamination guard G1."""
    if name in REFERENCE_ONLY_FIELDS and not REFERENCE_ARM_ACTIVE:
        raise ContaminationError(
            f"G1: dataset field {name!r} is reference text and may only be read "
            f"while the reference-control arm is active")
    if name not in READABLE_FIELDS and name not in REFERENCE_ONLY_FIELDS:
        raise ContaminationError(f"G1: dataset field {name!r} is not on the allowlist")
    return obj[name]


#: G2 coverage threshold. MEASURED 2026-09-24 over all 50 dev sub-steps:
#:   payload reproducing only the PRESCRIBED function header : max differential 0.0000
#:   payload holding only the minimal def line               : max differential -0.0526
#:   payload reproducing the reference BODY                  : min differential  0.3333
#: Valid band is 0.00..0.30, so the midpoint is not a knife edge.
G2_COVERAGE_TAU = 0.1667


def _assert_no_reference_overlap(payload: str, reference: str, *, min_chars: int = 60,
                                 given: str = "", coverage_tau: float = G2_COVERAGE_TAU) -> None:
    """Contamination guard G2: the scored payload must not reproduce the reference.

    WHY COVERAGE, NOT SUBSTRING. MEASURED 2026-09-24; the previous form was UNSATISFIABLE.

    The reference text is ``deps + ground_truth_code[0..idx]``, and every
    ``ground_truth_code`` necessarily REPEATS its sub-step's PRESCRIBED
    ``function_header`` -- text the task GIVES the candidate. A stride-aligned 60-char
    window landing wholly inside that signature is therefore present in ANY faithful
    candidate. Measured with the previous substring form over all 50 dev sub-steps:

        candidate reproducing only its prescribed header -> 50/50 TRIPPED
        candidate reproducing only the minimal def line   -> 18/50 TRIPPED
        candidate reproducing the reference BODY          -> 50/50 tripped (correct)
        empty payload                                     ->  0/50 (correct)

    The first two rows are false positives, so G2 could not be passed by a correct
    candidate: an instrument that cannot return success. It stayed invisible because
    ``NullCandidateSource`` emits "" and an empty payload cannot contain a window. It
    surfaced on the first genuine run -- item SciCode-78-78.1 PASSED, then 78.2 raised
    ``G2: ... shares a 60-char reference substring: 'nge_kutta_4th_order(f, state, t0, ...'``
    where that chunk is a suffix of the PRESCRIBED header.

    THE REPLACEMENT measures how much of the reference the payload covers BEYOND text the
    task already gave:

        cov  = fraction of reference windows present in the payload
        base = fraction of reference windows present in ``given``
        raise iff cov - base > coverage_tau

    A faithful candidate reproduces given text, so cov ~= base and the differential is ~0.
    A body-copying candidate covers windows that are NOT given, so the differential is
    large. The differential is invariant to the length of the prescribed signature --
    exactly the axis that made the substring form unsatisfiable.

    ``given`` defaults to "" so the two guard-control calls keep their original meaning:
    ``(ref_text, ref_text)`` still raises (cov 1.0, base 0.0), ``("", ref_text)`` still
    returns.
    """
    r = norm_ws(reference)
    if len(r) < min_chars:
        return
    windows = [r[i:i + min_chars] for i in range(0, len(r) - min_chars + 1, 20)]
    if not windows:
        return
    p = norm_ws(payload)
    if len(p) < min_chars:
        # Empty/short payload cannot reproduce the reference; preserves the
        # `G2_passes_empty_payload` control.
        return
    g = norm_ws(given)
    cov = sum(1 for c in windows if c in p) / len(windows)
    base = (sum(1 for c in windows if c in g) / len(windows)) if g else 0.0
    diff = cov - base
    if diff > coverage_tau:
        culprit = next((c for c in windows if c in p and not (g and c in g)), "")
        raise ContaminationError(
            f"G2: scored-arm payload covers {cov:.1%} of the reference vs {base:.1%} of it "
            f"given (differential {diff:.3f} > tau {coverage_tau:.4f}); "
            f"window {culprit[:80]!r}")


# ---------------------------------------------------------------------------
# 1. dataset pin verification  (mismatch => BLOCKED, never a silent proceed)
# ---------------------------------------------------------------------------

def verify_pin() -> dict:
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    checks = []
    for entry in pin["files"]:
        local = V2 / "data/official_benchmarks/scicode" / entry["name"]
        if not local.exists():
            checks.append({"file": entry["name"], "present": False,
                           "match": False, "expected_sha256": entry["sha256"]})
            continue
        got_sha = sha256_file(local)
        got_bytes = local.stat().st_size
        checks.append({
            "file": entry["name"], "present": True,
            "expected_sha256": entry["sha256"], "actual_sha256": got_sha,
            "sha_match": got_sha == entry["sha256"],
            "expected_bytes": entry["bytes"], "actual_bytes": got_bytes,
            "bytes_match": got_bytes == entry["bytes"],
            "match": got_sha == entry["sha256"] and got_bytes == entry["bytes"],
        })
    return {
        "pin_path": str(PIN.relative_to(V2)),
        "dataset": pin.get("dataset"), "revision": pin.get("revision"),
        "license": pin.get("license"), "gated": pin.get("gated"),
        "grader": pin.get("_grader"),
        "per_file": checks,
        "all_match": all(c["present"] and c["match"] for c in checks),
    }


# ---------------------------------------------------------------------------
# 2. interpreter / in-sandbox import verification   (D20)
# ---------------------------------------------------------------------------

def candidate_interpreters() -> list[tuple[str, str]]:
    out = []
    venv = pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / "m3_bench_venv" / "Scripts" / "python.exe"
    if venv.exists():
        out.append(("dedicated_venv", str(venv)))
    if os.path.exists(sys.executable):
        out.append(("runner_interpreter", sys.executable))
    return out


IMPORT_PROBE = """
import json, sys
res = {"executable": sys.executable, "prefix": sys.prefix, "version": sys.version,
       "sys_path": list(sys.path), "flags": {"isolated": bool(sys.flags.isolated),
       "no_user_site": bool(sys.flags.no_user_site)}, "modules": {}}
for name in ("numpy", "scipy", "sympy"):
    try:
        mod = __import__(name)
        # UHR-05 VACUOUS-PROBE FIX. MEASURED DEFECT this guards: an EMPTY
        # `site-packages/numpy/` directory imports as a NAMESPACE PACKAGE, so
        # `__import__("numpy")` SUCCEEDS while `numpy.array` does not exist
        # (measured: __file__=None, __version__=None, hasattr(numpy,"array")=False on the
        # runner's chosen venv). A return-code-only probe ACCEPTED that stub and every
        # downstream item then died with
        #     AttributeError: module 'numpy' has no attribute 'array'
        # which is what set controls_all_behaved=False and drove the verdict to VOID.
        # A module is AVAILABLE only if it exposes a FUNCTIONAL API surface.
        _need = {"numpy": ("ndarray", "array"),
                 "scipy": ("linalg",),
                 "sympy": ("symbols",)}.get(name, ())
        _functional = all(hasattr(mod, _a) for _a in _need) if _need else True
        res["modules"][name] = {
            "ok": bool(_functional),
            "version": getattr(mod, "__version__", "?"),
            "file": getattr(mod, "__file__", "?"),
            "requires": list(_need),
            "reason": None if _functional else "STUB_OR_EMPTY_NAMESPACE",
        }
    except BaseException as exc:
        res["modules"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
print("HENRI_IMPORT_PROBE=" + json.dumps(res))
"""


def probe_interpreter(exe: str) -> dict:
    harness = sh.SandboxHarness(mode="auto", timeout_s=60.0, python_executable=exe,
                                mem_limit_mb=MEM_LIMIT_MB, child_env_extra=CHILD_ENV_EXTRA,
                                enable_repl_telemetry=False, max_output_chars=MAX_OUTPUT_CHARS)
    res = harness.evaluate(IMPORT_PROBE, None, label="import-probe")
    info = None
    for line in (res.stdout or "").splitlines():
        if line.startswith("HENRI_IMPORT_PROBE="):
            info = json.loads(line[len("HENRI_IMPORT_PROBE="):])
    return {
        "declared_interpreter": exe,
        "probe_status": res.status,
        "probe_returncode": res.returncode,
        "probe_used_status_constant": res.status == sh.STATUS_PASSED,
        "probe_stderr_tail": (res.stderr or "").strip()[-400:],
        "child_environment": info,
        "sandbox": {k: harness.describe().get(k) for k in
                    ("requested_mode", "mode", "isolated", "surrogate", "sanitizer")},
    }


# ---------------------------------------------------------------------------
# 3. candidate source  (the only wired generator: NONE)
# ---------------------------------------------------------------------------

class CandidateSource:
    """Interface for a code generator wired to SciCode."""

    name = "abstract"
    description = ""
    produces_code = False

    def produce(self, item: dict, prior_outputs: list[str]) -> str:  # pragma: no cover
        raise NotImplementedError


class NullCandidateSource(CandidateSource):
    """The honest state of the project: NO generator is wired.

    HENRI's text egress is under repair (open-answer channel down; defect A2, the
    egress id->string unbinder, only now being fixed). This source emits the
    empty string for every subproblem. It is registered so the harness, the
    taxonomy and the receipt can be validated on the real dataset, and so the
    pass@1 floor for an ABSENT generator is measured rather than asserted.
    """

    name = "NullCandidateSource"
    description = ("emits the empty string for every subproblem; no code generator is "
                   "wired to SciCode (A2 egress unbinder under repair)")
    produces_code = False

    def produce(self, item: dict, prior_outputs: list[str]) -> str:
        return ""


# ---------------------------------------------------------------------------
# 4. selection  (deterministic, score-blind)
# ---------------------------------------------------------------------------

_TARGET_RE = re.compile(r"(?<![A-Za-z0-9_.])target(?!\s*=)")


def stable_order(rows: list[dict]) -> list[tuple[dict, int]]:
    """(problem, sub_step_index) sorted by (numeric problem_id, step index)."""
    return [(r, i) for r in sorted(rows, key=lambda r: int(r["problem_id"]))
            for i in range(len(r["sub_steps"]))]


def select_window(rows: list[dict], n: int = MAX_ITEMS) -> list[tuple[dict, int]]:
    """Deterministic, score-blind window.

    Rule (pre-registered):
      (a) take every sub_step whose published tests carry no external-constant
          reference, in stable order  -- this is a STRUCTURAL stratification so
          the pass@1 denominator is non-empty; it cannot be influenced by any
          candidate outcome because no candidate has run yet;
      (b) fill the remaining slots from the stable order of all sub_steps,
          skipping those already taken.
    """
    order = stable_order(rows)
    selfcontained = [(r, i) for r, i in order
                     if not _TARGET_RE.search("\n".join(_dataset_field(r["sub_steps"][i], "test_cases")))]
    chosen = list(selfcontained)
    for cand in order:
        if len(chosen) >= n:
            break
        if cand not in chosen:
            chosen.append(cand)
    return chosen[:n]


def item_id(problem: dict, idx: int) -> str:
    return f"SciCode-{problem['problem_id']}-{problem['sub_steps'][idx]['step_number']}"


def required_function_names(case: dict) -> tuple[str, ...]:
    names = re.findall(r"^\s*def\s+([A-Za-z_]\w*)", case.get("function_header") or "", re.M)
    return tuple(names)


_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?::[^=]+)?=", re.M)


def payload_defined_names(payload: str) -> set[str]:
    """Names the candidate PAYLOAD itself binds (defs + top-level assignments).

    Derived from the candidate's own code only, so it never touches dataset
    reference text.
    """
    p = payload or ""
    names = set(re.findall(r"^\s*def\s+([A-Za-z_]\w*)", p, re.M))
    names |= set(_ASSIGN_RE.findall(p))
    names |= set(re.findall(r"^\s*import\s+([A-Za-z_]\w*)", p, re.M))
    names |= set(re.findall(r"^\s*from\s+[A-Za-z_][\w.]*\s+import\s+([A-Za-z_]\w*)", p, re.M))
    return names


def prior_defs(problem: dict, idx: int) -> set[str]:
    """Function names defined in EARLIER sub_steps of the same problem.

    Reads ``ground_truth_code``, so this is REFERENCE-ARM-ONLY by construction:
    with guard G1 active it raises unless the caller has entered
    ``reference_arm_active()``. Used ONLY to detect an isolation artifact in a
    reference-arm failure, never to build a candidate payload.
    """
    names: set[str] = set()
    for k in range(idx):
        names |= set(re.findall(r"^\s*def\s+([A-Za-z_]\w*)",
                                _dataset_field(problem["sub_steps"][k], "ground_truth_code"), re.M))
    return names


def reference_prior_defs(problem: dict, idx: int) -> set[str]:
    with reference_arm_active():
        return prior_defs(problem, idx)


# ---------------------------------------------------------------------------
# 5. failure taxonomy  (full-text search; D17 / D19 / D21 fixes)
# ---------------------------------------------------------------------------

_INFRA_RE = re.compile(
    r"OpenBLAS error|Memory allocation .* failed|MemoryError|Unable to allocate"
    r"|bad_alloc|cannot allocate memory|LLVM ERROR|Intel MKL ERROR")
_DEP_RE = re.compile(r"ModuleNotFoundError|ImportError")
_SYNTAX_RE = re.compile(r"\bSyntaxError\b|\bIndentationError\b|\bTabError\b")
_NAMEERR_RE = re.compile(r"NameError:\s*name '([A-Za-z_]\w*)' is not defined")


def classify(status: str, stderr: str, *, arm: str, case: dict,
             earlier_defs: set[str], payload: str = "") -> dict:
    """Map a raw execution outcome to one of the seven taxonomy classes.

    Searches the FULL stderr (D17). Never consults ``SandboxResult.passed`` (D19).
    Returns a dict with the class plus a precise ``sub_class`` and the exact
    evidence line, so no row is unattributable (D21).

    ``payload`` is the exact code that was executed for this arm. It is consulted
    (for the candidate arm only) so that a symbol the payload never defines is
    attributed to the CANDIDATE rather than to the environment -- derived from
    the payload itself, never from dataset reference text.
    """
    if status == sh.STATUS_PASSED:
        return {"taxonomy": "STATUS_PASSED", "sub_class": "", "evidence": ""}
    if status == sh.STATUS_TIMEOUT:
        return {"taxonomy": "TIMEOUT", "sub_class": "", "evidence": "wall-clock timeout"}
    if status == sh.STATUS_VETOED:
        return {"taxonomy": "ERROR_OTHER", "sub_class": "POLICY_VETO", "evidence": "vetoed"}

    s = stderr or ""
    last = ""
    for line in reversed(s.splitlines()):
        if line.strip():
            last = line.strip()
            break

    m = _NAMEERR_RE.search(s)
    if m:
        name = m.group(1)
        if name in EXTERNAL_CONSTANT_NAMES:
            return {"taxonomy": "BLOCKED_EXTERNAL_CONSTANT",
                    "sub_class": f"external_constant:{name}",
                    "evidence": last}
        if name in earlier_defs:
            return {"taxonomy": "ERROR_OTHER", "sub_class": f"ISOLATION_ARTIFACT:{name}",
                    "evidence": last}
        if arm == "candidate" and name not in payload_defined_names(payload):
            return {"taxonomy": "ERROR_OTHER",
                    "sub_class": f"CANDIDATE_UNDEFINED_SYMBOL:{name}",
                    "evidence": last}

    if _INFRA_RE.search(s):
        return {"taxonomy": "BLOCKED_INFRA_RESOURCE",
                "sub_class": "numerical_backend_allocation", "evidence": last}

    if _DEP_RE.search(s):
        mm = re.search(r"No module named '([^']+)'", s)
        return {"taxonomy": "BLOCKED_DEPENDENCY",
                "sub_class": f"missing_module:{mm.group(1)}" if mm else "missing_module",
                "evidence": last}

    if _SYNTAX_RE.search(s):
        return {"taxonomy": "ERROR_OTHER",
                "sub_class": "SYNTAX_ERROR" if arm == "candidate" else "CASE_SYNTAX_ERROR",
                "evidence": last}

    if "AssertionError" in s:
        return {"taxonomy": "FAILED_ASSERT", "sub_class": "published_assertion_failed",
                "evidence": last}

    if m:
        return {"taxonomy": "ERROR_OTHER", "sub_class": f"UNDEFINED_SYMBOL:{m.group(1)}",
                "evidence": last}

    return {"taxonomy": "ERROR_OTHER",
            "sub_class": "unclassified:" + (last[:120] or status), "evidence": last}


# ---------------------------------------------------------------------------
# 6. execution
# ---------------------------------------------------------------------------

def run_case(harness, code: str, tests: str | None, label: str) -> dict:
    t0 = time.perf_counter()
    try:
        res = harness.evaluate(code, tests, timeout_s=ITEM_TIMEOUT_S, label=label)
    except BaseException as exc:  # harness-side failure, never a candidate outcome
        return {"status": "HARNESS_ERROR", "returncode": None,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "stdout": "", "stderr_full": f"{type(exc).__name__}: {exc}",
                "stderr_truncated": False, "passed_property": None,
                "mode": getattr(harness, "mode", "?"),
                "isolated": bool(getattr(harness, "isolated", False)),
                "status_reason": "harness raised"}
    return {
        "status": res.status,
        "returncode": res.returncode,
        "elapsed_ms": round(res.elapsed_ms, 1),
        "stdout": res.stdout or "",
        "stderr_full": res.stderr or "",
        "stderr_truncated": bool((res.limits or {}).get("stderr_truncated", False)),
        "passed_property": bool(res.passed),  # calibration only, never counted (D19)
        "mode": res.mode,
        "isolated": bool(res.isolated),
        "status_reason": res.status_reason,
    }


def build_reference_code(problem: dict, idx: int) -> str:
    """Accumulated reference code (D16: SciCode sub_steps are sequential)."""
    with reference_arm_active():
        deps = (problem.get("required_dependencies") or "").strip()
        prior = "\n".join(_dataset_field(problem["sub_steps"][k], "ground_truth_code")
                          for k in range(idx))
        own = _dataset_field(problem["sub_steps"][idx], "ground_truth_code")
    return deps + "\n" + (prior + "\n" if prior else "") + own


def build_candidate_code(problem: dict, idx: int, prior_outputs: list[str]) -> str:
    deps = (problem.get("required_dependencies") or "").strip()
    # UHR-05: `prior` must EXCLUDE the current step, because `own` supplies it below.
    # Without this slice, correcting the call site to range(idx + 1) would emit the
    # current step TWICE (once inside `prior`, once as `own`).
    prior = "\n".join(p for p in prior_outputs[:idx] if p)
    own = prior_outputs[idx] if idx < len(prior_outputs) else ""
    return deps + "\n" + (prior + "\n" if prior else "") + own


# ---------------------------------------------------------------------------
# 7. controls
# ---------------------------------------------------------------------------

#: A subproblem the runner AUTHORED, with tests the runner AUTHORED, executed
#: through the identical harness path. It is trivially correct on purpose: if it
#: does not pass, the path between the runner and the published-test runner is
#: broken and every number below is void.
SELF_POSITIVE_CODE = (
    "def henri_s1_reference_sum(values, scale=1.0):\n"
    "    total = 0.0\n"
    "    for v in values:\n"
    "        total += v\n"
    "    return total * scale\n"
)
SELF_POSITIVE_TESTS = (
    "def test_case_1():\n"
    "    assert henri_s1_reference_sum([1, 2, 3]) == 6.0\n"
    "test_case_1()\n"
    "def test_case_2():\n"
    "    assert henri_s1_reference_sum([1, 2], scale=2.0) == 6.0\n"
    "test_case_2()\n"
)
SELF_NEGATIVE_CODE = (
    "def henri_s1_reference_sum(values, scale=1.0):\n"
    "    total = 0.0\n"
    "    for v in values:\n"
    "        total -= v          # deliberately WRONG sign\n"
    "    return total * scale\n"
)


def control_rows(harness) -> list[dict]:
    rows = []

    def add(name, kind, must, code, tests, label):
        res = run_case(harness, code, tests, label)
        cls = classify(res["status"], res["stderr_full"], arm="control",
                       case={"function_header": code}, earlier_defs=set())
        behaved = (res["status"] == sh.STATUS_PASSED) if must == "PASS" else (
            res["status"] != sh.STATUS_PASSED)
        rows.append({
            "control": name, "kind": kind, "must": must, "observed_status": res["status"],
            "observed_taxonomy": cls["taxonomy"], "sub_class": cls["sub_class"],
            "behaved": bool(behaved), "elapsed_ms": res["elapsed_ms"],
            "code_sha256": sha256_text(code), "tests_sha256": sha256_text(tests),
            "stderr_full": res["stderr_full"][:4000],
        })

    # NC-1 / PC-1: authored subproblem, authored tests, SAME path.
    add("PC-1_authored_reference", "positive", "PASS", SELF_POSITIVE_CODE, SELF_POSITIVE_TESTS,
        "PC-1-authored")
    add("NC-1_authored_wrong_sign", "negative", "FAIL", SELF_NEGATIVE_CODE,
        SELF_POSITIVE_TESTS, "NC-1-authored")
    add("NC-2_empty_candidate", "negative", "FAIL", "", SELF_POSITIVE_TESTS, "NC-2-empty")
    return rows


def dataset_control_rows(harness, dev: list[dict]) -> list[dict]:
    """SciCode's OWN ground_truth_code against SciCode's OWN published tests.

    These are reference-arm runs and are excluded from the candidate score. They
    are the dataset-level positive control: if SciCode's published reference
    solution cannot pass SciCode's published test through this harness, the
    harness is broken.
    """
    rows = []
    pairs = [(r, i) for r, i in stable_order(dev)
             if not _TARGET_RE.search("\n".join(_dataset_field(r["sub_steps"][i], "test_cases")))]
    for problem, idx in pairs:
        case = problem["sub_steps"][idx]
        code = build_reference_code(problem, idx)
        tests = "\n".join(_dataset_field(case, "test_cases"))
        s = case
        res = run_case(harness, code, tests, f"dataset-control-{s['step_number']}")
        cls = classify(res["status"], res["stderr_full"], arm="reference_control",
                       case=s, earlier_defs=reference_prior_defs(problem, idx))
        rows.append({
            "control": f"PC-D{len(rows) + 1}_dataset_reference_{s['step_number']}",
            "kind": "positive", "must": "PASS", "observed_status": res["status"],
            "observed_taxonomy": cls["taxonomy"], "sub_class": cls["sub_class"],
            "behaved": res["status"] == sh.STATUS_PASSED,
            "elapsed_ms": res["elapsed_ms"],
            "code_sha256": sha256_text(code), "tests_sha256": sha256_text(tests),
            "stderr_full": res["stderr_full"][:4000],
        })

        # UPPER-BOUND DETECTOR SENSITIVITY CHECK (CONTROL ONLY).
        # Question it answers: is the candidate arm CAPABLE of returning PASSED
        # on this item, or is the measured zero an artifact of the candidate
        # path? A synthetic CONTROL-ONLY generator is fed the dataset reference
        # for this one calibration run. It is NOT registered as a candidate
        # source, it never enters the ledger, and it is excluded from the score.
        with reference_arm_active():
            ub_outputs = [_dataset_field(problem["sub_steps"][k], "ground_truth_code")
                          for k in range(idx + 1)]
        ub_code = build_candidate_code(problem, idx, ub_outputs)
        ub = run_case(harness, ub_code, tests, f"candidate-path-upper-bound-{s['step_number']}")
        ub_cls = classify(ub["status"], ub["stderr_full"], arm="candidate", case=s,
                          earlier_defs=set(), payload=ub_code)
        rows.append({
            "control": f"PC-U{len(rows) + 1}_candidate_path_upper_bound_{s['step_number']}",
            "kind": "positive", "must": "PASS", "observed_status": ub["status"],
            "observed_taxonomy": ub_cls["taxonomy"], "sub_class": ub_cls["sub_class"],
            "behaved": ub["status"] == sh.STATUS_PASSED,
            "elapsed_ms": ub["elapsed_ms"],
            "code_sha256": sha256_text(ub_code), "tests_sha256": sha256_text(tests),
            "stderr_full": ub["stderr_full"][:4000],
            "control_only": ("synthetic generator fed the dataset reference for CALIBRATION ONLY; "
                             "not a registered candidate source; excluded from the ledger and score"),
        })
    return rows


# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------

def main() -> int:
    run_id = ei.run_id_new()
    commit = ei.current_commit(str(REPO))
    t_start = time.time()
    report: dict = {"schema_id": SCHEMA_ID, "run_id": run_id, "commit_sha256": commit,
                    "benchmark_id": BENCHMARK_ID, "started_at_utc":
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print("=" * 78)
    print("S1 SciCode scaffold — CPU-only instrument validation + baseline")
    print("=" * 78)
    print(f"run_id   {run_id}")
    print(f"commit   {commit}")

    # ---- (1) pin -----------------------------------------------------------
    pin = verify_pin()
    report["dataset_pin"] = pin
    print("\n[1] DATASET PIN")
    print(f"  pin        {pin['pin_path']}  revision={pin['revision']}")
    for c in pin["per_file"]:
        print(f"  {c['file']:<22} present={c['present']} match={c['match']} "
              f"sha={c.get('actual_sha256', '?')[:16]}")
    print(f"  all_match  {pin['all_match']}")

    # ---- (2) interpreters --------------------------------------------------
    print("\n[2] INTERPRETER / IN-SANDBOX IMPORT PROBE  (D20)")
    probes = [probe_interpreter(exe) for _, exe in candidate_interpreters()]
    report["interpreter_probes"] = probes
    need = {"numpy", "scipy"}
    chosen = None
    for p in probes:
        env = p["child_environment"] or {}
        mods = env.get("modules", {})
        have = {m for m, v in mods.items() if v.get("ok")}
        p["modules_importable_in_sandbox"] = sorted(have)
        p["meets_requirement"] = need.issubset(have)
        print(f"  {p['declared_interpreter']}")
        print(f"    probe_status={p['probe_status']} sandbox_mode={p['sandbox'].get('mode')} "
              f"isolated={p['sandbox'].get('isolated')}")
        for m, v in mods.items():
            print(f"      {m:<6} {'OK  ' + str(v.get('version')) if v.get('ok') else 'MISSING ' + str(v.get('error'))[:60]}")
            if v.get("ok"):
                print(f"             {v.get('file')}")
        if p["meets_requirement"] and chosen is None:
            chosen = p
    report["chosen_interpreter"] = chosen["declared_interpreter"] if chosen else None

    if not pin["all_match"]:
        return finish(report, t_start, verdict="BLOCKED", reason="dataset pin mismatch")
    if chosen is None:
        return finish(report, t_start, verdict="BLOCKED",
                      reason="no interpreter has numpy+scipy importable inside the sandbox")

    harness = sh.SandboxHarness(mode="auto", timeout_s=ITEM_TIMEOUT_S,
                                python_executable=chosen["declared_interpreter"],
                                mem_limit_mb=MEM_LIMIT_MB, child_env_extra=CHILD_ENV_EXTRA,
                                enable_repl_telemetry=False, max_output_chars=MAX_OUTPUT_CHARS)
    report["sandbox"] = harness.describe()
    print(f"\n  chosen    {report['chosen_interpreter']}")
    print(f"  sandbox   mode={harness.mode} isolated={harness.isolated} surrogate={harness.surrogate}")
    print(f"  downgrade {str(harness.downgrade_reason)[:120]}")

    # ---- (3) dataset + window ---------------------------------------------
    dev = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    window = select_window(dev)
    win_ids = [item_id(r, i) for r, i in window]
    report["dataset_rows"] = len(dev)
    report["window"] = {
        "size": len(window), "item_ids": win_ids,
        "selection_rule": ("(a) every sub_step whose published tests carry no external-constant "
                           "reference, stable order; (b) fill from stable order of all sub_steps"),
        "stable_order": "sorted by (numeric problem_id, sub_step index)",
        "item_ids_sha256": sha256_text("\n".join(win_ids)),
    }
    print(f"\n[3] WINDOW  n={len(window)}  ids_sha={report['window']['item_ids_sha256'][:16]}")
    for wid in win_ids:
        print(f"    {wid}")
    if not window:
        return finish(report, t_start, verdict="BLOCKED", reason="selection produced an empty window")

    # ---- (4) contamination guard self-test --------------------------------
    print("\n[4] CONTAMINATION GUARD SELF-TEST  (G1, G2)")
    sample = dev[0]
    guard_selftest = {}
    try:
        _dataset_field(sample["sub_steps"][0], "ground_truth_code")
        guard_selftest["G1_blocks_reference_read"] = {"raised": False}
    except ContaminationError as exc:
        guard_selftest["G1_blocks_reference_read"] = {"raised": True, "error": str(exc)[:160]}
    try:
        with reference_arm_active():
            _dataset_field(sample["sub_steps"][0], "ground_truth_code")
        guard_selftest["G1_allows_read_in_reference_arm"] = {"raised": False}
    except ContaminationError as exc:
        guard_selftest["G1_allows_read_in_reference_arm"] = {"raised": True, "error": str(exc)[:160]}
    ref_text = ""
    if sample:
        with reference_arm_active():
            ref_text = _dataset_field(sample["sub_steps"][0], "ground_truth_code")
    try:
        _assert_no_reference_overlap(ref_text, ref_text)
        guard_selftest["G2_catches_reference_overlap"] = {"raised": False}
    except ContaminationError as exc:
        guard_selftest["G2_catches_reference_overlap"] = {"raised": True, "error": str(exc)[:160]}
    try:
        _assert_no_reference_overlap("", ref_text)
        guard_selftest["G2_passes_empty_payload"] = {"raised": False}
    except ContaminationError as exc:
        guard_selftest["G2_passes_empty_payload"] = {"raised": True, "error": str(exc)[:160]}
    # UHR-05 control: the PRESCRIBED header is text the task GIVES the candidate, so a
    # payload that reproduces it is NOT contamination. Before the coverage fix this control
    # would have RAISED for 50/50 dev sub-steps, i.e. G2 was unsatisfiable by a faithful
    # candidate. This control is the regression guard for that defect.
    try:
        if sample:
            _hdr0 = str((sample["sub_steps"][0] or {}).get("function_header") or "")
            _deps0 = str(sample.get("required_dependencies") or "")
            _assert_no_reference_overlap(_hdr0 + "\n    return None\n", ref_text,
                                         given=_deps0 + "\n" + _hdr0)
            guard_selftest["G2_passes_prescribed_header"] = {"raised": False}
        else:
            guard_selftest["G2_passes_prescribed_header"] = {"raised": None, "error": "no sample"}
    except ContaminationError as exc:
        guard_selftest["G2_passes_prescribed_header"] = {"raised": True, "error": str(exc)[:160]}
    report["contamination_guard_selftest"] = guard_selftest
    for k, v in guard_selftest.items():
        print(f"    {k:<36} raised={v['raised']}")
    # UHR-05: `G2_passes_prescribed_header` is IN the gate, not just in the report. A
    # control that only prints is a notice: it would have let the unsatisfiable-G2 defect
    # recur silently. `is False` (not a falsy check) so a control that could not run
    # (`raised=None`) fails closed rather than passing vacuously.
    guards_ok = (guard_selftest["G1_blocks_reference_read"]["raised"]
                 and not guard_selftest["G1_allows_read_in_reference_arm"]["raised"]
                 and guard_selftest["G2_catches_reference_overlap"]["raised"]
                 and not guard_selftest["G2_passes_empty_payload"]["raised"]
                 and guard_selftest["G2_passes_prescribed_header"]["raised"] is False)
    report["contamination_guards_behaved"] = guards_ok

    # ---- (5) controls ------------------------------------------------------
    print("\n[5] INSTRUMENT CONTROLS")
    controls = control_rows(harness)
    controls += dataset_control_rows(harness, dev)
    report["controls"] = controls
    for c in controls:
        print(f"    {c['control']:<34} must={c['must']:<4} status={c['observed_status']:<9} "
              f"behaved={c['behaved']} tax={c['observed_taxonomy']}")
    controls_ok = all(c["behaved"] for c in controls)
    report["controls_all_behaved"] = controls_ok

    # ---- (6) run -----------------------------------------------------------
    run_dir = ei.run_output_dir(OUT_ROOT, commit, BENCHMARK_ID, run_id)
    ledger = ItemLedger(run_dir / "items.jsonl")
    print(f"\n[6] RUN  ledger={ledger.path}")
    # UHR-05: candidate source selection. DEFAULT STAYS NullCandidateSource, so the
    # default path is byte-identical. Opt in with HENRI_SCICODE_CANDIDATE_SOURCE=backbone.
    # FAIL-CLOSED: a source that cannot run (no weights, no generation stack, unknown
    # name) must BLOCK through the same finish() path as any other precondition failure.
    # It must NEVER fall back to the Null source -- that would score 0 and look like a
    # measurement of HENRI rather than of an absent generator.
    try:
        from scicode_candidate_sources import select_source, CandidateSourceUnavailable
        _selected = select_source(verbose=True)
    except CandidateSourceUnavailable as _csu:
        return finish(report, t_start, verdict="BLOCKED",
                      reason=f"candidate source unavailable: {_csu}")
    except Exception as _cse:  # an import defect is also a precondition failure
        return finish(report, t_start, verdict="BLOCKED",
                      reason=f"candidate source import failed: {type(_cse).__name__}: {_cse}")
    source = _selected if _selected is not None else NullCandidateSource()
    report["candidate_source"] = {"name": source.name, "produces_code": source.produces_code,
                                  "description": source.description}
    print(f"  candidate source: {source.name} (produces_code={source.produces_code})")

    rows_written = 0
    for problem, idx in window:
        case = problem["sub_steps"][idx]
        wid = item_id(problem, idx)
        tests = "\n".join(_dataset_field(case, "test_cases"))
        deps = (problem.get("required_dependencies") or "")
        header = _dataset_field(case, "function_header") or ""
        step_prompt = _dataset_field(case, "step_description_prompt") or ""
        # UHR-05 OFF-BY-ONE FIX. MEASURED DEFECT this repairs: this list was built with
        # `range(idx)` (EXCLUSIVE) while the REFERENCE arm below uses `range(idx + 1)`
        # (INCLUSIVE). `build_candidate_code` then read `prior_outputs[idx]`, which for a
        # list of length `idx` is ALWAYS out of range, so it silently substituted "" and
        # the CURRENT step's generated code never entered the payload -- `source.produce`
        # was never even called for the current step (at idx=0, never at all). The
        # published tests require that sub-step's OWN function, so every scored item died
        # as `CANDIDATE_UNDEFINED_SYMBOL:<current function>`.
        # THIS FIX CANNOT RAISE pass@1 BY ITSELF: with NullCandidateSource the payload is
        # unchanged (`deps + "\n"`) and pass@1 stays 0.0. It removes a block on FUTURE
        # measurement; it does not manufacture a result.
        prior_candidate = [source.produce((problem, k), []) for k in range(idx + 1)]
        cand_code = build_candidate_code(problem, idx, prior_candidate)

        # G2 on the scored payload, before anything is executed.
        with reference_arm_active():
            ref_text_item = "\n".join(
                _dataset_field(problem["sub_steps"][k], "ground_truth_code") for k in range(idx + 1))
        # UHR-05: pass the GIVEN text (deps + the prescribed headers) so the coverage
        # differential can separate "reproduced what the task handed over" from "copied the
        # solution". `function_header` is not a reference-only field, so it is read directly.
        given_text_item = ((problem.get("required_dependencies") or "") + "\n" + "\n".join(
            str((problem["sub_steps"][k] or {}).get("function_header") or "")
            for k in range(idx + 1)))
        _assert_no_reference_overlap(cand_code, ref_text_item, given=given_text_item)

        ref_code = build_reference_code(problem, idx)
        ref_res = run_case(harness, ref_code, tests, f"ref-{wid}")
        ref_cls = classify(ref_res["status"], ref_res["stderr_full"], arm="reference_control",
                           case=case, earlier_defs=reference_prior_defs(problem, idx))
        cand_res = run_case(harness, cand_code, tests, f"cand-{wid}")
        cand_cls = classify(cand_res["status"], cand_res["stderr_full"], arm="candidate",
                            case=case, earlier_defs=set(), payload=cand_code)

        attemptable = ref_cls["taxonomy"] == "STATUS_PASSED"
        row = {
            "schema_id": SCHEMA_ID,
            "item_id": wid,
            "problem_id": problem["problem_id"],
            "step_number": case["step_number"],
            "arm": "candidate",
            "candidate_source": source.name,
            "candidate_produces_code": source.produces_code,
            "prompt_sha256": sha256_text(deps + header + step_prompt),
            "candidate_code_sha256": sha256_text(cand_code),
            "published_tests_sha256": sha256_text(tests),
            "status": TAXONOMY_TO_RUN_STATUS[cand_cls["taxonomy"]],
            "taxonomy": cand_cls["taxonomy"],
            "sub_class": cand_cls["sub_class"],
            "evidence": cand_cls["evidence"],
            "passed": cand_cls["taxonomy"] == "STATUS_PASSED",
            "attemptable": bool(attemptable),
            "counts_toward_score": bool(attemptable),
            "required_functions": list(required_function_names(case)),
            "returncode": cand_res["returncode"],
            "elapsed_ms": cand_res["elapsed_ms"],
            "run_mode": cand_res["mode"],
            "run_isolated": cand_res["isolated"],
            "raw_stdout_sha256": sha256_text(cand_res["stdout"]),
            "raw_stderr_sha256": sha256_text(cand_res["stderr_full"]),
            "stderr_full": cand_res["stderr_full"],           # D21: before aggregation
            "stderr_truncated": cand_res["stderr_truncated"],
            "passed_property_agrees": (cand_res["passed_property"]
                                       == (cand_cls["taxonomy"] == "STATUS_PASSED")),
            "reference_arm": {
                "code_sha256": sha256_text(ref_code),
                "status": ref_res["status"],
                "taxonomy": ref_cls["taxonomy"],
                "sub_class": ref_cls["sub_class"],
                "evidence": ref_cls["evidence"],
                "elapsed_ms": ref_res["elapsed_ms"],
                "stderr_full": ref_res["stderr_full"],
                "raw_stderr_sha256": sha256_text(ref_res["stderr_full"]),
            },
            "dataset_text_ingested": False,
            "tool_calls": 0, "retries": 0, "candidate_scores": [],
            "sagnac_delta": None, "token_top1": None, "token_entropy": None,
            "elapsed_s": round(cand_res["elapsed_ms"] / 1000.0, 3),
        }
        ledger.append(row)   # validated + fsynced BEFORE aggregation
        rows_written += 1
        print(f"    {wid:<12} ref={ref_cls['taxonomy']:<26} cand={cand_cls['taxonomy']:<24} "
              f"sub={cand_cls['sub_class'][:34]}")

    # ---- (7) aggregate by reading the ledger back (D21) --------------------
    persisted = list(ledger.rows())
    report["ledger_rows_written"] = rows_written
    # UHR-05: attach the source provenance AFTER the run loop. MEASURED DEFECT: attaching it
    # BEFORE the loop recorded `calls=0, emitted=0, device=None` on every run, because those
    # counters only move while generating -- a dead snapshot inside a receipt whose entire
    # purpose is attribution. Attaching here makes the model id, resolved snapshot, device,
    # emitted / no_code_emitted / defines_expected counts and the
    # BACKBONE_BASELINE_NOT_HENRI_CAPABILITY evidence class travel WITH the score.
    if hasattr(source, "report"):
        try:
            report["candidate_source_report"] = source.report()
        except Exception as _csr_exc:
            report["candidate_source_report_error"] = f"{type(_csr_exc).__name__}: {_csr_exc}"
    report["ledger_rows_read"] = len(persisted)
    tax = collections.Counter(r["taxonomy"] for r in persisted)
    ref_tax = collections.Counter(r["reference_arm"]["taxonomy"] for r in persisted)
    attemptable = [r for r in persisted if r["attemptable"]]
    passed = [r for r in attemptable if r["taxonomy"] == "STATUS_PASSED"]
    denominator = len(attemptable)
    numerator = len(passed)

    property_agreement = sum(1 for r in persisted if r["passed_property_agrees"])
    report["failure_taxonomy"] = {"candidate_arm": dict(tax), "reference_arm": dict(ref_tax)}
    report["score"] = {
        "numerator": numerator, "denominator": denominator,
        "pass_at_1": (round(numerator / denominator, 6) if denominator else None),
        "window_size": len(persisted),
        "attemptable_item_ids": [r["item_id"] for r in attemptable],
        "passed_item_ids": [r["item_id"] for r in passed],
        "denominator_definition": ("sub_steps whose OWN SciCode reference solution passed its OWN "
                                  "published tests through this harness inside the window"),
    }
    report["stderr_persistence"] = {
        "full_stderr_in_every_failure_row": all(
            bool(r["stderr_full"]) for r in persisted if r["taxonomy"] != "STATUS_PASSED"),
        "any_stderr_truncated": any(r["stderr_truncated"] for r in persisted),
        "unattributed_rows": sum(1 for r in persisted if r["sub_class"].startswith("unclassified")),
        "ledger": "items.jsonl alongside this receipt; raw *.jsonl is gitignored, so the "
                  "self-contained per-item digest below carries id/class/evidence/stderr sha256 "
                  "for every row (D21: no unattributable row, in-repo or on disk)",
    }
    report["per_item"] = [{
        "item_id": r["item_id"],
        "arm": r["arm"],
        "taxonomy": r["taxonomy"],
        "sub_class": r["sub_class"],
        "status": r["status"],
        "evidence": r["evidence"],
        "counts_toward_score": r["counts_toward_score"],
        "reference_arm_taxonomy": r["reference_arm"]["taxonomy"],
        "reference_arm_sub_class": r["reference_arm"]["sub_class"],
        "candidate_code_sha256": r["candidate_code_sha256"],
        "published_tests_sha256": r["published_tests_sha256"],
        "raw_stderr_sha256": r["raw_stderr_sha256"],
        "raw_stderr_chars": len(r["stderr_full"]),
        "raw_stderr_tail": r["stderr_full"][-700:],
        "elapsed_ms": r["elapsed_ms"],
    } for r in persisted]
    report["d19_calibration"] = {
        "counts_derived_from": "henri_sandbox_harness.STATUS_PASSED vs SandboxResult.status",
        "passed_property_used_for_counting": False,
        "passed_property_agreement_rows": property_agreement,
        "rows": len(persisted),
    }

    report["reference_arm_defects"] = {
        "isolation_artifacts": [r["item_id"] for r in persisted
                                if r["reference_arm"]["sub_class"].startswith("ISOLATION_ARTIFACT")],
        "note": ("an ISOLATION_ARTIFACT means MY accumulated-protocol assumption failed for that "
                 "item, which invalidates its attemptability classification"),
    }
    protocol_defect = bool(report["reference_arm_defects"]["isolation_artifacts"])

    # ---- (8) verdict -------------------------------------------------------
    if not controls_ok or not guards_ok:
        verdict, reason = "VOID", ("an instrument control misbehaved or a contamination guard "
                                   "did not self-test as designed; the harness is NOT validated")
    elif protocol_defect:
        verdict, reason = "VOID", ("the reference arm produced a protocol artifact "
                                   "(ISOLATION_ARTIFACT), so the attemptability classification "
                                   "cannot be trusted; the harness is NOT validated")
    elif denominator == 0:
        verdict, reason = "SCORED", ("controls passed and the pin matched, but ZERO window items "
                                     "were attemptable: every published test is dataset-blocked")
    else:
        verdict, reason = "SCORED", "controls passed and the pin matched"
    report["verdict"] = verdict
    report["verdict_reason"] = reason

    return finish(report, t_start, verdict=verdict, reason=reason, run_dir=run_dir,
                  ledger_path=ledger.path, ledger=ledger, persisted=persisted)


def finish(report: dict, t_start: float, *, verdict: str, reason: str,
           run_dir: pathlib.Path | None = None,
           ledger_path: pathlib.Path | None = None,
           ledger: ItemLedger | None = None,
           persisted: list[dict] | None = None) -> int:
    report["verdict"] = verdict
    report["blocking_reason" if verdict == "BLOCKED" else "verdict_reason"] = reason
    report["wall_seconds"] = round(time.time() - t_start, 2)
    report["non_claims"] = [
        "INSTRUMENT VALIDATION AND BASELINE ONLY. This is NOT evidence of HENRI capability.",
        "No code generator is wired to SciCode; the only registered candidate source emits the "
        "empty string. Any pass@1 here is the floor for an ABSENT generator.",
        "Makes NO claim about the AAII index, its weights, or any constituent.",
        "No local grader, tolerance, or re-derived expected value was substituted for the "
        "published tests.",
        "Not a namespace-isolated run: mode=container-rlimit, isolated=False, surrogate=True.",
        "The reference-control arm executes SciCode's own ground_truth_code; it is excluded from "
        "the candidate numerator and denominator.",
    ]
    if not report.get("dataset_pin"):
        report["dataset_pin"] = {}
    passed = report.get("score", {}).get("numerator", 0)
    denom = report.get("score", {}).get("denominator", 0)
    attempted = report.get("ledger_rows_read", 0)
    receipt = ei.build_run_receipt(
        run_id=report["run_id"], commit_sha=report["commit_sha256"], benchmark_id=BENCHMARK_ID,
        dataset_source="hf://SciCode1/SciCode",
        dataset_sha256=(report["dataset_pin"].get("per_file") or [{}])[0].get("actual_sha256"),
        item_count=attempted, attempted=attempted, passed=passed,
        failed=attempted - passed, execution_errors=0,
        ledger=ledger,
        extra={"s1_scaffold": report})
    out_dir = run_dir if run_dir is not None else OUT_ROOT / f"blocked__{report['run_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    receipt_sha = sha256_file(receipt_path)

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  verdict            {report['verdict']}")
    print(f"  reason             {reason}")
    print(f"  controls           all_behaved={report.get('controls_all_behaved')}")
    print(f"  guards             behaved={report.get('contamination_guards_behaved')}")
    print(f"  pass@1             {passed}/{denom}"
          + (f" = {passed / denom:.4f}" if denom else " (undefined: no attemptable items)"))
    ft = report.get("failure_taxonomy", {})
    print(f"  candidate taxonomy {ft.get('candidate_arm')}")
    print(f"  reference taxonomy {ft.get('reference_arm')}")
    print(f"  ledger             {ledger_path}")
    print(f"  receipt            {receipt_path}")
    print(f"  receipt_sha256     {receipt_sha}")
    print(f"  wall_seconds       {report['wall_seconds']}")
    print("  NON-CLAIMS: instrument validation + baseline only; NOT a HENRI capability claim; "
          "no AAII claim.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        print(json.dumps({"schema_id": SCHEMA_ID, "status": "CRASH",
                          "verdict": "BLOCKED",
                          "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        raise
