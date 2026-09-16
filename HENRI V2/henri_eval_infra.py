"""S0 evidence + run infrastructure for the AAII v4.3 R&D pipeline.

Spec: SPEC-2026-09-16-AAII-V43-RD-PIPELINE.json sections 6.2 and 6.3.

WHY EACH PIECE EXISTS (session-measured defects, 2026-09-15/16)

1. ``run_output_dir`` - a unique per-run directory keyed to
   (commit_sha, benchmark_id, run_id).
   DEFECT IT PREVENTS: two runs of the same harness wrote to the SAME filename.
   The later run overwrote the earlier one, so an earlier verdict survived only
   as prose in a document. A verification number that a later job can overwrite
   is not evidence.

2. ``ItemLedger`` - append-only per-item JSONL, flushed AND fsync'd per row.
   DEFECT IT PREVENTS: a pipeline that persisted telemetry only at the END lost
   every row when aggregation crashed after the arms had executed.

3. ``reconcile`` - Contract A arithmetic invariants.
   ``passed + failed == attempted`` and
   ``attempted + execution_errors == item_count``.
   A violation is INVALID_RUN, never a warning.

4. ``sha256_text_lf`` vs ``sha256_bytes`` - two different digests on purpose.
   Text manifests hash CANONICAL LF BYTES, because this repository sets
   core.autocrlf=true and the committed blob differs from the scratch file.
   Dataset .jsonl files hash RAW bytes. Mixing these two rules produced a
   published receipt hash that no fresh clone could reproduce.

5. ``scan_contamination`` - K-D. Benchmark task rows must never reach Zone C or
   a model store before evaluation runs.

This module EXTENDS the existing evidence surface. It does not replace
``henri_benchmark_registry``, which already declares
``henri.benchmark-registry.v1``, ``henri.run-evidence.v1`` and
``henri.arc-episode-trace.v1``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator

# --- schema identity -------------------------------------------------------

RUN_ITEM_SCHEMA = "henri.run-item.v1"

#: Required keys for every per-item telemetry row. A row missing any key raises.
RUN_ITEM_FIELDS: tuple[str, ...] = (
    "item_id",
    "prompt_sha256",
    "raw_stdout_sha256",
    "raw_stderr_sha256",
    "status",
    "elapsed_ms",
    "tool_calls",
    "retries",
    "candidate_scores",
    "sagnac_delta",
    "token_top1",
    "token_entropy",
)

STATUS_VALUES: frozenset[str] = frozenset(
    {"PASSED", "FAILED", "EXECUTION_ERROR", "VETOED", "SKIPPED", "TIMEOUT", "ABSTAINED"}
)

SAGNAC_LO, SAGNAC_HI = 0.0, 2.0


# --- digests ---------------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    """Raw-bytes digest. Use for datasets, binaries, and JSONL corpora."""
    return hashlib.sha256(b).hexdigest()


def sha256_text_lf(text: str) -> str:
    """Canonical LF-byte digest. Use ONLY for text artifacts and manifests.

    This repository sets core.autocrlf=true. A scratch file with CRLF endings
    and its committed blob with LF endings produce DIFFERENT raw digests. Text
    contracts therefore hash the LF-normalized form so that a fresh clone can
    reproduce the published value.
    """
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def sha256_file_raw(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


# --- run identity and unique output paths ----------------------------------

def run_id_new() -> str:
    """UTC timestamp plus a short random suffix. Unique across processes."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]


def run_output_dir(root: str | Path, commit_sha: str, benchmark_id: str,
                   run_id: str) -> Path:
    """Create and return a unique per-run directory.

    Keyed to (commit_sha, benchmark_id, run_id) so no two runs can overwrite
    each other's evidence. ``exist_ok=False`` is deliberate: silently reusing an
    existing run directory would reintroduce the overwrite defect.
    """
    safe_bench = re.sub(r"[^A-Za-z0-9._-]", "-", str(benchmark_id))
    safe_commit = re.sub(r"[^A-Za-z0-9._-]", "-", str(commit_sha))[:12]
    safe_run = re.sub(r"[^A-Za-z0-9._-]", "-", str(run_id))
    p = Path(root) / f"{safe_bench}__{safe_commit}__{safe_run}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def current_commit(repo_root: str | Path | None = None) -> str:
    """Best-effort commit SHA. Never raises; returns 'UNKNOWN' on failure."""
    import subprocess

    cwd = str(repo_root) if repo_root else None
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, cwd=cwd, timeout=20)
        sha = r.stdout.strip()
        return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


# --- append-only per-item ledger -------------------------------------------

class ItemLedger:
    """Append-only JSONL ledger for per-item telemetry.

    Each ``append`` validates the row, writes one line, flushes, and fsyncs
    before returning. Rows therefore reach disk BEFORE any aggregation step.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0
        if self.path.exists():
            # Count pre-existing rows so a resumed run reports honestly.
            self._count = sum(1 for line in self.path.read_text(
                encoding="utf-8", errors="replace").splitlines() if line.strip())

    def validate(self, row: dict[str, Any]) -> None:
        missing = [k for k in RUN_ITEM_FIELDS if k not in row]
        if missing:
            raise ValueError(f"{RUN_ITEM_SCHEMA}: missing required fields {missing}")
        if row["status"] not in STATUS_VALUES:
            raise ValueError(
                f"{RUN_ITEM_SCHEMA}: invalid status {row['status']!r}; "
                f"allowed={sorted(STATUS_VALUES)}")
        sd = row.get("sagnac_delta")
        if sd is not None:
            sd = float(sd)
            if not (SAGNAC_LO <= sd <= SAGNAC_HI):
                raise ValueError(
                    f"{RUN_ITEM_SCHEMA}: sagnac_delta={sd} outside "
                    f"[{SAGNAC_LO}, {SAGNAC_HI}]")

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        self.validate(row)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._count += 1
        return row

    @property
    def count(self) -> int:
        return self._count

    def rows(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def sha256(self) -> str:
        if not self.path.exists():
            return sha256_bytes(b"")
        return sha256_bytes(self.path.read_bytes())


# --- Contract A invariants --------------------------------------------------

def reconcile(item_count: int, attempted: int, passed: int, failed: int,
              execution_errors: int) -> dict[str, Any]:
    """Contract A arithmetic. Returns {'valid': bool, 'problems': [...]}.

    A caller that finds ``valid`` False MUST emit INVALID_RUN and stop; the
    numbers may not be reported as a score.
    """
    problems: list[str] = []
    if passed + failed != attempted:
        problems.append(
            f"passed({passed}) + failed({failed}) = {passed + failed} "
            f"!= attempted({attempted})")
    if attempted + execution_errors != item_count:
        problems.append(
            f"attempted({attempted}) + execution_errors({execution_errors}) = "
            f"{attempted + execution_errors} != item_count({item_count})")
    for name, val in (("item_count", item_count), ("attempted", attempted),
                      ("passed", passed), ("failed", failed),
                      ("execution_errors", execution_errors)):
        if int(val) < 0:
            problems.append(f"{name} is negative ({val})")
    return {"valid": not problems, "problems": problems}


def assert_sagnac(value: float) -> float:
    """Normalized Sagnac delta is bounded in [0, 2]."""
    v = float(value)
    if not (SAGNAC_LO <= v <= SAGNAC_HI):
        raise ValueError(f"normalized sagnac delta {v} outside [0,2]")
    return v


# --- K-D contamination scanner ---------------------------------------------

#: Markers that identify a row as originating from an evaluation benchmark.
CONTAMINATION_MARKERS: tuple[str, ...] = (
    "aaii", "scicode", "terminal-bench", "terminal_bench", "terminalbench",
    "automationbench", "gdpval", "gdp-val", "omniscience",
    "humanity's last exam", "humanitys last exam", "critpt", "aa-lcr",
    "gdp.pdf", "mbpp", "humaneval", "hle",
)

#: Field names that indicate a task/item identity rather than prose.
_ID_FIELD_RE = re.compile(r"item|task|problem|question|example|case", re.I)


def scan_contamination(store_rows: Iterable[dict[str, Any]],
                       text_fields: tuple[str, ...] = ("prompt", "text",
                                                       "content", "body")) -> dict[str, Any]:
    """K-D: detect benchmark task rows inside Zone C or a model store.

    A row is a HIT when BOTH conditions hold:
      * a text field contains a benchmark marker, AND
      * the row carries an identifier-like field holding a digit.

    Requiring both conditions avoids flagging documentation or telemetry prose
    that merely names a benchmark. Callers pass rows READ FROM A STORE, never
    documentation.

    Returns {'contaminated', 'hits', 'rows_scanned'}. ``contaminated=True``
    voids the run: BLOCKED_CONTAMINATION.
    """
    hits: list[dict[str, Any]] = []
    n = 0
    for i, row in enumerate(store_rows):
        n = i + 1
        if not isinstance(row, dict):
            continue
        blob = " ".join(str(row.get(f, "")) for f in text_fields).lower()
        found = [m for m in CONTAMINATION_MARKERS if m in blob]
        if not found:
            continue
        ident = " ".join(
            str(v) for k, v in row.items()
            if isinstance(k, str) and _ID_FIELD_RE.search(k))
        if re.search(r"\d", ident):
            hits.append({"row_index": i, "markers": found,
                         "id_fields": {k: str(v)[:60] for k, v in row.items()
                                       if isinstance(k, str) and _ID_FIELD_RE.search(k)}})
    return {"contaminated": bool(hits), "hits": hits, "rows_scanned": n}


# --- run receipt ------------------------------------------------------------

def build_run_receipt(*, run_id: str, commit_sha: str, benchmark_id: str,
                      dataset_source: str | None = None,
                      dataset_sha256: str | None = None,
                      item_count: int, attempted: int, passed: int, failed: int,
                      execution_errors: int, ledger: ItemLedger | None = None,
                      extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the extended ``henri.run-evidence.v1`` receipt.

    Field names match the eight populated ``run_evidence.json`` files already in
    this repository so the new pipeline is comparable to the MBPP receipts.
    """
    rec = reconcile(item_count, attempted, passed, failed, execution_errors)
    receipt: dict[str, Any] = {
        "schema_id": "henri.run-evidence.v1",
        "run_id": run_id,
        "commit_sha256": commit_sha,
        "benchmark_id": benchmark_id,
        "dataset_source": dataset_source,
        "dataset_sha256": dataset_sha256,
        "item_count": int(item_count),
        "attempted_count": int(attempted),
        "passed_count": int(passed),
        "failed_count": int(failed),
        "execution_error_count": int(execution_errors),
        "item_results_sha256": ledger.sha256() if ledger else None,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "OBSERVED" if rec["valid"] else "INVALID",
        "accounting": rec,
    }
    if extra:
        receipt.update(extra)
    return receipt
