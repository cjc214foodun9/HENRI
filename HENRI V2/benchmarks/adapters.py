"""Task adapters: a uniform surface over heterogeneous benchmark families.

THE CONTRACT
    An adapter yields Task records and turns a submitted answer into a score.
    The harness core never needs to know what the benchmark means.

THE TWO HARD RULES ENFORCED HERE
    1. An adapter with no pinned dataset CANNOT be scored. It reports
       SpecStatus.METHODOLOGY_ONLY and score() raises. This is the anti-laundering
       rule: no fabricated AAII number can leave the harness.
    2. A pinned dataset MUST declare a sha256 for every file it reads, verified
       at load. Unpinned data is never scored.

WHY THE AAII ADAPTERS ARE STUBS WITH REAL METADATA
    benchmarks/spec_registry.py holds the pinned methodology (prompts, metrics,
    judges, runs, turns) taken from the operator's PDFs. The DATASETS and the
    HARNESSES were NOT in those documents -- only URLs. So each AAII adapter
    carries its real spec metadata and raises on score() until its dataset is
    fetched and pinned. That is the honest state, encoded rather than promised.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from . import spec_registry as SR
from .harness_core import SpecStatus, sha256_file


class AdapterNotReady(RuntimeError):
    """Raised when an adapter is asked to score without a pinned dataset."""


@dataclass
class Task:
    """One benchmark item."""
    task_id: str
    prompt: str
    answers: list[str] = field(default_factory=list)   # gold, when disclosed
    meta: dict = field(default_factory=dict)


class TaskAdapter:
    """Base class. Subclasses must declare id, spec_status, dataset_url."""

    id: str = "abstract"
    spec_status: str = SpecStatus.METHODOLOGY_ONLY
    dataset_url: str | None = None
    harness_url: str | None = None
    requires_network: bool = False

    # -------------------------------------------------------------- dataset
    def dataset_files(self) -> dict[str, str]:
        """{path: sha256} for every file this adapter read. MUST be non-empty
        before score() is allowed to return a number."""
        return {}

    def load(self):
        raise NotImplementedError

    def tasks(self) -> list[Task]:
        raise NotImplementedError

    # --------------------------------------------------------------- scoring
    def score(self, task: Task, submission: str) -> float:
        raise NotImplementedError

    def assert_ready(self):
        """Anti-laundering gate. No pinned dataset => no score, ever."""
        if self.spec_status != SpecStatus.IMPLEMENTED:
            raise AdapterNotReady(
                f"{self.id}: spec_status={self.spec_status}. There is no pinned "
                f"dataset at {self.dataset_url!r} in this repository, so no score "
                "can be produced. Fetch the dataset, record its sha256, then set "
                "spec_status=IMPLEMENTED. Until then any number would be fabricated.")
        files = self.dataset_files()
        if not files:
            raise AdapterNotReady(
                f"{self.id}: spec_status is IMPLEMENTED but dataset_files() is "
                "empty. Unpinned data is never scored.")
        for path, want in files.items():
            if not os.path.isfile(path):
                raise AdapterNotReady(f"{self.id}: dataset file missing: {path}")
            got = sha256_file(path)
            if got != want:
                raise AdapterNotReady(
                    f"{self.id}: dataset hash mismatch for {path}: want {want}, "
                    f"got {got}. Refusing to score against unpinned data.")


# --------------------------------------------------------------------- AAII
class AAIIAdapter(TaskAdapter):
    """One AAII v4.3.2 evaluation. Real spec metadata; dataset NOT present.

    Constructed from spec_registry.EVALS so the pinned methodology and this
    adapter can never drift apart.
    """

    def __init__(self, eval_id: str):
        rec = SR.by_id(eval_id)
        self.eval_id = eval_id
        self.id = f"aaii-v{SR.INDEX_VERSION}/{eval_id}"
        self.rec = rec
        self.dataset_url = rec.get("dataset_url")
        self.harness_url = rec.get("harness_url")
        self.metric = rec.get("metric")
        self.grading = rec.get("grading")
        self.evidence = rec.get("evidence")
        self.index_weight = rec.get("index_weight")
        # spec_status comes from the registry; nothing is assumed here.
        self.spec_status = rec["spec_status"]
        self.requires_network = bool(self.dataset_url and
                                     "huggingface.co" in (self.dataset_url or ""))

    def load(self):
        self.assert_ready()

    def tasks(self):
        self.assert_ready()
        return []

    def score(self, task, submission):
        self.assert_ready()
        raise AdapterNotReady("unreachable: assert_ready raises first")

    def descriptor(self) -> dict:
        """Everything known about this eval, with the honest status attached."""
        return {
            "id": self.id, "eval_id": self.eval_id,
            "name": self.rec.get("name"), "version": self.rec.get("version"),
            "track": self.rec.get("track"),
            "spec_status": self.spec_status,
            "dataset_url": self.dataset_url, "harness_url": self.harness_url,
            "metric": self.metric, "grading": self.grading,
            "runs_per_task": self.rec.get("runs_per_task"),
            "turns": self.rec.get("turns"),
            "index_weight": self.index_weight,
            "evidence": self.evidence,
            "scoreable": self.spec_status == SpecStatus.IMPLEMENTED,
            "blocker": (None if self.spec_status == SpecStatus.IMPLEMENTED
                        else self._blocker()),
        }

    def _blocker(self) -> str:
        if self.spec_status == SpecStatus.UNAVAILABLE:
            return ("needs an external grading API (not self-scoreable): "
                    + (self.rec.get("grading") or ""))
        return ("dataset is not in the attached documents; only the URL is. "
                "Fetch and pin sha256, then set IMPLEMENTED.")


def aaii_adapters() -> dict[str, AAIIAdapter]:
    """All ten AAII v4.3.2 evaluations named in the operator's documents."""
    return {e["id"]: AAIIAdapter(e["id"]) for e in SR.EVALS}


def spec_readiness_table() -> list[dict]:
    """One row per eval: what is known, and what blocks a score."""
    return [a.descriptor() for a in aaii_adapters().values()]


if __name__ == "__main__":
    rows = spec_readiness_table()
    print(json.dumps({
        "n_evals": len(rows),
        "scoreable_now": sum(1 for r in rows if r["scoreable"]),
        "blocked": [r["id"] for r in rows if not r["scoreable"]],
        "rows": rows,
    }, indent=2))
