"""Small typed models for the HENRI graph boundary."""
from dataclasses import dataclass, field
from typing import Any

TERMINAL_STATES = {"complete", "partial", "blocked", "failed", "budget_exhausted"}

@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    status: str
    artifact_ref: str
    artifact_sha256: str | None = None

@dataclass
class HolonResult:
    task_id: str
    status: str
    claim_status: str
    finding: str
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    next_action: str = "defer"
    risks: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def validate_claim(self) -> None:
        if self.claim_status == "verified" and not self.evidence_refs:
            raise ValueError("verified claim requires at least one evidence reference")
        if self.status not in TERMINAL_STATES | {"active", "dispatched", "repair"}:
            raise ValueError(f"invalid result status: {self.status}")
