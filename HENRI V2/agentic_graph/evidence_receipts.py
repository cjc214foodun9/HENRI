"""Deterministic evidence receipts. A receipt proves only its stated check."""
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import hashlib, re

class ReceiptError(ValueError): pass
_HASH_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
@dataclass(frozen=True)
class EvidenceReceipt:
    schema_version: str
    receipt_id: str
    kind: str
    status: str
    subject_ref: str
    subject_sha256: str | None
    tool: str
    command_or_query_hash: str
    summary: str
    artifact_ref: str
    created_at_utc: str
    command_or_method: str = ""
    artifact_sha256: str | None = None
    details_ref: str | None = None
    exit_code: int | None = None
    def to_dict(self):
        """Return a payload that conforms to the evidence-receipt schema."""
        return asdict(self)
    def validate(self):
        if self.kind not in {"test","hash","schema","citation","query","lint","diff"}: raise ReceiptError("invalid receipt kind")
        if self.status not in {"pass","fail","unavailable"}: raise ReceiptError("invalid receipt status")
        if len(self.command_or_query_hash)!=64: raise ReceiptError("command hash must be SHA-256")
        for value in (self.subject_sha256, self.artifact_sha256):
            if value is not None and not _HASH_RE.fullmatch(value): raise ReceiptError("invalid SHA-256")
        if not self.command_or_method.strip(): raise ReceiptError("command_or_method is required")
        return self

def _sha(data: bytes): return hashlib.sha256(data).hexdigest()
def file_hash_receipt(path: str, receipt_id: str, tool: str="sha256") -> EvidenceReceipt:
    p=Path(path); digest=_sha(p.read_bytes())
    tagged="sha256:"+digest
    return EvidenceReceipt("1.0",receipt_id,"hash","pass",str(p),tagged,tool,_sha(tool.encode()),f"SHA-256 computed for {p.name}",str(p),datetime.now(timezone.utc).isoformat(),tool, tagged).validate()
def verify_artifact_hash(*, receipt_id, subject_ref, expected_sha256, artifact_ref, actual_bytes, tool="sha256"):
    actual = _sha(actual_bytes)
    expected = expected_sha256.removeprefix("sha256:")
    status = "pass" if actual == expected else "fail"
    return EvidenceReceipt("1.0",receipt_id,"hash",status,subject_ref,"sha256:"+actual,tool,_sha(tool.encode()),f"expected={expected}; actual={actual}",artifact_ref,datetime.now(timezone.utc).isoformat(),tool,"sha256:"+actual).validate()

def receipt_from_result(*,receipt_id,kind,status,subject_ref,summary,artifact_ref,tool,command_or_method,subject_sha256=None,artifact_sha256=None,details_ref=None,exit_code=None):
    r=EvidenceReceipt("1.0",receipt_id,kind,status,subject_ref,subject_sha256,tool,_sha(command_or_method.encode()),summary,artifact_ref,datetime.now(timezone.utc).isoformat(),command_or_method,artifact_sha256,details_ref,exit_code)
    return r.validate()
