"""Routing and receipt promotion decisions for deterministic-first execution."""
from dataclasses import asdict, is_dataclass

def select_route(task_type: str, risk_class: str="routine", explicit_moa: bool=False) -> str:
    if risk_class == "load_bearing" or explicit_moa:
        return "bounded_moa"
    if task_type in {"verification", "telemetry", "governance"}:
        return "deterministic"
    return "single_worker"

def _status(receipt):
    if isinstance(receipt, dict):
        return receipt.get("status")
    if is_dataclass(receipt):
        return asdict(receipt).get("status")
    return getattr(receipt, "status", None)

def can_promote_verified(claim_status: str, receipts: list[object]) -> bool:
    return claim_status == "verified" and bool(receipts) and all(_status(r) == "pass" for r in receipts)
