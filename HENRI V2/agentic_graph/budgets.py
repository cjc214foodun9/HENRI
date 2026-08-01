"""Hard graph budgets and leaf-only retry control. No model calls."""
from dataclasses import dataclass, field
import hashlib, json

MAX_GRAPH_DEPTH = 2
MAX_TOTAL_MODEL_CALLS = 4
MAX_DEFAULT_WORKERS = 1
MAX_HIGH_RISK_WORKERS = 3
MAX_LEAF_RETRIES = 1
MAX_COE_REPAIR_CYCLES = 2
MAX_DYNAMIC_CONTEXT_CHARS = 96000
MAX_WORKER_OUTPUT_CHARS = 12000
MAX_SYNTHESIS_INPUT_CHARS = 48000

class BudgetExceeded(RuntimeError): pass

@dataclass
class GraphBudget:
    max_depth: int = 2
    max_model_calls: int = MAX_TOTAL_MODEL_CALLS
    max_workers: int = MAX_DEFAULT_WORKERS
    max_hard_workers: int = MAX_HIGH_RISK_WORKERS
    max_leaf_retries: int = MAX_LEAF_RETRIES
    max_coe_repairs: int = MAX_COE_REPAIR_CYCLES
    model_calls: int = 0
    workers: int = 0
    depth: int = 0
    coe_repairs: int = 0

    def consume_model_call(self, count: int = 1) -> None:
        if self.model_calls + count > self.max_model_calls:
            raise BudgetExceeded("model-call budget exhausted")
        self.model_calls += count

    def admit_workers(self, count: int, high_risk: bool = False) -> None:
        limit = self.max_hard_workers if high_risk else self.max_workers
        if self.workers + count > limit:
            raise BudgetExceeded("worker fan-out budget exhausted")
        self.workers += count

    def enter_depth(self, depth: int) -> None:
        if depth > self.max_depth:
            raise BudgetExceeded("graph-depth budget exhausted")
        self.depth = max(self.depth, depth)

    def consume_coe_repair(self) -> None:
        if self.coe_repairs >= self.max_coe_repairs:
            raise BudgetExceeded("CoE repair budget exhausted")
        self.coe_repairs += 1

@dataclass
class RetryController:
    max_leaf_retries: int = MAX_LEAF_RETRIES
    retries: dict[str, int] = field(default_factory=dict)
    fingerprints: dict[str, set[str]] = field(default_factory=dict)

    @staticmethod
    def fingerprint(leaf_id: str, task: object, repair_instruction: str) -> str:
        payload=json.dumps({"leaf":leaf_id,"task":task,"repair":repair_instruction},sort_keys=True,default=str,separators=(",",":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def allow(self, leaf_id: str, task: object, repair_instruction: str) -> bool:
        if not repair_instruction.strip(): return False
        if self.retries.get(leaf_id, 0) >= self.max_leaf_retries: return False
        fp=self.fingerprint(leaf_id,task,repair_instruction)
        seen=self.fingerprints.setdefault(leaf_id,set())
        if fp in seen: return False
        seen.add(fp); self.retries[leaf_id]=self.retries.get(leaf_id,0)+1
        return True

    def retry_targets(self, failed_leaf_id: str) -> list[str]: return [failed_leaf_id]
