"""Deterministic four-layer context packing with raw-artifact rejection."""
from dataclasses import dataclass
from pathlib import PurePosixPath
import hashlib, json

class ContextPackError(ValueError): pass
@dataclass(frozen=True)
class ContextPack:
    static_prefix: str
    dynamic_tail: str
    static_prefix_sha256: str
    dynamic_tail_sha256: str
    estimated_dynamic_tokens: int
    omitted_artifact_refs: list[str]
    rejection_reasons: list[str]

class ContextPacker:
    def __init__(self, max_dynamic_chars: int=96000, raw_artifact_chars: int=12000, max_worker_output_chars: int=12000):
        self.max_dynamic_chars=max_dynamic_chars; self.raw_artifact_chars=raw_artifact_chars; self.max_worker_output_chars=max_worker_output_chars
    @staticmethod
    def _stable(value): return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"))
    @staticmethod
    def _hash(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()
    @staticmethod
    def _allowed(ref: str, allowed: list[str]) -> bool:
        if not allowed: return True
        ref_norm=ref.replace("\\","/").lstrip("./")
        for candidate in allowed:
            c=candidate.replace("\\","/").lstrip("./")
            if ref_norm == c or ref_norm.startswith(c.rstrip("/") + "/"): return True
        return False
    def pack(self, *, policy, contracts, skills, tools, task: dict, state_delta: dict|None=None, receipts: list[dict]|None=None, excerpts: list[dict]|None=None, allow_full_source: bool=False) -> ContextPack:
        static="\n".join(["[LAYER 1: POLICY]\n"+self._stable(policy),"[LAYER 2: CONTRACTS]\n"+self._stable(contracts),"[LAYER 3: SKILL_ADAPTERS_AND_TOOLS]\n"+self._stable({"skills":skills,"tools":tools})])
        dynamic={"task":task,"state_delta":state_delta or {},"receipts":receipts or [],"excerpts":[]}
        allowed_paths=list(task.get("allowed_paths", [])) + list(task.get("scope", {}).get("allowed_paths", []))
        omitted=[]; reasons=[]
        for item in sorted(excerpts or [],key=lambda x:str(x.get("ref", ""))):
            text=str(item.get("text", "")); ref=str(item.get("ref", "")); kind=str(item.get("kind", "excerpt"))
            if not self._allowed(ref, allowed_paths):
                omitted.append(ref); reasons.append(f"reference outside allowed paths: {ref}"); continue
            permitted=bool(item.get("bounded",False)) or (allow_full_source and kind=="source")
            if kind in {"raw_log", "raw_ast"} and not item.get("bounded",False):
                omitted.append(ref); reasons.append(f"raw {kind} rejected: {ref}"); continue
            if len(text)>self.raw_artifact_chars and not permitted:
                omitted.append(ref); reasons.append(f"raw artifact exceeds limit: {ref}"); continue
            if len(text)>self.max_worker_output_chars and kind=="worker_output":
                omitted.append(ref); reasons.append(f"worker output exceeds limit: {ref}"); continue
            dynamic["excerpts"].append({"ref":ref,"sha256":self._hash(text),"text":text})
        tail="[LAYER 4: DYNAMIC TAIL]\n"+self._stable(dynamic)
        if len(tail)>self.max_dynamic_chars: raise ContextPackError("dynamic context budget exceeded")
        return ContextPack(static,tail,self._hash(static),self._hash(tail),max(1,(len(tail)+3)//4),omitted,reasons)
