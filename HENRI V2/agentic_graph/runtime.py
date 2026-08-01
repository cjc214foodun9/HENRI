"""Fail-closed state transitions for the HENRI graph control plane."""
from .budgets import GraphBudget, BudgetExceeded, RetryController
from .routing import can_promote_verified

TERMINAL={"complete","partial","blocked","failed","budget_exhausted"}
ALLOWED={"intake":{"classified","blocked"},"classified":{"collected","dispatched","blocked"},"collected":{"dispatched","blocked"},"dispatched":{"verified","repair","partial","blocked","failed","budget_exhausted"},"repair":{"dispatched","blocked","budget_exhausted"},"verified":{"complete","partial","blocked"},"complete":set(),"partial":set(),"blocked":set(),"failed":set(),"budget_exhausted":set()}
class GraphRuntimeError(RuntimeError): pass

class GraphRuntime:
    def __init__(self,budget=None,retries=None,task_id="runtime-task"):
        self.budget=budget or GraphBudget(); self.retries=retries or RetryController(); self.state="intake"; self.completed_leaves=set(); self.failed_node=None; self.task_id=task_id
    def _delta(self, old, new, reason, changed_refs=None, status=None, unmet_checks=None, next_safe_action=""):
        return {"schema_version":"1.0","task_id":self.task_id,"from_state":old,"to_state":new,"state":new,"status":status or (new if new in TERMINAL else "partial"),"reason":reason,"changed_refs":changed_refs or [],"completed_checks":sorted(self.completed_leaves),"unmet_checks":unmet_checks or [],"failed_node":self.failed_node,"retry_count":sum(self.retries.retries.values()),"next_safe_action":next_safe_action,"budget_remaining":{"model_calls":self.budget.max_model_calls-self.budget.model_calls,"leaf_retries":self.retries.max_leaf_retries-sum(self.retries.retries.values()),"coe_repairs":self.budget.max_coe_repairs-self.budget.coe_repairs}}
    def transition(self,new_state,reason="",changed_refs=None,unmet_checks=None,next_safe_action=""):
        old=self.state
        if new_state not in ALLOWED.get(old,set()): raise GraphRuntimeError(f"invalid transition {old}->{new_state}: {reason}")
        self.state=new_state
        return self._delta(old,new_state,reason,changed_refs,unmet_checks=unmet_checks,next_safe_action=next_safe_action)
    def promote(self,claim_status,receipts):
        if not can_promote_verified(claim_status,receipts): raise GraphRuntimeError("verified promotion requires passing deterministic receipts")
        return self.transition("verified","receipts passed",next_safe_action="complete after final policy check")
    def retry_leaf(self,leaf_id,task,repair):
        if self.state not in {"dispatched","repair"}: raise GraphRuntimeError("leaf retry requires dispatched or repair state")
        if not self.retries.allow(leaf_id,task,repair): raise BudgetExceeded("leaf retry denied")
        self.failed_node=leaf_id; old=self.state; self.state="repair"
        return self._delta(old,"repair",f"repair failed leaf {leaf_id}",changed_refs=[leaf_id],unmet_checks=[repair],next_safe_action="dispatch failed leaf only")
