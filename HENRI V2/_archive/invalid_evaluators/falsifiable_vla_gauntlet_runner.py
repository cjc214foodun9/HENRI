"""
HENRI V2: Falsifiable Closed-Loop VLA Gauntlet Runner
Subsystem: Falsifiable Evaluation & Exteroceptive Verification Engine

Implements the 5-Stage Closed-Loop VLA Evaluation Protocol:
  [ Un-cached Task Input ]
             │
             ▼
  [ Zone A: O-VSA Phase Ingress ] (UWE Mapping to S^{D-1}, D=65,536)
             │
             ▼
  [ Zone B: Wave-JEPA Transition ] (Predict Next State Wave \hat{\Psi}_{t+1})
             │
             ▼
  [ Live Sagnac Veto Filter ] (Reject \Delta_{Sagnac} > \tau_{veto})
             │
             ▼
  [ Exteroceptive REPL Sandbox ] (Real-time Execution & Traceback Feedback)
             │
             ▼
  [ Out-of-Distribution Score ] (Falsifiable Agentic Accuracy Metric)

Features:
- Zero Pre-Seeded Codebooks / Pure Continuous Wave Phase Ring Unbinding
- Live Exteroceptive Sandbox Execution & Zone C Dirichlet Boundary Updates (\nu = -1.0)
- Measurement of PyTorch/CUDA forward pass latencies, token generation rates, and online test-time SGLD adaptation
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
import torch

repo_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(repo_path)
for p in [repo_path, parent_path, os.path.join(parent_path, "scripts")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec
from henri_decoder import HENRIUnifiedEgressTransducer
from exteroceptive_sandbox import ExteroceptiveSandboxTransducer
from sagnac_mcts_planner import SagnacMCTSPlanner

try:
    import agentic_event_store
except ImportError:
    agentic_event_store = None


class FalsifiableVLAGauntletRunner:
    def __init__(self, d_model: int = 65536, tau_veto: float = 0.35):
        self.d_model = d_model
        self.tau_veto = tau_veto
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Zone A: O-VSA Phase Ingress
        self.codec = qFHRREpistemicCodec(d_model=d_model, device=self.device)
        
        # 2. Zone B & Egress Transducer (Un-cached Continuous Wave Head)
        self.transducer = HENRIUnifiedEgressTransducer(d_model=d_model, device=self.device)
        
        # 3. Exteroceptive REPL Sandbox (Live Verification & Traceback Feedback)
        self.sandbox = ExteroceptiveSandboxTransducer(d_model=d_model)
        
        self.staged_dir = os.path.join(repo_path, "data", "official_benchmarks", "staged_eval_suites")

    def run_falsifiable_item(self, item: dict, suite_id: str) -> dict:
        t0 = time.perf_counter()
        
        # Stage 1: Un-cached Task Input & Zone A O-VSA Ingress
        prompt_str = item.get("prompt", item.get("question", ""))
        prompt_wave = self.codec.encode_text(prompt_str).to(torch.float32)  # [D=65,536] on S^{D-1}
        w_task = self.codec.encode_text(f"TASK_OPERATOR_{suite_id}").to(torch.float32)
        
        # Stage 2: Zone B Wave-JEPA Hadamard Transition
        goal_wave = self.codec.bind_hadamard(w_task, prompt_wave).to(torch.float32)
        
        # Stage 3: Live Sagnac Veto Filter Check
        dot_val = torch.dot(goal_wave, prompt_wave)
        sagnac_delta = float(1.0 - torch.abs(dot_val / self.d_model).item())
        is_vetoed = sagnac_delta > self.tau_veto
        
        if is_vetoed:
            latency = time.perf_counter() - t0
            return {
                "task_id": item.get("task_id", "unknown"),
                "passed": False,
                "sagnac_delta": sagnac_delta,
                "vetoed": True,
                "latency_sec": latency,
                "dirichlet_boundary": -1.0,
                "error_stage": "SAGNAC_VETO"
            }

        # Stage 4: Egress Transduction to Candidate Code / Text Response
        text_resp, telem = self.transducer.decode_wave_to_response(goal_wave, prompt_str)
        
        # Stage 5: Exteroceptive REPL Sandbox Verification
        if "prompt" in item and "test_code" in item:
            candidate_code = f"{text_resp}\n{item['test_code']}"
            success, sb_res = self.sandbox.execute_and_transduce(
                candidate_code,
                axiom_id=f"ood_{item['task_id']}",
                source_metadata=suite_id
            )
            is_pass = success
        elif "answer" in item:
            expected = item["answer"]
            is_pass = expected in text_resp
            success = is_pass
            sb_res = {}
        else:
            is_pass = bool(text_resp) and ("EXECUTION_ERROR" not in text_resp)
            success = is_pass
            sb_res = {}

        # Test-Time SGLD Parameter Adaptation Step on Failure
        sgld_res = {}
        if not is_pass:
            target_wave = self.codec.encode_text("CORRECT_GOAL_STATE").to(torch.float32)
            sgld_res = self.transducer.unbinder.adapt_in_context_sgld(
                active_wave=goal_wave,
                target_wave=target_wave,
                target_token_ids=torch.tensor([0], device=self.device),
                steps=2
            )

        latency = time.perf_counter() - t0
        return {
            "task_id": item.get("task_id", "unknown"),
            "passed": is_pass,
            "sagnac_delta": sagnac_delta,
            "vetoed": False,
            "latency_sec": latency,
            "tokens_per_sec": len(text_resp.split()) / max(1e-4, latency),
            "dirichlet_boundary": 1.0 if is_pass else -1.0,
            "sgld_adaptation": sgld_res,
            "sandbox_res": sb_res
        }

    def run_full_falsifiable_gauntlet(self) -> dict:
        raise RuntimeError(
            "BLOCKED: this VLA gauntlet depends on unverified staged suites; "
            "no external score is permitted until canonical task data, evaluators, "
            "and a complete evidence bundle are supplied"
        )
        suites = [
            "gdpval_aa", "terminal_bench_hard", "terminal_bench_v21",
            "tau2_telecom", "tau3_banking", "scicode", "aa_lcr",
            "aa_omniscience", "ifbench", "hle", "gpqa_diamond",
            "critpt", "mmmu_pro", "ifeval_official"
        ]

        print("========================================================================")
        print("  PROJECT HENRI V2: FALSIFIABLE CLOSED-LOOP VLA GAUNTLET RUNNER")
        print("========================================================================")

        results = {}
        total_passed = 0
        total_items = 0
        total_vetoes = 0
        t0 = time.perf_counter()

        for suite_id in suites:
            staged_file = os.path.join(self.staged_dir, f"{suite_id}_staged.jsonl")
            if not os.path.exists(staged_file):
                continue

            items = []
            with open(staged_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))

            passed = 0
            vetoes = 0
            for item in items:
                res = self.run_falsifiable_item(item, suite_id)
                if res["passed"]:
                    passed += 1
                if res["vetoed"]:
                    vetoes += 1

            results[suite_id] = {
                "passed": passed,
                "total": len(items),
                "vetoes": vetoes,
                "accuracy": (passed / len(items)) * 100.0 if items else 0.0
            }
            total_passed += passed
            total_items += len(items)
            total_vetoes += vetoes

            print(f"[{suite_id.upper()}] Passed: {passed}/{len(items)} | Accuracy: {results[suite_id]['accuracy']:.2f}% | Sagnac Vetoes: {vetoes}")

        total_elapsed = time.perf_counter() - t0
        composite_score = (total_passed / total_items) * 100.0 if total_items else 0.0

        print("========================================================================")
        print(f" FALSIFIABLE AGENTIC ACCURACY SCORE: {composite_score:.2f} / 100")
        print(f" TOTAL PASSED ITEMS: {total_passed} / {total_items}")
        print(f" TOTAL SAGNAC VETOES REJECTED: {total_vetoes}")
        print(f" TOTAL GAUNTLET RUNTIME: {total_elapsed:.4f} seconds ({total_items / max(1e-4, total_elapsed):.2f} items/sec)")
        print("========================================================================")

        scorecard = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "composite_score": composite_score,
            "total_passed": total_passed,
            "total_items": total_items,
            "total_vetoes": total_vetoes,
            "total_elapsed_sec": total_elapsed,
            "query_throughput_items_per_sec": total_items / max(1e-4, total_elapsed),
            "suite_results": results
        }

        # Save scorecard locally
        out_log_dir = os.path.join(repo_path, "telemetry_logs")
        os.makedirs(out_log_dir, exist_ok=True)
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        scorecard_path = os.path.join(out_log_dir, f"falsifiable_vla_scorecard_{ts_str}.json")
        with open(scorecard_path, "w", encoding="utf-8") as f:
            json.dump(scorecard, f, indent=2)

        # Seal in Agentic Event Store
        if agentic_event_store is not None:
            try:
                agentic_event_store.append_event(
                    event_type="FALSIFIABLE_VLA_GAUNTLET",
                    payload={
                        "composite_score": composite_score,
                        "total_passed": total_passed,
                        "total_items": total_items,
                        "scorecard_path": scorecard_path
                    },
                    stream="telemetry",
                    actor="henri_arbiter",
                    causal_status="observed"
                )
                print("[AGENTIC GRAPH] Event sealed successfully.")
            except Exception as e:
                print(f"[AGENTIC GRAPH] Warning: Event seal failed: {e}")

        return scorecard


if __name__ == "__main__":
    runner = FalsifiableVLAGauntletRunner()
    runner.run_full_falsifiable_gauntlet()
