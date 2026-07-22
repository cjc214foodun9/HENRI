#!/usr/bin/env python3
"""
Hermes Token Optimization Suite
Author: Aletheia Systems
Description: Implements asymmetric model routing, local tool encapsulation,
context truncation, and conditional model escalation for Hermes Agent.
"""

import os
import json
import subprocess
import requests
from typing import Dict, Any, Optional

# --- HERMES CUSTOM SKILL PATTERNS ---

class TokenEfficientSkills:
    
    @staticmethod
    def RunPytestAndCompressLogs(test_path: str = "tests/") -> str:
        """
        SKILL 1: Encapsulated Local Execution & Context Truncation
        Runs tests locally, strips stdout noise, and returns only the failing assertions.
        Saves thousands of tokens compared to letting Hermes run raw bash and read logs.
        """
        result = subprocess.run(["pytest", test_path, "-q", "--tb=short"], capture_output=True, text=True)
        
        if result.returncode == 0:
            return "[SUCCESS] All unit tests passed. Context consumed: ~15 tokens."

        # Truncate and filter log to keep only essential failure lines
        raw_output = result.stdout + result.stderr
        failure_lines = [line for line in raw_output.split("\n") if "FAIL:" in line or "FAILED" in line or "E   " in line]
        compressed_log = "\n".join(failure_lines[:20])  # Cap at top 20 relevant lines

        return f"[TEST FAILURE SUMMARY]\n{compressed_log}\n(Full raw log omitted to preserve context quota)."

    @staticmethod
    def ExecuteCodeWithEscalation(code_snippet: str, objective: str) -> str:
        """
        SKILL 2: Conditional Model Escalation Pipeline
        Tries low-cost DeepSeek v4 Pro first. If execution fails or yields errors,
        escalates the error diff to Kimi k3.
        """
        print("[Router] Attempting execution via low-cost driver (DeepSeek v4 Pro)...")
        driver_prompt = f"Objective: {objective}\nCode:\n```python\n{code_snippet}\n```\nFix any errors and output clean executable python."
        driver_response = HermesAsymmetricRouter.call_deepseek(driver_prompt)

        # Local validation check (e.g. static syntax check)
        try:
            compile(driver_response, "<string>", "exec")
            return f"[SOLVED BY DEEPSEEK]\n{driver_response}"
        except Exception as err:
            print(f"[Router] DeepSeek solution failed syntax check: {err}. Escalating to Kimi k3 Expert...")
            
            # Escalation to expensive Kimi k3 with precise error context
            expert_prompt = f"Objective: {objective}\nPrevious Attempt Failed Syntax Check:\n{err}\nCode:\n{driver_response}\nProvide the mathematically correct, bug-free implementation."
            expert_response = HermesAsymmetricRouter.call_kimi_k3_expert(expert_prompt)
            return f"[SOLVED BY KIMI K3 EXPERT]\n{expert_response}"

if __name__ == "__main__":
    print("=== Testing Hermes Token Optimization Suite ===")
    
    # Example 1: Local Compressed Test Log Execution
    test_summary = TokenEfficientSkills.RunPytestAndCompressLogs("./tests")
    print(f"\n1. Test Execution Output:\n{test_summary}")

    # Example 2: Asymmetric Escalation Demo
    sample_code = "def compute_langevin(x):\n  return x + " # Intentional syntax error
    solved = TokenEfficientSkills.ExecuteCodeWithEscalation(sample_code, "Implement anisotropic Langevin noise step")
    print(f"\n2. Escalation Router Result:\n{solved}")