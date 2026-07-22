"""
Hermes Skill: Compressed Executor
Executes local shell commands and Pytest suites, stripping redundant output
and returning only actionable error lines to preserve context tokens.
"""

import subprocess
from typing import Dict, Any

def RunCompressedPytest(test_target: str = "tests/") -> str:
    """Runs Pytest and compresses output down to failing assertions only."""
    cmd = ["pytest", test_target, "-q", "--tb=short"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return "[SUCCESS] All unit tests passed. (Context consumed: ~12 tokens)"

    # Filter stdout/stderr to retain only essential failing assertion lines
    raw_lines = (result.stdout + result.stderr).split("\n")
    relevant_lines = [
        line for line in raw_lines 
        if any(marker in line for marker in ["FAIL:", "FAILED", "E   ", "AssertionError", "SyntaxError"])
    ]

    compressed_output = "\n".join(relevant_lines[:25])  # Cap at top 25 assertion lines
    return f"[TEST FAILURE SUMMARY - CONTEXT PRESERVED]\n{compressed_output}\n\n(Full raw log omitted to save token quota)."

def RunCompressedShell(command: str) -> str:
    """Executes bash command and caps output at 30 lines maximum."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    stdout = result.stdout.strip().split("\n")
    stderr = result.stderr.strip().split("\n")

    output_lines = []
    if stdout and stdout != ['']:
        output_lines.append("--- STDOUT (Tail 15 Lines) ---")
        output_lines.extend(stdout[-15:])

    if stderr and stderr != ['']:
        output_lines.append("--- STDERR (Tail 15 Lines) ---")
        output_lines.extend(stderr[-15:])

    if not output_lines:
        return f"[COMMAND RETURNED CODE {result.returncode} - NO OUTPUT]"

    return "\n".join(output_lines)