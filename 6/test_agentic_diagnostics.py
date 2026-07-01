import os
import sys
import torch
import pytest

# Insert current file's parent folders to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_telemetry_harness import HenriTelemetryHarness

def test_logical_ast_quality():
    print("\n[DIAGNOSTICS] Verifying AST Linewidth Phase Retention (Gamma_phi)...")
    harness = HenriTelemetryHarness(dim=4096)
    
    # Test valid syntax layout mapping vs completely orthogonal noise
    canonical = "def transform(grid):\n    return np.rot90(grid)"
    generated_good = "def transform(grid):\n    # Rotate\n    return np.rot90(grid)"
    generated_broken = "def transform(grid):\n    x = 1\n    return grid"
    
    res_good = harness.calculate_ast_topo(generated_good, canonical)
    res_broken = harness.calculate_ast_topo(generated_broken, canonical)
    
    print(f" -> Good AST Variance: {res_good['gamma_phi']:.4f}")
    print(f" -> Broken AST Variance: {res_broken['gamma_phi']:.4f}")
    
    # Programmatic CI/CD gating: verify lower phase variance for accurate translations
    assert res_good['gamma_phi'] < res_broken['gamma_phi'], "AST phase variance ordering failed!"
    assert res_good['gamma_phi'] < 1.5, f"Structural drift exceeds tolerance: {res_good['gamma_phi']}"

def test_sagnac_velocity_decay():
    print("\n[DIAGNOSTICS] Verifying Sagnac relaxation velocity decay...")
    harness = HenriTelemetryHarness(dim=4096)
    
    # Mocking Sagnac error delta progression over time steps
    deltas = [1.0, 0.8, 0.5, 0.2, 0.05]
    v_sr = harness.calculate_sagnac_velocity(deltas, len(deltas))
    print(f" -> Sagnac Relaxation Decay Velocity (V_sr): {v_sr:.4f}")
    
    assert v_sr > 0.0, "Decay velocity should track positive stabilization!"

def test_repl_compilation_safety():
    print("\n[DIAGNOSTICS] Verifying REPL compilation safety...")
    # Safe syntax validation gate
    valid_code = "def process():\n    return 'OK'"
    invalid_code = "def process():\n    return 'OK' °" # Has invalid character
    
    # Simple compilation checks as performed inside sandbox
    try:
        compile(valid_code, "<string>", "exec")
        compiled_valid = True
    except SyntaxError:
        compiled_valid = False
        
    try:
        compile(invalid_code, "<string>", "exec")
        compiled_invalid = True
    except SyntaxError:
        compiled_invalid = False
        
    assert compiled_valid, "Valid code compilation failed!"
    assert not compiled_invalid, "Invalid character was not vetoed by the compiler!"
    print(" -> [PASS] Syntax parser successfully vetoed code anomalies.")
