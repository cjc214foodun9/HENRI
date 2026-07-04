import ast
import math
import time
import numpy as np
import torch
from typing import List, Tuple, Dict, Any

# Dynamic import configuration to support standalone testing
try:
    from vsa_transducer import HenriASTTransducer
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from vsa_transducer import HenriASTTransducer

class HenriTelemetryHarness:
    """
    Substrate-independent telemetry harness that computes:
    1. AST-Topo (Abstract Syntax Tree Topology Preservation & Phase Linewidth Retention)
    2. Sagnac Relaxation Velocity (V_sr)
    3. Epiplexity Density per Joule (E_J)
    """
    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.transducer = HenriASTTransducer(cores=16, channels=dim // 16)

    def calculate_ast_topo(self, generated_code: str, canonical_code: str) -> Dict[str, Any]:
        """
        Computes Phase Linewidth Retention (Gamma_phi) between the generated and
        canonical AST vectors in complex space.
        """
        try:
            # Generate target waves (quantized phase-locked hypervectors)
            gen_wave = self.transducer.generate_psi_target(generated_code).flatten()
            can_wave = self.transducer.generate_psi_target(canonical_code).flatten()
            
            # Calculate phase angle difference between the two waves
            # Theta = angle(W_gen * conj(W_can))
            phase_diff = torch.angle(gen_wave * torch.conj(can_wave))
            
            # Gamma_phi is the phase variance
            gamma_phi = torch.var(phase_diff).item()
            status = "PASSED" if gamma_phi < 1.5 else "DEGRADED"
            
            return {
                "gamma_phi": gamma_phi,
                "status": status,
                "error": None
            }
        except Exception as e:
            return {
                "gamma_phi": 999.0,
                "status": "FAILED",
                "error": str(e)
            }

    def calculate_sagnac_velocity(self, sagnac_deltas: List[float], tau_steps: int) -> float:
        r"""
        Calculates the Sagnac Relaxation Velocity (V_sr) representing the rate of
        Sagnac error delta decay: V_sr = \partial \Delta / \partial \tau
        """
        if len(sagnac_deltas) < 2 or tau_steps <= 0:
            return 0.0
            
        # Compute discrete differences: Delta_{i} - Delta_{i+1}
        diffs = []
        for i in range(len(sagnac_deltas) - 1):
            diffs.append(sagnac_deltas[i] - sagnac_deltas[i+1])
            
        # V_sr is the average rate of decay per step
        v_sr = float(np.mean(diffs))
        return v_sr

    def calculate_epiplexity_density(self, 
                                     code_str: str, 
                                     runtime_sec: float, 
                                     simulated_wattage: float = 1.5) -> Dict[str, float]:
        """
        Computes Epiplexity Density per Joule (E_J) using character-level Shannon entropy
        and roughness metrics as a proxy for structural density.
        """
        from collections import Counter
        epsilon = 1e-9
        
        # 1. Entropy (Complexity)
        counts = Counter(code_str)
        total = len(code_str)
        if total == 0:
            entropy = 0.0
        else:
            entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
            
        # 2. Roughness (Total Variation of Ordinal values)
        ordinals = [ord(c) for c in code_str]
        if len(ordinals) < 2:
            roughness = 0.0
        else:
            roughness = sum(abs(ordinals[i] - ordinals[i-1]) for i in range(len(ordinals)-1)) / (len(ordinals) - 1)
            
        # Birkhoff measure estimate: 1.0 / (entropy + 0.01 * roughness + epsilon)
        epiplexity = 1.0 / (entropy + 0.01 * roughness + epsilon)
        
        # Joules Expended = runtime_sec * wattage
        joules_expended = max(runtime_sec * simulated_wattage, 1e-6)
        
        # Epiplexity density
        e_j = epiplexity / joules_expended
        
        return {
            "epiplexity": epiplexity,
            "joules_expended": joules_expended,
            "e_j": e_j
        }

if __name__ == "__main__":
    print("[TEST] Initializing Telemetry Harness verification...")
    harness = HenriTelemetryHarness(dim=4096)
    
    # 1. Test AST-Topo Calculation
    canonical = "def transform(grid):\n    return np.rot90(grid)"
    generated_good = "def transform(grid):\n    # Rotate\n    return np.rot90(grid)"
    generated_broken = "def transform(grid):\n    x = 1\n    y = 2\n    z = 3\n    return grid"
    
    res_good = harness.calculate_ast_topo(generated_good, canonical)
    res_broken = harness.calculate_ast_topo(generated_broken, canonical)
    
    print(f"  Good AST Gamma_phi: {res_good['gamma_phi']:.4f} (Status: {res_good['status']})")
    print(f"  Broken AST Gamma_phi: {res_broken['gamma_phi']:.4f} (Status: {res_broken['status']})")
    assert res_good['gamma_phi'] < res_broken['gamma_phi'], "Symmetry metric failure: good code should have lower phase variance!"
    
    # 2. Test Sagnac Velocity
    deltas = [1.0, 0.8, 0.5, 0.2, 0.05]
    v_sr = harness.calculate_sagnac_velocity(deltas, len(deltas))
    print(f"  Sagnac Relaxation Velocity: {v_sr:.4f}")
    assert v_sr > 0, "Decay velocity should be positive"
    
    # 3. Test Epiplexity Density per Joule
    metrics = harness.calculate_epiplexity_density(generated_good, runtime_sec=0.5, simulated_wattage=1.5)
    print(f"  Epiplexity: {metrics['epiplexity']:.4f}")
    print(f"  Joules Expended: {metrics['joules_expended']:.4f} J")
    print(f"  E_J: {metrics['e_j']:.4f} Epiplexity/J")
    assert metrics['e_j'] > 0, "E_J score must be positive"
    
    print("[TEST] All telemetry harness validations PASSED!")
