"""  
Project HENRI: System Verification and Validation Suite  
Target: Phase 3 Chronological H-MPC Latent Steering Loop  
"""

import sys
import os

# Add paths to enable correct package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "henri_core"))

from henri_core.h_mpc_steering import run_phase_3_validation

if __name__ == "__main__":  
    run_phase_3_validation()
