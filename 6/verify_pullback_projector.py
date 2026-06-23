"""  
Project HENRI: System Verification and Validation Suite  
Target: Phase 5 Pullback Repair Head and Dual-Block Constraints  
"""

import sys
import os

# Add paths to enable correct package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "henri_core"))

from henri_core.pullback_projector import run_phase_5_validation

if __name__ == "__main__":  
    run_phase_5_validation()
