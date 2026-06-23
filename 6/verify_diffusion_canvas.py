"""  
Project HENRI: System Verification and Validation Suite  
Target: Phase 4 Non-Autoregressive Diffusion Canvas Relaxation  
"""

import sys
import os

# Add paths to enable correct package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "henri_core"))

from henri_core.diffusion_canvas import run_phase_4_validation

if __name__ == "__main__":  
    run_phase_4_validation()
