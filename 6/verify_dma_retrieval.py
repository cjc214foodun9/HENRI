"""  
Project HENRI: System Verification and Validation Suite  
Target: Phase 1 Holographic Associative DMA Lookup Interface  
"""

import sys
import os

# Add paths to enable correct package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "henri_core"))

from henri_core.retrieval_core import run_clean_room_validation

if __name__ == "__main__":  
    run_clean_room_validation()
