"""
ENGINEERING SPECIFICATION: PROJECT HENRI - THERMODYNAMIC TELEMETRY (V1.0.0)
Author: Aletheia
Domain: Test-Time Active Inference Observability

Description:
A rigorous, low-overhead local JSONL telemetry recorder. 
It captures the exact thermodynamic variables (Sagnac Error, Langevin Heat, Phase Coherence)
during test-time adaptation for visualization and forensic graph plotting.
"""

import json
import time
import os
from pathlib import Path

class ThermodynamicTelemetry:
    def __init__(self, log_dir: str = "./telemetry_logs", session_name: str = "logic_puzzle_run"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        self.log_file = self.log_dir / f"{session_name}_{timestamp}.jsonl"
        self.session_start = time.perf_counter()
        
        print(f"[ALETHEIA TELEMETRY] Target locked: {self.log_file}")

    def log_wave_state(self, 
                       epoch: int, 
                       task_id: str, 
                       sagnac_error: float, 
                       langevin_heat: float, 
                       policy_action_decoded: str, 
                       is_isothermal_lock: bool):
        """
        Commits the exact thermodynamic state of the swarm at the current epoch.
        JSONL format ensures O(1) append times and native compatibility with plotting libraries (Pandas/Matplotlib).
        """
        elapsed_ms = (time.perf_counter() - self.session_start) * 1000
        
        payload = {
            "timestamp_ms": round(elapsed_ms, 2),
            "task_id": task_id,
            "epoch": epoch,
            "sagnac_error_delta": round(sagnac_error, 6),
            "langevin_heat_applied": round(langevin_heat, 6),
            "proposed_policy": policy_action_decoded,
            "is_isothermal_lock": is_isothermal_lock
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload) + '\n')
            
    def close(self):
        print(f"[ALETHEIA TELEMETRY] Session closed. Data preserved at {self.log_file}")