import torch  
import torch.nn as nn  
import torch.nn.functional as F

class HenriDynamicGearShifter(nn.Module):  
    def __init__(self, full_dimensions: int = 4096, max_experts: int = 16):  
        super().__init__()  
        self.full_dimensions = full_dimensions  
        self.max_experts = max_experts  
          
        # Track historical stress trends to apply smooth transition buffers  
        self.register_buffer("stress_momentum", torch.tensor(0.5, dtype=torch.float32))  
        self.momentum_coefficient = 0.85

    def shift_transmission_gears(self, latest_stress: float, latest_sigreg: float,   
                                  available_vram_gb: float = 24.0) -> dict:  
        """  
        Computes the target hardware parameter scaling matrix using Kuramoto R.
        """  
        import math
        # Calculate mathematically exact Kuramoto Order Parameter R from potential v (latest_stress)
        R = math.sqrt(max(0.0, 1.0 - latest_stress))
        
        # Smooth out R instead of stress using exponential moving average  
        self.stress_momentum = (self.momentum_coefficient * self.stress_momentum) + ((1.0 - self.momentum_coefficient) * R)  
        smoothed_R = self.stress_momentum.item()  
          
        # Invariant safety override: If VRAM headroom is critical, force Gear 1 contraction  
        if available_vram_gb < 4.0:  
            print("[GEAR SHIFTER] Critical memory threshold reached. Forcing Gear 1 contraction.")  
            return self._compile_gear_payload(gear=1, R=smoothed_R, horizon=2, history=2, active_experts=4, context_size=1024)
            
        # R -> 1 (High sync): Gear 1 (compress representation, low cache overhead)
        # R -> 0.5 (Moderate sync): Gear 2
        # R -> 0 (High entropy dispersion): Gear 3 (expand degrees of freedom)
        if smoothed_R >= 0.75 and latest_sigreg < 2.5:  
            return self._compile_gear_payload(gear=1, R=smoothed_R, horizon=3, history=2, active_experts=4, context_size=2048)  
              
        elif smoothed_R >= 0.35 and smoothed_R < 0.75:  
            return self._compile_gear_payload(gear=2, R=smoothed_R, horizon=6, history=4, active_experts=8, context_size=4096)  
              
        else:  
            target_horizon = 12 if available_vram_gb < 16.0 else 16  
            target_context = 8192 if available_vram_gb < 16.0 else 16384  
            return self._compile_gear_payload(gear=3, R=smoothed_R, horizon=target_horizon, history=8, active_experts=self.max_experts, context_size=target_context)

    def _compile_gear_payload(self, gear: int, R: float, horizon: int, history: int,   
                               active_experts: int, context_size: int) -> dict:  
        return {  
            "current_transmission_gear": gear,  
            "R_macro": R,
            "h_mpc_horizon_steps": horizon,  
            "jepa_history_truncation": history,  
            "active_expert_count": active_experts,  
            "allocated_context_window": context_size,  
            "diffusion_guidance_scale": 1.5 + (0.5 * gear) # Intensify core focus in higher gears  
        }

class AdaptiveSwarmOrchestratorBridge:  
    """  
    Integration Connector: Dynamically slices token sequences and weights  
    mid-flight to match the active gear targets.  
    """  
    def __init__(self, orchestrator_reference):  
        self.orchestrator = orchestrator_reference  
        self.shifter = HenriDynamicGearShifter(max_experts=orchestrator_reference.num_streams)

    def execute_synchronized_gear_shift(self, current_telemetry: dict):  
        """  
        Modulates the active serving configurations inside the running distillation loop.  
        """  
        stress = current_telemetry.get("thermodynamic_stress_cost", 0.5)  
        sigreg = current_telemetry.get("sigreg_disentanglement_score", 3.0)  
          
        # Calculate optimal parameter scales  
        gear_payload = self.shifter.shift_transmission_gears(stress, sigreg)  
          
        print(f"\n[TRANSMISSION] Shifted to Gear {gear_payload['current_transmission_gear']} // Coherence R: {gear_payload['R_macro']:.4f} // Adjusting Compute Matrix.")  
        print(f"  - Active Context Horizon:   {gear_payload['allocated_context_window']} tokens")  
        print(f"  - Forward Lookahead Depth: {gear_payload['h_mpc_horizon_steps']} steps")  
        print(f"  - Functor Expert Slices:    {gear_payload['active_expert_count']} threads")

        # 1. Update Lookahead Truncation Limits (Bypasses deep loops on simple targets)  
        self.orchestrator.h_mpc_horizon = gear_payload["h_mpc_horizon_steps"]  
        if hasattr(self.orchestrator, "harness") and hasattr(self.orchestrator.harness, "sandbox"):  
            # Update the inner JEPA sandbox depth parameter on the fly  
            self.orchestrator.harness.sandbox.trajectory_truncation = gear_payload["jepa_history_truncation"]

        # 2. Dynamically adjust active expert allocations  
        self.orchestrator.current_active_experts = gear_payload["active_expert_count"]  
          
        # 3. Apply memory-mapped page cache truncation bounds to prevent OOM  
        self.orchestrator.max_context_len = gear_payload["allocated_context_window"]  
          
        # 4. Modulate output crystallization relaxation focus  
        if hasattr(self.orchestrator, "canvas_sampler"):  
            self.orchestrator.canvas_sampler.guidance_scale = gear_payload["diffusion_guidance_scale"]
