import numpy as np
import torch
import torch.nn.functional as F


class EntropicSurvivalEngine:
    def __init__(self, num_experts=16, wave_dim=4096, survival_threshold=0.35):
        self.num_experts = num_experts
        self.wave_dim = wave_dim
        # The minimum resonance required to escape the death penalty
        self.survival_threshold = survival_threshold 

    @torch.no_grad()
    def evaluate_entropic_fitness(self, expert_waves, zone_c_attractors, zone_c_repellers):
        """
        Passes the 16 expert waves through the Zone B emulator simulation to find
        the closest vector to the stable boundary axioms.
        """
        def to_real(t):
            if t is None: return None
            if torch.is_complex(t):
                return torch.cat([t.real, t.imag], dim=-1).to(torch.float32)
            if t.shape[-1] == 4096:
                zeros = torch.zeros_like(t)
                return torch.cat([t, zeros], dim=-1).to(torch.float32)
            return t.to(torch.float32)
            
        z_attr = to_real(zone_c_attractors)
        z_rep = to_real(zone_c_repellers)
        
        num_candidates = len(expert_waves)
        fitness_scores = torch.zeros(num_candidates, device='cpu')
        
        for i in range(num_candidates):
            if i >= len(expert_waves):
                break
            wave = expert_waves[i].unsqueeze(0) # Shape: [1, 4096]
            wave = to_real(wave)
            
            # 1. Calculate constructive interference with stable axioms (Reward)
            if z_attr is not None and len(z_attr) > 0:
                attractor_resonance = F.cosine_similarity(wave, z_attr)
                max_attraction = torch.max(attractor_resonance)
            else:
                max_attraction = 0.0
            
            # 2. Calculate destructive interference with forbidden axioms (Penalty)
            if z_rep is not None and len(z_rep) > 0:
                repeller_resonance = F.cosine_similarity(wave, z_rep)
                max_repulsion = torch.max(repeller_resonance)
            else:
                max_repulsion = 0.0
                
            # The Entropic Fitness: High attraction, low repulsion
            fitness_scores[i] = max_attraction - (1.5 * max_repulsion)
            
        return fitness_scores

    @torch.no_grad()
    def apply_viscoelastic_apoptosis(self, manager, fitness):
        """
        Applies dynamic Langevin damping (Viscous Drag) to an expert matrix during generation.
        """
        decay_severity = 1.0 - max(fitness, 0.0) 
        death_penalty = 0.80 * decay_severity # Rapid decay
        manager.lora_A.data *= (1.0 - death_penalty)
        manager.lora_B.data *= (1.0 - death_penalty)
        print(f"[ZONE B] DAMPING APPLIED: Resonance {fitness:.4f} < Threshold. Matrix dragged by {death_penalty*100:.1f}%.")

    @torch.no_grad()
    def execute_survival_creep(self, lora_managers, fitness_scores, residual_wave, learning_rate=0.05):
        """
        The Survivor's Reward. Applies synaptic growth to the most efficient physical solution.
        """
        if residual_wave is None:
            return
            
        if isinstance(residual_wave, np.ndarray):
            residual_wave = torch.tensor(residual_wave)
            
        if torch.is_complex(residual_wave):
            residual_wave = torch.cat([residual_wave.real.flatten(), residual_wave.imag.flatten()], dim=-1).to(torch.float32)
        else:
            residual_wave = residual_wave.flatten().to(torch.float32)


            
        # Sort experts by their entropic fitness
        ranked_indices = torch.argsort(fitness_scores, descending=True)
        alpha_winner = ranked_indices[0]
        
        if isinstance(lora_managers, dict):
            lora_items = list(lora_managers.items())
        else:
            lora_items = list(enumerate(lora_managers))
            
        for i, manager in lora_items:
            if i >= len(fitness_scores):
                break
                
            fitness = fitness_scores[i].item()
            
            if i == alpha_winner and fitness > 0:
                # [SURVIVAL REWARD] The closest vector.
                # It efficiently navigated the complexity. Reward it with synaptic growth
                # by pulling its weights toward the orthogonal residual.
                
                # Align residual wave dimension to manager's gemma_dim if needed
                res_tensor = residual_wave.clone().to(device=manager.lora_A.device, dtype=manager.lora_A.dtype)
                if len(res_tensor) < manager.gemma_dim:
                    res_tensor = F.pad(res_tensor, (0, manager.gemma_dim - len(res_tensor)))
                elif len(res_tensor) > manager.gemma_dim:
                    res_tensor = res_tensor[:manager.gemma_dim]
                    
                update_A = torch.outer(res_tensor, res_tensor[:manager.rank])
                update_A = F.normalize(update_A, p=2, dim=1)
                
                growth_step = learning_rate * fitness * update_A
                manager.lora_A.data += growth_step
                print(f"[ZONE B] Expert {i} SURVIVED. Resonance: {fitness:.4f}. Synaptic growth applied.")
                
            else:
                # [MARGINAL CREEP]
                # Gradual viscoelastic relaxation.
                manager.lora_A.data *= 0.98
