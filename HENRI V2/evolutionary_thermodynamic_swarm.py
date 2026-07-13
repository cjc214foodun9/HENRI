import torch
import torch.fft as fft
import torch.nn.functional as F
import hashlib

class SwarmExpert:
    """
    A single biological/computational cell in the HENRI Swarm.
    Inspired by GENREG V4: Each expert self-organizes its own reproductive 
    and mutational strategy (its Langevin heat tolerance).
    """
    def __init__(self, dim=4096, device=None, parent_a=None, parent_b=None):
        self.dim = dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if parent_a is not None and parent_b is not None:
            # GENREG V5: Neuron-Level (Phase-Level) Crossover
            # Randomly inherit phase angles from parent A or B
            mask = torch.rand(self.dim, device=self.device) > 0.5
            self.wave = torch.where(mask, parent_a.wave, parent_b.wave)
            
            # Inherit and slightly mutate evolutionary strategies
            self.mutation_rate = parent_a.mutation_rate + (torch.randn(1).item() * 0.05)
            self.mutation_scale = parent_b.mutation_scale + (torch.randn(1).item() * 0.05)
        else:
            # Genesis generation
            phases = (torch.rand(self.dim, device=self.device) * 2 * torch.pi) - torch.pi
            self.wave = torch.polar(torch.ones(self.dim, device=self.device), phases)
            self.mutation_rate = torch.rand(1).item()   # Probability of mutation
            self.mutation_scale = torch.rand(1).item()  # Magnitude of Langevin heat
            
        # Ensure bounds
        self.mutation_rate = max(0.01, min(1.0, self.mutation_rate))
        self.mutation_scale = max(0.01, min(2.0, self.mutation_scale))
        self.fitness = 0.0

    def apply_gradient_free_mutation(self):
        """
        Replaces Viscoelastic Creep gradients with Darwinian Thermal Noise.
        Experts with high mutation_scale make "fewer but bigger bets".
        """
        if torch.rand(1).item() < self.mutation_rate:
            thermal_noise = torch.randn_like(self.wave.real) + 1j * torch.randn_like(self.wave.imag)
            # Inject heat based on the expert's own self-organized strategy
            self.wave = self.wave + (thermal_noise * self.mutation_scale)
            # Re-project to the Unitary hypersphere (energy conservation)
            self.wave = self.wave / (torch.abs(self.wave) + 1e-9)

class GradientFreeThermostat:
    """
    The Evolution Engine. Replaces autograd with Trust-Based Selection.
    """
    def __init__(self, swarm_size=16, dim=4096):
        self.swarm_size = swarm_size
        self.dim = dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the population
        self.population = [SwarmExpert(dim, self.device) for _ in range(swarm_size)]
        
    def _calculate_sagnac_fitness(self, wave: torch.Tensor, target_axiom: torch.Tensor) -> float:
        """Fitness is defined as the phase resonance with the Zone C Axiom."""
        resonance = torch.real(torch.sum(wave.conj() * target_axiom)).item() / self.dim
        return resonance

    def evolutionary_epoch(self, target_axiom: torch.Tensor) -> SwarmExpert:
        """
        Executes one complete generation of Gradient-Free Evolution.
        """
        # 1. Evaluate Fitness (Sagnac Resonance)
        for expert in self.population:
            expert.fitness = self._calculate_sagnac_fitness(expert.wave, target_axiom)
            
        # 2. Sort population by fitness (highest first)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        best_expert = self.population[0]
        
        # 3. Trust-Based Selection (Elitism)
        # Keep the top 25% of the population untouched
        elite_count = max(2, self.swarm_size // 4)
        elites = self.population[:elite_count]
        
        new_population = list(elites)
        
        # 4. Phase-Level Crossover and Reproduction
        # Fill the rest of the population by breeding the elites
        while len(new_population) < self.swarm_size:
            parent_a = elites[torch.randint(0, elite_count, (1,)).item()]
            parent_b = elites[torch.randint(0, elite_count, (1,)).item()]
            
            child = SwarmExpert(dim=self.dim, device=self.device, parent_a=parent_a, parent_b=parent_b)
            child.apply_gradient_free_mutation()
            new_population.append(child)
            
        self.population = new_population
        return best_expert

# --- Execution Test ---
if __name__ == "__main__":
    # Mocking the Unitary Wave Generator
    def generate_uwe(seed: str, dim=4096):
        hash_digest = hashlib.sha256(seed.encode('utf-8')).digest()
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(int.from_bytes(hash_digest[:4], 'little'))
        phases = (torch.rand(dim, generator=generator) * 2 * torch.pi) - torch.pi
        return torch.polar(torch.ones(dim), phases)

    print("🚀 INITIATING GRADIENT-FREE EVOLUTIONARY SWARM")
    
    # The absolute truth the system is trying to discover (The ARC-AGI rule)
    target_axiom = generate_uwe("ARC_AGI_HIDDEN_RULE")
    
    # Instantiate the 16-expert population
    swarm_engine = GradientFreeThermostat(swarm_size=16)
    
    # Run the Darwinian simulation
    max_generations = 150
    for gen in range(1, max_generations + 1):
        alpha_expert = swarm_engine.evolutionary_epoch(target_axiom)
        
        if gen % 10 == 0 or gen == 1:
            print(f"[Gen {gen:03d}] Best Fitness: {alpha_expert.fitness:.4f} "
                  f"| Strategy (Rate: {alpha_expert.mutation_rate:.2f}, Scale: {alpha_expert.mutation_scale:.2f})")
            
        if alpha_expert.fitness > 0.995:
            print(f"\n✅ ABSOLUTE ATTRACTOR LOCK ACHIEVED AT GENERATION {gen}.")
            print("The Swarm has successfully evolved the answer without computing a single gradient.")
            break