import torch
import torch.nn as torch_nn

from emergent_topological_manifold import EmergentManifold
from autotelic_cognitive_engine import IMGEP_Manager
from neurosymbolic_program_induction import ProgramInductor
from active_experimentation_engine import ClosedLoopScientist, PhysicalSubstrateInterface

class SwarmAgent(torch_nn.Module):
    """
    An individual autonomous epistemic agent within the swarm.
    Each agent develops its own symbolic theories about the wave mechanics
    and sets its own intrinsic learning goals.
    """
    def __init__(self, agent_id: int, state_dim: int, action_dim: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.agent_id = agent_id
        
        # Phase 2: Autotelic Goal Generation
        self.imgep = IMGEP_Manager(state_dim, action_dim, vocab_size, embed_dim)
        
        # Phase 3: Neurosymbolic Logic Induction
        self.inductor = ProgramInductor(state_dim)
        
        # Phase 4: Active Experimentation
        self.scientist = ClosedLoopScientist(self.inductor, state_dim)
        
        # Internal state tracking
        self.current_concept_focus = (0, 1) # Default starting Vygotskian concepts

    def propose_experiment(self, current_global_state: torch.Tensor):
        """
        Agent imagines a goal, formulates a hypothesis, and proposes an experiment.
        """
        # 1. Imagine a goal state (Vygotskian recombination)
        goal_state = self.imgep.generate_goal(
            torch.tensor([self.current_concept_focus[0]]), 
            torch.tensor([self.current_concept_focus[1]])
        )
        
        # 2. Design the optimal experiment to test its current neurosymbolic theories
        if not self.scientist.active_theories:
            # Bootstrap if no theories exist yet
            dummy_x = torch.randn(5, current_global_state.size(-1))
            dummy_y = torch.randn(5, current_global_state.size(-1))
            self.scientist.bootstrap_hypotheses(dummy_x, dummy_y)
            
        callables = [t['callable'] for t in self.scientist.active_theories]
        proposed_x = self.scientist.designer.design_optimal_experiment(callables, current_global_state)
        
        return proposed_x, goal_state

    def assimilate_results(self, state, action, next_state, goal_state, concept_key):
        """Update internal intrinsic motivations based on the physical result."""
        metrics = self.imgep.internalize_experience(state, action, next_state, goal_state, concept_key)
        return metrics


class GlobalCognitiveSwarm(torch_nn.Module):
    """
    The orchestrator. Manages the shared topological manifold and arbitrates
    physical experiments on the silicon photonic / Barium Titanate substrate.
    """
    def __init__(self, num_agents: int, state_dim: int, action_dim: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.state_dim = state_dim
        
        # Phase 1: Shared Global Memory (Replaces rigid 4x4 matrix zones)
        self.shared_manifold = EmergentManifold(in_features=state_dim, hidden_features=state_dim)
        
        # Hardware Interface
        self.physical_substrate = PhysicalSubstrateInterface()
        
        # Initialize the Swarm
        self.agents = torch_nn.ModuleList([
            SwarmAgent(i, state_dim, action_dim, vocab_size, embed_dim) 
            for i in range(num_agents)
        ])

    def swarm_step(self, raw_sensory_input: torch.Tensor):
        """
        The Core Loop of HENRI.
        1. Crystallize raw data into the manifold.
        2. Agents propose experiments.
        3. Swarm executes the highest-value experiment on the hardware.
        4. Agents assimilate the results and rewrite their internal logic.
        """
        # 1. Manifold Updates (Entropy Reduction & Topological Closure)
        # The shared environment self-organizes based on incoming wave data
        structured_state = self.shared_manifold(raw_sensory_input).detach()
        
        proposed_experiments = []
        
        # 2. Epistemic Foraging (Agents think)
        for agent in self.agents:
            exp_x, goal_state = agent.propose_experiment(structured_state)
            proposed_experiments.append({
                'agent': agent,
                'exp_x': exp_x,
                'goal_state': goal_state
            })
            
        # 3. Hardware Arbitration (Risk Minimization)
        # For this example, we simply average the proposed experimental parameters,
        # but in a deeper physical system, you could select the experiment with the 
        # highest mathematical variance across the ENTIRE swarm's theories.
        consensus_action = torch.stack([p['exp_x'] for p in proposed_experiments]).mean(dim=0)
        
        # Limit phase modulation voltages to safe theoretical bounds
        MAX_SAFE_MODULATION = 0.5
        safe_consensus_action = consensus_action * MAX_SAFE_MODULATION
        
        # Physical Execution (Interacting with the Barium Titanate waveguide)
        emergent_next_state = self.physical_substrate.execute(safe_consensus_action)
        
        # Map the raw physical result back through our shared topology
        structured_next_state = self.shared_manifold(emergent_next_state).detach()
        
        # 4. Swarm Assimilation (Learning)
        for prop in proposed_experiments:
            agent = prop['agent']
            goal_state = prop['goal_state']
            concept_key = str(agent.current_concept_focus)
            
            # The agent updates its autotelic drivers based on the group's action
            metrics = agent.assimilate_results(
                structured_state, safe_consensus_action, structured_next_state, goal_state, concept_key
            )
            
            # The agent updates its symbolic logic trees based on the empirical truth
            # (Phase 4 integration)
            agent.scientist.empirical_observations.append((safe_consensus_action, structured_next_state))
            # Trigger a re-evaluation of theories if enough data is collected
            if len(agent.scientist.empirical_observations) % 5 == 0:
                 agent.scientist.run_discovery_cycle(structured_next_state)
                 
            # If learning progress stalls, shift Vygotskian focus (curriculum climbing)
            if metrics["learning_progress"] < 0.01:
                agent.current_concept_focus = (
                    (agent.current_concept_focus[0] + 1) % 10,
                    (agent.current_concept_focus[1] + 1) % 10
                )
                
        return structured_next_state

# =====================================================================
# Execution Mockup
# =====================================================================
if __name__ == "__main__":
    print("Initializing HENRI Emergent Cognitive Swarm...")
    
    STATE_DIM = 128
    ACTION_DIM = 128
    VOCAB_SIZE = 50
    EMBED_DIM = 8
    
    henri_swarm = GlobalCognitiveSwarm(
        num_agents=3, 
        state_dim=STATE_DIM, 
        action_dim=ACTION_DIM, 
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM
    )
    
    # Simulate a stream of continuous sensory data from a waveguide
    for step in range(10):
        mock_wave_data = torch.randn(1, STATE_DIM)
        new_state = henri_swarm.swarm_step(mock_wave_data)
        print(f"Swarm Step {step+1} Completed. New Topological State Formed.")