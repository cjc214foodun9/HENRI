import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import collections
from typing import List, Callable, Dict, Any

class ExperimentDesigner(torch_nn.Module):
    """
    Identifies parameter regions where competing mathematical theories disagree.
    It uses gradient ascent on the input space to maximize the prediction variance
    across an ensemble of hypotheses, effectively targeting the topological 
    boundaries of the network's current knowledge.
    """
    def __init__(self, state_dim: int, lr: float = 0.05, optimization_steps: int = 50):
        super().__init__()
        self.state_dim = state_dim
        self.lr = lr
        self.optimization_steps = optimization_steps

    def design_optimal_experiment(self, hypothesis_ensemble: List[Callable], seed_state: torch.Tensor) -> torch.Tensor:
        """
        Takes a seed state and perturbs it until the hypotheses disagree the most.
        Returns the optimal 'x' (experimental parameters) to test in reality.
        """
        # We optimize the input state, NOT the network weights.
        experimental_x = seed_state.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([experimental_x], lr=self.lr)

        for _ in range(self.optimization_steps):
            optimizer.zero_grad()
            
            predictions = []
            for hypothesis in hypothesis_ensemble:
                predictions.append(hypothesis(experimental_x))
            stacked_preds = torch.stack(predictions) # (Num_Theories, Batch, State_Dim)
            
            # 1. Physics-Aware Epistemic Uncertainty (Phase Disagreement)
            # Calculate how much the theories disagree on the wave vector directions
            mean_pred = stacked_preds.mean(dim=0, keepdim=True)
            # Cosine distance between each theory and the mean ensemble prediction
            cos_sim = F.cosine_similarity(stacked_preds, mean_pred.expand_as(stacked_preds), dim=-1)
            # We want to MAXIMIZE disagreement, which means MINIMIZING cosine similarity
            disagreement = -cos_sim.mean() 
            
            # 2. Hardware Boundary Regularization (Soft Clamp)
            # Penalize the optimizer if it tries to push the experimental parameters 
            # beyond the safe normalized bounds [-1, 1].
            boundary_penalty = torch.relu(torch.abs(experimental_x) - 1.0).sum() * 10.0
            
            # Loss to minimize: -disagreement + penalty
            loss = disagreement + boundary_penalty 
            loss.backward()
            optimizer.step()
            
        # Apply a hard physical clamp at the very end just to be perfectly safe
        return torch.clamp(experimental_x.detach(), min=-1.0, max=1.0)


class PhysicalSubstrateInterface:
    """
    Mock interface representing the physical boundary. 
    In deployment, this sends telemetry out to the Barium Titanate waveguides, 
    triggers the physical wave mechanics, and records the emergent topological state.
    """
    def execute(self, experimental_params: torch.Tensor) -> torch.Tensor:
        # Simulate an unknown physical ground truth (e.g., a non-linear wave interaction)
        # The agent does not have access to this equation.
        noise = torch.randn_like(experimental_params) * 0.05
        return torch.sin(experimental_params * 2.0) + torch.tanh(experimental_params) + noise


class ClosedLoopScientist(torch_nn.Module):
    """
    The orchestrator of Phase 4. 
    Couples hypothesis generation (Phase 3) with hypothesis-conditioned 
    experiment selection, actively probing the environment to resolve uncertainty.
    """
    def __init__(self, program_inductor, state_dim: int, ensemble_size: int = 5, memory_size: int = 1000):
        super().__init__()
        self.inductor = program_inductor # Instance of ProgramInductor from Phase 3
        self.designer = ExperimentDesigner(state_dim)
        self.physical_world = PhysicalSubstrateInterface()
        self.ensemble_size = ensemble_size
        
        self.active_theories = [] 
        
        # Use a bounded deque to prevent OOM and forget outdated thermodynamic states
        self.empirical_observations = collections.deque(maxlen=memory_size)
        
        # ModuleList to register active theory modules for serialization & device casting
        self.active_theory_modules = torch_nn.ModuleList()

    def _sync_theory_modules(self):
        """Registers all local primitive modules from active ASTs to a ModuleList for serialization/device casts."""
        modules = []
        for theory in self.active_theories:
            modules.extend(self._collect_ast_modules(theory['ast']))
        self.active_theory_modules = torch_nn.ModuleList(modules)

    def _collect_ast_modules(self, ast: Dict[str, Any]) -> List[torch_nn.Module]:
        modules = []
        if ast["type"] == "branch":
            modules.append(ast["branch_logic"])
            modules.append(ast["condition_primitive"])
            modules.extend(self._collect_ast_modules(ast["true_tree"]))
            modules.extend(self._collect_ast_modules(ast["false_tree"]))
        elif ast["type"] == "primitive":
            modules.append(ast["op"])
            for inp in ast["inputs"]:
                modules.extend(self._collect_ast_modules(inp))
        return modules

    def bootstrap_hypotheses(self, x_init: torch.Tensor, y_init: torch.Tensor):
        """Generates the initial competing theories based on early data."""
        self.active_theories = []
        for _ in range(self.ensemble_size):
            ast, loss = self.inductor.fit_program_to_data(x_init, y_init, num_architectures=3)
            # Create a callable wrapper for the AST
            callable_theory = lambda x, ast=ast: self.inductor.execute_program(ast, x)
            self.active_theories.append({'ast': ast, 'callable': callable_theory, 'score': loss})
        self._sync_theory_modules()

    def run_discovery_cycle(self, seed_state: torch.Tensor):
        """
        The core active inference loop.
        """
        # 1. DESIGN: Find where theories disagree most
        callables = [t['callable'] for t in self.active_theories]
        optimal_x = self.designer.design_optimal_experiment(callables, seed_state)
        
        # 2. EXECUTE: Query the physical substrate
        true_y = self.physical_world.execute(optimal_x)
        self.empirical_observations.append((optimal_x, true_y))
        
        # 3. VERIFY & UPDATE: Cull weak theories, mutate strong ones
        # Re-evaluate all theories on the new, highly-informative data point
        for theory in self.active_theories:
            pred_y = theory['callable'](optimal_x)
            error = F.mse_loss(pred_y, true_y).item()
            # Exponential moving average to update the theory's trust score
            theory['score'] = (0.7 * theory['score']) + (0.3 * error)
            
        # Sort theories by how well they survived the physical experiment
        self.active_theories.sort(key=lambda t: t['score'])
        
        # 4. ITERATE: Discard the worst theory and synthesize a new one using RECENT data
        # Grab up to the last 50 highly-informative experiments to train the new theory
        recent_obs = list(self.empirical_observations)[-50:]
        best_x_history = torch.cat([obs[0] for obs in recent_obs])
        best_y_history = torch.cat([obs[1] for obs in recent_obs])
        
        # Induce a new program to replace the weakest link
        new_ast, new_loss = self.inductor.fit_program_to_data(best_x_history, best_y_history, num_architectures=5)
        new_callable = lambda x, ast=new_ast: self.inductor.execute_program(ast, x)
        
        # Replace the worst performing theory (Darwinian selection on the logic space)
        self.active_theories[-1] = {'ast': new_ast, 'callable': new_callable, 'score': new_loss}
        self._sync_theory_modules()
        
        return optimal_x, true_y, self.active_theories[0] # Return the resulting best theory