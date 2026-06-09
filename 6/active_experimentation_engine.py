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
        with torch.enable_grad():
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


class ActiveExperimentationEngine:
    def __init__(self, orchestrator, max_revisions=3):
        self.orchestrator = orchestrator  # Points to the base model wrapper
        self.max_revisions = max_revisions
        
        from neurosymbolic_program_induction import ProgramInductor
        from zone_b_emulator import ZoneBPhysicalEmulator
        
        self.inductor = ProgramInductor(state_dim=128)
        self.emulator = ZoneBPhysicalEmulator(self.orchestrator)

    def execute_task_manifold(self, task_dict):
        """
        Coordinates the multi-turn ACE loop: playbook generation, steering wave compilation,
        decentralized generation, syntax checking, physical wave validation, and playbook curation.
        """
        playbook_dict = self.orchestrator.initialize_empty_playbook()
        
        from run_arc_benchmark import build_arc_prompt
        task_prompt, _ = build_arc_prompt(task_dict)
        
        for revision_step in range(1, self.max_revisions + 1):
            print(f"\n--- [ACE TURN {revision_step}] ---")
            
            # 1. Compile the active playbook definitions into a steering wave
            playbook_wave = self.orchestrator.compile_playbook_to_wave(playbook_dict)
            
            # 2. Delegate generation to the decentralized 16-agent fabric
            candidate_code = self.orchestrator.swarm_fabric.generate_swarm_hypothesis(
                task_dict=task_dict,
                playbook_wave=playbook_wave,
                playbook_dict=playbook_dict
            )
            
            print(f"[ENGINE] Swarm Hypothesis generated.")
            
            # 3. Validate candidate through the symbolic AST compiler
            ast_valid, syntax_feedback = self.inductor.verify_syntax(candidate_code)
            if not ast_valid:
                print(f"[ENGINE] Syntax check failed: {syntax_feedback}")
                insight = self.orchestrator.reflect_on_failure(task_prompt, candidate_code, syntax_feedback)
                playbook_dict = self.orchestrator.curate_playbook(playbook_dict, insight)
                print(f"[ENGINE] Playbook curated based on syntax error.")
                continue
                
            # 4. Verify logic through the physical GPU-accelerated optical emulator
            wave_valid, physical_feedback = self.emulator.evaluate_wavefront(candidate_code, target_label="SCADA_Pressure_Control")
            if not wave_valid:
                print(f"[ENGINE] Physical wave check failed: {physical_feedback}")
                insight = self.orchestrator.reflect_on_failure(task_prompt, candidate_code, physical_feedback)
                playbook_dict = self.orchestrator.curate_playbook(playbook_dict, insight)
                print(f"[ENGINE] Playbook curated based on boundary violation.")
                continue
                
            # Both validation gates pass: extract final test grid prediction
            common_imports = (
                "import math\n"
                "import collections\n"
                "from collections import defaultdict, deque, Counter\n"
                "import itertools\n"
                "import copy\n"
                "import numpy as np\n"
                "import torch\n"
                "import torch.nn as torch_nn\n"
            )
            
            code_block = candidate_code
            if "<|python_begin" in candidate_code:
                idx_begin = candidate_code.find("<|python_begin")
                idx_end = candidate_code.find("<|python_end|>")
                if idx_end == -1:
                    idx_end = candidate_code.find("</|python_end|>")
                idx_close = candidate_code.find("|>", idx_begin)
                if idx_close != -1 and idx_close < idx_end:
                    code_block = candidate_code[idx_close + 2 : idx_end].strip()
            elif "```python" in candidate_code:
                parts = candidate_code.split("```python")
                if len(parts) > 1:
                    code_block = parts[1].split("```")[0].strip()

            exec_code = common_imports + code_block + "\n\n"
            self.orchestrator.repl.execute_block(exec_code)
            
            test_input = task_dict["test"][0]["input"]
            test_runner = (
                f"input_val = {test_input}\n"
                f"output_val = transform(input_val)\n"
                f"print('TEST_RESULT:', output_val)\n"
            )
            test_res = self.orchestrator.repl.execute_block(test_runner)
            test_pred = None
            if test_res["success"]:
                stdout = test_res["stdout"].strip()
                if "TEST_RESULT:" in stdout:
                    try:
                        result_str = stdout.split("TEST_RESULT:")[1].strip()
                        import ast
                        test_pred = ast.literal_eval(result_str)
                    except:
                        pass

            if playbook_wave is not None:
                self.orchestrator.l3_router.update_expert_centroids(playbook_wave)
            
            self.orchestrator.flush_lora_and_context_to_db(domain_tag="ARC_Task")
            
            return test_pred, revision_step, "SUCCESS"
            
        self.orchestrator.flush_lora_and_context_to_db(domain_tag="ARC_Task")
        return None, self.max_revisions, "FAILED"