import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import math
import random
from typing import List, Callable, Dict, Any

# =====================================================================
# 1. Smooth Interpretation (Differentiable Control Flow)
# =====================================================================
class SmoothBranch(torch_nn.Module):
    """
    Implements Smooth Interpretation for discrete control flow (If/Else).
    Approximates the convolution of a step function with a Gaussian kernel.
    This allows gradients to flow across discrete branch boundaries, critical
    for tuning neural parameters inside complex, generated logical programs.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        # Temperature controls the entropy/smoothness of the branch boundary
        # As T -> 0, it approaches a discrete, non-differentiable step function.
        self.temperature = temperature

    def forward(self, condition_val: torch.Tensor, true_branch_out: torch.Tensor, false_branch_out: torch.Tensor):
        """
        condition_val: A continuous tensor where > 0 implies True, < 0 implies False.
        """
        # Gaussian CDF via Error Function (erf)
        # Represents the probability mass of the condition being true under Gaussian noise
        p_true = 0.5 * (1.0 + torch.erf(condition_val / (self.temperature * math.sqrt(2))))
        p_false = 1.0 - p_true
        
        # Smoothly blend the outputs based on the Gaussian probability mass
        return (p_true * true_branch_out) + (p_false * false_branch_out)


# =====================================================================
# 2. HOUDINI-Style Primitives Library (Strongly Typed)
# =====================================================================
class SymbolicPrimitive(torch_nn.Module):
    """Base class for differentiable functional primitives."""
    def __init__(self, arity: int):
        super().__init__()
        self.arity = arity # Number of inputs it takes

class DifferentiableWaveTransform(SymbolicPrimitive):
    """A parameterized physical/topological primitive (e.g., Phase Shift)."""
    def __init__(self):
        super().__init__(arity=1)
        # Neural parameter tuned end-to-end via gradient descent
        self.phase_shift = torch_nn.Parameter(torch.tensor([0.0]))
        self.amplitude = torch_nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.amplitude * torch.sin(x + self.phase_shift)

class LinearTopologicalFold(SymbolicPrimitive):
    """A neural parameterized linear map for manifold folding."""
    def __init__(self, dim: int):
        super().__init__(arity=1)
        self.linear = torch_nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)

class SymbolicAdd(SymbolicPrimitive):
    """Parameter-free symbolic composition."""
    def __init__(self):
        super().__init__(arity=2)
        
    def forward(self, x, y):
        return x + y


# =====================================================================
# 3. HOUDINI Program Synthesizer
# =====================================================================
class ProgramInductor(torch_nn.Module):
    """
    Neurosymbolic Synthesizer. 
    Performs a combinatorial search over program architectures (the symbolic module),
    while tuning the embedded neural parameters via gradient descent (the neural module).
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.smooth_branch = SmoothBranch(temperature=0.1)
        
        # In a full implementation, this library would be dynamically expanded
        # by the Vygotskian Imagination module from Phase 2.
        self.library = torch_nn.ModuleDict({
            'wave_transform': DifferentiableWaveTransform(),
            'topo_fold': LinearTopologicalFold(state_dim),
            'add': SymbolicAdd()
        })
        
    def clone_primitive(self, op_name: str, device: torch.device) -> torch_nn.Module:
        """Creates a localized, independent instance of a primitive on the correct device."""
        if op_name == 'wave_transform':
            return DifferentiableWaveTransform().to(device)
        elif op_name == 'topo_fold':
            return LinearTopologicalFold(self.state_dim).to(device)
        elif op_name == 'add':
            return SymbolicAdd().to(device)
        elif op_name == 'smooth_branch':
            return SmoothBranch(temperature=0.1).to(device)
        else:
            raise ValueError(f"Unknown primitive: {op_name}")

    def generate_random_program_tree(self, depth: int, device: torch.device) -> Dict[str, Any]:
        """Stochastically generates an architecture with isolated parameters."""
        if depth == 0:
            return {"type": "input"}
            
        node_type = random.choice(["branch", "primitive"])
        
        if node_type == "branch":
            return {
                "type": "branch",
                # Instantiate local SmoothBranch and condition primitive
                "branch_logic": self.clone_primitive("smooth_branch", device),
                "condition_primitive": self.clone_primitive("topo_fold", device),
                "true_tree": self.generate_random_program_tree(depth - 1, device),
                "false_tree": self.generate_random_program_tree(depth - 1, device)
            }
        else:
            op_name = random.choice(['wave_transform', 'topo_fold', 'add'])
            # Instantiate local operational primitive
            op_instance = self.clone_primitive(op_name, device)
            inputs = [self.generate_random_program_tree(depth - 1, device) for _ in range(op_instance.arity)]
            return {
                "type": "primitive",
                "op": op_instance,
                "inputs": inputs
            }

    def execute_program(self, ast: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
        """Recursively evaluates the generated differentiable program."""
        if ast["type"] == "input":
            return x
            
        elif ast["type"] == "branch":
            # 1. Evaluate condition using the local primitive
            cond_val = ast["condition_primitive"](x).sum(dim=-1, keepdim=True)
            # 2. Recursively evaluate branches
            true_out = self.execute_program(ast["true_tree"], x)
            false_out = self.execute_program(ast["false_tree"], x)
            # 3. Apply local Smooth Interpretation
            return ast["branch_logic"](cond_val, true_out, false_out)
            
        elif ast["type"] == "primitive":
            op = ast["op"]
            # Recursively evaluate all inputs to this function
            evaluated_inputs = [self.execute_program(inp, x) for inp in ast["inputs"]]
            return op(*evaluated_inputs)

    def get_ast_parameters(self, ast: Dict[str, Any]) -> List[torch_nn.Parameter]:
        """Recursively collects parameters ONLY for this specific candidate tree."""
        params = []
        if ast["type"] == "branch":
            params.extend(list(ast["branch_logic"].parameters()))
            params.extend(list(ast["condition_primitive"].parameters()))
            params.extend(self.get_ast_parameters(ast["true_tree"]))
            params.extend(self.get_ast_parameters(ast["false_tree"]))
        elif ast["type"] == "primitive":
            params.extend(list(ast["op"].parameters()))
            for inp in ast["inputs"]:
                params.extend(self.get_ast_parameters(inp))
        return params

    def fit_program_to_data(self, x: torch.Tensor, target_y: torch.Tensor, num_architectures: int = 5):
        best_ast = None
        best_loss = float('inf')
        
        for _ in range(num_architectures):
            # 1. Generate AST with modules already pushed to x.device
            ast = self.generate_random_program_tree(depth=2, device=x.device)
            
            # 2. Extract ONLY the local parameters for this specific tree
            ast_params = self.get_ast_parameters(ast)
            
            # If the tree has no learnable parameters (e.g., just "add" ops), skip tuning
            if not ast_params:
                continue
                
            with torch.enable_grad():
                optimizer = torch.optim.Adam(ast_params, lr=0.01)
                
                for step in range(10): 
                    optimizer.zero_grad()
                    # execute_program no longer calls .to(x.device), it just runs the math
                    pred_y = self.execute_program(ast, x)
                    loss = F.mse_loss(pred_y, target_y)
                    loss.backward()
                    optimizer.step()
                
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_ast = ast
                
        return best_ast, best_loss

    def verify_syntax(self, code_str: str) -> tuple:
        """
        Statically compiles candidate code string to AST to detect syntax errors.
        """
        import ast
        import traceback
        
        code_to_parse = code_str
        if "<|python_begin" in code_str:
            idx_begin = code_str.find("<|python_begin")
            idx_end = code_str.find("<|python_end|>")
            if idx_end == -1:
                idx_end = code_str.find("</|python_end|>")
            idx_close = code_str.find("|>", idx_begin)
            if idx_close != -1 and idx_close < idx_end:
                code_to_parse = code_str[idx_close + 2 : idx_end].strip()
        elif "```python" in code_str:
            parts = code_str.split("```python")
            if len(parts) > 1:
                code_to_parse = parts[1].split("```")[0].strip()
                
        try:
            ast.parse(code_to_parse)
            return True, "Python AST compiled successfully."
        except SyntaxError as e:
            error_msg = f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset}\nCode: {e.text}"
            return False, error_msg
        except Exception as e:
            return False, f"AST Compilation Error: {str(e)}"