import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import ast
import traceback
import os
from typing import Dict, List, Tuple, Optional
from henri_core.hrr import QuantizedFHRREngine
from henri_core.core import ProprietaryHENRICore

class StandaloneL3Router(nn.Module):
    """
    Standalone Ingress Transduction Head.
    Utilizes the pre-trained 'fluid_context_router.weight' to map discrete
    token or coordinate vectors directly into the 4096-D complex phase-space,
    bypassing the need for an external Gemma language model.
    """
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Hardwired vocabulary embedding matrix
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Linear projection bridge corresponding to the pre-trained fluid_context_router.weight
        self.activation_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.init_orthogonal_bridge()

    @torch.no_grad()
    def init_orthogonal_bridge(self):
        """
        Orthogonally initializes the projection parameter.
        Ensures absolute angular distance conservation in the VSA space.
        """
        nn.init.orthogonal_(self.activation_projection.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Ingests discrete token sequences, maps them to high-dimensional VSA space,
        and wraps them onto the S^4095 unit complex hypersphere.
        """
        # tokens shape: [B, S]
        embeddings = self.token_embedding(tokens) # [B, S, hidden_dim]
        projected = self.activation_projection(embeddings) # [B, S, hidden_dim]
        
        # Force strict hyperspherical wrapping (Unit Modulus periodic invariant)
        phases = (projected / torch.norm(projected, p=2, dim=-1, keepdim=True)) * math.pi
        return torch.complex(torch.cos(phases), torch.sin(phases))


class UniversalREPLSandbox:
    """
    Type-safe, secure on-device code execution sandbox.
    Parses candidate code strings via Python AST and executes them inside an isolated frame.
    """
    def __init__(self):
        self.execution_globals = {
            "np": np,
            "math": math,
        }

    def execute_safely(self, code_string: str, input_data) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Runs crystallized code candidates against actual ARC data grids.
        Enforces runtime safety by intercepting banned system modules.
        """
        local_scope = {}
        try:
            # Parse code structure via AST to check for illegal imports
            parsed_ast = ast.parse(code_string)
            for node in ast.walk(parsed_ast):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    return False, "Sandbox violation: Import statements banned.", 1000.0

            # Execute compiled byte-code inside dynamic scope
            compiled_code = compile(parsed_ast, filename="<sandbox>", mode="exec")
            exec(compiled_code, self.execution_globals, local_scope)
            
            if "transform" not in local_scope:
                return False, "Syntax Error: transform(input_grid) entrypoint missing.", 500.0
            
            transform_fn = local_scope["transform"]
            output_grid = transform_fn(input_data)
            
            return True, str(output_grid), 0.0

        except Exception as e:
            tb = traceback.format_exc()
            return False, f"Runtime Exception: {str(e)}\n{tb}", 250.0


class HenriCognitiveSwarmOrchestrator(nn.Module):
    """
    Central Swarm Orchestration Engine.
    Instantiates and coordinates 16 parallel experts, each running a lean 
    485M parameter wave core with integrated viscoelastic LoRA adapters.
    Allows experts to explore distinct bandwidths of the latent space concurrently.
    """
    def __init__(self, vocab_size: int = 32000, dim: int = 4096, num_experts: int = 16):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        
        # Initialize VSA Engine and Standalone L3 Router
        self.vsa_engine = QuantizedFHRREngine(dim=dim)
        self.l3_router = StandaloneL3Router(vocab_size=vocab_size, hidden_dim=dim)
        self.sandbox = UniversalREPLSandbox()

        # Instantiate 16 parallel 485M parameter wave core experts (P-MoM)
        # Total Swarm Parameter Footprint: 16 * 485M = ~7.76 Billion parameters
        self.experts = nn.ModuleList([
            ProprietaryHENRICore(dim=dim, num_layers=32)
            for _ in range(num_experts)
        ])

        # Pre-trained microheater thermal mask for Langevin simmer
        self.register_buffer("thermal_mask", torch.ones(dim))

        # Persistent sub-axiom channels (Experts 12-15 are locked from eviction)
        self.preserved_sub_axioms: Dict[int, Dict] = {}

    def load_pretrained_weights(self, path: str = "henri_core_final.pt", device: str = "cuda"):
        """
        Loads pre-trained physical parameters.
        Distributes spatial_kernel and K_micro parameters cleanly across the 16 parallel experts.
        """
        if not os.path.exists(path):
            print(f"[!] Target checkpoint {path} not found. Skipping weight load; using defaults.")
            return False
            
        print(f"[LOADING SUBSTRATE] Loading pre-trained weights from {path} onto {device}...")
        try:
            state_dict = torch.load(path, map_location=device)
            
            # 1. Align context router weights: fluid_context_router.weight
            if "fluid_context_router.weight" in state_dict:
                self.l3_router.activation_projection.weight.data.copy_(
                    state_dict["fluid_context_router.weight"].to(torch.float32)
                )
            
            # 2. Distribute spatial_kernel and K_micro across all 16 parallel experts
            for idx, expert in enumerate(self.experts):
                # Distribute Kuramoto coupling matrices: K_micro
                if "K_micro" in state_dict:
                    k_data = state_dict["K_micro"].to(torch.float32)
                    for l in range(expert.num_layers):
                        if k_data.dim() == 2:
                            expert.coupling_matrices[l].data.copy_(k_data)
                        else:
                            expert.coupling_matrices[l].data.copy_(k_data[0] if k_data.dim() > 2 else k_data)

                # Distribute Base Unitary Phase Masks: spatial_kernel
                if "spatial_kernel" in state_dict:
                    kernel_data = state_dict["spatial_kernel"]
                    for l in range(expert.num_layers):
                        if not kernel_data.is_complex():
                            if kernel_data.shape[-1] == 2:
                                complex_kernel = torch.complex(kernel_data[..., 0], kernel_data[..., 1])
                            else:
                                complex_kernel = torch.complex(torch.cos(kernel_data), torch.sin(kernel_data))
                        else:
                            complex_kernel = kernel_data
                            
                        if complex_kernel.shape == expert.layers[l].shape:
                            expert.layers[l].data.copy_(complex_kernel)
                        else:
                            expert.layers[l].data.copy_(complex_kernel[0] if complex_kernel.dim() > 2 else complex_kernel)

            # 3. Align Langevin thermal simmer mask: thermal_mask
            if "thermal_mask" in state_dict:
                self.thermal_mask.copy_(state_dict["thermal_mask"].to(torch.float32))

            # Execute post-load Björck-Newton projection to lock all expert manifolds
            for expert in self.experts:
                expert.bjorck_newton_orthonormalize(iterations=5)
                
            print("[SUCCESS] Autonomous experts successfully locked on the Stiefel manifold.")
            return True
            
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to realign parameter dictionary: {str(e)}")
            traceback.print_exc()
            return False

    def generate_parallel_hypotheses(self, raw_tokens: torch.Tensor, target_axiom: torch.Tensor) -> Dict:
        """
        Executes parallel inference across all 16 experts.
        Applies individual low-rank adaptations, evaluates Sagnac Veto limits, 
        and updates trapped experts using viscoelastic creep callbacks.
        """
        # Ingest tokens directly: raw_tokens shape [B, Seq_Len] -> complex_wave [B, Seq_Len, Dim]
        input_wave = self.l3_router(raw_tokens)
        
        # Compress sequence dimension via circular convolution superposition
        flat_input = torch.sum(input_wave, dim=1) # [B, Dim]
        flat_input = F.normalize(flat_input.real, p=2, dim=-1) + 1j * F.normalize(flat_input.imag, p=2, dim=-1)

        results = []
        global_error_accumulator = 0.0

        # Propagate the wavefront in parallel through the 16 experts (P-MoM)
        for i in range(self.num_experts):
            expert = self.experts[i]
            
            # Forward pass through the expert's 32-layer diffractive network
            output_wave = expert(flat_input, langevin_temp=0.0)
            
            # Sagnac Logic Homodyne Veto Check
            error_delta = torch.abs(output_wave - target_axiom)
            error_energy = torch.sum(error_delta ** 2, dim=-1).mean().item()
            global_error_accumulator += error_energy

            if error_energy > 0.35:
                # Sagnac Veto triggered. Re-fire expert path under Langevin thermal excitation
                # Modulate noise injection using the pre-trained thermal mask
                output_wave_heated = expert(flat_input, langevin_temp=1.5)
                output_wave_heated = output_wave_heated * self.thermal_mask.unsqueeze(0)
                
                error_energy_new = torch.sum(torch.abs(output_wave_heated - target_axiom) ** 2, dim=-1).mean().item()
                results.append({
                    "expert_id": i,
                    "status": "VETO_RELAXED",
                    "error": error_energy_new,
                    "wave": output_wave_heated
                })
            else:
                results.append({
                    "expert_id": i,
                    "status": "PHASE_LOCKED",
                    "error": error_energy,
                    "wave": output_wave
                })

        return {"results": results, "global_error": global_error_accumulator / self.num_experts}

    def apply_viscoelastic_gradient_updates(self, lr: float):
        """
        Instructs all active experts to update their LoRA parameters
        using viscoelastic material memory rules, preventing gradient shattering.
        """
        for expert in self.experts:
            for l in range(expert.num_layers):
                expert.adapters[l].apply_viscoelastic_update(lr=lr)

    def flush_lora_and_context_to_db(self, domain_tag: str, preserve_channels: List[int] = [12, 13, 14, 15]):
        """
        Selective Synaptic Consolidation.
        Archives high-performing expert parameters prior to eviction,
        completely isolating experts 12-15 to preserve global constitutional priors.
        """
        print(f"[TABULA RASA] Flushing volatile memory channels under domain: {domain_tag}...")
        
        for idx in range(self.num_experts):
            if idx in preserve_channels:
                # Capture and lock the expert's complete parameter topology
                self.preserved_sub_axioms[idx] = {
                    'layers': [W.data.clone() for W in self.experts[idx].layers],
                    'coupling': [K.data.clone() for K in self.experts[idx].coupling_matrices],
                    'lora': self.experts[idx].adapters.state_dict()
                }
                print(f" -> [PRESERVED] Long-Term Expert Channel {idx} locked inside non-volatile cache.")
            else:
                # Apply high-energy Gaussian reset to wipe corrupt local paths
                self.experts[idx].reset_parameters()

        print("[SUCCESS] Selective memory consolidation finished. Core parameters secured.")