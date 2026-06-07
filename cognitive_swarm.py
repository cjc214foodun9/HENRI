import os
import sys
import time
import math
import queue
import threading
import concurrent.futures
import numpy as np
import torch
import torch.nn as torch_nn
import psutil
import psycopg
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add paths to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "6"))

from dynamic_lora import DynamicLoraManager
from l3_router_model import L3SwarmRouter
from zone_b import HenriOpticalCoreD2NN
from hopfield_cleanup import HopfieldSemanticCleanup
from boundary_validator import BoundaryAxiomValidator
from universal_repl import UniversalREPL
from memory_cache import CachedHRRMemoryEngine
from telemetry_server import telemetry_register, NonBlockingTelemetryServer

from emergent_topological_manifold import EmergentManifold
from autotelic_cognitive_engine import IMGEP_Manager
from neurosymbolic_program_induction import ProgramInductor
from active_experimentation_engine import ClosedLoopScientist, PhysicalSubstrateInterface

try:
    import llama_cpp
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

class GemmaRAMSwarmMock:
    """
    High-fidelity fallback mock of Gemma E4B model running from RAM
    when llama_cpp cannot find the weights or fails to initialize.
    """
    def __init__(self, gemma_dim=2048):
        self.gemma_dim = gemma_dim
        print("[MOCK SWARM] Initialized Gemma E4B RAM Swarm in Mock Mode.")

    def create_embedding(self, prompt: str) -> dict:
        rng = np.random.default_rng(seed=hash(prompt) & 0xffffffff)
        embedding = rng.normal(loc=0.0, scale=0.02, size=self.gemma_dim).tolist()
        return {"data": [{"embedding": embedding}]}

    def create_chat_completion(self, messages, max_tokens=128, temperature=0.7, **kwargs):
        # Simulate structured reasoning or code generation
        prompt = messages[-1]["content"]
        if "Code Template:" in prompt:
            if "lambda_plus" in prompt:
                # Challenge 2 style with parameters
                text = (
                    "To solve the challenge, here is the complete Python solution:\n"
                    "<|python_begin: heat=0.0|>\n"
                    "import sympy as sp\n"
                    "lambda_plus, lambda_minus = sp.symbols('lambda_plus lambda_minus')\n"
                    "k_plus, k_minus = sp.symbols('k_plus k_minus')\n"
                    "alpha = sp.symbols('alpha')\n"
                    "vbar_b = sp.symbols('vbar_b')\n"
                    "beta = sp.symbols('beta')\n"
                    "sigma2 = sp.symbols('sigma2')\n"
                    "def answer(lambda_plus, lambda_minus, k_plus, k_minus, alpha, vbar_b, beta, sigma2):\n"
                    "    Lambda = lambda_plus + lambda_minus\n"
                    "    answer_beta = 'A'\n"
                    "    answer_sigma2 = 'B'\n"
                    "    return Lambda, answer_beta, answer_sigma2\n"
                    "<|python_end|>\n"
                )
            else:
                # Challenge 1 / Generic style without parameters
                text = (
                    "To solve the challenge, here is the complete Python solution:\n"
                    "<|python_begin: heat=0.0|>\n"
                    "def answer():\n"
                    "    coeffs = [0.0] * 11\n"
                    "    return coeffs\n"
                    "<|python_end|>\n"
                )
        elif "SCADA" in prompt or "thermodynamic" in prompt:
            text = (
                "To optimize the thermodynamic pressure loop:\n"
                "<|python_begin: heat=0.4|>\n"
                "import sympy as sp\n"
                "p, v, t = sp.symbols('p v t')\n"
                "eq = p * v - 8.314 * t\n"
                "print(sp.solve(eq, p)[0])\n"
                "<|python_end|>\n"
                "Using the ideal gas law, pressure is 8.314*t/v."
            )
        else:
            text = f"[Mock response for: '{prompt[:30]}'] Reasoning path verified."
        return {
            "choices": [
                {
                    "message": {
                        "content": text
                    }
                }
            ]
        }


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


class SynapticConsolidationManager:
    """
    Manages online checkpointing, semantic routing, and synaptic consolidation
    for domain-specific LoRA adapters using TimescaleDB.
    """
    def __init__(self, db_url=None):
        self.db_url = db_url or os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")
        self.adapters_dir = Path("archive/domain_adapters")
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.is_connected = False
        
        # Test connection
        if self.db_url:
            try:
                with psycopg.connect(self.db_url, connect_timeout=2) as conn:
                    self.is_connected = True
                print("[SYNAPTIC] Connected to TimescaleDB for LoRA registry.")
            except Exception as e:
                print(f"[SYNAPTIC] Warning: TimescaleDB unreachable for registry ({e}). Running in offline mode.")
                
    def route_and_load_adapter(self, domain_tag: str, lora_manager: DynamicLoraManager) -> bool:
        """
        Semantic Router: Pre-fetches the specialized LoRA adapter associated with the
        domain_tag from the database registry and loads it into the lora_manager.
        """
        if not self.is_connected:
            # Offline fallback: try to load directly from domain path if exists
            fallback_path = self.adapters_dir / f"{domain_tag}.bin"
            if fallback_path.exists():
                lora_manager.lora_path = str(fallback_path)
                lora_manager.load_weights()
                print(f"[SYNAPTIC] Semantic Router: Loaded offline fallback adapter for {domain_tag}")
                return True
            return False
            
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT adapter_path, sagnac_error_delta 
                        FROM lora_adapters_registry 
                        WHERE domain_tag = %s;
                    """, (domain_tag,))
                    row = cur.fetchone()
                    if row:
                        adapter_path, error_delta = row[0], row[1]
                        if os.path.exists(adapter_path):
                            lora_manager.lora_path = adapter_path
                            lora_manager.load_weights()
                            print(f"[SYNAPTIC] Semantic Router: Successfully routed and loaded adapter for domain '{domain_tag}' (Sagnac error: {error_delta:.4f})")
                            return True
                        else:
                            print(f"[SYNAPTIC] Registry path '{adapter_path}' does not exist on disk.")
        except Exception as e:
            print(f"[SYNAPTIC] Error during semantic routing: {e}")
        return False

    def consolidate_and_save_adapter(self, domain_tag: str, lora_manager: DynamicLoraManager, error_delta: float) -> bool:
        """
        Synaptic Consolidation: Persists the stream's current LoRA weights to the
        domain-specific adapter path on disk and registers it in the database.
        """
        adapter_file = self.adapters_dir / f"{domain_tag}.bin"
        
        # Save weights to domain path on disk
        old_path = lora_manager.lora_path
        try:
            lora_manager.lora_path = str(adapter_file)
            lora_manager.save_weights()
        except Exception as save_err:
            print(f"[SYNAPTIC] Error writing adapter to disk: {save_err}")
            return False
        finally:
            lora_manager.lora_path = old_path
            
        if not self.is_connected:
            print(f"[SYNAPTIC] Consolidated adapter for '{domain_tag}' locally on disk (offline mode).")
            return True
            
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO lora_adapters_registry (domain_tag, adapter_path, sagnac_error_delta, last_updated)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (domain_tag)
                        DO UPDATE SET 
                            adapter_path = EXCLUDED.adapter_path,
                            sagnac_error_delta = EXCLUDED.sagnac_error_delta,
                            last_updated = EXCLUDED.last_updated;
                    """, (domain_tag, str(adapter_file), float(error_delta)))
                    print(f"[SYNAPTIC] Synaptic Consolidation: Registered/Updated adapter for domain '{domain_tag}' in database registry.")
                    return True
        except Exception as e:
            print(f"[SYNAPTIC] Error during database synaptic consolidation: {e}")
        return False


class HenriCognitiveSwarmOrchestrator:
    """
    Ties together Zone A (Gemma E4B RAM Swarm + LoRAs), the L3 SRAM Cache Router,
    Zone B (D2NN Physics + Sagnac boundary validation), and Zone C (Hopfield Cleanup + REPL).
    Synchronizes the asynchronous timed loop to fire in series with the RAM cycles
    to build a continuous, coherent HRR vector stream.
    """
    def __init__(self, model_path="gemma-4-E4B-it-Q4_0.gguf", num_streams=16, gemma_dim=2048, hrr_dim=4096):
        self.num_streams = num_streams
        self.gemma_dim = gemma_dim
        self.hrr_dim = hrr_dim
        
        # 0. Synaptic Consolidation Manager
        self.synaptic_manager = SynapticConsolidationManager()
        self.gemma_dim = gemma_dim
        self.hrr_dim = hrr_dim
        
        # 1. Core Pinning & Affinity setup
        self.set_core_affinity()

        # 2. Initialize Gemma E4B model strictly in RAM (CPU, use_mlock, use_mmap=False)
        self.model_path = model_path
        self.base_model = None
        self.gen_model = None
        self.is_mock = False
        
        if HAS_LLAMA_CPP and os.path.exists(model_path):
            try:
                print(f"[SYSTEM] Loading Gemma E4B Base weights directly into RAM from: {model_path}")
                # Load embedding model (n_ctx=4096, n_batch=256, embedding=True)
                print("[SYSTEM] Loading embedding engine...")
                self.base_model = llama_cpp.Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_batch=256,
                    n_threads=4,      # Optimized EPYC thread occupancy
                    embedding=True,   # Enabled for embeddings
                    use_mmap=False,   # Disable memory mapping (bypass pagefile swaps)
                    use_mlock=False,  # Set to False to avoid Windows privilege errors
                    n_gpu_layers=0    # Force CPU execution (RAM swarm)
                )
                
                # Load generation model (n_ctx=4096, n_batch=256, embedding=False)
                print("[SYSTEM] Loading generation engine...")
                self.gen_model = llama_cpp.Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_batch=256,
                    n_threads=4,
                    embedding=False,  # Disabled for text generation cache slots
                    use_mmap=False,
                    use_mlock=False,
                    n_gpu_layers=0
                )
            except Exception as e:
                print(f"[WARNING] Failed to load GGUF model via llama_cpp: {e}. Falling back to mock model.")
                self.base_model = GemmaRAMSwarmMock(gemma_dim=gemma_dim)
                self.gen_model = self.base_model
                self.is_mock = True
        else:
            print("[INFO] GGUF weight file not found or llama_cpp missing. Booting in high-fidelity mock mode.")
            self.base_model = GemmaRAMSwarmMock(gemma_dim=gemma_dim)
            self.gen_model = self.base_model
            self.is_mock = True

        # Measure dynamic embedding dimension to support different GGUF sizes (e.g. E4B has 2560)
        try:
            test_emb = self.base_model.create_embedding("test")['data'][0]['embedding']
            if isinstance(test_emb, list) and len(test_emb) > 0 and isinstance(test_emb[0], list):
                self.gemma_dim = len(test_emb[0])
            else:
                self.gemma_dim = len(test_emb)
            print(f"[SYSTEM] Measured model latent dimension: {self.gemma_dim}")
        except Exception as e:
            print(f"[WARNING] Failed to measure embedding dimension: {e}. Defaulting to {gemma_dim}")
            self.gemma_dim = gemma_dim

        # 3. Create 16 stream-specific Dynamic LoRA Managers
        self.lora_managers = {}
        for i in range(num_streams):
            lora_file = f"archive/dynamic_lora_stream_{i}.bin"
            self.lora_managers[i] = DynamicLoraManager(
                gemma_dim=self.gemma_dim, 
                rank=16, 
                lora_path=lora_file
            )

        # 4. Initialize L3 Cache Router & Translator (pinned to Cores 4-7)
        self.l3_router = L3SwarmRouter(
            vocab_size=64000, 
            hidden_dim=1024, 
            num_layers=2, 
            num_heads=4, 
            pf_dim=512, 
            activation_dim=self.gemma_dim
        )
        
        # Enforce VSA unit-modulus invariants on Swarm Master signatures
        self.l3_router.enforce_vsa_invariants()

        # 5. Initialize Zone B D2NN model
        self.optical_core = HenriOpticalCoreD2NN(num_channels=self.hrr_dim, num_layers=5)

        # 6. Initialize Boundary Axiom Validator (64-D CFT boundary validator)
        self.boundary_validator = BoundaryAxiomValidator(
            bulk_dim=self.hrr_dim, 
            boundary_dim=64, 
            epsilon_spine=0.35
        )

        # 7. Initialize Hopfield Network Cleanup and pre-populate with seed concepts
        self.hopfield = HopfieldSemanticCleanup(dim=self.hrr_dim, beta=35.0, max_iterations=5)
        self.populate_hopfield_lexicon()

        # 8. Stateful REPL sandbox
        self.repl = UniversalREPL()

        # 8b. Initialize stream-specific memory caches (INT8 phase-only growing cache)
        self.memory_engines = {
            i: CachedHRRMemoryEngine(wave_dim=self.hrr_dim, coherence_threshold=0.70, accumulation_limit=15)
            for i in range(num_streams)
        }
        self.stream_position_indices = {i: 0 for i in range(num_streams)}
        
        # Start telemetry server dynamically
        self.telemetry_server = None
        self.start_telemetry_server()

        # Phase 2, 3, 4: Distributed swarm agents (one per stream)
        self.agents = torch_nn.ModuleList([
            SwarmAgent(agent_id=i, state_dim=128, action_dim=128, vocab_size=50, embed_dim=8)
            for i in range(num_streams)
        ])

        # 9. Asynchronous timed loop control structures
        self.stream_contexts = {i: [] for i in range(num_streams)}
        self.stream_prompts = {i: "" for i in range(num_streams)}
        self.active_wave_queue = queue.Queue()
        self.stop_loop = threading.Event()
        self.timed_loop_thread = None

    def set_core_affinity(self):
        """Pins Swarm execution (Cores 0-3) and L3 Router execution (Cores 4-7)."""
        p = psutil.Process(os.getpid())
        try:
            # On Windows, we can restrict process affinity or set thread affinity
            # Lock the overall process affinity to Cores 0-7
            num_cores = psutil.cpu_count()
            if num_cores >= 8:
                p.cpu_affinity(list(range(8)))
                print("[HARDWARE] Process pinned to CPU Cores 0-7 (Firewalled: Swarm [0-3], L3 Router [4-7]).")
            else:
                p.cpu_affinity(list(range(num_cores)))
                print(f"[HARDWARE] Process pinned to available Cores: {list(range(num_cores))}")
        except Exception as e:
            print(f"[WARNING] Core affinity setting not supported or failed: {e}")

    def populate_hopfield_lexicon(self):
        """Fills the Hopfield Network vocabulary with pristine orthogonal axioms and labels."""
        # Add physics constants labels
        concepts = [
            "Planck_Constant", "Boltzmann_Constant", "Speed_of_Light",
            "Thermodynamic_Conservation", "Sagnac_Threshold_Limit", "Lipschitz_Bound",
            "Dirichlet_Invariant", "Neumann_Active_Flow", "Optics_Resonance",
            "SCADA_Pressure_Control", "SCADA_Thermal_Clamping", "Attractor_Converged"
        ]
        # Generate pristine unit-magnitude complex vectors
        for label in concepts:
            phases = (torch.rand(self.hrr_dim) * 2 * math.pi) - math.pi
            vector = torch.polar(torch.ones(self.hrr_dim), phases)
            self.hopfield.register_concept(label, vector)

    def get_stream_address(self, stream_id: int) -> torch.Tensor:
        """Retrieves a unique orthogonal address wave vector for a given stream ID."""
        # Fetch or generate address for the stream
        label = f"Stream_Address_{stream_id}"
        if label not in self.hopfield.vocabulary:
            phases = (torch.rand(self.hrr_dim) * 2 * math.pi) - math.pi
            vector = torch.polar(torch.ones(self.hrr_dim), phases)
            self.hopfield.register_concept(label, vector)
        return self.hopfield.vocabulary[label]

    def process_repl_blocks(self, stream_id: int, generated_text: str) -> str:
        """Parses generated text for python blocks and executes them statefully in the REPL."""
        if "<|python_begin" in generated_text and "<|python_end|>" in generated_text:
            idx_begin = generated_text.find("<|python_begin")
            idx_end = generated_text.find("<|python_end|>")
            idx_close_bracket = generated_text.find("|>", idx_begin)
            
            if idx_close_bracket != -1 and idx_close_bracket < idx_end:
                code_block = generated_text[idx_close_bracket + 2 : idx_end].strip()
                print(f"\n[REPL SANDBOX - Stream {stream_id}] Executing block in stateful REPL...")
                
                # Run the REPL
                res = self.repl.execute_block(code_block)
                stdout = res["stdout"].strip()
                stderr = res["stderr"].strip()
                
                output_content = stdout if res["success"] else f"Error: {stderr or res['error_message']}"
                output_tag = f"\n<|output_begin|>\n{output_content}\n<|output_end|>\n"
                
                # Append output to text representation
                new_text = generated_text[:idx_end + len("<|python_end|>")] + output_tag + generated_text[idx_end + len("<|python_end|>"):]
                
                # Log telemetry
                print(f"[REPL SANDBOX - Stream {stream_id}] Output: {output_content[:60]}...")
                return new_text
        return generated_text

    def get_position_key(self, position_idx: int) -> torch.Tensor:
        """Generates a deterministic position phase key vector for VSA binding."""
        rng = torch.Generator().manual_seed(position_idx)
        phases = (torch.rand(self.hrr_dim, generator=rng) * 2 * math.pi) - math.pi
        return torch.polar(torch.ones(self.hrr_dim), phases)

    def step_stream(self, stream_id: int, prompt: str) -> tuple:
        """Runs a single forward step on one stream thread, returning lensed 4096-D wave."""
        # 1. Retrieve the base embedding (Gemma E4B hidden activations)
        emb_res = self.base_model.create_embedding(prompt)
        h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)

        # 2. Apply the stream-specific LoRA weights to activations
        h_7b_lora = self.lora_managers[stream_id].apply_lora(h_7b_raw)
        if len(h_7b_lora.shape) == 2:
            h_7b_lora = torch.mean(h_7b_lora, dim=0)

        # 3. Project to global 6324x6324 wavefront by replicating activation to all 16 streams
        activations_stack = h_7b_lora.unsqueeze(0).unsqueeze(0).repeat(16, 1, 1)
        global_wavefront, _, _ = self.l3_router(activations=activations_stack)
        wave_2d_in = global_wavefront.squeeze(0)
        
        # 4. Apply analytical optical lens focusing to downsample [6324, 6324] -> [64, 64]
        N = wave_2d_in.size(-1)
        x = torch.linspace(-1.0, 1.0, N, device=wave_2d_in.device)
        y = torch.linspace(-1.0, 1.0, N, device=wave_2d_in.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        lens_phase = -50.0 * (X**2 + Y**2)
        lens = torch.polar(torch.ones_like(lens_phase), lens_phase)
        
        wave_lensed = wave_2d_in * lens
        focal_plane = torch.fft.fft2(wave_lensed, norm='ortho')
        focal_plane_shifted = torch.fft.fftshift(focal_plane)
        
        start = (N - 64) // 2
        end = start + 64
        focused_64 = focal_plane_shifted[start:end, start:end]
        
        mags = torch.abs(focused_64)
        mags = torch.clamp(mags, min=1e-8)
        focused_64_norm = focused_64 / mags
        focused_64_flat = focused_64_norm.flatten()
        
        # --- Memory Cache Integration ---
        # A. Update stream's memory engine
        token_activation = focused_64_flat.clone()
        pos_idx = self.stream_position_indices[stream_id]
        position_key = self.get_position_key(pos_idx).to(token_activation.device)
        signature_key = token_activation.clone()
        
        self.memory_engines[stream_id].update_active_memory(
            token_activation=token_activation,
            position_key=position_key,
            signature_key=signature_key
        )
        self.stream_position_indices[stream_id] += 1
        
        # B. Retrieve blended context and perform superposition
        retrieved_wave = self.memory_engines[stream_id].retrieve_from_cache(query_key=token_activation)
        blended_focused = focused_64_flat + retrieved_wave
        blended_mags = torch.abs(blended_focused).clamp(min=1e-8)
        resolved_focused = blended_focused / blended_mags
        
        return resolved_focused, h_7b_lora

    def run_continuous_wave_timed_loop(self, interval_seconds=0.25):
        """
        Asynchronous timed loop running on a background thread.
        Uses parallel tiled wave synthesis to directly generate the global 6324x6324 wave.
        """
        print(f"\n[SWARM LOOP] Starting asynchronous tiled swarm loop (Tick Interval: {interval_seconds}s)...")
        tick_count = 0
        
        while not self.stop_loop.is_set():
            start_tick = time.perf_counter()
            tick_count += 1
            
            # Retrieve active prompts
            prompts = [self.stream_prompts[i] for i in range(self.num_streams)]
            
            # Step 1: Run the 16 Swarm streams to gather raw activations and apply dynamic LoRA
            stream_activations = []
            for i in range(self.num_streams):
                emb_res = self.base_model.create_embedding(prompts[i])
                h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
                h_7b_lora = self.lora_managers[i].apply_lora(h_7b_raw)
                if len(h_7b_lora.shape) == 2:
                    h_7b_lora = torch.mean(h_7b_lora, dim=0)
                stream_activations.append(h_7b_lora)
                
            # Step 2: Stack activations to shape [16, 1, gemma_dim]
            activations_stack = torch.stack(stream_activations).unsqueeze(1) # [16, 1, gemma_dim]
            
            # Step 3: Call L3 router to synthesize the global 6324x6324 wave directly
            global_wavefront, _, _ = self.l3_router(activations=activations_stack)
            psi_bulk = global_wavefront.squeeze(0) # shape [6324, 6324]

            # Step 4: Put into the queue for the Zone B physical core to pull
            self.active_wave_queue.put({
                "tick": tick_count,
                "psi_bulk": psi_bulk,
                "activations": stream_activations
            })
            
            # Maintain series synchronization with RAM fetch cycles
            elapsed = time.perf_counter() - start_tick
            sleep_time = max(0.005, interval_seconds - elapsed)
            time.sleep(sleep_time)

    def process_next_wave(self, target_label="SCADA_Pressure_Control") -> dict:
        """
        Pulls the constructed bulk wave, runs physical Zone B emulation,
        verifies boundary CFT constraints, applies rehypothecation, and cleanup.
        Integrates all four phases under the Epistemic Swarm paradigm.
        """
        try:
            payload = self.active_wave_queue.get(timeout=2.0)
        except queue.Empty:
            return {"status": "TIMEOUT", "msg": "No wave constructed in timed loop queue."}

        tick = payload["tick"]
        psi_bulk = payload["psi_bulk"]
        activations = payload["activations"]

        print(f"\n--- [COGNITIVE CYCLE TICK {tick}] Intercepting Wavefront ---")

        # 1. Fetch the target manifold vector from Hopfield registered axioms
        target_vector = self.hopfield.vocabulary.get(target_label)
        if target_vector is None:
            target_vector = self.get_stream_address(0)  # fallback
            
        target_np = target_vector.detach().numpy().astype(np.complex64)

        # 1b. Manifold Entropy Reduction (Phase 1)
        # Downsample psi_bulk if it is 2D
        if psi_bulk.ndim == 2:
            N = psi_bulk.size(-1)
            x = torch.linspace(-1.0, 1.0, N, device=psi_bulk.device)
            y = torch.linspace(-1.0, 1.0, N, device=psi_bulk.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            lens_phase = -50.0 * (X**2 + Y**2)
            lens = torch.polar(torch.ones_like(lens_phase), lens_phase)
            
            wave_lensed = psi_bulk * lens
            focal_plane = torch.fft.fft2(wave_lensed, norm='ortho')
            focal_plane_shifted = torch.fft.fftshift(focal_plane)
            
            start = (N - 64) // 2
            end = start + 64
            focused_64 = focal_plane_shifted[start:end, start:end]
            
            mags = torch.abs(focused_64)
            mags = torch.clamp(mags, min=1e-8)
            psi_flat = (focused_64 / mags).flatten()
        else:
            if psi_bulk.size(0) == 4096:
                psi_flat = psi_bulk
            else:
                psi_flat = psi_bulk.flatten()

        # Convert complex boundary wave to 128-D Real/Imag tensor before passing to manifold
        raw_boundary_wave = self.boundary_validator.bulk_to_boundary(psi_flat)
        wave_real, wave_imag = raw_boundary_wave.real, raw_boundary_wave.imag
        euclidean_wave = torch.cat([wave_real, wave_imag], dim=-1).unsqueeze(0)
        structured_state = self.boundary_validator.shared_manifold(euclidean_wave).detach()

        proposed_experiments = []

        # 2. Epistemic Foraging (Phase 2 & 3)
        for agent in self.agents:
            exp_x, goal_state = agent.propose_experiment(structured_state)
            
            # Calculate how much the ENTIRE swarm disagrees on this specific agent's proposal
            callables = [t['callable'] for a in self.agents for t in a.scientist.active_theories]
            if not callables:
                 disagreement = 1.0  # Default to high if uninitialized
            else:
                 predictions = torch.stack([c(exp_x) for c in callables])
                 mean_pred = predictions.mean(dim=0, keepdim=True)
                 # Use Negative Cosine Similarity for Phase-Aware Variance
                 cos_sim = torch.nn.functional.cosine_similarity(predictions, mean_pred.expand_as(predictions), dim=-1)
                 disagreement = -cos_sim.mean().item()
                 
            proposed_experiments.append({
                'agent': agent,
                'exp_x': exp_x,
                'goal_state': goal_state,
                'disagreement_score': disagreement
            })

        # 3. The Epistemic Auction (Fixing the Averaging Paradox)
        # Select the single experiment that causes the most uncertainty across the swarm
        winning_proposal = max(proposed_experiments, key=lambda x: x['disagreement_score'])
        winning_action = winning_proposal['exp_x']
        max_disagreement = winning_proposal['disagreement_score']

        # 4. Hardware Actuation Mapping
        # Clamp safely and reshape from 128-D back to 64-D Complex for the optical core
        safe_action = torch.clamp(winning_action, min=-0.5, max=0.5) 
        action_real, action_imag = torch.chunk(safe_action, 2, dim=-1)
        complex_actuation = torch.complex(action_real, action_imag)

        bypassed_physical = False

        # 5. Hardware Bypass & Empirical Assimilation Logic
        if max_disagreement >= 0.02:
            # EXECUTE PHYSICAL HARDWARE
            print(f"[SWARM] Theories disagree (Max Disagreement: {max_disagreement:.4f}). Triggering physical experiment...")
            
            # Project complex actuation back to bulk 4096-D wavefront space
            psi_modulated = torch.mv(torch.conj(self.boundary_validator.P.T), complex_actuation)
            psi_modulated_np = psi_modulated.detach().numpy().astype(np.complex64)

            # Fire the bulk wave into Zone B physical emulator (D2NN layers)
            truth_np, delta_np, alignment = self.optical_core.forward(
                hr_wavefront=psi_modulated_np, 
                target_manifold=target_np,
                langevin_heat=0.0
            )
            
            # Map physical wave back through manifold
            truth_tensor = torch.tensor(truth_np, dtype=torch.complex64)
            is_valid, veto_reason, error_energy, h_cft = self.boundary_validator.validate_boundary(truth_tensor)
            
            next_real, next_imag = h_cft.real, h_cft.imag
            structured_next_state = torch.cat([next_real, next_imag], dim=-1).unsqueeze(0).detach()

            # ASSIMILATE ONLY ON TRUE GROUND TRUTH
            for prop in proposed_experiments:
                agent = prop['agent']
                concept_hash = str(agent.current_concept_focus)
                metrics = agent.assimilate_results(
                    structured_state, winning_action, structured_next_state, prop['goal_state'], concept_hash
                )
                
                # Record empirical observation
                agent.scientist.empirical_observations.append((winning_action, structured_next_state))
                if len(agent.scientist.empirical_observations) % 5 == 0:
                    agent.scientist.run_discovery_cycle(structured_next_state)

                # If learning progress stalls, shift Vygotskian focus (curriculum climbing)
                if metrics["learning_progress"] < 0.01:
                    agent.current_concept_focus = (
                        (agent.current_concept_focus[0] + 1) % 10,
                        (agent.current_concept_focus[1] + 1) % 10
                    )
        else:
            # BYPASS PHYSICAL HARDWARE (The Epistemic Clutch)
            # Use the mean prediction of the swarm to simulate the next state
            print(f"[SWARM] Epistemic agreement achieved (Max Disagreement: {max_disagreement:.4f}). Bypassing physical execution to save hardware wear.")
            bypassed_physical = True
            
            all_preds = torch.stack([c(winning_action) for c in callables])
            structured_next_state = all_preds.mean(dim=0).detach()
            
            # Reconstruct bulk wave from CFT boundary wave
            real_part, imag_part = torch.chunk(structured_next_state.squeeze(0), 2, dim=-1)
            h_cft = torch.complex(real_part, imag_part)
            truth_tensor = torch.mv(torch.conj(self.boundary_validator.P.T), h_cft)
            
            truth_np = truth_tensor.detach().numpy().astype(np.complex64)
            delta_np = np.zeros_like(truth_np)
            alignment = np.ones(1)
            is_valid = True
            veto_reason = None
            error_energy = 0.0

        if not is_valid:
            print(f"[!] SAGNAC VETO: {veto_reason} | Error Energy: {error_energy:.4f}")
            
            # Langevin Shockwave thermal injection
            langevin_heat = 0.8
            self.optical_core.apply_langevin_noise(langevin_heat)
            
            # Convert alignment vector to a single scalar score
            alignment_scalar = alignment.mean().item() if isinstance(alignment, (np.ndarray, torch.Tensor)) else alignment
            
            # Update Active Neumann Boundary CFT sector
            delta_tensor = torch.tensor(delta_np, dtype=torch.complex64)
            delta_cft = self.boundary_validator.bulk_to_boundary(delta_tensor)
            self.boundary_validator.update_neumann_boundary(delta_cft, alignment_scalar)
            
            # Steer Swarm LoRA weights using the error delta
            for i in range(self.num_streams):
                self.lora_managers[i].update_with_rehypothecated_tensors(delta_np, alignment_scalar)
                
            # Consolidate dynamic LoRA adapter weights in TimescaleDB
            self.synaptic_manager.consolidate_and_save_adapter(
                domain_tag=target_label,
                lora_manager=self.lora_managers[0],
                error_delta=error_energy
            )
            
            # Update thread-safe telemetry register
            truth_tensor_2d = truth_tensor.reshape(64, 64)
            telemetry_register.update(
                active_tiles=[True] * 16,
                coupling=1.0,
                veto_intensity=error_energy,
                langevin_heat=langevin_heat,
                status="VETOED",
                error_energy=error_energy,
                lora_scale=1.0,
                phase_data=torch.angle(truth_tensor_2d),
                intensity_data=torch.abs(truth_tensor_2d) ** 2
            )
                
            return {
                "status": "VETOED",
                "reason": veto_reason,
                "heat": langevin_heat,
                "error": error_energy
            }
        else:
            # 4. Success state: Hopfield Network semantic cleanup back into English
            if bypassed_physical:
                print(f"[+] Dirichlet Boundary Bypassed (Consensus Achieved). Running Hopfield cleanup...")
            else:
                print(f"[+] Dirichlet Boundary Verified. Sagnac Delta: {error_energy:.4f}. Running Hopfield cleanup...")
            clean_wave, best_concept, confidence = self.hopfield.cleanup(truth_tensor)
            
            print(f"[+] Hopfield Cleanup Resolved: Concept '{best_concept}' (Confidence: {confidence * 100:.2f}%)")
            
            # Consolidate dynamic LoRA adapter weights in TimescaleDB
            self.synaptic_manager.consolidate_and_save_adapter(
                domain_tag=target_label,
                lora_manager=self.lora_managers[0],
                error_delta=error_energy
            )
            
            # Update thread-safe telemetry register
            truth_tensor_2d = truth_tensor.reshape(64, 64)
            telemetry_register.update(
                active_tiles=[True] * 16,
                coupling=1.0,
                veto_intensity=0.0,
                langevin_heat=0.0,
                status="CONVERGED" if not bypassed_physical else "BYPASSED",
                error_energy=error_energy,
                lora_scale=1.0,
                phase_data=torch.angle(truth_tensor_2d),
                intensity_data=torch.abs(truth_tensor_2d) ** 2
            )
            
            return {
                "status": "CONVERGED",
                "concept": best_concept,
                "confidence": confidence,
                "error": error_energy
            }

    def start_swarm_loop(self, initial_prompts, interval=0.25, target_label="SCADA_Pressure_Control"):
        """Starts the timed background loop."""
        for i in range(self.num_streams):
            self.stream_prompts[i] = initial_prompts.get(i, "Solve SCADA thermodynamic pressure equations.")
            # Route and load specialized domain adapter
            self.synaptic_manager.route_and_load_adapter(target_label, self.lora_managers[i])
        
        # Start telemetry server dynamically
        self.start_telemetry_server()

        self.stop_loop.clear()
        self.timed_loop_thread = threading.Thread(
            target=self.run_continuous_wave_timed_loop, 
            args=(interval,), 
            daemon=True
        )
        self.timed_loop_thread.start()

    def stop_swarm_loop(self):
        """Stops the timed background loop."""
        self.stop_loop.set()
        if self.timed_loop_thread and self.timed_loop_thread.is_alive():
            self.timed_loop_thread.join(timeout=2.0)
        print("[SWARM LOOP] Timed loop stopped.")
        
        # Shutdown telemetry server gracefully
        if hasattr(self, "telemetry_server") and self.telemetry_server:
            self.telemetry_server.stop()
            self.telemetry_server = None

    def start_telemetry_server(self):
        """Starts the telemetry server on port 8000 if not already running."""
        if not hasattr(self, "telemetry_server") or self.telemetry_server is None:
            self.telemetry_server = NonBlockingTelemetryServer(host='0.0.0.0', port=8000)
            self.telemetry_server.start()



class AletheiaAgent:
    """
    Implements Google's Aletheia math research agent loop.
    Contains three sub-agents: Generator, Verifier, and Reviser.
    Prevents logical cheating by executing and verifying all steps statefully.
    """
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.current_temperature = 0.4

    def generate(self, prompt, history=[]):
        """Generator Sub-agent: Generates a candidate solution using CoT and tool execution."""
        messages = [
            {"role": "system", "content": (
                "You are the Generator sub-agent. Parse the mathematical/physics problem "
                "down into digestible steps. Show your detailed Chain-of-Thought (CoT) reasoning. "
                "If symbolic or tensor algebra calculation is required (such as deriving Fefferman-Graham or Weyl expansions), "
                "you must delegate it by writing a Python block enclosed in <|python_begin|> and <|python_end|> tags. "
                "For example, write code using SymPy or Cadabra to algebraically perform the tensor expansions and print the resulting coefficients. "
                "The sandbox will execute the code and return the output. Then, you must parse the execution output and write the final def answer() function. "
                "DO NOT guess, bluff, or invent results. Every step must be self-contained."
            )}
        ]
        for h in history:
            messages.append(h)
        messages.append({"role": "user", "content": prompt})

        # Loop to allow multi-turn symbolic execution (tool use)
        max_tool_turns = 3
        for turn in range(max_tool_turns):
            res = self.orchestrator.gen_model.create_chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=self.current_temperature
            )
            content = res["choices"][0]["message"]["content"]
            
            # Check if there is a python block to execute that is NOT just the final answer
            if "<|python_begin" in content and "<|python_end|>" in content:
                idx_begin = content.find("<|python_begin")
                idx_end = content.find("<|python_end|>")
                idx_close_bracket = content.find("|>", idx_begin)
                if idx_close_bracket != -1 and idx_close_bracket < idx_end:
                    code_block = content[idx_close_bracket + 2 : idx_end].strip()
                    
                    # If this block is only defining answer(), it is the final solution, so return it directly
                    if "def answer(" in code_block and "return " in code_block:
                        return content
                        
                    # Run code block statefully
                    exec_res = self.orchestrator.repl.execute_block(code_block)
                    stdout = exec_res["stdout"].strip()
                    stderr = exec_res["stderr"].strip()
                    output_content = stdout if exec_res["success"] else f"Error: {stderr or exec_res['error_message']}"
                    
                    # Append assistant message and tool response message
                    messages.append({"role": "assistant", "content": content})
                    output_tag = f"<|output_begin|>\n{output_content}\n<|output_end|>"
                    messages.append({"role": "user", "content": f"Sandbox Execution Output:\n{output_tag}\n\nBased on these results, please complete your derivation and write the final solution with def answer() return values."})
                    print(f"[TOOL USE] Executed sandbox code block. Output:\n{output_content[:300]}...")
                    continue
            
            return content

    def verify(self, candidate, target_label="SCADA_Pressure_Control") -> tuple:
        """
        Verifier Sub-agent: Tests code in stateful REPL, projects logic
        to VSA wave space, and validates Dirichlet/Neumann boundaries.
        """
        has_code = False
        repl_output = ""
        repl_success = True
        
        if "<|python_begin" in candidate and "<|python_end|>" in candidate:
            has_code = True
            idx_begin = candidate.find("<|python_begin")
            idx_end = candidate.find("<|python_end|>")
            idx_close_bracket = candidate.find("|>", idx_begin)
            if idx_close_bracket != -1 and idx_close_bracket < idx_end:
                code_block = candidate[idx_close_bracket + 2 : idx_end].strip()
                res = self.orchestrator.repl.execute_block(code_block)
                repl_success = res["success"]
                if res["success"]:
                    repl_output = res["stdout"].strip()
                else:
                    repl_output = res["error_message"].strip() or res["stderr"].strip()

        if has_code and not repl_success:
            return False, f"REPL Execution Error: The code block failed with error: {repl_output}", None

        # 2. Project candidate solution to 4096-D complex wave using L3 Router
        emb_res = self.orchestrator.base_model.create_embedding(candidate)
        h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
        h_7b_lora = self.orchestrator.lora_managers[0].apply_lora(h_7b_raw)
        if len(h_7b_lora.shape) == 2:
            h_7b_lora = torch.mean(h_7b_lora, dim=0)
            
        # Check if the task is a strict mathematical or symbolic derivation
        is_symbolic_derivation = any(kw in target_label.lower() or kw in candidate.lower() for kw in ["symbolic", "mathematical", "derivation", "weyl", "anomaly", "expansion", "coefficient", "thermodynamic", "conservation", "coeff"])
        
        if is_symbolic_derivation:
            print(f"[SWARM] Bypassing tiled wave superposition for Strict Symbolic Derivation ('{target_label}'). Routing to isolated high-fidelity stream.")
            psi_candidate_focused = self.orchestrator.l3_router.activation_to_wave(h_7b_lora)
            if len(psi_candidate_focused.shape) == 2:
                psi_candidate_focused = torch.mean(psi_candidate_focused, dim=0)
            psi_candidate_focused = psi_candidate_focused.reshape(64, 64)
        else:
            # Replicate candidate activation across all 16 streams to create [16, 1, gemma_dim]
            activations_stack = h_7b_lora.unsqueeze(0).unsqueeze(0).repeat(16, 1, 1)
            psi_candidate, _, _ = self.orchestrator.l3_router(activations=activations_stack)
            
            # Apply lens downsampling
            N = psi_candidate.size(-1)
            x = torch.linspace(-1.0, 1.0, N, device=psi_candidate.device)
            y = torch.linspace(-1.0, 1.0, N, device=psi_candidate.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            lens_phase = -50.0 * (X**2 + Y**2)
            lens = torch.polar(torch.ones_like(lens_phase), lens_phase)
            
            wave_lensed = psi_candidate.squeeze(0) * lens
            focal_plane = torch.fft.fft2(wave_lensed, norm='ortho')
            focal_plane_shifted = torch.fft.fftshift(focal_plane)
            
            start = (N - 64) // 2
            end = start + 64
            focused_64 = focal_plane_shifted[start:end, start:end]
            
            mags = torch.abs(focused_64)
            mags = torch.clamp(mags, min=1e-8)
            psi_candidate_focused = focused_64 / mags

        # 3. Fire the wave in Zone B
        target_vector = self.orchestrator.hopfield.vocabulary.get(target_label)
        if target_vector is None:
            target_vector = self.orchestrator.get_stream_address(0)
            
        target_np = target_vector.detach().numpy().astype(np.complex64)
        psi_candidate_flat = psi_candidate_focused.flatten()
        # Query memory cache for stream 0 and blend history
        retrieved_wave = self.orchestrator.memory_engines[0].retrieve_from_cache(query_key=psi_candidate_flat)
        blended_focused = psi_candidate_flat + retrieved_wave
        blended_mags = torch.abs(blended_focused).clamp(min=1e-8)
        psi_candidate_resolved = blended_focused / blended_mags
        
        psi_cand_np = psi_candidate_resolved.detach().numpy().astype(np.complex64)

        truth_np, delta_np, alignment = self.orchestrator.optical_core.forward(
            hr_wavefront=psi_cand_np,
            target_manifold=target_np,
            langevin_heat=0.0
        )

        # 4. Boundary Validation
        truth_tensor = torch.tensor(truth_np, dtype=torch.complex64)
        is_valid, veto_reason, error_energy, h_cft = self.orchestrator.boundary_validator.validate_boundary(truth_tensor)

        if not is_valid:
            feedback = f"Sagnac Veto: The candidate logic violated Dirichlet boundary axioms. Reason: {veto_reason} | Error Energy: {error_energy:.4f}"
            # Update thread-safe telemetry register
            truth_tensor_2d = truth_tensor.reshape(64, 64)
            telemetry_register.update(
                active_tiles=[True] * 16,
                coupling=1.0,
                veto_intensity=error_energy,
                langevin_heat=0.8,
                status="VETOED",
                error_energy=error_energy,
                lora_scale=1.0,
                phase_data=torch.angle(truth_tensor_2d),
                intensity_data=torch.abs(truth_tensor_2d) ** 2
            )
            return False, feedback, delta_np
            
        # Update thread-safe telemetry register
        truth_tensor_2d = truth_tensor.reshape(64, 64)
        telemetry_register.update(
            active_tiles=[True] * 16,
            coupling=1.0,
            veto_intensity=0.0,
            langevin_heat=0.0,
            status="CONVERGED",
            error_energy=error_energy,
            lora_scale=1.0,
            phase_data=torch.angle(truth_tensor_2d),
            intensity_data=torch.abs(truth_tensor_2d) ** 2
        )
        return True, "Dirichlet boundaries verified. Sagnac alignment achieved.", delta_np

    def revise(self, prompt, candidate, feedback) -> str:
        """Reviser Sub-agent: Corrects the candidate solution based on feedback."""
        messages = [
            {"role": "system", "content": (
                "You are the Reviser sub-agent. Your role is to correct the candidate solution "
                "based on the feedback and errors provided by the Verifier. "
                "Ensure you correct all python syntax/logical errors, do not guess, "
                "and adhere strictly to mathematical invariants. Output the updated solution."
            )},
            {"role": "user", "content": f"Problem: {prompt}\n\nCandidate Solution:\n{candidate}\n\nVerifier Feedback:\n{feedback}"}
        ]
        res = self.orchestrator.gen_model.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=self.current_temperature
        )
        return res["choices"][0]["message"]["content"]

    def execute_reasoning_loop(self, prompt, target_label="SCADA_Pressure_Control", max_revisions=3):
        """Orchestrates Generator, Verifier, and Reviser in a closed cognitive loop."""
        # Route and load specialized domain adapter
        self.orchestrator.synaptic_manager.route_and_load_adapter(target_label, self.orchestrator.lora_managers[0])
        
        self.current_temperature = 0.4 # Reset to baseline
        history = []
        candidate = self.generate(prompt, history)
        print(f"\n[Aletheia Agent] Candidate generated:\n---Candidate Begin---\n{candidate}\n---Candidate End---")
        
        for revision in range(1, max_revisions + 1):
            is_valid, feedback, delta_np = self.verify(candidate, target_label)
            print(f"[Aletheia Agent - Revision {revision}] Verification: {'PASS' if is_valid else 'FAIL'} | Feedback: {feedback}")
            
            if is_valid:
                # Consolidate and save adapter when verified successfully
                error_val = 0.0
                self.orchestrator.synaptic_manager.consolidate_and_save_adapter(
                    domain_tag=target_label, 
                    lora_manager=self.orchestrator.lora_managers[0], 
                    error_delta=error_val
                )
                self.current_temperature = 0.4 # Reset to baseline
                return candidate, revision, "CONVERGED"
                
            # If invalid, update the LoRA weights to "bend" future reasoning vectors
            if delta_np is not None:
                alignment_scalar = 0.1
                for i in range(self.orchestrator.num_streams):
                    self.orchestrator.lora_managers[i].update_with_rehypothecated_tensors(delta_np, alignment_scalar)
                    
                # Consolidate dynamic LoRA adapter weights in TimescaleDB
                error_val = 0.5
                try:
                    import re
                    match = re.search(r"Error Energy:\s*([\d\.]+)", feedback)
                    if match:
                        error_val = float(match.group(1))
                except:
                    pass
                self.orchestrator.synaptic_manager.consolidate_and_save_adapter(
                    domain_tag=target_label, 
                    lora_manager=self.orchestrator.lora_managers[0], 
                    error_delta=error_val
                )
                
                # Langevin Temperature Boosting: Boost temperature proportionally to Sagnac error energy
                boost = min(0.5, error_val * 0.5)
                self.current_temperature = min(1.0, 0.4 + boost)
                print(f"[LANGEVIN BOOST] Sagnac Veto detected (Error Energy: {error_val:.4f}). Boosting temperature to {self.current_temperature:.2f} for revision {revision}.")
                    
            # Reviser phase
            candidate = self.revise(prompt, candidate, feedback)
            print(f"\n[Aletheia Agent - Revision {revision}] Revised Candidate:\n---Revised Begin---\n{candidate}\n---Revised End---")
            
        return candidate, max_revisions, "TIMEOUT"
