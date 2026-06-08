import os
import sys

# Force llama.cpp to bypass Vulkan and execute purely on CPU to prevent VRAM KV cache crashes
os.environ["GGML_VULKAN_DISABLE"] = "1"
os.environ["GGML_DISABLE_VULKAN"] = "1"
os.environ["GGML_OPENCL_DISABLE"] = "1"

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

def pin_current_thread_to_core_7():
    """
    Pins the current execution thread to logical processors representing physical Core 7 (for HRR math).
    Adapts based on total system logical processor count to prevent out-of-bounds masks.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            num_logical = psutil.cpu_count(logical=True)
            if num_logical >= 16:
                # 8-core CPU with SMT: physical Core 7 is logical 14, 15
                mask = (1 << 14) | (1 << 15)
                cores_str = "14, 15"
                num_threads = 2
            elif num_logical >= 8:
                # 4-core SMT or 8-core without SMT: pin to last core
                mask = (1 << 6) | (1 << 7)
                cores_str = "6, 7"
                num_threads = 2
            else:
                # Small CPU: pin to last logical core
                mask = (1 << (num_logical - 1))
                cores_str = str(num_logical - 1)
                num_threads = 1
            
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadAffinityMask(handle, ctypes.c_size_t(mask))
            torch.set_num_threads(num_threads)
        except Exception as e:
            pass

class LookaheadExpertPrefetcher:
    """
    Asynchronously prefetches chunks of the model file into physical RAM (OS Page Cache)
    to mask NVMe-to-RAM page fault latency during token generation cycles.
    """
    def __init__(self, model_path, file_size):
        self.model_path = model_path
        self.file_size = file_size
        self.prefetch_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.prefetch_thread = None
        
        if os.path.exists(model_path):
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self.prefetch_thread.start()

    def _prefetch_worker(self):
        try:
            # Open GGUF file in binary read mode
            fd = os.open(self.model_path, os.O_RDONLY | getattr(os, 'O_BINARY', 0))
        except Exception as e:
            print(f"[PREFETCH] Failed to open model file for reading: {e}")
            return
            
        while not self.stop_event.is_set():
            try:
                # Wait for prefetch requests (offset, size)
                item = self.prefetch_queue.get(timeout=0.1)
                offset, size = item
                
                # Align offset and size to 64KB boundary for Windows performance
                offset = (offset // 65536) * 65536
                size = ((size + 65535) // 65536) * 65536
                
                if offset + size > self.file_size:
                    size = self.file_size - offset
                
                if size > 0:
                    # Perform file read to populate OS cache
                    os.lseek(fd, offset, os.SEEK_SET)
                    bytes_to_read = size
                    while bytes_to_read > 0 and not self.stop_event.is_set():
                        chunk_size = min(4 * 1024 * 1024, bytes_to_read)
                        _ = os.read(fd, chunk_size)
                        bytes_to_read -= chunk_size
                self.prefetch_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                pass
                
        try:
            os.close(fd)
        except:
            pass

    def trigger_prefetch_for_experts(self, token_ids):
        """
        Translates projected expert selections/token IDs into file block offsets
        and pushes them to the prefetch queue.
        """
        if not os.path.exists(self.model_path) or self.file_size <= 0:
            return
            
        for tok in token_ids:
            expert_id = hash(tok) % 8
            block_size = 50 * 1024 * 1024  # 50 MB block
            offset = int((self.file_size * 0.2) + expert_id * (self.file_size * 0.08) + (tok % 10) * block_size)
            if offset + block_size <= self.file_size:
                self.prefetch_queue.put((offset, block_size))

    def stop(self):
        self.stop_event.set()
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)

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


class SteeredLlama:
    def __init__(self, llama_instance, orchestrator):
        self.llama = llama_instance
        self.orchestrator = orchestrator

    def __getattr__(self, name):
        return getattr(self.llama, name)

    def __call__(self, prompt, *args, **kwargs):
        dynamic_lora_weights = kwargs.pop("dynamic_lora_weights", None)
        logits_processor = kwargs.pop("logits_processor", None)
        
        if dynamic_lora_weights is not None:
            # 1. Retrieve the base embedding of the prompt
            emb_res = self.orchestrator.base_model.create_embedding(prompt)
            h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
            
            # 2. Compute expert residuals and blend them
            with torch.no_grad():
                device = self.orchestrator.l3_router.w_down.weight.device
                h_7b_raw = h_7b_raw.to(device)
                
                blended_residual = torch.zeros_like(h_7b_raw)
                for idx in range(self.orchestrator.num_streams):
                    w = dynamic_lora_weights[idx].item() if torch.is_tensor(dynamic_lora_weights) else dynamic_lora_weights[idx]
                    if w > 0.0001:
                        manager = self.orchestrator.lora_managers[idx]
                        lora_A = manager.lora_A.to(device)
                        lora_B = manager.lora_B.to(device)
                        residual_i = torch.matmul(torch.matmul(h_7b_raw, lora_A), lora_B)
                        blended_residual += w * residual_i
                
                # 3. Project blended residual to logit bias space
                h_1024 = torch.matmul(blended_residual, self.orchestrator.l3_router.activation_projection.weight.T)
                logit_bias = torch.matmul(h_1024, self.orchestrator.l3_router.token_embedding.weight.T)
                logit_bias_np = logit_bias.cpu().numpy()
                
            # 4. Create custom logits processor to apply logit bias
            def steered_logits_processor(input_ids, logits):
                n_vocab_model = logits.shape[-1]
                n_vocab_bias = len(logit_bias_np)
                if n_vocab_model == n_vocab_bias:
                    logits[:] = logits + logit_bias_np
                elif n_vocab_model > n_vocab_bias:
                    padded_bias = np.zeros(n_vocab_model, dtype=np.float32)
                    padded_bias[:n_vocab_bias] = logit_bias_np
                    logits[:] = logits + padded_bias
                else:
                    logits[:] = logits + logit_bias_np[:n_vocab_model]
                return logits
                
            import llama_cpp
            new_lp = steered_logits_processor
            if logits_processor is None:
                logits_processor = llama_cpp.LogitsProcessorList([new_lp])
            else:
                logits_processor.append(new_lp)
                
            print(f"[STEERING] Steered generation model using routing weights: {[round(float(w), 4) for w in dynamic_lora_weights.tolist()] if torch.is_tensor(dynamic_lora_weights) else [round(w, 4) for w in dynamic_lora_weights]}")
            
        return self.llama(prompt, *args, logits_processor=logits_processor, **kwargs)

    def create_chat_completion(self, messages, *args, **kwargs):
        dynamic_lora_weights = kwargs.pop("dynamic_lora_weights", None)
        logits_processor = kwargs.pop("logits_processor", None)
        
        if dynamic_lora_weights is not None:
            prompt_text = "".join(m["content"] for m in messages)
            
            # 1. Retrieve the base embedding of the prompt
            emb_res = self.orchestrator.base_model.create_embedding(prompt_text)
            h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
            
            # 2. Compute expert residuals and blend them
            with torch.no_grad():
                device = self.orchestrator.l3_router.w_down.weight.device
                h_7b_raw = h_7b_raw.to(device)
                
                blended_residual = torch.zeros_like(h_7b_raw)
                for idx in range(self.orchestrator.num_streams):
                    w = dynamic_lora_weights[idx].item() if torch.is_tensor(dynamic_lora_weights) else dynamic_lora_weights[idx]
                    if w > 0.0001:
                        manager = self.orchestrator.lora_managers[idx]
                        lora_A = manager.lora_A.to(device)
                        lora_B = manager.lora_B.to(device)
                        residual_i = torch.matmul(torch.matmul(h_7b_raw, lora_A), lora_B)
                        blended_residual += w * residual_i
                
                # 3. Project blended residual to logit bias space
                h_1024 = torch.matmul(blended_residual, self.orchestrator.l3_router.activation_projection.weight.T)
                logit_bias = torch.matmul(h_1024, self.orchestrator.l3_router.token_embedding.weight.T)
                logit_bias_np = logit_bias.cpu().numpy()
                
            # 4. Create custom logits processor to apply logit bias
            def steered_logits_processor(input_ids, logits):
                n_vocab_model = logits.shape[-1]
                n_vocab_bias = len(logit_bias_np)
                if n_vocab_model == n_vocab_bias:
                    logits[:] = logits + logit_bias_np
                elif n_vocab_model > n_vocab_bias:
                    padded_bias = np.zeros(n_vocab_model, dtype=np.float32)
                    padded_bias[:n_vocab_bias] = logit_bias_np
                    logits[:] = logits + padded_bias
                else:
                    logits[:] = logits + logit_bias_np[:n_vocab_model]
                return logits
                
            import llama_cpp
            new_lp = steered_logits_processor
            if logits_processor is None:
                logits_processor = llama_cpp.LogitsProcessorList([new_lp])
            else:
                logits_processor.append(new_lp)
                
            print(f"[STEERING] Steered chat completion using routing weights: {[round(float(w), 4) for w in dynamic_lora_weights.tolist()] if torch.is_tensor(dynamic_lora_weights) else [round(w, 4) for w in dynamic_lora_weights]}")
            
        return self.llama.create_chat_completion(messages, *args, logits_processor=logits_processor, **kwargs)


class HenriCognitiveSwarmOrchestrator:
    """
    Ties together Zone A (Gemma E4B RAM Swarm + LoRAs), the L3 SRAM Cache Router,
    Zone B (D2NN Physics + Sagnac boundary validation), and Zone C (Hopfield Cleanup + REPL).
    Synchronizes the asynchronous timed loop to fire in series with the RAM cycles
    to build a continuous, coherent HRR vector stream.
    """
    def __init__(self, model_path="Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf", num_streams=16, gemma_dim=2048, hrr_dim=4096):
        self.num_streams = num_streams
        self.gemma_dim = gemma_dim
        self.hrr_dim = hrr_dim
        
        # 0. Synaptic Consolidation Manager
        self.synaptic_manager = SynapticConsolidationManager()
        self.gemma_dim = gemma_dim
        self.hrr_dim = hrr_dim
        
        # Prefer 12B model, then fall back to older ones
        self.model_path = model_path
        if model_path in ["Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf", "gemma-4-E4B-it-Q4_0.gguf", "gemma-4-26B-A4B-it-uncensored-Q8_0.gguf", "mock_only.gguf"]:
            for c_path in ["Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf", "gemma-4-26B-A4B-it-uncensored-Q8_0.gguf", "gemma-4-E4B-it-Q4_0.gguf"]:
                if os.path.exists(c_path):
                    self.model_path = c_path
                    break

        # 1. Core Pinning & Affinity setup
        self.set_core_affinity()

        # 2. Try to expand Windows process working set size based on GGUF sizes
        if sys.platform == "win32":
            try:
                import ctypes
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                
                # Determine paths for working set computation
                parent_dir = Path(__file__).parent
                gen_model_path = str(parent_dir / self.model_path)
                emb_model_path = str(parent_dir / "Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf")
                if not os.path.exists(emb_model_path):
                    emb_model_path = gen_model_path
                
                # Sum the file sizes
                model_size = 0
                if os.path.exists(gen_model_path) and self.model_path != "mock_only.gguf":
                    model_size += os.path.getsize(gen_model_path)
                if os.path.exists(emb_model_path) and emb_model_path != gen_model_path and emb_model_path != "mock_only.gguf":
                    model_size += os.path.getsize(emb_model_path)
                    
                if model_size == 0:
                    model_size = 6 * 1024 * 1024 * 1024
                
                # Add 4 GB safety margin for context window, L3 router, and PyTorch tensors
                min_size = model_size + 4 * 1024 * 1024 * 1024
                max_size = model_size + 8 * 1024 * 1024 * 1024
                
                # SetProcessWorkingSetSize returns non-zero on success
                ret = ctypes.windll.kernel32.SetProcessWorkingSetSize(
                    handle, 
                    ctypes.c_size_t(min_size), 
                    ctypes.c_size_t(max_size)
                )
                if ret != 0:
                    print(f"[HARDWARE] Windows process working set expanded successfully to {min_size/(1024**3):.2f}GB-{max_size/(1024**3):.2f}GB based on model sizes.")
                else:
                    print("[HARDWARE] Warning: Windows SetProcessWorkingSetSize returned failure code.")
            except Exception as e:
                print(f"[HARDWARE] Warning: Failed to adjust Windows working set size: {e}")

        # Initialize Lookahead Expert Prefetcher
        if os.path.exists(self.model_path) and self.model_path != "mock_only.gguf":
            file_size = os.path.getsize(self.model_path)
            self.prefetcher = LookaheadExpertPrefetcher(self.model_path, file_size)
        else:
            self.prefetcher = None

        # Determine if we should attempt memory locking (mlock)
        attempt_mlock = (sys.platform != "win32")

        # 2b. Initialize Gemma models strictly in RAM (CPU, use_mlock, use_mmap=True)
        self.base_model = None
        self.gen_model = None
        self.is_mock = False
        
        # Determine paths for generation and embedding models to prevent memory-slot decode issues.
        # Autoregressive generation fails when sharing a context with embedding=True in llama.cpp-python.
        # To strictly use the new 12B model, we point both pathways to the same 12B model file.
        # Under use_mmap=True, weights are shared in page cache, but context allocations are separate.
        parent_dir = Path(__file__).parent
        gen_model_path = str(parent_dir / self.model_path)
        emb_model_path = str(parent_dir / "Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf")
        
        if not os.path.exists(emb_model_path):
            emb_model_path = gen_model_path # Fallback to same model if not present
            
        # Calculate llama.cpp threads (Cores 0-6 for logical 16 core CPU)
        num_logical = psutil.cpu_count(logical=True)
        llama_threads = 7 if num_logical >= 16 else max(1, num_logical - 1)
        
        if HAS_LLAMA_CPP and os.path.exists(gen_model_path) and self.model_path != "mock_only.gguf":
            try:
                # 1. Load the Generation Engine (embedding=False)
                print(f"[SYSTEM] Loading Generation engine via mmap from: {gen_model_path}")
                self.gen_model = llama_cpp.Llama(
                    model_path=gen_model_path,
                    n_ctx=32768,       # Scale up from 16384 to allow active revision loops
                    n_batch=512,       # Increase processing throughput for dense prompt evals
                    n_threads=llama_threads,
                    embedding=False,   # Must be False to allow KV cache decoding slot allocation
                    use_mmap=True,
                    use_mlock=attempt_mlock,
                    n_gpu_layers=0
                )
                
                # 2. Load the Embedding Engine (embedding=True)
                if os.path.exists(emb_model_path) and emb_model_path != "mock_only.gguf":
                    print(f"[SYSTEM] Loading Embedding engine via mmap from: {emb_model_path}")
                    self.base_model = llama_cpp.Llama(
                        model_path=emb_model_path,
                        n_ctx=8192,      # Increased to 8192 to support embedding candidate solutions with long reasoning chains
                        n_batch=32,
                        n_threads=llama_threads,
                        embedding=True,   # Must be True for create_embedding to work
                        use_mmap=True,
                        use_mlock=attempt_mlock,
                        n_gpu_layers=0
                    )
                else:
                    print(f"[SYSTEM] Embedding engine '{emb_model_path}' not found. Sharing generation context (limited mode).")
                    self.base_model = self.gen_model
                    
                print("[SYSTEM] Model contexts loaded successfully.")
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

        # Wrap gen_model and base_model with SteeredLlama to support in-memory MoE steering
        if self.gen_model is not None:
            self.gen_model = SteeredLlama(self.gen_model, self)
        if self.base_model is not None:
            self.base_model = SteeredLlama(self.base_model, self)

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

        # Load expert centroids from disk if present
        self.load_router_centroids()

        # 5. Initialize Zone B D2NN model
        # Use GPU (CUDA/DirectML) for Zone B physical emulation if available to offload heavy wave propagation
        try:
            import torch_directml
            d2nn_device = torch_directml.device()
        except ImportError:
            d2nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optical_core = HenriOpticalCoreD2NN(num_channels=self.hrr_dim, num_layers=5, device=d2nn_device)

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

        # 10. Set all PyTorch modules to evaluation mode to prevent graph/gradient tracking and on-line routing training
        self.l3_router.eval()
        self.boundary_validator.eval()
        self.agents.eval()
        if hasattr(self.optical_core, "emulator") and self.optical_core.emulator is not None:
            self.optical_core.emulator.eval()

        # 11. Warm up memory caches
        self.warm_up_caches()

        # 12. Active inference/benchmark session caches for predictive context management
        self.evicted_text_registry = {}  # mapping of axiom_label -> raw_text
        self.active_block_embeddings = {}  # mapping of block_identifier -> 4096-D complex wave tensor

    def set_core_affinity(self):
        """Pins process affinity to all logical cores to allow thread-level partitioning."""
        p = psutil.Process(os.getpid())
        try:
            num_logical = psutil.cpu_count(logical=True)
            p.cpu_affinity(list(range(num_logical)))
            print(f"[HARDWARE] Process affinity set to logical cores 0-{num_logical-1}.")
        except Exception as e:
            print(f"[WARNING] Process affinity setting failed: {e}")

    def warm_up_caches(self):
        """
        Runs dummy forward passes through the PyTorch networks (L3SwarmRouter, 
        HenriOpticalCoreD2NN, etc.) to warm up CPU L3 caches and load weights 
        from DRAM into local CPU caches.
        """
        print("[HARDWARE] Warming up CPU L3/V-Cache memory cache layers...")
        start_time = time.perf_counter()
        
        # 1. Warm up L3SwarmRouter (projects embedding activations to wavefront grid)
        # Dummy activations stack: shape [16, 1, gemma_dim]
        dummy_activations = torch.randn(16, 1, self.gemma_dim)
        for _ in range(5):
            with torch.no_grad():
                _ = self.l3_router(activations=dummy_activations)
        
        # 2. Warm up HenriOpticalCoreD2NN (Zone B optical core layers)
        # Input wavefront: shape [hrr_dim] (complex)
        dummy_wave = torch.complex(torch.randn(self.hrr_dim), torch.randn(self.hrr_dim))
        dummy_wave = dummy_wave / (torch.norm(dummy_wave) + 1e-8)
        dummy_wave_np = dummy_wave.numpy().astype(np.complex64)
        dummy_target_np = np.ones(self.hrr_dim, dtype=np.complex64)
        for _ in range(5):
            _ = self.optical_core.forward(
                hr_wavefront=dummy_wave_np,
                target_manifold=dummy_target_np,
                langevin_heat=0.0
            )
            
        # 3. Warm up BoundaryAxiomValidator shared manifold networks
        # Input: shape [1, bulk_dim] which is [1, boundary_dim * 2] = [1, 128]
        dummy_boundary = torch.randn(1, 128)
        for _ in range(5):
            with torch.no_grad():
                _ = self.boundary_validator.shared_manifold(dummy_boundary)
                
        # 4. Warm up Hopfield semantic cleanups
        # Hopfield cleanup query: shape [hrr_dim] complex
        for _ in range(5):
            _ = self.hopfield.cleanup(dummy_wave)
            
        elapsed = time.perf_counter() - start_time
        print(f"[HARDWARE] L3 V-Cache warming complete in {elapsed:.4f} seconds.")

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

    @torch.no_grad()
    def step_stream(self, stream_id: int, prompt: str) -> tuple:
        """Runs a single forward step on one stream thread, returning lensed 4096-D wave."""
        # Pin current thread to physical Core 7 for HRR math
        pin_current_thread_to_core_7()
        
        # Trigger predictive prefetch for upcoming tokens
        if self.prefetcher:
            pseudo_tokens = [ord(c) for c in prompt[:10]]
            self.prefetcher.trigger_prefetch_for_experts(pseudo_tokens)
            
        # 1. Retrieve the base embedding (Gemma hidden activations)
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
        
        return resolved_focused.detach().cpu(), h_7b_lora.detach().cpu()

    def run_continuous_wave_timed_loop(self, interval_seconds=0.25):
        """
        Asynchronous timed loop running on a background thread.
        Uses parallel tiled wave synthesis to directly generate the global 6324x6324 wave.
        """
        # Pin current thread to physical Core 7 for HRR math
        pin_current_thread_to_core_7()
        torch.set_grad_enabled(False)
        
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
                stream_activations.append(h_7b_lora.detach())
                
            # Step 2: Stack activations to shape [16, 1, gemma_dim]
            activations_stack = torch.stack(stream_activations).unsqueeze(1) # [16, 1, gemma_dim]
            
            # Step 3: Call L3 router to synthesize the global 6324x6324 wave directly
            global_wavefront, _, _ = self.l3_router(activations=activations_stack)
            psi_bulk = global_wavefront.squeeze(0).detach() # shape [6324, 6324]

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

    @torch.no_grad()
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
            
            # Project 4096-D complex wave to 3840-D real activations using L3 router
            delta_projected = self.l3_router.wave_to_activation(delta_np)
            
            # Steer Swarm LoRA weights using the error delta
            for i in range(self.num_streams):
                self.lora_managers[i].update_with_rehypothecated_tensors(delta_projected, alignment_scalar)
                
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
            
            # --- CENTROID DRIFT ANCHORING ---
            winner_idx = self.l3_router.update_expert_centroids(truth_tensor)
            print(f"[ANCHOR] Wave converged! Centroid of expert {winner_idx} drifted toward this concept topology.")
            self.save_router_centroids()
            with torch.no_grad():
                self.lora_managers[winner_idx].lora_A.copy_(self.lora_managers[0].lora_A)
                self.lora_managers[winner_idx].lora_B.copy_(self.lora_managers[0].lora_B)
            self.lora_managers[winner_idx].save_weights()
            
            # Consolidate dynamic LoRA adapter weights in database registry
            self.synaptic_manager.consolidate_and_save_adapter(
                domain_tag=f"Dynamic_Expert_{winner_idx}",
                lora_manager=self.lora_managers[winner_idx],
                error_delta=0.0
            )

            # Consolidate dynamic LoRA adapter weights in TimescaleDB for baseline target
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

    def save_router_centroids(self):
        try:
            os.makedirs("archive", exist_ok=True)
            torch.save(self.l3_router.expert_centroids, "archive/l3_router_centroids.bin")
            print("[SYSTEM] Saved expert centroids to archive/l3_router_centroids.bin")
        except Exception as e:
            print(f"[SYSTEM] Warning: Failed to save expert centroids: {e}")

    def load_router_centroids(self):
        path = "archive/l3_router_centroids.bin"
        if os.path.exists(path):
            try:
                loaded = torch.load(path, map_location="cpu")
                if loaded.shape == self.l3_router.expert_centroids.shape:
                    with torch.no_grad():
                        self.l3_router.expert_centroids.copy_(loaded)
                    print(f"[SYSTEM] Loaded expert centroids from {path}")
                else:
                    print(f"[SYSTEM] Warning: Shape mismatch for expert centroids in {path}")
            except Exception as e:
                print(f"[SYSTEM] Warning: Failed to load expert centroids: {e}")

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
        
        # Stop lookahead prefetcher
        if hasattr(self, "prefetcher") and self.prefetcher:
            self.prefetcher.stop()
        
        # Shutdown telemetry server gracefully
        if hasattr(self, "telemetry_server") and self.telemetry_server:
            self.telemetry_server.stop()
            self.telemetry_server = None

    def start_telemetry_server(self):
        """Starts the telemetry server on port 8000 if not already running."""
        if not hasattr(self, "telemetry_server") or self.telemetry_server is None:
            self.telemetry_server = NonBlockingTelemetryServer(host='0.0.0.0', port=8000)
            self.telemetry_server.start()

    def flush_cognitive_manifold(self):
        """
        Executes a Hard Manifold Reset to prevent context bloat, OOM timeouts,
        and cross-task/cross-turn causal leakage.
        """
        import gc
        import torch
        print("\n[SYSTEM] Executing Hard Manifold Reset...")

        # 1. Sever the KV Cache (The Semantic Guillotine)
        if hasattr(self, "gen_model") and self.gen_model is not None:
            if hasattr(self.gen_model, "reset"):
                try:
                    self.gen_model.reset()
                    print("  - KV cache reset.")
                except Exception as e:
                    print(f"  - Error resetting llama_cpp KV cache: {e}")

        # 2. Reset the HENRI Wave State & Memory Caches
        if hasattr(self, "memory_engines") and self.memory_engines:
            for engine in self.memory_engines.values():
                if hasattr(engine, "clear_cache"):
                    try:
                        engine.clear_cache()
                    except:
                        pass
                elif hasattr(engine, "active_memories"):
                    try:
                        engine.active_memories.clear()
                    except:
                        pass
            print("  - Cleared continuous wave-space active memories.")
            
        if hasattr(self, "stream_position_indices"):
            for k in self.stream_position_indices.keys():
                self.stream_position_indices[k] = 0

        # 3. Clear context containers and active queue references
        if hasattr(self, "stream_contexts") and self.stream_contexts:
            for k in list(self.stream_contexts.keys()):
                self.stream_contexts[k].clear()
            print("  - Cleared stream contexts.")

        if hasattr(self, "active_wave_queue") and self.active_wave_queue is not None:
            q_cleared = 0
            while not self.active_wave_queue.empty():
                try:
                    self.active_wave_queue.get_nowait()
                    q_cleared += 1
                except:
                    break
            if q_cleared > 0:
                print(f"  - Evicted {q_cleared} wavefronts from active queue.")

        # 4. Aggressive Garbage Collection and VRAM Purging
        import gc
        collected = gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        print(f"[SYSTEM] Manifold flushed. GC cleared {collected} objects.")

    def flush_lora_and_context_to_db(self, domain_tag="SCADA_Pressure_Control"):
        """
        Translates LoRA adapters' current state into 4096-D complex HRR waves using L3 router,
        inserts them into the hrr_canonical_lexicon table in TimescaleDB (Zone C) as boundary axioms,
        and executes a Hard Manifold Reset to wipe context window memory and stabilize RAM.
        """
        import uuid
        import psycopg
        import torch
        import numpy as np

        print(f"\n[SYSTEM] Flushing LoRA adapters & context to Zone C database for domain '{domain_tag}'...")

        # 1. Check if we have a connection to the database
        db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")
        
        # 2. Iterate through all streams to convert LoRA state to 4096D complex vectors
        for i in range(self.num_streams):
            try:
                lora_manager = self.lora_managers[i]
                
                # Probe vector: ones vector of size self.gemma_dim
                probe = torch.ones(self.gemma_dim, dtype=torch.float32)
                
                # Apply LoRA steering to the probe activation
                h_lora = lora_manager.apply_lora(probe).detach().cpu()
                
                # Run through the L3 SRAM model (l3_router) to translate to 4096-D complex wave
                with torch.no_grad():
                    psi = self.l3_router.activation_to_wave(h_lora).detach().cpu()
                    
                # Convert complex vector elements to database format (real / imag split)
                psi_np = psi.numpy()
                db_vec = np.zeros(4096, dtype=np.float32)
                # Map 2048 complex elements to 4096 real elements (first 2048 real, last 2048 imag)
                db_vec[:2048] = psi_np[:2048].real
                db_vec[2048:] = psi_np[:2048].imag
                
                vector_str = "[" + ",".join(map(str, db_vec)) + "]"
                
                semantic_label = f"LoRA_Stream_{i}_{domain_tag}_{int(time.time())}"
                concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, semantic_label))
                
                # Insert into hrr_canonical_lexicon table
                try:
                    with psycopg.connect(db_url, connect_timeout=3) as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront)
                                VALUES (%s, %s, %s, %s::vector)
                                ON CONFLICT (concept_hash) DO NOTHING;
                                """,
                                (concept_hash, semantic_label, domain_tag, vector_str),
                            )
                            # Register in Hopfield lexicon so it becomes immediately fetchable in RAM
                            mags = torch.abs(psi)
                            mags = torch.clamp(mags, min=1e-8)
                            self.hopfield.register_concept(semantic_label, psi / mags)
                except Exception as db_err:
                    print(f"  - Database insert failed for stream {i}: {db_err}. Registering in-memory only.")
                    mags = torch.abs(psi)
                    mags = torch.clamp(mags, min=1e-8)
                    self.hopfield.register_concept(semantic_label, psi / mags)
                    
                # Re-initialize the LoRA manager's weights to clear the memory footprint and start fresh
                lora_manager.lora_A = torch.randn(self.gemma_dim, lora_manager.rank) * 0.02
                lora_manager.lora_B = torch.zeros(lora_manager.rank, self.gemma_dim)
                lora_manager.save_weights()
                
                # Explicitly delete references
                del probe
                del h_lora
                del psi
                del psi_np
                del db_vec
                del vector_str
                
            except Exception as e:
                print(f"  - Error flushing LoRA stream {i}: {e}")

        # 3. Wipe the llama.cpp context windows (Hard Manifold Reset)
        self.flush_cognitive_manifold()
        print("[SYSTEM] LoRA adapters consolidated to Zone C database and contexts fully flushed.")

    def compact_prompt_history(self, prompt: str) -> str:
        """
        Compacts the prompt history when context window threshold is breached.
        Tokenizes the oldest segment, routes it through L3SwarmRouter to generate
        a 4096-D complex wave, evicts it to TimescaleDB as a boundary axiom,
        and replaces the segment in the prompt with a concise axiom reference.
        """
        import re
        import uuid
        import torch
        import numpy as np
        import psycopg
        import time

        print("[SYSTEM] Compacting prompt history...")

        # 1. Identify the block of text to evict
        pattern_demo = re.compile(
            r"(--- DEMONSTRATION PAIR \d+ ---\n.*?)(?=\n--- DEMONSTRATION PAIR \d+ ---\n|\n--- TEST INPUT GRID ---)",
            re.DOTALL
        )
        match = pattern_demo.search(prompt)
        
        evicted_text = None
        if match:
            evicted_text = match.group(1)
        else:
            # Fallback to evicting older conversation turns
            turns = list(re.finditer(r"<\|turn\>(system|user|model)\n", prompt))
            evict_idx = -1
            for idx, m in enumerate(turns):
                role = m.group(1)
                if role in ("user", "model"):
                    if idx < len(turns) - 1:
                        next_turn_text = prompt[turns[idx+1].start():]
                        if "<|reasoning_begin|>" not in next_turn_text[:150]:
                            evict_idx = idx
                            break
            if evict_idx != -1:
                start_pos = turns[evict_idx].start()
                end_pos = turns[evict_idx+1].start()
                evicted_text = prompt[start_pos:end_pos]

        if not evicted_text:
            print("  - Warning: No suitable text block found for eviction.")
            return prompt

        # Remove trailing/leading whitespaces for clean replacement match
        evicted_text_clean = evicted_text.strip()
        if not evicted_text_clean:
            return prompt

        # 2. Tokenize and project to 4096-D complex wave via L3SwarmRouter
        try:
            tokens = self.gen_model.tokenize(evicted_text_clean.encode('utf-8'))
            tokens_tensor = torch.tensor(tokens, dtype=torch.long)
            
            with torch.no_grad():
                psi = self.l3_router.text_to_wave(tokens_tensor).detach().cpu()
                
            print(f"  - Projected evicted context to 4096-D wave (tokens: {len(tokens)})")
        except Exception as e:
            print(f"  - Error projecting evicted context: {e}. Using fallback random wave.")
            phases = (torch.rand(self.hrr_dim) * 2 * np.pi) - np.pi
            psi = torch.polar(torch.ones(self.hrr_dim), phases)

        # 3. Evict to TimescaleDB (Zone C)
        semantic_label = f"Evicted_Context_{uuid.uuid4().hex[:8]}"
        concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, semantic_label))
        
        psi_np = psi.numpy()
        db_vec = np.zeros(4096, dtype=np.float32)
        db_vec[:2048] = psi_np[:2048].real
        db_vec[2048:] = psi_np[:2048].imag
        vector_str = "[" + ",".join(map(str, db_vec)) + "]"
        
        db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")
        try:
            with psycopg.connect(db_url, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront)
                        VALUES (%s, %s, %s, %s::vector)
                        ON CONFLICT (concept_hash) DO NOTHING;
                        """,
                        (concept_hash, semantic_label, "evicted_context", vector_str),
                    )
            print(f"  - Successfully evicted context axiom '{semantic_label}' to TimescaleDB.")
        except Exception as db_err:
            print(f"  - Database insert failed for evicted context: {db_err}. Registering in-memory only.")

        # Register in-memory Hopfield cleanup lexicon
        try:
            mags = torch.abs(psi)
            mags = torch.clamp(mags, min=1e-8)
            self.hopfield.register_concept(semantic_label, psi / mags)
        except Exception as h_err:
            print(f"  - Hopfield registration failed: {h_err}")

        # 4. Replace the raw text segment in the prompt with the axiom symbol
        axiom_symbol = f" [AXIOM: {semantic_label}] "
        new_prompt = prompt.replace(evicted_text_clean, axiom_symbol, 1)
        
        original_tokens = len(tokens)
        new_tokens = len(self.gen_model.tokenize(axiom_symbol.encode('utf-8')))
        print(f"  - Compaction complete: Reduced prompt size by {original_tokens - new_tokens} tokens.")
        
        return new_prompt

    def proactive_eviction(self, prompt: str, watermark: int = 27800) -> str:
        """
        Intelligently evicts context blocks with the lowest semantic resonance
        until the total prompt token count falls below the watermark.
        """
        import re
        import uuid
        import torch
        import numpy as np
        import psycopg
        import hashlib

        # 1. Tokenize prompt and check if we are below watermark
        tokens = self.gen_model.tokenize(prompt.encode('utf-8'))
        current_len = len(tokens)
        if current_len <= watermark:
            return prompt

        print(f"[SYSTEM] Prompt size ({current_len} tokens) exceeds watermark ({watermark}). Starting proactive eviction...")

        # 2. Loop to evict lowest-resonance candidates until below watermark
        while current_len > watermark:
            # 2.1 Identify eviction candidates
            candidate_blocks = []
            
            # Pattern for demonstration pairs
            demo_pattern = re.compile(
                r"(--- DEMONSTRATION PAIR \d+ ---\n.*?)(?=\n--- DEMONSTRATION PAIR \d+ ---\n|\n--- TEST INPUT GRID ---|\n<\|turn\>|(?:\n\s*)?\[AXIOM:|$)",
                re.DOTALL
            )
            for match in demo_pattern.finditer(prompt):
                block_text = match.group(1).strip()
                if block_text and not block_text.startswith("[AXIOM:") and "[AXIOM:" not in block_text:
                    candidate_blocks.append(block_text)

            # Pattern for conversation turns
            turn_pattern = re.compile(
                r"(<\|turn\>(user|model)\n.*?<turn\|>)",
                re.DOTALL
            )
            turns = list(turn_pattern.finditer(prompt))
            if len(turns) > 2:
                # Keep last two turns (query + active response turn)
                for turn in turns[:-2]:
                    block_text = turn.group(1).strip()
                    if block_text and not block_text.startswith("[AXIOM:") and "[AXIOM:" not in block_text:
                        if "<|turn>system" not in block_text:
                            candidate_blocks.append(block_text)

            # Filter out duplicates and ensure we actually have candidates
            candidate_blocks = list(set(candidate_blocks))
            if not candidate_blocks:
                print("  - Warning: No more raw context blocks available for eviction.")
                break

            # 2.2 Generate the 4096-D query wave for the active reasoning trace
            idx = prompt.find("--- TEST INPUT GRID ---")
            if idx != -1:
                active_trace = prompt[idx:]
            else:
                active_trace = prompt[-2000:]

            try:
                emb_res = self.base_model.create_embedding(active_trace)
                q_3840 = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
                with torch.no_grad():
                    q_4096 = self.l3_router.activation_to_wave(q_3840).detach().cpu()
            except Exception as e:
                print(f"  - Error embedding active reasoning trace: {e}. Aborting eviction.")
                break

            # 2.3 Score candidates entirely in 4096-D complex wave space
            candidate_scores = []
            for block_text in candidate_blocks:
                block_hash = hashlib.sha256(block_text.encode('utf-8')).hexdigest()
                
                # Retrieve from cache or generate
                if block_hash not in self.active_block_embeddings:
                    try:
                        emb_res_block = self.base_model.create_embedding(block_text)
                        block_3840 = torch.tensor(emb_res_block["data"][0]["embedding"], dtype=torch.float32)
                        with torch.no_grad():
                            block_4096 = self.l3_router.activation_to_wave(block_3840).detach().cpu()
                        self.active_block_embeddings[block_hash] = block_4096
                    except Exception as e:
                        print(f"  - Warning: Failed to embed block. Using fallback random wave. Error: {e}")
                        phases = (torch.rand(self.hrr_dim) * 2 * np.pi) - np.pi
                        block_4096 = torch.polar(torch.ones(self.hrr_dim), phases)
                        self.active_block_embeddings[block_hash] = block_4096

                block_4096 = self.active_block_embeddings[block_hash]

                # Compute complex cosine similarity
                dot_product = torch.dot(q_4096, torch.conj(block_4096))
                similarity = dot_product.real / (torch.norm(q_4096) * torch.norm(block_4096) + 1e-8)
                similarity_val = similarity.item()
                print(f"    * Candidate {block_hash[:8]}: similarity = {similarity_val:.4f} | text = {block_text[:50].replace('\n', ' ')}...")
                candidate_scores.append({
                    'text': block_text,
                    'hash': block_hash,
                    'similarity': similarity_val
                })

            # 2.4 Identify candidate with the lowest resonance score
            lowest_candidate = min(candidate_scores, key=lambda x: x['similarity'])
            evicted_text_clean = lowest_candidate['text']
            block_hash = lowest_candidate['hash']
            psi = self.active_block_embeddings[block_hash]

            # 2.5 Evict to TimescaleDB
            semantic_label = f"Evicted_Context_{block_hash[:8]}"
            concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, semantic_label))
            
            psi_np = psi.numpy()
            db_vec = np.zeros(4096, dtype=np.float32)
            db_vec[:2048] = psi_np[:2048].real
            db_vec[2048:] = psi_np[:2048].imag
            vector_str = "[" + ",".join(map(str, db_vec)) + "]"

            db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")
            try:
                with psycopg.connect(db_url, connect_timeout=3) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront, raw_text)
                            VALUES (%s, %s, %s, %s::vector, %s)
                            ON CONFLICT (concept_hash) DO UPDATE 
                            SET raw_text = EXCLUDED.raw_text, hrr_wavefront = EXCLUDED.hrr_wavefront;
                            """,
                            (concept_hash, semantic_label, "evicted_context", vector_str, evicted_text_clean),
                        )
                print(f"  - Proactively evicted '{semantic_label}' to DB (Resonance similarity: {lowest_candidate['similarity']:.4f}).")
            except Exception as db_err:
                print(f"  - Database insert failed for evicted context: {db_err}. Local caching only.")

            # Update session caches and register in Hopfield cleanup
            self.evicted_text_registry[semantic_label] = evicted_text_clean
            self.active_block_embeddings[semantic_label] = psi
            try:
                mags = torch.abs(psi)
                mags = torch.clamp(mags, min=1e-8)
                self.hopfield.register_concept(semantic_label, psi / mags)
            except Exception as h_err:
                print(f"  - Hopfield registration failed: {h_err}")

            # 2.6 Replace in prompt
            axiom_symbol = f" [AXIOM: {semantic_label}] "
            prompt = prompt.replace(evicted_text_clean, axiom_symbol, 1)

            # Re-tokenize and check length
            tokens = self.gen_model.tokenize(prompt.encode('utf-8'))
            current_len = len(tokens)
            print(f"  - Prompt compacted to {current_len} tokens.")

        return prompt

    def rehydrate_prompt(self, prompt: str, watermark: int = 27800, threshold: float = 0.82) -> str:
        """
        Scans for all [AXIOM: ...] tags in the prompt, queries waves/text,
        scores them simultaneously against current query trace, sorts descending,
        and rehydrates highest-scoring first up to watermark headroom.
        """
        import re
        import torch
        import numpy as np
        import psycopg

        # 1. Scan the prompt for [AXIOM: ...] tags
        axiom_pattern = re.compile(r"\[AXIOM:\s*(Evicted_Context_[a-f0-9]+)\]")
        found_tags = list(axiom_pattern.finditer(prompt))
        if not found_tags:
            return prompt

        # 2. Retrieve corresponding 4096-D vectors and raw texts
        retrieved_axioms = {}
        labels_to_fetch = []
        for match in found_tags:
            label = match.group(1)
            if label in self.evicted_text_registry and label in self.active_block_embeddings:
                retrieved_axioms[label] = (self.active_block_embeddings[label], self.evicted_text_registry[label])
            else:
                labels_to_fetch.append(label)

        if labels_to_fetch:
            db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")
            try:
                with psycopg.connect(db_url, connect_timeout=3) as conn:
                    with conn.cursor() as cur:
                        for label in labels_to_fetch:
                            cur.execute(
                                """
                                SELECT hrr_wavefront, raw_text 
                                FROM hrr_canonical_lexicon 
                                WHERE semantic_label = %s;
                                """,
                                (label,),
                            )
                            row = cur.fetchone()
                            if row:
                                vector_str = row[0]
                                raw_text = row[1]
                                arr = np.fromstring(vector_str.strip('[]'), sep=',')
                                half = len(arr) // 2
                                real_part = torch.tensor(arr[:half], dtype=torch.float32)
                                imag_part = torch.tensor(arr[half:], dtype=torch.float32)
                                c_wave = torch.complex(real_part, imag_part)
                                if len(c_wave) < self.hrr_dim:
                                    psi = torch.zeros(self.hrr_dim, dtype=torch.complex64)
                                    psi[:len(c_wave)] = c_wave
                                else:
                                    psi = c_wave[:self.hrr_dim]
                                
                                self.active_block_embeddings[label] = psi
                                self.evicted_text_registry[label] = raw_text
                                retrieved_axioms[label] = (psi, raw_text)
            except Exception as e:
                print(f"  - Database fetch failed for rehydration: {e}")

        if not retrieved_axioms:
            return prompt

        # 3. Generate 4096-D query wave for active reasoning trace
        idx = prompt.find("--- TEST INPUT GRID ---")
        if idx != -1:
            active_trace = prompt[idx:]
        else:
            active_trace = prompt[-2000:]

        try:
            emb_res = self.base_model.create_embedding(active_trace)
            q_3840 = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
            with torch.no_grad():
                q_4096 = self.l3_router.activation_to_wave(q_3840).detach().cpu()
        except Exception as e:
            print(f"  - Error embedding reasoning trace for rehydration: {e}")
            return prompt

        # 4. Score all found axioms against active trace simultaneously
        candidate_scores = []
        for label, (psi, raw_text) in retrieved_axioms.items():
            dot_product = torch.dot(q_4096, torch.conj(psi))
            similarity = dot_product.real / (torch.norm(q_4096) * torch.norm(psi) + 1e-8)
            similarity_val = similarity.item()
            print(f"    * Rehydration Candidate {label}: similarity = {similarity_val:.4f}")
            candidate_scores.append({
                'label': label,
                'text': raw_text,
                'similarity': similarity_val
            })

        # 5. Sort descending (highest similarity first)
        candidate_scores.sort(key=lambda x: x['similarity'], reverse=True)

        # 6. Rehydrate until we hit watermark threshold
        current_tokens = len(self.gen_model.tokenize(prompt.encode('utf-8')))
        print(f"    * Rehydration loop starting: current_tokens={current_tokens}, watermark={watermark}, threshold={threshold}")
        
        self.last_routing_weights = None
        skipped_salient_waves = []
        
        for cand in candidate_scores:
            if cand['similarity'] < threshold:
                print(f"    * Skipping {cand['label']}: similarity {cand['similarity']:.4f} < threshold {threshold}")
                continue
                
            raw_tokens = len(self.gen_model.tokenize(cand['text'].encode('utf-8')))
            tag_tokens = len(self.gen_model.tokenize(f" [AXIOM: {cand['label']}] ".encode('utf-8')))
            net_change = raw_tokens - tag_tokens
            
            print(f"    * Candidate {cand['label']}: raw_tokens={raw_tokens}, tag_tokens={tag_tokens}, net_change={net_change}")
            if current_tokens + net_change <= watermark:
                # Perform replace
                pattern = re.compile(rf"\s*\[AXIOM:\s*{cand['label']}\]\s*")
                prompt = pattern.sub(f"\n{cand['text']}\n", prompt)
                current_tokens += net_change
                print(f"  - Rehydrated '{cand['label']}' (similarity score: {cand['similarity']:.4f})")
            else:
                print(f"    * Skipping {cand['label']}: current_tokens {current_tokens} + net_change {net_change} > watermark {watermark} (out of headroom)")
                # Collect skipped salient waves
                candidate_wave = self.active_block_embeddings.get(cand['label'])
                if candidate_wave is not None:
                    skipped_salient_waves.append(candidate_wave)
        
        if skipped_salient_waves:
            print(f"  - Superposition of {len(skipped_salient_waves)} skipped salient context waves for MoE routing.")
            with torch.no_grad():
                # Stack and sum the 4096-D continuous waves
                superposition_wave = torch.sum(torch.stack(skipped_salient_waves), dim=0)
                # Compute MoE weights based on the combined semantic intent
                # We unsqueeze to batch size 1: [1, 4096]
                self.last_routing_weights = self.l3_router.compute_routing_weights(superposition_wave.unsqueeze(0), temperature=0.8).squeeze(0)

        return prompt


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

    @torch.no_grad()
    def verify(self, candidate, target_label="SCADA_Pressure_Control") -> tuple:
        """
        Verifier Sub-agent: Tests code in stateful REPL, projects logic
        to VSA wave space, and validates Dirichlet/Neumann boundaries.
        """
        # Pin current thread to physical Core 7 for verification and HRR math
        pin_current_thread_to_core_7()
        
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

    @torch.no_grad()
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
                self.orchestrator.flush_lora_and_context_to_db(domain_tag=target_label)
                return candidate, revision, "CONVERGED"
                
            # If invalid, update the LoRA weights to "bend" future reasoning vectors
            if delta_np is not None:
                alignment_scalar = 0.1
                # Project 4096-D complex wave to 3840-D real activations using L3 router
                delta_projected = self.orchestrator.l3_router.wave_to_activation(delta_np)
                for i in range(self.orchestrator.num_streams):
                    self.orchestrator.lora_managers[i].update_with_rehypothecated_tensors(delta_projected, alignment_scalar)
                    
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
            
        self.orchestrator.flush_lora_and_context_to_db(domain_tag=target_label)
        return candidate, max_revisions, "TIMEOUT"
