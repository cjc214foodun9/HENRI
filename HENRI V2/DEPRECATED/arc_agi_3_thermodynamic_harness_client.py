import os
import torch
import torch.nn.functional as F
import math
import logging
import asyncio

# The official ARC-AGI SDK (Must be installed via pip)
try:
    import arc_agi
except ImportError:
    logging.warning("arc_agi module not found. Please run: pip install arc-agi")

# Import the native biophysical architecture
from morphogenetic_syncytium import SyncytiumCore
from sensory_transducer import ChemicalSensorAnchor
from phylogenetic_memory import EngramStore

# Strict biophysical constraints
torch.set_default_dtype(torch.float32)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [ARC-AGI HARNESS] %(message)s')

class HolographicSpatialPointer:
    """
    Translates the thermodynamic intensity of the 4096-dimensional complex wave
    directly into 2D Euclidean spatial coordinates (X, Y).
    Bypasses discrete tokenization entirely.
    """
    def __init__(self, grid_dim: int = 64):
        self.grid_dim = grid_dim

    def extract_coordinates(self, psi_out: torch.Tensor):
        # 1. Compute L2 Magnitude of the complex wavefront
        psi_magnitude = torch.abs(psi_out).squeeze()
        
        # 2. Find the index of maximum energy (constructive interference peak)
        index_peak = torch.argmax(psi_magnitude).item()
        
        # 3. Translate the 1D peak index into 2D spatial coordinates natively
        y = index_peak // self.grid_dim
        x = index_peak % self.grid_dim
        
        return x, y


from phylogenetic_memory import EngramStore

async def run_arc_benchmark():
    logging.info("Initializing Thermodynamic Harness ARC-AGI-3 Client")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize the Core (Bone)
    dimension = 4096
    core = SyncytiumCore(dimension=dimension, depth=8).to(device)
    sensor = ChemicalSensorAnchor(dim=dimension).to(device)
    
    # Load the Epistemic Fossilized Weights
    fossil_path = "henri_fossilized_core.pt"
    if os.path.exists(fossil_path):
        try:
            state_dict = torch.load("henri_fossilized_core.pt", map_location=device, weights_only=True)
            core.load_state_dict(state_dict)
            # Force all parameters to complex64 / float32 to prevent FP64 fallback from the fossil checkpoint
            for name, param in core.named_parameters():
                if param.is_complex():
                    param.data = param.data.to(torch.complex64)
                elif param.is_floating_point():
                    param.data = param.data.to(torch.float32)
            for name, buf in core.named_buffers():
                if buf.is_complex():
                    buf.data = buf.data.to(torch.complex64)
                elif buf.is_floating_point():
                    buf.data = buf.data.to(torch.float32)
            logging.info("Successfully loaded Epistemic Fossilized Core (The Bone).")
        except Exception as e:
            logging.warning(f"Fossilized core failed to load: {e}. Running with untrained bone.")
    else:
        logging.warning("Fossilized core NOT FOUND. Running with untrained bone.")
        
    core.eval() # Bone is frozen
    
    # Spawn Cartilage for test-time adaptation
    for layer in core.layers:
        layer.spawn_cartilage(rank=64)
    
    # Ensure newly spawned cartilage parameters are moved to GPU
    core.to(device)
    
    pointer = HolographicSpatialPointer(grid_dim=64)
    
    # Initialize the Holographic Cache Manager
    import sys
    # Ensure HENRI_V2 is in path so we can import from it on vast.ai or local
    sys.path.append(os.path.join(os.path.dirname(__file__), "HENRI V2"))
    sys.path.append(os.path.join(os.path.dirname(__file__), "HENRI_V2"))
    
    from holographic_cache_manager import HolographicCacheManager
    from phylogenetic_memory import EngramStore
    
    dsn = os.environ.get("POSTGRES_DSN", "postgres://postgres:password@localhost:5432/henri")
    store = EngramStore(dsn)
    await store.initialize_schema()
    
    cache_manager = HolographicCacheManager(store, max_size=10)
    cache_manager.start_worker()
    
    # 2. Connect to the Arcade Environment
    try:
        from arcengine import GameAction
        arc = arc_agi.Arcade()
        
        games = arc.get_environments()
        logging.info(f"Initiating FULL SCALE Unlimited Benchmark across {len(games)} environments.")
        
        for game_info in games:
            game_id = getattr(game_info, 'game_id', None) or getattr(game_info, 'id', None) or game_info
            
            env = arc.make(game_id, render_mode="terminal")
            obs = env.reset()
            
            logging.info(f"Started Game: {game_id}")
            
            done = False
            steps = 0
            while not done and steps < 50:
                frame = getattr(obs, 'frame', [])
                try:
                    frame_tensor = torch.tensor(frame, dtype=torch.float32, device=device).flatten()
                except Exception:
                    import numpy as np
                    frame_tensor = torch.tensor(np.array(frame), dtype=torch.float32, device=device).flatten()
                    
                I_t = torch.zeros(1, dimension, dtype=torch.float32, device=device)
                flat_len = min(frame_tensor.shape[0], dimension)
                if flat_len > 0:
                    I_t[0, :flat_len] = frame_tensor[:flat_len]
                
                # Transduce the 2D grid frame into a thermodynamic complex wave via ChemicalSensorAnchor
                psi_in = sensor(I_t)
                psi_in = F.normalize(psi_in, p=2, dim=-1)
                
                # --- WAVE-JEPA & VISCOELASTIC CREEP ---
                cache_manager.request_prefetch(psi_in)
                engram_target = await cache_manager.get_prefetched_engram(timeout=5.0)
                
                if engram_target is not None:
                    engram_target = engram_target.to(device)
                    # Run the continuous thermodynamic creep
                    from darwinian_selection_loop import run_darwinian_inference
                    psi_out = run_darwinian_inference(core, psi_in, engram_target, max_epochs=250, dt=1.0)
                else:
                    logging.warning("Cache miss! Starvation on Tensor Cores. Falling back to zero-shot.")
                    psi_out = core(psi_in)
                
                # Extract coordinates via thermodynamic energy peak
                x, y = pointer.extract_coordinates(psi_out)
                
                logging.info(f"Step {steps} | Transduced Frame -> Peak Energy Coordinates: X={x}, Y={y}")
                
                try:
                    obs = env.step(GameAction.ACTION6, data={"x": x, "y": y})
                except Exception as e:
                    logging.error(f"Failed to step environment: {e}")
                    break
                
                if hasattr(obs, 'levels_completed') and getattr(obs, 'levels_completed', 0) > 0:
                    logging.info(f"Game {game_id} Solved! Levels Completed: {obs.levels_completed}")
                    
                    # Cache the survival trait for evolutionary memory
                    context_hash = str(game_id)
                    try:
                        await store.cache_survival_trait(context_hash, psi_out)
                        logging.info(f"Successfully crystallized survival engram to TimescaleDB for {game_id}")
                    except Exception as e:
                        logging.error(f"Failed to cache survival trait: {e}")
                    
                    done = True
                    
                steps += 1
            
        scorecard = arc.get_scorecard()
        logging.info(f"Final Scorecard:\n{scorecard}")
        
    except Exception as e:
        logging.error(f"Benchmark execution failed: {e}")
    finally:
        await cache_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(run_arc_benchmark())