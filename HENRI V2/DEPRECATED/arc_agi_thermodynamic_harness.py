import torch
import torch.nn.functional as F
import math
import asyncio
import logging
import sys
import os

# Link to core components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sensory_transducer import ChemicalSensorAnchor
from morphogenetic_syncytium import SyncytiumCore
from darwinian_selection_loop import run_darwinian_inference
from phylogenetic_memory import EngramStore
from holographic_cache_manager import HolographicCacheManager
from motor_egress_crystallizer import HolographicActionCrystallizer

torch.set_default_dtype(torch.float32)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class ARCThermodynamicHarness:
    """
    The Master Execution Loop for solving ARC-AGI.
    Orchestrates the Sensory Ingress, Memory Pre-fetch, Darwinian Yielding, 
    and the final Motor Egress crystallization.
    """
    def __init__(self, dim: int = 4096, depth: int = 8, vocab_size: int = 32000):
        self.dim = dim
        self.depth = depth
        
        # 1. Initialize Biological Components
        logging.info("Initializing Autopoietic ARC-AGI Ecosystem...")
        self.sensor = ChemicalSensorAnchor(dim=dim)
        self.syncytium = SyncytiumCore(dimension=dim, depth=depth)
        
        # We manually spawn the cartilage here (simulating that the bone is already etched/frozen)
        # In a full run, we would call syncytium.etch_invariants() then spawn_cartilage()
        for layer in self.syncytium.layers:
            layer.weight.requires_grad = False
            layer.spawn_cartilage(rank=64)
            
        # Motor Egress module (Unmodified Vocab Size as instructed)
        self.egress = HolographicActionCrystallizer(dim=dim, vocab_size=vocab_size)
        
        # Database connection mapping
        self.store = EngramStore("postgresql://user:pass@localhost:5432/henri")
        self.cache_manager = HolographicCacheManager(self.store)

    async def solve_grid(self, grid_input: torch.Tensor, target_prior_hash: str) -> torch.Tensor:
        """
        Executes the continuous test-time learning loop to solve an ARC grid.
        grid_input: [batch, dim] 
        """
        logging.info(">>> PHASE 1: SENSORY INGESTION")
        # Transduce raw discrete grid into continuous chemical phase wave
        psi_in = self.sensor(grid_input)
        logging.info(f"Sensory Wave Transduced. Modulus: {torch.norm(psi_in, p=2, dim=-1).item():.6f}")
        
        logging.info(">>> PHASE 2: ASYNCHRONOUS MEMORY PRE-FETCH")
        # Spin up async worker to fetch the structural thermodynamic bound
        self.cache_manager.start_worker()
        
        # Instead of directly querying the real tensor (which would be passed), we mock the query 
        # for execution since we are retrieving via hash lookup in a full prod DB.
        # For the physics to execute, we submit a dummy query to the cache buffer.
        query = torch.randn(self.dim, dtype=torch.complex64)
        self.cache_manager.request_prefetch(query)
        
        psi_target = await self.cache_manager.get_prefetched_engram(timeout=2.0)
        await self.cache_manager.shutdown()
        
        # Fallback to simulated constraint if DB is unavailable
        if psi_target is None:
            logging.warning("Zone C unreachable. Simulating absolute target bound for test execution.")
            psi_target = torch.randn(1, self.dim, dtype=torch.complex64)
            psi_target = psi_target / torch.norm(psi_target, p=2, dim=-1, keepdim=True)
        else:
            # Reformat batched
            if psi_target.dim() == 1:
                psi_target = psi_target.unsqueeze(0)
            psi_target = psi_target / torch.norm(psi_target, p=2, dim=-1, keepdim=True)

        logging.info(">>> PHASE 3: DARWINIAN TEST-TIME LEARNING (CARTILAGE YIELD)")
        # run_darwinian_inference expects raw angle tensors to compute the complex wave natively.
        psi_out = run_darwinian_inference(
            self.syncytium, 
            torch.angle(psi_in), 
            torch.angle(psi_target), 
            max_epochs=50
        )
        
        logging.info(">>> PHASE 4: MOTOR EGRESS CRYSTALLIZATION")
        # Construct a strict GBNF logical sieve mask blocking all non-ARC tokens
        # Assuming token indices 15-24 correspond to colors '0'-'9', 25='[', 26=']', 27=','
        valid_indices = list(range(15, 28)) 
        mask = torch.zeros(1, self.egress.vocab_size, dtype=torch.float32)
        for idx in valid_indices:
            mask[0, idx] = 1.0
            
        probabilities, selection = self.egress(psi_out, mask)
        
        logging.info(f"Crystallized Token Selection ID: {selection.item()}")
        return selection

async def execute_harness():
    print("=" * 80)
    print("   PROJECT HENRI: ARC-AGI THERMODYNAMIC HARNESS")
    print("=" * 80)
    
    DIMENSION = 256 # Executing localized physics scale
    VOCAB_SIZE = 32000
    
    harness = ARCThermodynamicHarness(dim=DIMENSION, depth=8, vocab_size=VOCAB_SIZE)
    
    # Simulate a flattened 16x16 ARC grid input padded to 256
    mock_grid_env = torch.randn(1, DIMENSION) * 10.0
    
    try:
        final_token = await harness.solve_grid(mock_grid_env, "arc_spatial_translation")
        print(f"\n[SUCCESS] ARC-AGI Master Execution Loop successfully compiled. Token Emitted: {final_token.item()}")
    except Exception as e:
        logging.error(f"Execution Error: {e}")

if __name__ == "__main__":
    asyncio.run(execute_harness())
