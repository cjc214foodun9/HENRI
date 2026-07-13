import asyncio
import torch
import asyncpg
import numpy as np
import logging
import math
from typing import Optional, List, Dict, Any
from asyncpg.pool import Pool

# Enforce strict float32 and complex64 logic globally
torch.set_default_dtype(torch.float32)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhylogeneticStore")

class EngramStore:
    """
    Integrates asyncpg with a TimescaleDB + pgvector backend.
    Serves as the genetic engram store to cache and retrieve stabilized 
    wave trajectories (geodesics) across evolutionary lineage.
    """
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[Pool] = None

    async def _connect_with_backoff(self) -> None:
        """
        Establishes connection pool with exponential backoff.
        Guarantees zero-trust robustness against database drops.
        """
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if self.pool is None:
                    self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=10)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to connect to TimescaleDB after {max_retries} attempts. Fatal error.")
                    raise e
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Connection dropped. Retrying in {delay} seconds... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(delay)

    async def initialize_schema(self) -> None:
        """
        Creates the hypertable `phylogenetic_engrams` and necessary pgvector bindings.
        """
        await self._connect_with_backoff()
        
        async with self.pool.acquire() as conn:
            # Ensure pgvector extension is available for Holographic Hashing
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # TimescaleDB requires the partition key (timestamp) to be part of the primary key
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS phylogenetic_engrams (
                    id UUID DEFAULT gen_random_uuid(),
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    environmental_context_hash VARCHAR(256),
                    engram_wave vector(8192),
                    PRIMARY KEY (id, timestamp)
                );
            """)
            
            # Convert to TimescaleDB chronological hypertable
            await conn.execute("""
                SELECT create_hypertable('phylogenetic_engrams', 'timestamp', if_not_exists => TRUE);
            """)
            
            # HNSW index omitted because pgvector limits HNSW to 2000 dimensions (we use 8192)
            # Exact search (vector_cosine_ops) will be used instead.
        logger.info("Phylogenetic Schema Initialization Complete.")

    def _serialize_wave(self, complex_tensor: torch.Tensor) -> str:
        """
        Normalizes the 4096-D PyTorch complex64 tensor to L2=1.0 and serializes 
        it into an 8192-float array string for pgvector.
        Format: [Real0, Imag0, Real1, Imag1...]
        """
        assert complex_tensor.shape[-1] == 4096, "Expected 4096-D complex tensor"
        
        # Flatten batch dimension if present
        vec = complex_tensor.view(-1)
        
        # Strictly normalize L2=1.0 as the engram metric invariant
        norm = torch.norm(vec, p=2)
        vec = vec / (norm + 1e-16)
        
        # Interleave real and imaginary components
        stacked = torch.stack((vec.real, vec.imag), dim=-1)
        flat_real_array = stacked.view(-1).detach().cpu().numpy().astype(np.float32)
        
        # Format string array required by pgvector: '[0.1, 0.2, ...]'
        return '[' + ','.join(map(str, flat_real_array)) + ']'

    def _deserialize_wave(self, vector_string: str) -> torch.Tensor:
        """
        Maps an 8192-float pgvector string back into a PyTorch complex64 tensor [4096].
        """
        clean_str = vector_string.strip('[]')
        float_array = np.array([float(x) for x in clean_str.split(',')], dtype=np.float32)
        
        # Reshape to 4096 x 2
        interleaved = float_array.reshape(4096, 2)
        
        # Project back to complex 128
        complex_tensor = torch.tensor(interleaved[:, 0] + 1j * interleaved[:, 1], dtype=torch.complex64)
        return complex_tensor

    async def cache_survival_trait(self, context_hash: str, complex_tensor: torch.Tensor) -> None:
        """
        Serializes and inserts a stabilized wave trajectory (Engram) into the 
        evolutionary lineage database.
        """
        await self._connect_with_backoff()
        
        vector_str = self._serialize_wave(complex_tensor)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO phylogenetic_engrams (environmental_context_hash, engram_wave)
                VALUES ($1, $2::vector)
            """, context_hash, vector_str)

    async def retrieve_ancestral_engram(self, query_complex_tensor: torch.Tensor, k: int = 1) -> List[Dict[str, Any]]:
        """
        Executes a Holographic Hash Cosine Distance search (<=>) on the pgvector index, 
        returning the closest historical tensor mapped back into a PyTorch complex64 shape.
        """
        await self._connect_with_backoff()
        
        query_str = self._serialize_wave(query_complex_tensor)
        
        async with self.pool.acquire() as conn:
            # Query pgvector utilizing the <=> operator for Cosine Distance
            rows = await conn.fetch(f"""
                SELECT id, timestamp, environmental_context_hash, engram_wave::text, 
                       (engram_wave <=> $1::vector) as distance
                FROM phylogenetic_engrams
                ORDER BY distance ASC
                LIMIT $2
            """, query_str, k)
            
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "context_hash": row["environmental_context_hash"],
                    "distance": row["distance"],
                    "engram_wave": self._deserialize_wave(row["engram_wave"])
                })
                
            return results

    async def close(self):
        """Clean shutdown of the database pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Engram Store connection pool closed.")


if __name__ == "__main__":
    print("Testing Phylogenetic Engram Store Serialization Boundaries...")
    
    # Instantiate dummy target
    store = EngramStore("postgresql://user:pass@localhost:5432/henri")
    
    # 1. Create a raw 4096-D complex tensor (representing a physical state from the Chemical Sensor)
    dim = 4096
    original_tensor = torch.randn(dim, dtype=torch.float32) + 1j * torch.randn(dim, dtype=torch.float32)
    
    # 2. Serialize (simulates caching to TimescaleDB)
    vector_str = store._serialize_wave(original_tensor)
    
    # Verify dimensions for Postgres
    float_count = vector_str.count(',') + 1
    print(f"Serialized Array Length: {float_count} floats (Expected: 8192)")
    assert float_count == 8192, "Vectorization did not accurately map to 8192-D."
    
    # 3. Deserialize (simulates retrieval from pgvector)
    restored_tensor = store._deserialize_wave(vector_str)
    
    print(f"Restored Tensor Shape: {restored_tensor.shape}")
    print(f"Restored Tensor Type:  {restored_tensor.dtype}")
    assert restored_tensor.dtype == torch.complex64, "Loss of precision! Failed to restore strictly to complex64."
    
    # Validate strictly normalized L2 modulus
    restored_norm = torch.norm(restored_tensor, p=2).item()
    print(f"Restored Tensor L2 Norm: {restored_norm:.16f} (Expected: ~1.0000000000000000)")
    assert math.isclose(restored_norm, 1.0, rel_tol=1e-9), "L2 Normalization constraint failed!"
    
    print("\n[SUCCESS] Phylogenetic Engram vector translation logic passes strict biophysical checks.")
