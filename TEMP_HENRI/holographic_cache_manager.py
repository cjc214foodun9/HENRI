import torch
import asyncio
import logging
from typing import Optional

# Enforce strict float32 and complex64 logic globally
torch.set_default_dtype(torch.float32)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HolographicCacheManager")

class HolographicCacheManager:
    """
    Asynchronous Ring Buffer designed to break the I/O Memory Wall.
    Pre-fetches boundary engrams from TimescaleDB in the background and 
    allocates them directly into pinned CUDA memory (if available) for 
    zero-latency offload to the Syncytium during the synchronous forward pass.
    """
    def __init__(self, store, max_size: int = 10):
        # EngramStore instance (from phylogenetic_memory.py)
        self.store = store
        self.max_size = max_size
        
        # Async queues for managing pre-fetch requests and pinned results
        self.request_queue = asyncio.Queue()
        self.cache = asyncio.Queue(maxsize=max_size)
        
        # Background worker task
        self.worker_task: Optional[asyncio.Task] = None
        
    def start_worker(self):
        """Spawns the background asyncio pre-fetch worker."""
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._prefetch_loop())
            logger.info("Holographic Cache Manager background worker started.")

    async def _prefetch_loop(self):
        """
        Continuously listens for query tensors, fetches closest engrams via 
        TimescaleDB asyncpg, and pins them into memory.
        """
        while True:
            try:
                # Wait for the next active query phase tensor
                query_tensor = await self.request_queue.get()
                
                # Fetch closest phylogenetic ancestor (network I/O bounded)
                results = await self.store.retrieve_ancestral_engram(query_tensor, k=1)
                
                if results:
                    engram = results[0]["engram_wave"]
                    
                    # Allocate into pinned memory for zero-latency host-to-device transfer
                    # (Fallback to standard memory if CUDA is not available on host)
                    if torch.cuda.is_available():
                        engram = engram.pin_memory()
                    
                    # Push to ready cache, dropping oldest if full (ring buffer logic)
                    if self.cache.full():
                        _ = self.cache.get_nowait()
                        
                    await self.cache.put(engram)
                    
                self.request_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Cache Manager worker terminated.")
                break
            except Exception as e:
                logger.error(f"Error in prefetch loop: {e}")

    def request_prefetch(self, query_tensor: torch.Tensor):
        """
        Non-blocking request from the synchronous forward loop to enqueue a fetch.
        """
        try:
            self.request_queue.put_nowait(query_tensor)
        except asyncio.QueueFull:
            pass

    async def get_prefetched_engram(self, timeout: float = 0.5) -> Optional[torch.Tensor]:
        """
        Pulls the next ready pinned engram from the ring buffer.
        """
        try:
            return await asyncio.wait_for(self.cache.get(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Cache miss! GPU forward pass starved waiting for I/O.")
            return None
            
    async def shutdown(self):
        """Gracefully kills the background worker."""
        if self.worker_task:
            self.worker_task.cancel()
            await asyncio.gather(self.worker_task, return_exceptions=True)
            self.worker_task = None


if __name__ == "__main__":
    print("Testing Holographic Cache Manager Logic...")
    
    # Mock Engram Store interface to verify async flow
    class MockEngramStore:
        async def retrieve_ancestral_engram(self, query_tensor, k=1):
            await asyncio.sleep(0.1) # Simulate I/O latency
            mock_tensor = torch.randn(4096, dtype=torch.complex64)
            return [{"engram_wave": mock_tensor}]
            
    async def run_test():
        store = MockEngramStore()
        cache_manager = HolographicCacheManager(store)
        
        # Start background worker
        cache_manager.start_worker()
        
        # Main Thread: Submit a non-blocking request early
        query = torch.randn(4096, dtype=torch.complex64)
        cache_manager.request_prefetch(query)
        
        print("Pre-fetch requested. Main thread doing other synchronous GPU work...")
        await asyncio.sleep(0.05) 
        
        print("Main thread needs engram now. Accessing cache buffer...")
        # Since DB takes 0.1s and we only slept 0.05s, it will wait remaining 0.05s
        engram = await cache_manager.get_prefetched_engram(timeout=1.0)
        
        if engram is not None:
            is_pinned = engram.is_pinned() if torch.cuda.is_available() else "N/A (No CUDA)"
            print(f"Success! Engram retrieved from ring buffer. Pinned: {is_pinned}")
        else:
            print("Failure! Cache miss.")
            
        await cache_manager.shutdown()

    asyncio.run(run_test())
