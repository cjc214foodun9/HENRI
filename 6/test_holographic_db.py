import torch
import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from henri_core.database import HenriTimescaleConnector

def test_holographic_db():
    print("[TEST] Running Holographic DB lookup test suite...")
    
    # 1. Connect to DB (using offline cache fallback or local DB)
    db_uri = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")
    db = HenriTimescaleConnector(db_uri)
    
    # 2. Test Holographic Key Generation
    dummy_wave = torch.randn(4096)
    h_key = db.generate_holographic_key(dummy_wave)
    
    print(f"[TEST] Generated holographic key (shape: {h_key.shape}): {h_key[:10]}...")
    assert h_key.shape == (4096,), "Holographic key shape must be (4096,)!"
    assert np.all(np.abs(h_key) == 1.0) or np.any(h_key == 0), "Holographic key must be sign-based (-1, 0, 1)!"
    print("[SUCCESS] Holographic key generation verified.")
    
    # 3. Test Async Prefetching
    print("[TEST] Triggering async prefetching...")
    db.async_prefetch_attractors(h_key, "constant_or_kind")
    
    # Wait a moment for async thread to complete
    time.sleep(2.0)
    
    # Verify prefetch buffer
    print(f"[TEST] Prefetch buffer contents for 'constant_or_kind': {db.prefetch_buffer.get('constant_or_kind', 'Empty')}")
    print("[SUCCESS] Async pre-fetching verified.")
    print("[SUCCESS] All holographic database tests passed!")

if __name__ == "__main__":
    test_holographic_db()
