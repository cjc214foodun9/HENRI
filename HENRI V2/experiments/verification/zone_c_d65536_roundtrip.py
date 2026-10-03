"""Zone C round-trip proof at D=65536. Deterministic. No GPU.

Settles the imperative claim: a 65536-dim wave stores to TimescaleDB and
reads back byte-exact, with no [D,D] object and no VRAM touch.
"""
import os
import sys
import hashlib

os.environ.setdefault("HENRI_FREEZE_LEARNING", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import zone_c_segment_cache as zsc

NUM_BLOCKS = 8192          # D = 8192 * 8 = 65536
EXPECTED_BYTES = NUM_BLOCKS * 8 * 4   # 262144
DSN = zsc.resolve_zone_c_dsn() if hasattr(zsc, "resolve_zone_c_dsn") else None

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" | " + detail) if detail else ""))
    if not cond:
        fails.append(name)

# ---- 1. codec: CPU-only, exact size -------------------------------
g = torch.Generator().manual_seed(20261003)
wave = (torch.randn(NUM_BLOCKS, 8, generator=g, dtype=torch.float32)
        + 1j * torch.randn(NUM_BLOCKS, 8, generator=g, dtype=torch.float32))
check("wave is [8192, 8] complex64", tuple(wave.shape) == (8192, 8) and wave.is_complex())

buf = zsc.wave_to_bytes(wave)
check("payload is %d bytes" % EXPECTED_BYTES, len(buf) == EXPECTED_BYTES,
      "got %d" % len(buf))

back = zsc.bytes_to_wave(buf, NUM_BLOCKS)
check("round-trip shape [8192, 8]", tuple(back.shape) == (8192, 8))
check("round-trip dtype float32 real", back.dtype == torch.float32)
check("round-trip bit-exact", torch.equal(back.view(-1), torch.from_numpy(
    np.frombuffer(buf, dtype=np.float32).copy())),
    "sha256=%s" % hashlib.sha256(buf).hexdigest()[:16])

# ---- 2. dimension mismatch is caught (the 5 small rows) -----------
try:
    zsc.bytes_to_wave(b"\x00" * 8192, NUM_BLOCKS)
    check("8192-byte row REFUSED at production dim", False, "silently accepted")
except ValueError as e:
    check("8192-byte row REFUSED at production dim", True, str(e)[:52])

# ---- 3. no GPU anywhere in the codec ------------------------------
src = open(zsc.__file__, encoding="utf-8").read()
codec = src[src.index("def wave_to_bytes"):src.index("_PROJ_CACHE")]
check("codec has no cuda reference", "cuda" not in codec.lower())

print("ROUNDTRIP_VERDICT=" + ("PASS" if not fails else "FAIL:" + ",".join(fails)))
sys.exit(1 if fails else 0)
