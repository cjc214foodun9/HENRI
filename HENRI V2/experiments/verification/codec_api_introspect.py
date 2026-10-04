"""Codec API introspection -- no guessing at signatures."""
import hashlib
import os
import sys

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)

import numpy as np
import zone_c_world_knowledge_codec as C

c = C.get_codec()
print("module constants:")
for k in ("BLOCK_DIM", "WAVE_DIM", "WAVE_EXPAND", "NUM_BLOCKS", "PROJ_DIM",
          "MAX_WORDS", "LCG_MUL", "LCG_ADD"):
    print(f"   {k} = {getattr(C, k, None)}")

print("\ncodec attrs:")
for a in ("wave_dim", "proj_dim", "geometry"):
    v = getattr(c, a, None)
    print(f"   {a!r} = {v!r}  (callable={callable(v)})")

b, w = c.encode("the alpha report")
w = np.asarray(w)
print(f"\nencode('the alpha report'):")
print(f"   bytes len     = {len(b)}")
print(f"   wave dtype    = {w.dtype}   shape = {w.shape}   size = {w.size}")
flat = w.ravel()
nnz = int(np.count_nonzero(flat))
print(f"   nnz           = {nnz} / {flat.size}   frac = {nnz/flat.size:.4f}")
print(f"   min / max     = {float(flat.min()):.4f} / {float(flat.max()):.4f}")

print(f"\ntokenize('the alpha report') = {C.tokenize('the alpha report')}")

V = 512
for tok in ("alpha", "bravo", "the"):
    ident = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % V
    print(f"   sha256('{tok}') % {V} = {ident}")

# feature-block footprint: which 16-slots-per-feature blocks carry content
print("\nfeature-block footprint:")
fb = None
try:
    feats = C.features_of("the alpha report")
    print(f"   features_of -> {len(feats)} features: {feats[:6]}")
    EX = int(getattr(C, "WAVE_EXPAND", 16))
    BL = int(getattr(C, "BLOCK_DIM", 8))
    fb = EX
    print(f"   WAVE_EXPAND={EX} 1 feature occupies {EX} blocks of {BL} slots")
except Exception as e:
    print(f"   features_of unavailable: {e}")

print(f"\ncodebook memory arithmetic:")
for K in (256, 512):
    for D in (8192, 65536):
        nbytes = K * D * 8
        print(f"   K={K:>4} D={D:>6} complex64 = {nbytes/2**20:>8.1f} MiB")
