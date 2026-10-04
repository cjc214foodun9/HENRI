"""Decode the codec engram layout exactly, then build the content-grounded corpus.

FACTS ALREADY ESTABLISHED (codec_api_introspect.py, RC=0):
    WAVE_DIM=65536  NUM_BLOCKS=8192  BLOCK_DIM=8  WAVE_EXPAND=16  PROJ_DIM=2000
    encode(text) -> (bytes len=262144, ndarray shape=(2000,) float32)
    262144 = 8192 * 8 * 4  -> the bytes ARE the full [8192,8] float32 engram
    2000-float array is the HNSW search projection, unit-norm, nnz=5

This script answers, with numbers and no guessing:
    L1  what geometry() reports
    L2  the exact byte -> [8192,8] decode
    L3  per-block nonzero count (is filler dominant, as measured earlier?)
    L4  which blocks a KNOWN word occupies (signal footprint)
    L5  does the same word in a different template share feature blocks?
    L6  the complex convention: [8192,8] float32 -> [8192,4] complex64 = 32768 complex
"""
import os
import sys
import hashlib
import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)
import zone_c_world_knowledge_codec as C

c = C.get_codec()
NB = int(C.NUM_BLOCKS)      # 8192
BD = int(C.BLOCK_DIM)       # 8
EX = int(C.WAVE_EXPAND)     # 16

print("L1 geometry()")
try:
    g = c.geometry()
    print(f"   {g}")
except Exception as e:
    print(f"   geometry() raised: {e}")

b, proj = c.encode("the alpha report")
eng = np.frombuffer(b, dtype="<f4").reshape(NB, BD)
print(f"\nL2 decode")
print(f"   bytes            = {len(b)}")
print(f"   reshape          = {eng.shape} {eng.dtype}")
print(f"   round-trip ok    = {eng.tobytes() == b}")
print(f"   projection       = {np.asarray(proj).shape} nnz={int(np.count_nonzero(proj))}")

nzb = np.count_nonzero(eng, axis=1)
print(f"\nL3 per-block nonzero counts")
print(f"   blocks           = {NB}")
print(f"   blocks with 0 nz = {int((nzb == 0).sum())}")
print(f"   blocks with 1 nz = {int((nzb == 1).sum())}")
print(f"   blocks with >1 nz= {int((nzb > 1).sum())}")
print(f"   total nnz        = {int(nzb.sum())} / {NB*BD}  frac={nzb.sum()/(NB*BD):.4f}")


def blocks_of(text):
    bb, _ = c.encode(text)
    e = np.frombuffer(bb, dtype="<f4").reshape(NB, BD)
    return set(np.nonzero(np.count_nonzero(e, axis=1))[0].tolist())


A = blocks_of("the alpha report")
B = blocks_of("the alpha bravo report")
Cw = blocks_of("alpha is the word")
print(f"\nL4 signal footprint")
print(f"   'the alpha report'      active blocks = {len(A)}")
print(f"   'the alpha bravo report' active blocks = {len(B)}")
print(f"   'alpha is the word'      active blocks = {len(Cw)}")
print(f"   A & B shared            = {len(A & B)}   (alpha common)")
print(f"   A - B                   = {len(A - B)}")

print(f"\nL5 same word, different template")
for w in ("alpha", "bravo"):
    s1 = blocks_of(f"the {w} report")
    s2 = blocks_of(f"{w} is the word")
    print(f"   {w:>6}: template1={len(s1):>5}  template2={len(s2):>5}  shared={len(s1 & s2):>5}"
          f"  jaccard={len(s1 & s2)/len(s1 | s2):.4f}")

print(f"\nL6 complex convention")
cplx = eng.reshape(NB, BD // 2, 2).view(np.complex64).reshape(NB, BD // 2)
print(f"   [8192,8] f32 -> view complex64 -> {cplx.shape} = {cplx.size} complex values")
print(f"   codebook K=512 x {cplx.size} complex64 = "
      f"{512*cplx.size*8/2**20:.1f} MiB")
