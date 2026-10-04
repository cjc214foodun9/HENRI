"""Verify encode_egress(): additive patch must not change encode(), and must
reproduce the K-SNR probe's ZERO variant through the REAL codec method.

K-SNR probe (codec_snr_fill_probe.py) built the ZERO variant by calling the private
C._wave_accum and zeroing rows by hand. That proved the MECHANISM, not the SHIPPED
METHOD. This checks the shipped method:

  A1 encode() is UNCHANGED      -> geometry gates still pass (retrieval contract safe)
  A2 encode_egress() == probe ZERO variant, byte-exact
  A3 encode() != encode_egress() on a text with empty rows (the patch is real)
  A4 stored-engram compatibility: encode() bytes still round-trip [8192,8]
"""
import os
import sys

import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)
import zone_c_world_knowledge_codec as C

NB, BD = int(C.NUM_BLOCKS), int(C.BLOCK_DIM)
codec = C.get_codec()
ok = True

# ---- A1: encode() unchanged -> geometry gates ----
g = codec.geometry()
gates = {
    "identical>=0.999": g["identical"] >= 0.999,
    "end_1edit>=0.80": g["end_1edit"] >= 0.80,
    "reversal<=0.60": g["reversal"] <= 0.60,
    "random_cross<=0.20": g["random_cross"] <= 0.20,
    "fragment>2*random": g["fragment_retrieval"] > 2.0 * g["random_cross"],
}
print("A1 encode() RETRIEVAL CONTRACT")
for k, v in gates.items():
    print(f"   {k:<22} {v}")
ok &= all(gates.values())
print(f"   geometry = { {k: round(v,6) for k,v in g.items()} }")


def probe_zero(text):
    """The K-SNR probe's ZERO construction, reproduced exactly."""
    acc = C._wave_accum(C.features_of(text, ngram_max=3))
    rows = acc.reshape(NB, BD).copy()
    n = np.linalg.norm(rows, axis=1, keepdims=True)
    np.divide(rows, n, out=rows, where=n > 1e-9)
    return rows


TEXT = "the alpha report"
shipped_egress = codec.encode_egress(TEXT)
probe_z = probe_zero(TEXT)
same = np.array_equal(shipped_egress, probe_z)
print(f"\nA2 encode_egress() == probe ZERO variant, byte-exact: {same}")
print(f"      max abs diff = {float(np.abs(shipped_egress - probe_z).max()):.3e}")
ok &= bool(same)

b, _ = codec.encode(TEXT)
stored = np.frombuffer(b, dtype="<f4").reshape(NB, BD)
differs = not np.array_equal(stored, shipped_egress)
nz_stored = int((np.count_nonzero(stored, axis=1) == 0).sum())
nz_egress = int((np.count_nonzero(shipped_egress, axis=1) == 0).sum())
print(f"\nA3 patch is REAL (encode != encode_egress): {differs}")
print(f"      rows all-zero : stored={nz_stored}  egress={nz_egress}")
print(f"      egress active rows = {NB - nz_egress} of {NB} "
      f"({(NB-nz_egress)/NB*100:.2f}%)")
ok &= bool(differs)

print(f"\nA4 stored engram round-trip")
print(f"      bytes={len(b)}  reshape={stored.shape}  round-trip={stored.tobytes() == b}")
ok &= (stored.tobytes() == b and len(b) == 262144)

print(f"\nALL CHECKS PASS = {ok}")
