"""
Stage-0b-rev frozen nonlinear encoder — RFF lift + circular-conv VSA binding.
=============================================================================
Reference 3 (gpt-5.6-sol) binding; pre-registration vla_stage0b_rev_contract.md
(sha c2ca66e7...).

Pipeline (frozen, numpy-only):
  x (4D) --standardize (calib_mean/std)--> x_hat
       --RFF phi=[cos(W x_hat + b); sin(W x_hat + b)]--> (384,)
       --circular conv with slot key k_i--> z_i (384,)
       --per-slot unit norm--> z (1, 16, 384) float32   [sphere geometry]

Contracts:
  C1 default OFF -> encode() returns input byte-identical (np.array_equal)
  C2 zero trainable state (no torch, no Parameter/backward/optimizer)
  C3 frozen cross-process output hash (identical npz bytes)
  C4 shape (1,16,384) float32; per-slot unit-norm sphere err <= 1e-6
  C5 deterministic restart (same npz + input -> byte-identical)
  C6 sensitivity over DEDUPLICATED real observations (>=99% distinct pairs
     L2 > 1e-3; zero collisions)
  C7 non-collapse (flat calib SVD rank >= 16; min slot std >= 1e-3)
  C8 no env/learner access (no gymnasium, no wrapper, no learner imports)
  C9 verification uses real corpus observations (never synthetic arrays)
"""
import hashlib, os, pathlib, sys

import numpy as np

NPZ = pathlib.Path(__file__).resolve().parent / "vla_stage0b_rev_params.npz"
EXPECTED_NPZ_SHA = "766e607ad0bc739ea0a139172dd34e16d01a268cca80e990af5aab01006cfcd7"
CORPUS_DIR = pathlib.Path(__file__).resolve().parent / "vla_stage0c_corpus"
N_SLOTS, D_SLOT = 16, 384


class Stage0bRevEncoder:
    def __init__(self, npz_path=NPZ):
        raw = open(npz_path, "rb").read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != EXPECTED_NPZ_SHA:
            raise RuntimeError(
                f"npz SHA mismatch: {digest[:24]} != {EXPECTED_NPZ_SHA[:24]}")
        z = np.load(npz_path, allow_pickle=False)
        self.W = z["W"]                       # (192, 4) float32
        self.b = z["b"]                       # (192,)
        self.k = z["k"]                       # (16, 384) float32
        self.calib_mean = z["calib_mean"]     # (4,)
        self.calib_std = z["calib_std"]       # (4,)
        self._enabled = os.environ.get("HENRI_STAGE0B_REV_ENABLE") == "1"

    def encode(self, obs):
        """Encode obs -> (1,16,384) float32; default OFF -> byte-identical input."""
        x = np.asarray(obs, dtype=np.float32)
        if not self._enabled:
            return x
        if x.shape[-1] != 4:
            raise ValueError(f"expected last dim 4, got {x.shape[-1]}")
        batch = x.reshape(-1, 4)
        xs = (batch - self.calib_mean) / self.calib_std
        phase = xs @ self.W.T + self.b                 # (n, 192)
        phi = np.concatenate([np.cos(phase), np.sin(phase)], axis=1)  # (n,384)
        phi = phi.astype(np.float64)
        out = np.zeros((batch.shape[0], N_SLOTS, D_SLOT), dtype=np.float32)
        kf = np.fft.rfft(self.k.astype(np.float64), axis=1)   # (16,193)
        for i in range(N_SLOTS):
            for n in range(batch.shape[0]):
                conv = np.fft.irfft(np.fft.rfft(phi[n]) * kf[i], n=D_SLOT)
                norm = np.linalg.norm(conv)
                out[n, i] = (conv / norm).astype(np.float32)
        return out


def load_corpus_lex():
    """Lexicographic filename order -> (calib_records, eval_records)."""
    manifest = json_load((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    files = sorted(manifest["files"].items())
    calib, evald = [], []
    for fname, finfo in files:
        recs = []
        for line in (CORPUS_DIR / fname).read_text(encoding="utf-8").splitlines():
            recs.append(json_parse(line))
        (calib if len(calib) < 171 else evald).extend(recs)
    return calib, evald


import json as _json
json_load = _json.loads
json_parse = _json.loads


def _class_source(cls):
    """Return the dedented executable source of a single class, docstring stripped.

    Raw-module scans false-fail because the audit harness itself contains the
    forbidden literals ('import torch', 'Parameter(', 'GymWrapper') in the
    forbidden-list expressions. Scan ONLY the audited class (skill rule:
    dedent + ast.parse + docstrings removed explicitly).
    """
    import ast, inspect, textwrap
    raw = textwrap.dedent(inspect.getsource(cls))
    tree = ast.parse(raw)
    if tree.body and isinstance(tree.body[0], ast.Expr) \
            and isinstance(tree.body[0].value, ast.Constant) \
            and isinstance(tree.body[0].value.value, str):
        tree.body = tree.body[1:]
    return ast.unparse(tree)


def main():
    src = _class_source(Stage0bRevEncoder)
    results = {}

    # C2: zero trainable state — source-level audit (assembled strings avoid
    # self-referential hits from the audit expressions themselves)
    forbidden = ["import torch", "Parameter(", "backward(", "optimizer",
                 "randn", "g" + "ym"]
    c2 = all(f not in src for f in forbidden)
    results["C2_zero_trainable"] = c2

    # C8: no env/learner access
    c8 = ("Stage0" + "GymWrapper" not in src) and ("gym" + "nasium" not in src) and ("redmd" not in src)
    results["C8_no_env_learner"] = c8

    # C1: default OFF bypass
    enc_off = Stage0bRevEncoder()
    probe = np.asarray([0.1, 0.2, -0.3, 0.4], dtype=np.float32)
    bypass = enc_off.encode(probe)
    results["C1_bypass"] = bool(np.array_equal(bypass, probe)) and bypass.dtype == probe.dtype

    # real observations for the remaining contracts
    calib, evald = load_corpus_lex()
    assert len(calib) == 171 and len(evald) == 133, (len(calib), len(evald))

    os.environ["HENRI_STAGE0B_REV_ENABLE"] = "1"
    enc = Stage0bRevEncoder()

    # C4: geometry + shape
    z = enc.encode(np.asarray(calib[0]["obs_t"], dtype=np.float32))
    norms = np.linalg.norm(z[0], axis=1)
    c4 = (z.shape == (1, N_SLOTS, D_SLOT)) and (z.dtype == np.float32) and \
         (float(np.max(np.abs(norms - 1.0))) <= 1e-6)
    results["C4_shape_geometry"] = c4
    results["C4_max_norm_err"] = float(np.max(np.abs(norms - 1.0)))

    # C6/C7: deduplicate real observations across calib obs_t + obs_next
    obs_list = []
    for rec in calib:
        obs_list.append(np.asarray(rec["obs_t"], dtype=np.float32).tobytes())
        obs_list.append(np.asarray(rec["obs_next"], dtype=np.float32).tobytes())
    uniq = {b: True for b in obs_list}.keys()
    arrs = [np.frombuffer(b, dtype=np.float32) for b in uniq]
    Z = enc.encode(np.stack(arrs)).reshape(len(arrs), N_SLOTS * D_SLOT).astype(np.float64)
    # pairwise L2 (upper triangle)
    d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    iu = np.triu_indices(len(arrs), k=1)
    pair_d = d[iu]
    frac = float(np.mean(pair_d > 1e-3))
    coll = int(np.sum(pair_d < 1e-12))
    results["C6_dedup_n"] = len(arrs)
    results["C6_frac_gt_1e-3"] = frac
    results["C6_min_l2"] = float(pair_d.min())
    results["C6_collisions"] = coll
    results["C6_sensitivity"] = frac >= 0.99 and coll == 0

    # C7: non-collapse on flat calib matrix
    Zc = enc.encode(np.stack([np.asarray(r["obs_t"], dtype=np.float32) for r in calib]))
    Zf = Zc.reshape(171, N_SLOTS * D_SLOT).astype(np.float64)
    s = np.linalg.svd(Zf, compute_uv=False)
    rank16 = int(np.sum(s > 1e-3 * s[0]))
    slot_std = Zc.reshape(171, N_SLOTS, D_SLOT).std(axis=0)
    min_std = float(slot_std.min())
    results["C7_rank_1e-3"] = rank16
    results["C7_min_slot_std"] = min_std
    results["C7_noncollapse"] = rank16 >= 16 and min_std >= 1e-3

    # C3/C5: cross-process output hash (compare across the two --hash runs)
    if "--hash" in sys.argv:
        hset = np.stack(arrs[:4])
        out = enc.encode(hset).tobytes()
        print("OUTPUT_SHA", hashlib.sha256(out).hexdigest())

    ok = all(v is True for k, v in results.items() if k.startswith("C") and isinstance(v, bool))
    print("CONTRACT_RESULTS", results)
    print("ALL_PASS", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
