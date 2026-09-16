#!/usr/bin/env python3
"""OBSERVED: is the torus encoder invertible? (Phase 10.4 directive 4)

Measures, on local CPU:
  0. VERIFICATION  -- does my explicit operator reproduce `enc.encode` exactly?
  1. FREQUENCY CENSUS -- does the S x S lattice the directive's IDFT spec assumes exist?
  2. RANK          -- is the pre-normalisation operator full column rank?
  3. MP ROUNDTRIP  -- A+ A n == n with the KNOWN accumulator (theory check)
  4. STORED-WAVE DECODE -- recover the grid from the normalised wave (the real test)
  5. RECEIPT

Writes experiments/verification/torus_encoder_adjoint_observed.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(ROOT)) if os.path.basename(ROOT) == "verification" \
    else os.path.dirname(ROOT)
sys.path.insert(0, V2)

from o_vsa_torus_encoder import TorusIngressEncoder          # noqa: E402
import torus_encoder_adjoint as TEA                          # noqa: E402

ARC_ROOT = os.environ.get("ARC_CORPUS", "C:/Users/chan/henri_data/ARC-AGI/data")
# Receipt path. `ADJ_OUT` exists so TESTS can redirect the write into tmp_path: the
# canonical path is SEALED and bound to the doc by validate_seal_consistency.PAIRS,
# so no test may write it.
OUT = os.environ.get("ADJ_OUT") or os.path.join(
    ROOT, "torus_encoder_adjoint_observed.json")
N_BLOCKS = int(os.environ.get("ARC_NB", 8192))
VOCAB = 256
MODULUS = 32
SEED = 20260914
TOL_VERIFY = float(os.environ.get("ADJ_TOL", 1e-3))   # enforced section-0 bound


def load_arc(root, n):
    tasks = []
    for split in ("training", "evaluation"):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            try:
                t = json.load(open(os.path.join(d, f), encoding="utf-8"))
            except Exception:
                continue
            tasks.append((f, t))
            if len(tasks) >= n:
                return tasks
    return tasks


def main():
    t0 = time.time()
    dev = "cpu"
    enc = TorusIngressEncoder(num_blocks=N_BLOCKS, vocab_size=VOCAB, modulus=MODULUS,
                              mode="TORUS_VAL", seed=SEED, device=dev)
    out = {
        "schema": "henri.arc.torus-adjoint-eval.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_class": "OBSERVED", "device_kind": dev,
        "torch": torch.__version__, "n_blocks": int(enc.num_blocks),
        "modulus_S": int(enc.modulus), "seed": SEED,
        "directive": {
            "doc_sha256": ("1ea527bd87c352839ee2aba0b88080b1aed9033c4268b5681cda6313e7a7b2ef"),
            "set_member_sha256_suffix1": (
                "615d6f63b5329098"  # the (1) copy that carries the Phase 10.4 pages
            ),
            "claim_tested": "spec 4.1: adjoint == 2D IDFT over an S x S lattice; "
                            "value channel is an 8-blade Clifford multivector",
        },
    }
    print(f"=== encoder num_blocks={enc.num_blocks} S={enc.modulus} "
          f"dc_slots={enc.dc_slots} dc_weight={enc.dc_weight:.3e} ===")

    # ---- 0. VERIFICATION: my operator must reproduce enc.encode exactly --------
    print("\n=== 0. operator verification (my algebra vs enc.encode) ===")
    ver = []
    for grid in ([[0, 1], [2, 3]], [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
                 [[0, 3, 5, 7], [2, 2, 8, 9], [1, 6, 4, 3], [0, 0, 5, 5]]):
        H, W = len(grid), len(grid[0])
        V = max(max(r) for r in grid) + 1
        n = TEA.indicator_from_grid(grid, V)
        mine = TEA.forward_wave(enc, n, H, W)
        theirs = enc.encode(grid)
        err = float((mine - theirs).abs().max())
        ver.append({"grid": f"{H}x{W}", "max_abs_err": err})
        print(f"  {H}x{W}: max|mine - enc.encode| = {err:.3e}")
    out["verification"] = ver
    out["verification_max_err"] = max(v["max_abs_err"] for v in ver)
    # ---- ENFORCED GATE (defect V4-NOGATE, found by audit) ------------------------
    # Section 0 previously COMPUTED and RECORDED this error but never CHECKED it, so
    # "verification-first" was a claim without a mechanism -- and adj1.log shows the
    # consequence: the pre-fix adjoint died on a tensor-SHAPE RuntimeError, not on a
    # verification failure, so the safeguard was never actually exercised.
    #     ERROR-PATH ONLY: the sealed run measures 1.581e-05 against this 1e-3 bound
    # (63x margin), so no emitted key or value changes and the scored content is
    # unaffected. Deliberately NOT added to `out`: an emitted key would change the
    # receipt schema. Firing is proven by tests/contract/test_torus_adjoint_gate.py.
    # (Note: the receipt FILE cannot reproduce byte-identically on a re-run because its
    #  `utc` and `elapsed_secs` fields always differ. The claim is unchanged CONTENT.)
    if not (out["verification_max_err"] < TOL_VERIFY):
        raise RuntimeError(
            f"operator verification FAILED: max err {out['verification_max_err']:.3e} "
            f">= tol {TOL_VERIFY:.0e}. The explicit algebra does NOT reproduce "
            f"enc.encode, so every downstream rank/decode number would be meaningless. "
            f"Refusing to continue (no receipt written).")

    # ---- 1. frequency census --------------------------------------------------
    print("\n=== 1. frequency census (does the S x S lattice exist?) ===")
    cen = TEA.frequency_census(enc)
    out["frequency_census"] = cen
    for k, v in cen.items():
        print(f"  {k:32s} {v}")

    # ---- 2. rank of the pre-normalisation operator ----------------------------
    print("\n=== 2. PRE-normalisation rank (small grids) ===")
    ranks = {}
    for (H, W, V) in ((4, 4, 3), (6, 6, 4), (8, 8, 4)):
        r = TEA.rank_test(enc, H, W, V)
        ranks[f"{H}x{W}xV{V}"] = r
        print(f"  {H}x{W} V={V}: rank={r['rank']}/{r['unknowns']} "
              f"eqs={r['equations']} full={r['full_column_rank']} "
              f"cond={r['cond']:.3e}")
    out["rank_pre_normalisation"] = ranks

    # ---- 3. Moore-Penrose round-trip (theory: is information there?) ----------
    print("\n=== 3. A+ A round-trip with the KNOWN accumulator ===")
    mp = []
    for grid in ([[0, 1, 2], [3, 2, 1], [0, 0, 1]], [[1, 0], [0, 2]]):
        r = TEA.mp_roundtrip(enc, grid, max(max(rw) for rw in grid) + 1)
        mp.append({"grid": f"{len(grid)}x{len(grid[0])}", **r})
        print(f"  {len(grid)}x{len(grid[0])}: exact={r['exact']} "
              f"max_err={r['max_abs_err']:.3e} grid_exact={r['grid_exact']}")
    out["mp_roundtrip_pre_norm"] = mp

    # ---- 4. decode from the STORED (normalised) wave --------------------------
    print("\n=== 4. decode from the STORED wave (the real test) ===")
    dec = []
    cases = [("synthetic3x3", [[0, 1, 2], [3, 2, 1], [0, 0, 1]]),
             ("synthetic4x4", [[0, 1, 0, 1], [2, 2, 3, 3], [1, 0, 1, 0], [3, 3, 2, 2]])]
    for tid, t in load_arc(ARC_ROOT, 40):
        for i, p in enumerate(t.get("train", [])[:1]):
            a = np.asarray(p["input"], dtype=np.int64)
            if a.ndim == 2 and 2 <= a.shape[0] <= 12 and 2 <= a.shape[1] <= 12:
                cases.append((f"{tid}#{i}", a.tolist()))
                break
        if len(cases) >= 8:
            break
    for name, grid in cases:
        H, W = len(grid), len(grid[0])
        V = max(max(r) for r in grid) + 1
        w = enc.encode(grid)
        t1 = time.time()
        r = TEA.decode_wave_to_grid(enc, w, H, W, V, steps=300)
        rec = r["grid"]
        exact = bool(torch.equal(rec, torch.as_tensor(grid, dtype=torch.long)))
        cell_acc = float((rec == torch.as_tensor(grid, dtype=torch.long)).float().mean())
        dec.append({"case": name, "H": H, "W": W, "V": V, "exact": exact,
                    "cell_accuracy": cell_acc, "mean_block_cos": r["mean_block_cos"],
                    "min_block_cos": r["min_block_cos"],
                    "secs": round(time.time() - t1, 2)})
        print(f"  {name:16s} {H}x{W} V={V}: exact={exact} "
              f"cell_acc={cell_acc:.4f} cos={r['mean_block_cos']:.6f} "
              f"({dec[-1]['secs']}s)")
    out["decode_from_stored_wave"] = dec
    n_exact = sum(d["exact"] for d in dec)
    out["decode_summary"] = {
        "n_cases": len(dec), "n_exact": n_exact,
        "exact_rate": n_exact / max(1, len(dec)),
        "mean_cell_accuracy": float(np.mean([d["cell_accuracy"] for d in dec])) if dec else 0.0,
    }

    out["elapsed_secs"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n=== elapsed {out['elapsed_secs']}s -> {OUT}")
    print(f"=== decode: {n_exact}/{len(dec)} exact, "
          f"mean cell acc {out['decode_summary']['mean_cell_accuracy']:.4f}")


if __name__ == "__main__":
    main()
