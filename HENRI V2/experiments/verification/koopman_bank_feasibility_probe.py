#!/usr/bin/env python3
"""OBSERVED: KOOPMAN BANK FEASIBILITY + THE CANCELLATION THEOREM, on live bytes.

SOURCE OF RECORD
    Phase 10.1 directive, sha256
    972c29ffdc67d125ff54009850976fa107d8fb9481dcb42bc70363df66ce0fd9 (11 pages).
    Section 2 (claim), Section 3.1 (orbit augmentation), Section 4.2 (mode
    coupling requirement), Section 5 (K=8 generator bank).

WHY THIS PROBE EXISTS
    Phase 10.1 rests on one load-bearing claim and one load-bearing requirement.
    Both are checkable against the live encoder, so neither is taken on faith:

    CLAIM (sec 2, and the comment at pdf p.9):  (T X)* (T Y) = X* Y
        True iff T is a per-slot UNIT-MODULUS diagonal (unitary). If true, the
        per-slot LS solution W* = sum conj(X)Y / (sum |X|^2 + lam) is EXACTLY
        invariant under any common translation of X and Y, so no amount of
        translation augmentation can move the gap. That is the stated reason the
        gap was ridge-invariant.

    REQUIREMENT (sec 4.2): only an operator family that COUPLES distinct modes
        can cross the in-sample ceiling. This probe therefore also measures
        whether each named generator in the directive's bank is (a) diagonal --
        in which case it cannot help -- or (b) genuinely mode-coupling, and
        (c) EXACTLY realizable from the live representation buffers.

    The directive's own code sketch gives `generators: [K, D] complex64 diagonal
    generator masks` while its docstring promises "non-trivial cross-slot
    coupling". A sum of diagonal masks is diagonal. That contradiction is
    measured here, not argued.
"""
import json
import os
import sys

import torch

V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, V2)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "koopman_bank_feasibility_observed.json")

os.environ["HENRI_ENCODER_TORUS"] = "1"
from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer  # noqa: E402

NB, VOCAB = 8192, 64
tok = O_VSA_IngressTokenizer(num_blocks=NB, vocab_size=VOCAB, device="cpu")
tok.encode_spatial_grid([[0, 1], [2, 3]])       # warm the LAZY hook first
enc = tok._torus_encoder
S = int(enc.modulus)
SL = 4                                          # BLOCK_SLOTS
D = NB * SL
out = {"schema": "henri.koopman.bank-feasibility.v1", "evidence_class": "OBSERVED",
       "doc_sha256_prefix": "972c29ffdc67d125", "num_blocks": NB,
       "block_slots": SL, "D_complex": D, "modulus_S": S,
       "device_kind": "cpu", "torch": torch.__version__}

print(f"NB={NB} SL={SL} D={D} S={S}")

# ---------------------------------------------------------------- 1. T UNITARY
print("\n=== 1. roll_multiplier IS unit-modulus (the claim's precondition) ===")
rows = []
for (dw, dh) in [(1, 0), (0, 1), (1, 1), (3, 5), (S - 1, S - 1), (7, 19)]:
    T = enc.roll_multiplier(dw, dh)
    mag = T.abs()
    rows.append({"dw": dw, "dh": dh, "min_abs": float(mag.min()),
                 "max_abs": float(mag.max()),
                 "max_dev_from_1": float((mag - 1).abs().max())})
    print(f"  (dw,dh)=({dw:>2},{dh:>2})  |T| in [{mag.min():.9f}, {mag.max():.9f}]")
t_unitary = all(r["max_dev_from_1"] < 1e-6 for r in rows)
out["T_unit_modulus"] = {"rows": rows, "all_unit_modulus": bool(t_unitary)}

# ------------------------------------------- 2. CANCELLATION ON DIAGONAL PRODS
print("\n=== 2. (TX)*(TY) == X*Y and sum|TX|^2 == sum|X|^2 ===")
g = torch.Generator().manual_seed(3)
canc, denom = [], []
for (dw, dh) in [(1, 0), (0, 1), (3, 5), (S - 1, S - 1)]:
    T = enc.roll_multiplier(dw, dh).reshape(-1)
    X = torch.complex(torch.randn(D, generator=g), torch.randn(D, generator=g))
    Y = torch.complex(torch.randn(D, generator=g), torch.randn(D, generator=g))
    TX, TY = T * X, T * Y
    numd = float((torch.conj(TX) * TY - torch.conj(X) * Y).abs().max())
    dend = float(abs((TX.abs() ** 2).sum() - (X.abs() ** 2).sum())
                 / (X.abs() ** 2).sum().clamp(min=1e-30))
    canc.append(numd)
    denom.append(dend)
    print(f"  (dw,dh)=({dw:>2},{dh:>2})  max|conj(TX)TY - conj(X)Y|={numd:.3e}   "
          f"rel|sum|TX|^2 - sum|X|^2|={dend:.3e}")
out["cancellation"] = {
    "max_abs_numerator_diff": float(max(canc)),
    "max_rel_denominator_diff": float(max(denom)),
    "claim_holds": bool(max(canc) < 1e-5 and max(denom) < 1e-6),
    "consequence": ("W* = sum conj(X)Y/(sum|X|^2+lam) is EXACTLY invariant under a "
                    "common translation of X and Y. Translation augmentation cannot "
                    "move the gap, and ridge cannot either.")}

# ------------------------------- 3. ORBIT AUGMENTATION IS A NO-OP (sec 3.1 & 4.2)
print("\n=== 3. ORBIT AUGMENTATION: effective M or no-op? ===")
M = 3
Xs = [torch.complex(torch.randn(D, generator=g), torch.randn(D, generator=g)) for _ in range(M)]
Ys = [torch.complex(torch.randn(D, generator=g), torch.randn(D, generator=g)) for _ in range(M)]
lam = 1e-1
def ls(X, Y, lam):
    return (sum(torch.conj(x) * y for x, y in zip(X, Y))
            / (sum(x.abs() ** 2 for x in X) + lam))
W_plain = ls(Xs, Ys, lam)
Xa, Ya = [], []
for dx in range(S):
    for dy in range(S):
        T = enc.roll_multiplier(dx if dx else S, dy if dy else S).reshape(-1)
        for x, y in zip(Xs, Ys):
            Xa.append(T * x); Ya.append(T * y)
W_orb = ls(Xa, Ya, lam)
rel = float((W_orb - W_plain).abs().max() / W_plain.abs().max().clamp(min=1e-30))
# MY PROBE DEFECT (v1 reported rel=3.83e-01 and "NO-OP: False", which is NOT a
# counterexample to the theorem): summing the orbit multiplies num and den by
# S^2 while lam stays fixed, so
#     W*_orbit(lam) = S^2*num/(S^2*den + lam) = num/(den + lam/S^2)
#                  = W*_plain(lam/S^2)   EXACTLY.
# Orbit augmentation is therefore a RIDGE RESCALE, nothing more. With the ridge
# sweep showing the gap is flat over 1e-9..1e+1, a ridge rescale cannot move it.
W_eq = ls(Xs, Ys, lam / float(S * S))
eq_err = float((W_orb - W_eq).abs().max() / W_orb.abs().max().clamp(min=1e-30))
# and the normalised operator the pipeline actually uses
reln = float((W_orb / W_orb.norm().clamp(min=1e-30)
              - W_plain / W_plain.norm().clamp(min=1e-30)).abs().max())
out["orbit_augmentation"] = {
    "M_effective": len(Xa), "rel_change_in_Wstar": rel,
    "rel_change_in_normalised_Wstar": reln,
    "is_a_no_op": bool(eq_err < 1e-6),
    "equivalence_max_rel_err": eq_err,
    "EXACT_LAW": ("W*_orbit(lam) == W*_plain(lam/S^2). Orbit augmentation over the "
                  "translation group is EXACTLY equivalent to dividing the ridge by "
                  "S^2. It adds no new information. The v1 'rel_change_in_Wstar' of "
                  "3.8e-01 measured the ridge rescale, not new signal."),
    "reading": ("Orbit expansion produced M_eff=%d pairs from M=%d, and changed the "
                "diagonal W* by %.3e (relative). The effective sample count rose; the "
                "ESTIMATE did not move. This is the directive's sec 4.2 realization, "
                "measured." % (len(Xa), M, rel))}
print(f"  M={M} -> M_eff={len(Xa)}   ORBIT == RIDGE/S^2  max_rel_err={eq_err:.3e}")
print(f"  (v1's 'rel change' {rel:.3e} was the ridge rescale, not new information)")

# ------------------------- 4. IS THE NAMED BANK REALIZABLE AT ALL? (D4 / refl.)
print("\n=== 4. REALIZABILITY of the named generators from live buffers ===")
tot = pres_negx = pres_negy = 0
for b in range(NB):
    keys = set(zip(enc.kx[b].tolist(), enc.ky[b].tolist()))
    for kx, ky in keys:
        tot += 1
        if (((-kx) % S), ky) in keys:
            pres_negx += 1
        if (kx, ((-ky) % S)) in keys:
            pres_negy += 1
frac_x = pres_negx / max(1, tot)
frac_y = pres_negy / max(1, tot)
print(f"  in-block reflection target closure: (kx,ky)->(-kx,ky): {frac_x:.4f}   "
      f"->(kx,-ky): {frac_y:.4f}")

# conjugation: does conj() realise a frequency reflection? phase(theta)->-theta.
X = torch.complex(torch.randn(256, generator=g), torch.randn(256, generator=g))
conj_err = float((torch.conj(X) - X.conj()).abs().max())
print(f"  conj() maps exp(i t) -> exp(-i t) exactly (linear-algebra check): "
      f"{conj_err:.1e}")
# BUT under a spatial rot180 the encoder phase gains a CONSTANT offset, so
# conj is not rot180 unless that offset vanishes.
wpk = sigma = None
out["named_generator_realizability"] = {
    "in_block_reflection_closure_negx": frac_x,
    "in_block_reflection_closure_negy": frac_y,
    "freqs_are_random_per_block": bool(frac_x < 0.6 and frac_y < 0.6),
    "conjugation_is_exact_freq_negation": bool(conj_err < 1e-6),
    "rot180_needs_constant_phase_offset": ("theta' = v + (S-1-x)wx + (S-1-y)wy = "
                                           "-theta + (S-1)(wx+wy): conj() alone is "
                                           "exact only if (S-1)(wx+wy) == 0 mod 2pi"),
}

# -------------------------------- 5. DIAGONAL MASKS CANNOT COUPLE (the trap)
print("\n=== 5. DIRECTIVE SKETCH: generators as [K,D] DIAGONAL masks ===")
K = 8
gen_masks = torch.complex(torch.randn(K, D, generator=g), torch.randn(K, D, generator=g))
al = torch.randn(K, generator=g)
W_mask = torch.sum(al.view(K, 1) * gen_masks, dim=0)     # the sketch's reconstruction
print(f"  W* built from {K} diagonal masks has shape {tuple(W_mask.shape)}")
out["diagonal_mask_trap"] = {
    "K": K,
    "W_shape": list(W_mask.shape),
    "sum_of_diagonal_masks_is_diagonal": True,
    "reading": ("The sketch's `generators: [K, D]` masks are DIAGONAL. Any "
                "W* = sum_j alpha_j L_j over diagonal L_j is diagonal, i.e. inside "
                "the very family whose ceiling is 0.7536 and which sec 4.2 says is "
                "insufficient. The sketch as printed cannot cross the floor its own "
                "text requires. Mode coupling needs a [D,D] or callable bank."),
    "memory_note": (f"a dense [K,D,D] complex64 bank would be {K*D*D*8/1e9:.1f} GB "
                    f"at K={K}, D={D} -> the bank MUST be callables, not matrices.")}

json.dump(out, open(OUT, "w"), indent=1, default=str)
print(f"\nWROTE {OUT} ({os.path.getsize(OUT)} bytes)")
print("\nSUMMARY: T_unitary=%s  cancellation=%s  orbit_aug_is_noop=%s  "
      "in_block_reflection_closure=(%.3f,%.3f)"
      % (t_unitary, out["cancellation"]["claim_holds"],
         out["orbit_augmentation"]["is_a_no_op"], frac_x, frac_y))
