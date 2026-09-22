#!/usr/bin/env python3
"""Ingress probe: is the random-phasor codebook metrically flat, and does RFSS fix it?

THE CLAIM UNDER TEST
    The audit states the live ingress builds tokens as independent random phasors,
    giving <Psi_i, Psi_j> ~ 0 for all i != j, so semantic topology is annihilated.
    The remedy is Role-Filler Sum Superposition. Both halves are MEASURED here
    against a matched control; neither is asserted.

INSTRUMENT DISCIPLINE
    The two bases are built at the SAME shape, SAMe dtype, and SAME normalization
    (unit L2 norm). Without that, a scale difference alone would fabricate a
    separation for whichever arm happened to have larger norms. That exact error is
    documented in henri_grounded_lexical_codec.py, where a scale conflation produced
    a fake +18 gap. A self-check asserts the norms match before any gap is reported.

GATES
  I1  self-check   both bases are unit-norm per row (no scale conflation)
  I2  self-check   the random control is metrically FLAT: its cosine distribution is
                   centred on 0, confirming the defect is real and not assumed
  I3  THE FIX      RFSS cosine correlates strongly POSITIVELY with the number of
                   shared role-value pairs. This is the operational definition of
                   "semantic topology exists".
  I4  siblings     tokens sharing class+bucket+parity are measurably closer than
                   tokens sharing nothing (e.g. 'a' vs 'b' vs 'a' vs 'Z')
  I5  unbinding    a role can be recovered from the RFSS token key (round trip)
  I6  determinism  the same token maps to the same wavefront across instances
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from henri_role_filler_ingress import (  # noqa: E402
    RoleFillerIngressCodebook, build_random_basis_control, char_roles)

OUT = os.path.join(HERE, "ingress_role_filler_probe_observed.json")


def cos_matrix(basis: torch.Tensor) -> torch.Tensor:
    """[V, V] absolute cosine between rows; basis is [V, nb, 8] real."""
    flat = basis.reshape(basis.shape[0], -1).to(torch.float64)
    flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (flat @ flat.t()).abs()


def main() -> int:
    V, NB = 256, 1024          # reduced num_blocks: CPU-cheap, construction is dim-independent
    res: dict = {}
    fails: list = []

    rf = RoleFillerIngressCodebook(vocab_size=V, num_blocks=NB)
    # MATCHED NORMALIZATION. Both bases are built at normalize="row" so ||row||=1 on
    # both sides before any cosine is computed. An earlier draft compared an RFSS
    # basis at row-norm 1 against a control at row-norm sqrt(num_blocks)=32 (the live
    # tokenizer's `dim=-1` convention) and the scale gate fired, correctly: any
    # consumer that thresholds a RAW inner product or L2 distance would have seen the
    # wrong magnitude. Cosine is scale-invariant, so the metric conclusions below were
    # never affected -- but the comparison is now unambiguous.
    rnd = build_random_basis_control(V, NB, seed=0, normalize="row")
    rfb = rf.build_basis(normalize="row")

    # Separately confirm the BLOCK mode reproduces the live tokenizer convention
    # exactly, so the module can be a drop-in replacement without changing scale.
    rfb_block = rf.build_basis(normalize="block")
    rnd_block = build_random_basis_control(V, NB, seed=0, normalize="block")
    res["I1b_rfss_block_row_norm_mean"] = float(
        rfb_block.norm(dim=(1, 2)).mean().item())
    res["I1b_random_block_row_norm_mean"] = float(
        rnd_block.norm(dim=(1, 2)).mean().item())
    res["I1b_expected_block_row_norm_sqrt_nb"] = math.sqrt(NB)
    for name, v_ in (("rfss", res["I1b_rfss_block_row_norm_mean"]),
                     ("random", res["I1b_random_block_row_norm_mean"])):
        if abs(v_ - math.sqrt(NB)) > 1e-2:
            fails.append(f"I1b: {name} block-mode row norm {v_:.4f} != sqrt(num_blocks) "
                         f"{math.sqrt(NB):.4f}; drop-in replacement would change scale")

    # ------------------------------------------------------------------ I1
    for name, b in (("rfss", rfb), ("random", rnd)):
        n = b.reshape(b.shape[0], -1).norm(dim=-1)
        res[f"I1_{name}_norm_min"] = float(n.min().item())
        res[f"I1_{name}_norm_max"] = float(n.max().item())
        if abs(float(n.mean().item()) - 1.0) > 1e-3:
            fails.append(f"I1: {name} rows are not unit-norm (mean {n.mean():.4f}); "
                         f"any gap measured below would be a scale artifact")
    if not torch.allclose(rfb.norm(dim=(1, 2)), rnd.norm(dim=(1, 2)), atol=1e-3):
        fails.append("I1: the two bases have different row norms; comparison invalid")

    # ------------------------------------------------------------------ I2
    C_rnd = cos_matrix(rnd)
    off = C_rnd[~torch.eye(V, dtype=torch.bool)]
    res["I2_random_offdiag_mean"] = float(off.mean().item())
    res["I2_random_offdiag_std"] = float(off.std().item())
    res["I2_random_offdiag_p99"] = float(torch.quantile(off, 0.99).item())
    if abs(res["I2_random_offdiag_mean"]) > 0.05:
        fails.append(f"I2: the random control is NOT flat (off-diagonal mean "
                     f"{res['I2_random_offdiag_mean']:.4f}); the stated defect does "
                     f"not reproduce and the remedy's premise must be re-derived")

    # ------------------------------------------------------------------ I3
    C_rf = cos_matrix(rfb)
    shares, sims = [], []
    for i in range(V):
        for j in range(i + 1, V):
            shares.append(rf.shared_roles(i, j))
            sims.append(float(C_rf[i, j].item()))
    s = torch.tensor(shares, dtype=torch.float64)
    y = torch.tensor(sims, dtype=torch.float64)
    # Pearson correlation between shared-role count and cosine similarity
    if float(s.std().item()) > 0 and float(y.std().item()) > 0:
        corr = float((((s - s.mean()) * (y - y.mean())).sum()
                      / (s.std() * y.std() * len(s))).item())
    else:
        corr = 0.0
    res["I3_rfss_shared_vs_cosine_pearson"] = corr
    res["I3_rfss_cosine_mean"] = float(y.mean().item())
    res["I3_rfss_shared_mean"] = float(s.mean().item())
    # Same correlation for the control: must be ~0.
    shares_r, sims_r = [], []
    for i in range(0, V, 7):
        for j in range(i + 1, V, 11):
            shares_r.append(rf.shared_roles(i, j))
            sims_r.append(float(C_rnd[i, j].item()))
    sr = torch.tensor(shares_r, dtype=torch.float64)
    yr = torch.tensor(sims_r, dtype=torch.float64)
    corr_r = (float((((sr - sr.mean()) * (yr - yr.mean())).sum()
                     / (sr.std() * yr.std() * len(sr))).item())
              if float(sr.std().item()) > 0 and float(yr.std().item()) > 0 else 0.0)
    res["I3_random_shared_vs_cosine_pearson"] = corr_r
    if corr < 0.30:
        fails.append(f"I3: RFSS shared/cosine correlation is only {corr:.4f}; the "
                     f"construction does not create usable metric structure")
    if abs(corr_r) > 0.15:
        fails.append(f"I3: the random control also shows correlation {corr_r:.4f}; "
                     f"the separation is not attributable to the construction")

    # ------------------------------------------------------------------ I4
    pairs = [("a", "b"), ("a", "c"), ("0", "1"), ("a", "Z"), ("0", "A"), (" ", "a")]
    detail = {}
    for x, y_ in pairs:
        i, j = ord(x), ord(y_)
        detail[f"{x!r} vs {y_!r}"] = {
            "shared_roles": rf.shared_roles(i, j),
            "rfss_cos": float(C_rf[i, j].item()),
            "random_cos": float(C_rnd[i, j].item()),
        }
    res["I4_pairs"] = detail
    sib = float(C_rf[ord("a"), ord("b")].item())
    dis = float(C_rf[ord("a"), ord("Z")].item())
    res["I4_sibling_ab"] = sib
    res["I4_distant_aZ"] = dis
    res["I4_sibling_minus_distant"] = sib - dis
    if sib <= dis:
        fails.append(f"I4: sibling 'a'/'b' ({sib:.4f}) is not closer than distant "
                     f"'a'/'Z' ({dis:.4f})")

    # ------------------------------------------------------------------ I5
    # Unbind the 'class' role from a token key and recover the value index.
    ri = rf._rindex["class"]
    k = rf.encode_token(ord("a"))            # complex [D]
    un = torch.fft.ifft(torch.fft.fft(k) * torch.conj(torch.fft.fft(rf.role_keys[ri])))
    vk = rf.value_keys["class"]
    sims_v = torch.einsum("d,vd->v", un.conj(), vk).abs()
    got = int(sims_v.argmax().item())
    want = rf._vindex["class"][char_roles(ord("a"))["class"]]
    res["I5_unbound_class_index"] = got
    res["I5_expected_class_index"] = want
    res["I5_unbinding_ok"] = got == want
    if got != want:
        fails.append(f"I5: unbinding recovered class index {got}, expected {want}")

    # ------------------------------------------------------------------ I6
    rf2 = RoleFillerIngressCodebook(vocab_size=V, num_blocks=NB)
    same = bool(torch.equal(rf.encode_token(ord("q")).real,
                            rf2.encode_token(ord("q")).real))
    res["I6_deterministic"] = same
    if not same:
        fails.append("I6: the same token produced a different wavefront across "
                     "instances; codebooks are not reproducible")

    verdict = "PASS" if not fails else "FAIL"
    out = {"module": "ingress_role_filler_probe", "evidence_class": "OBSERVED",
           "vocab_size": V, "num_blocks": NB, "results": res,
           "gate_failures": fails, "verdict": verdict,
           "claim": ("The live random-phasor ingress is metrically flat; RFSS makes "
                     "similarity track shared role structure, so byte tokens acquire "
                     "a usable metric.")}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 78)
    print(f"INGRESS ROLE-FILLER PROBE  (vocab={V}, num_blocks={NB})")
    print("=" * 78)
    print(f"  I1 unit-norm  rfss={res['I1_rfss_norm_mean' if 'I1_rfss_norm_mean' in res else 'I1_rfss_norm_min']:.6f} "
          f"min / random min={res['I1_random_norm_min']:.6f}")
    print(f"  I2 random control off-diagonal cosine: mean {res['I2_random_offdiag_mean']:+.5f} "
          f"std {res['I2_random_offdiag_std']:.5f} p99 {res['I2_random_offdiag_p99']:.4f}")
    print(f"     => live ingress is {'FLAT (defect confirmed)' if abs(res['I2_random_offdiag_mean'])<0.05 else 'NOT flat'}")
    print(f"  I3 shared-roles vs cosine: RFSS {corr:+.4f}   random {corr_r:+.4f}")
    print(f"  I4 sibling a/b {sib:.4f}  vs  distant a/Z {dis:.4f}  "
          f"(delta {sib-dis:+.4f})")
    for k_, v_ in res["I4_pairs"].items():
        print(f"       {k_:<12} shared={v_['shared_roles']}  "
              f"rfss={v_['rfss_cos']:.4f}  random={v_['random_cos']:.4f}")
    print(f"  I5 unbinding class from 'a': got {got} want {want} -> {res['I5_unbinding_ok']}")
    print(f"  I6 deterministic across instances: {same}")
    print()
    if fails:
        print("GATE FAILURES:")
        for f_ in fails:
            print(f"  - {f_}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
