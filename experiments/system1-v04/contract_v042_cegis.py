"""
Contract tests: system1_kernel_v042_cegis_beam.py (CPU, disposable).
===========================================================================
Verifies, on the REAL calibrated v0.4.1 checkpoint (energy head
rho=0.4383/AUROC=0.7531) and REAL tasks:

  C1  beta=0.0 decode is BYTE-IDENTICAL to System1KernelV04.beam_decode
      (same scores, same selected program) — standard beam preserved.
  C2  beta=0.40 changes at least one beam trajectory vs beta=0 (engagement)
      on a task set with non-degenerate energy.
  C3  FSA validity: every candidate expansion obeys the token FSA
      (grammar_compliance == 1.0 for final selected sequences).
  C4  Determinism: same checkpoint + same task -> identical outputs on
      repeat runs for both arms.
  C5  Energy sanity: candidate energies are in [0,1], not all equal
      (head is discriminative, not collapsed), and energy of a candidate
      differs from raw-embedding energy (OOD guard) — i.e. the decoder
      uses core-unrolled latents.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from system1_kernel_v041_energy_refactored import (  # noqa: E402
    TOK2ID, KernelV04Config, System1KernelV04, grammar_compliance,
    detokenize)
from system1_kernel_v042_cegis_beam import (  # noqa: E402
    CEGISBeamPriorityDecoder)
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sig_ids, sig_matrix, pad_tokens, fp_of)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(0)
    random.seed(0)

    cfg = KernelV04Config()
    model = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(st["model"])
    model.eval()
    dec = CEGISBeamPriorityDecoder(model)

    rng = random.Random(12345)
    tasks = [gen_task(rng) for _ in range(6)]

    z0 = model.encode_tokens(pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(model, tasks, 16, dev)

    # ---- C1: beta=0 == beam_decode (byte-identical programs) ----
    c1_ok = True
    for i in range(len(tasks)):
        ref = model.beam_decode(z0[i:i + 1], sp[i:i + 1], width=16)
        got, rec = dec.decode_cegis_beam(
            z0[i:i + 1], sp[i:i + 1], beam_width=16, beta_priority=0.0)
        if ref != got:
            c1_ok = False
            print(f"  C1 MISMATCH task {i}: ref={ref} got={got}")
    print(f"C1 beta=0 identical to beam_decode: {'PASS' if c1_ok else 'FAIL'}")

    # ---- C2: beta=0.40 engagement (trajectory change) ----
    changed = 0
    for i in range(len(tasks)):
        _, r0 = dec.decode_cegis_beam(z0[i:i + 1], sp[i:i + 1],
                                      beam_width=16, beta_priority=0.0)
        _, r1 = dec.decode_cegis_beam(z0[i:i + 1], sp[i:i + 1],
                                      beam_width=16, beta_priority=0.40)
        if r1["best_score"] != r0["best_score"] or \
                r1["final_candidates"][0][0] != r0["final_candidates"][0][0]:
            changed += 1
    print(f"C2 beta=0.40 trajectory change: {changed}/{len(tasks)} "
          f"({'PASS' if changed >= 1 else 'FAIL'})")

    # ---- C3: FSA validity of selected sequences ----
    c3_ok = True
    for i in range(len(tasks)):
        for beta in (0.0, 0.40):
            seq, _ = dec.decode_cegis_beam(z0[i:i + 1], sp[i:i + 1],
                                           beam_width=16, beta_priority=beta)
            if grammar_compliance(seq) != 1.0:
                c3_ok = False
                print(f"  C3 FSA violation beta={beta} task {i}: {seq}")
    print(f"C3 FSA validity (both arms): {'PASS' if c3_ok else 'FAIL'}")

    # ---- C4: determinism ----
    c4_ok = True
    for beta in (0.0, 0.40):
        s1, r1 = dec.decode_cegis_beam(z0[0:1], sp[0:1], beam_width=16,
                                       beta_priority=beta)
        s2, r2 = dec.decode_cegis_beam(z0[0:1], sp[0:1], beam_width=16,
                                       beta_priority=beta)
        if s1 != s2 or r1["best_score"] != r2["best_score"]:
            c4_ok = False
            print(f"  C4 nondeterminism beta={beta}")
    print(f"C4 determinism: {'PASS' if c4_ok else 'FAIL'}")

    # ---- C5: energy sanity + core-unrolled (OOD guard) ----
    e_core: list[float] = []
    e_raw: list[float] = []
    for i in range(len(tasks)):
        seq, _ = dec.decode_cegis_beam(z0[i:i + 1], sp[i:i + 1],
                                       beam_width=16, beta_priority=0.0)
        ids = [TOK2ID["BOS"]] + seq
        e_core.append(dec.candidate_energy(ids, dev))
        zraw = model.encode_tokens(
            torch.tensor([ids], dtype=torch.long, device=dev))
        e_raw.append(float(model.energy(zraw).item()))
    c5a = all(0.0 <= e <= 1.0 for e in e_core)
    c5b = len(set(round(e, 4) for e in e_core)) >= 2   # non-degenerate
    c5c = any(abs(a - b) > 1e-3 for a, b in zip(e_core, e_raw))  # OOD guard
    print(f"C5 energy sanity: range={'PASS' if c5a else 'FAIL'} "
          f"non-degenerate={'PASS' if c5b else 'FAIL'} "
          f"core!=raw={'PASS' if c5c else 'FAIL'}")

    ok = c1_ok and changed >= 1 and c3_ok and c4_ok and c5a and c5b and c5c
    print(f"CONTRACT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
