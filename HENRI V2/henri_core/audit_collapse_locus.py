"""LOCALIZE THE TEST-TIME COLLAPSE.

The audit showed: 6 distinct queries -> 1 identical answer, yet the decoder
emits 16/16 distinct macro-tokens. So the information dies in ONE of:
  A) ingress (token -> psi)          cos(psi_ab, psi_dog) measured 0.0 -> NO
  B) swarm + consensus (psi -> psi_conv)   energy identical -> SUSPECT
  C) snap_text (psi_conv -> ids)     decoder internals non-degenerate -> SUSPECT

Method: instrument each stage separately and compare across DIFFERENT inputs.
Decisive test for (B): solve with use_swarm=False. If the output becomes
input-dependent, the swarm is the collapse locus.

Read-only.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")
from henri_core import cli as H_cli

torch.set_num_threads(8)
QS = ["ab", "dog", "cat", "entity acts on object in context"]


def cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def sep(t):
    print("\n" + "-" * 76)
    print(t)
    print("-" * 76)


def main() -> int:
    system, tok = H_cli.build(small=True, vocab=512)
    system.build_axioms(H_cli.CORPUS, tok)

    print("=" * 76)
    print("LOCALIZING THE TEST-TIME COLLAPSE")
    print("=" * 76)

    # ---- stage A: ingress -----------------------------------------------------
    sep("A. INGRESS  (token -> psi)")
    with torch.no_grad():
        psi_in = {q: system.wave_of(q, tok) for q in QS}
    print(f"  {'query':38s} {'|psi|':>8s}")
    for q in QS:
        print(f"  {q[:38]:38s} {float(psi_in[q].norm()):8.5f}")
    print("  pairwise cos(psi):")
    for i, a in enumerate(QS):
        for b in QS[i + 1:]:
            print(f"    {a[:16]:16s} vs {b[:16]:16s} = {cos(psi_in[a], psi_in[b]): .6f}")

    # ---- stage B: swarm + consensus -------------------------------------------
    sep("B. SWARM + CONSENSUS  (psi -> psi_conv)")
    conv, energy = {}, {}
    with torch.no_grad():
        for q in QS:
            out = system.swarm(psi_in[q].unsqueeze(0)[0], patterns=system.axiom_bank)
            win = system.consensus(out)
            conv[q] = win["psi"]
            energy[q] = (win["energy"], win["delta_h"], out["distinct_seeds"])
    print(f"  {'query':38s} {'energy':>10s} {'delta_h':>12s} {'seeds':>6s}")
    for q in QS:
        e, dh, ds = energy[q]
        print(f"  {q[:38]:38s} {e:10.6f} {dh:12.3e} {ds:6d}")
    print(f"  energy spread across queries : "
          f"{max(e[0] for e in energy.values()) - min(e[0] for e in energy.values()):.3e}")
    print("  pairwise cos(psi_conv)  <-- if ~1.0, the swarm ERASES input identity:")
    for i, a in enumerate(QS):
        for b in QS[i + 1:]:
            print(f"    {a[:16]:16s} vs {b[:16]:16s} = {cos(conv[a], conv[b]): .6f}")

    # ---- stage C: decoder snap ------------------------------------------------
    sep("C. SNAP_TEXT  (psi -> token ids), from psi_conv vs from raw psi_in")
    with torch.no_grad():
        print(f"  {'query':38s} {'ids(conv)':>16s} {'ids(raw)':>16s}")
        for q in QS:
            i_conv, p_conv = system.decoder.snap_text(conv[q].unsqueeze(0))
            i_raw, p_raw = system.decoder.snap_text(psi_in[q].unsqueeze(0))
            print(f"  {q[:38]:38s} {str(i_conv.tolist()):>16s} {str(i_raw.tolist()):>16s}")
        # are the DECODER INTERNALS input-dependent?
        print("\n  macro-token divergence (internal, before the head):")
        tgts = {}
        for q in QS:
            b, t = system.decoder.encode_wave(psi_in[q].unsqueeze(0))
            tgts[q] = t
        for i, a in enumerate(QS):
            for b in QS[i + 1:]:
                print(f"    {a[:16]:16s} vs {b[:16]:16s} = {cos(tgts[a], tgts[b]): .6f}")

    # ---- decisive test: does removing the swarm restore input dependence? -----
    sep("D. DECISIVE: solve(use_swarm=False) -- bypass swarm+consensus")
    print(f"  {'query':38s} {'swarm=True':>22s} {'swarm=False':>22s}")
    for q in QS:
        r1 = system.solve(q, tok, use_swarm=True)
        r2 = system.solve(q, tok, use_swarm=False)
        print(f"  {q[:38]:38s} {str(r1['token_ids']):>22s} {str(r2['token_ids']):>22s}")
    ids_on = {q: tuple(system.solve(q, tok, use_swarm=True)["token_ids"]) for q in QS}
    ids_off = {q: tuple(system.solve(q, tok, use_swarm=False)["token_ids"]) for q in QS}
    print(f"\n  distinct answers WITH swarm    : {len(set(ids_on.values()))} of {len(QS)}")
    print(f"  distinct answers WITHOUT swarm : {len(set(ids_off.values()))} of {len(QS)}")
    print(f"  -> collapse locus: "
          f"{'SWARM/CONSENSUS' if len(set(ids_off.values())) > len(set(ids_on.values())) else 'NOT the swarm alone'}")

    # ---- is the veto gating the answer? --------------------------------------
    sep("E. SAGNAC VETO  (does a dark port suppress the answer?)")
    for q in QS:
        r = system.solve(q, tok, use_swarm=True)
        print(f"  {q[:30]:30s} allow={str(r['sagnac']['allow']):5s} "
              f"delta={r['sagnac']['delta']:.5f} thr={r['sagnac']['threshold']} "
              f"reason={r['sagnac']['reason']}")

    print("\nDONE (read-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
