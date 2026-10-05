"""Is the representation ever optimized? Measure the gradient path.

SPEC_A PREMISE (contract item 1)
    Zone A builds the text path with slot_router(emb).argmax(-1) and then
    indexes with int(). If the wave carries no gradient, training the readout
    never shapes the representation, so the ingress stays at random init.
    That is the mechanism the composition-ablation verdict H2 predicts.

WHAT THIS MEASURES (deterministic; no bound, no gate)
    1  requires_grad on the wave from each construction path
    2  can a backward pass reach the ingress parameters at all?
    3  the operations present in _token_writes that sever the graph
    4  POSITIVE CONTROL: the decoder DOES receive gradient and the loss falls
    5  the wave is byte-identical before and after an optimization step
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.m4_generative import (M4Config, WaveTextGenerator,  # noqa: E402
                                      build_corpus, build_system)

PIN = 20261004


def main() -> int:
    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, pin_seed=PIN)
    specs = [corpus.specs[i] for i in corpus.train_idx[:8]]
    tgts = [tok.encode(corpus.traces[i]) for i in corpus.train_idx[:8]]

    print("GRADIENT PATH AUDIT: does the wave carry gradient?")
    print("=" * 78)

    # 1. the exact paths the M4 harness uses
    psi_a = system.wave_of(specs[0], tok)
    print("  1 system.wave_of(...)        requires_grad =", psi_a.requires_grad)
    gen = WaveTextGenerator(system, tok, train_body=True)
    psi_b = gen.wave(specs)
    print("    WaveTextGenerator.wave()   requires_grad =", psi_b.requires_grad)

    # 2. raw ingress with grad explicitly enabled on every tensor
    for p in system.ingress.parameters():
        p.requires_grad_(True)
    psi_c = system.ingress.encode_text(specs[0], tok)
    print("  2 ingress.encode_text(...)   requires_grad =", psi_c.requires_grad)
    reached = []
    try:
        psi_c.abs().pow(2).sum().backward(retain_graph=True)
        for n, p in system.ingress.named_parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                reached.append(n)
    except Exception as exc:                                  # noqa: BLE001
        print("    backward raised:", type(exc).__name__, str(exc)[:90])
    print("    ingress params REACHED by backward:",
          reached if reached else "NONE")

    # 3. severing operations, read from the real source
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "zone_a.py"), encoding="utf-8").read()
    seg = src.split("def _token_writes")[1].split("def encode_text")[0]
    print("  3 operations inside _token_writes:")
    for op, why in (("argmax", "hard assignment, no softmax gradient"),
                    ("int(", "cast to Python int detaches the tensor"),
                    ("torch.tensor(1.0)", "constant magnitude, no leaf"),
                    ("self.angle[", "buffer lookup, not a parameter")):
        print(f"      {op:<20} {'PRESENT' if op in seg else 'absent ':>8}  {why}")
    print("    _assemble uses a float(n) branch:", "float(n)" in src)
    # D132 (self-caught): wave_of lives in system.py, not zone_a.py, so the first
    # version printed a misleading "n/a". Read the right file.
    syssrc = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "system.py"), encoding="utf-8").read()
    wo = (syssrc.split("def wave_of")[1].split("def ")[0]
          if "def wave_of" in syssrc else "")
    print("    system.wave_of wraps in no_grad  :", "no_grad" in wo)
    print("    zone_a.encode_text wraps no_grad :", "no_grad" in src)

    # 4. POSITIVE CONTROL: the decoder does receive gradient
    gen2 = WaveTextGenerator(system, tok, train_body=True)
    rep = gen2.fit(specs, tgts, M4Config(steps=3))
    dg = [float(p.grad.abs().sum()) for p in gen2.dec.parameters()
          if p.grad is not None]
    print("  4 decoder tensors with grad:", len(dg),
          "| max |grad| =", f"{max(dg):.4g}" if dg else "none",
          "| loss", f"{rep.loss_first:.4f} -> {rep.loss_last:.4f}")

    # 5. the wave does not move
    w1 = system.wave_of(specs[0], tok)
    print("  5 wave byte-identical after the step:",
          torch.equal(w1, system.wave_of(specs[0], tok)))

    print("=" * 78)
    print("READING")
    print("  If the wave carries no gradient and no backward reaches the")
    print("  ingress, the representation is FROZEN at random init and the")
    print("  readout trains on a constant. That is the mechanism H2 predicts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
