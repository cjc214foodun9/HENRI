"""GRANULAR TEST-TIME AUDIT of Project HENRI (small config, CPU).

Read-only instrumentation. Answers: what does the system ACTUALLY compute at
test time, and is the output a function of the input?

Stages:
  S1 tokenize          -> token ids
  S2 ingress encode    -> ALIASING test cos('ab','ba'), cos('dog','god')
  S3 wave_of           -> psi, norm, requires_grad
  S4 axioms + solve    -> the full closed loop, per query
  S5 decoder egress    -> shaped logits
  S6 input dependence  -> are two DIFFERENT queries distinguishable?
  S7 order sensitivity -> does swapping token order change the answer?
  S8 perturbations     -> is the answer stable under irrelevant input change?
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")
from henri_core import cli as H_cli
from henri_core import substrate as sub

torch.set_num_threads(8)


def sep(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    d = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / d)


def main() -> int:
    sep("BUILD (small=True, the shipped default path)")
    system, tok = H_cli.build(small=True, vocab=512)
    print(f"  system class   : {type(system).__name__}")
    print(f"  dim            : {system.dim}   (full model: {sub.DEFAULT_DIM})")
    print(f"  vocab          : {system.vocab}")
    print(f"  ingress        : {type(system.ingress).__name__}")
    print(f"  swarm workers  : {system.swarm.B}  steps={system.swarm.steps}")
    dcfg = system.decoder.cfg
    from henri_core.model3_decoder import resolve_n_mem
    n_mem = resolve_n_mem(dcfg.d_model, dcfg.n_mem, dcfg.dk_target)
    print(f"  decoder        : d_model={dcfg.d_model} n_mem={n_mem} "
          f"d_k={dcfg.d_model // n_mem} n_macro={dcfg.n_macro} "
          f"n_layers={dcfg.n_layers} dk_target={dcfg.dk_target}")

    sep("S1  TOKENIZE")
    for q in ["ab", "ba", "dog", "god"]:
        print(f"  {q!r:6s} -> ids {tok.encode(q)}")

    sep("S2  INGRESS ENCODE -- THE ALIASING TEST")
    with torch.no_grad():
        e = {q: system.ingress.encode_text(q, tok) for q in
             ["ab", "ba", "dog", "god", "cat", "act"]}
    print(f"  shape {tuple(e['ab'].shape)}  dtype {e['ab'].dtype}")
    for p in [("ab", "ba"), ("dog", "god"), ("cat", "act"), ("ab", "dog")]:
        c = cos(e[p[0]], e[p[1]])
        flag = "  <== ORDER LOST" if (c > 0.99 and p[1] != "dog") else ""
        print(f"  cos({p[0]!r:6s},{p[1]!r:6s}) = {c: .10f}{flag}")

    sep("S3  WAVE_OF")
    with torch.no_grad():
        p = {q: system.wave_of(q, tok) for q in ["ab", "ba", "dog", "god"]}
    print(f"  shape {tuple(p['ab'].shape)}  dtype {p['ab'].dtype}")
    print(f"  |psi('ab')|_2      : {float(p['ab'].reshape(-1).norm()):.10f}")
    print(f"  requires_grad      : {p['ab'].requires_grad}")
    print(f"  cos(psi ab, psi ba): {cos(p['ab'].real, p['ba'].real): .10f}")
    print(f"  cos(psi ab, psi dog): {cos(p['ab'].real, p['dog'].real): .10f}")

    sep("S4  BUILD AXIOMS + FULL SOLVE LOOP")
    waves = system.build_axioms(H_cli.CORPUS, tok)
    print(f"  axioms pinned  : {len(waves)}   veto.n_axioms={system.veto.n_axioms}")
    print(f"  threshold      : {system.veto.threshold}")

    printed = False
    results = {}
    for q in ["ab", "ba", "dog", "god", "cat", "act"]:
        r = system.solve(q, tok, use_swarm=True)
        results[q] = r
        if not printed:
            print(f"\n  --- full receipt for query 'ab' ---")
            for k in ["token_ids", "probability", "decoded"]:
                print(f"    {k:12s}: {r.get(k)}")
            print(f"    sagnac     : allow={r['sagnac']['allow']} "
                  f"delta={r['sagnac']['delta']:.6f} thr={r['sagnac']['threshold']}")
            print(f"    swarm      : {r['swarm']}")
            print(f"    memory     : n_axioms={r['memory']['n_axioms']} "
                  f"offdiag_max={r['memory']['offdiag_max']:.6f}")
            printed = True

    print(f"\n  {'query':7s} {'decoded':>12s} {'p':>8s} {'sagnac.allow':>13s} {'delta':>9s} {'E':>10s}")
    for q, r in results.items():
        print(f"  {q!r:7s} {str(r['decoded'])[:12]:>12s} {r['probability']:8.4f} "
              f"{str(r['sagnac']['allow']):>13s} {r['sagnac']['delta']:9.5f} "
              f"{str(r['swarm']['energy'])[:10]:>10s}")

    sep("S5  DECODER EGRESS (batched)")
    with torch.no_grad():
        psi_b = system.wave_of("ab", tok).unsqueeze(0)
        bands, tokens = system.decoder.encode_wave(psi_b)
        full = system.decoder(psi_b)
    print(f"  psi_b                 : {tuple(psi_b.shape)}")
    print(f"  macro-tokens          : {tuple(tokens.shape)}")
    print(f"  text_logits           : {tuple(full['text_logits'].shape)}")
    print(f"  action                : {tuple(full['action'].shape)}")
    print(f"  argmax token id       : {int(full['text_logits'].argmax(-1))}")
    print(f"  logits std (per dim)  : {float(full['text_logits'].std()):.6f}")
    print(f"  distinct macro-tokens : "
          f"{int(torch.unique(tokens.round(decimals=3), dim=1).shape[1])} of {tokens.shape[1]}")

    sep("S6  IS THE ANSWER A FUNCTION OF THE INPUT?")
    dec = {q: results[q]["decoded"] for q in results}
    ids = {q: tuple(results[q]["token_ids"]) for q in results}
    n_uniq_dec = len(set(dec.values()))
    n_uniq_ids = len(set(ids.values()))
    print(f"  distinct decoded  : {n_uniq_dec} of {len(dec)}")
    print(f"  distinct token_ids: {n_uniq_ids} of {len(ids)}")
    for q in ["ab", "ba", "dog", "god", "cat", "act"]:
        print(f"    {q!r:6s} -> ids {ids[q]}  dec {dec[q]!r}")

    sep("S7  ORDER SENSITIVITY (the intelligence-relevant question)")
    sw = [("ab", "ba"), ("dog", "god"), ("cat", "act")]
    for a, b in sw:
        same = ids[a] == ids[b]
        print(f"  {a!r} vs {b!r}: token_ids equal = {same}   "
              f"decoded equal = {dec[a] == dec[b]}")

    sep("S8  IRRELEVANT-INPUT PERTURBATION (stability check)")
    base = system.solve("dog", tok, use_swarm=True)["token_ids"]
    for q in ["a dog", "dog!", "dogo", "dogg"]:
        r = system.solve(q, tok, use_swarm=True)
        print(f"  {q!r:8s} -> ids {r['token_ids']}  delta={r['sagnac']['delta']:.6f}  "
              f"{'SAME as dog' if r['token_ids'] == base else 'differs'}")

    print("\nAUDIT COMPLETE (read-only; no files changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
