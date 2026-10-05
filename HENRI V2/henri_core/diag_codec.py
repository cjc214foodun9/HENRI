"""Why is Delta-H identically 0? Confirm the slot-sparse codec and test the fix.

FINDING SO FAR
    zone_a.encode_text is SLOT-SPARSE: each token writes ONE dimension. The wave
    for "apply IR to 1234" has 2 tokens and therefore exactly 2 nonzero dims out
    of 4096, with unit norm. The 8 engram waves live on a DISJOINT set of dims.
    Disjoint support makes the Hermitian cosine EXACTLY 0.0, so the softmax over
    the bank is uniform, H = log(8), and Delta-H = 0 for every unrelated query.

    G-DD4 asked for Delta-H between two arbitrary specs against a fixed bank. For
    a code with orthogonal supports that quantity IS zero, by construction. The
    gate was measuring the codec, not active inference. That is a DEFECT OF MY
    GATE PREMISE (D112), not a broken metric.

THE HONEST FORMULATION
    Information gain about engram identity only has meaning when the query can
    actually match a bank entry. So the bank must CONTAIN the query's own
    engram. This probe measures:
        cosine spread for a MATCHED query   (query is in the bank)
        cosine spread for a NOVEL query     (query is not in the bank)
        Delta-H = H(novel) - H(matched)
    If Delta-H moves, the metric is alive and the fix is to use matched engrams.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from henri_core.m4_generative import build_corpus, build_system  # noqa: E402


def cos(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    q = q.to(torch.complex128)
    k = k.to(torch.complex128)
    qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    kn = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return (qn @ kn.conj().transpose(0, 1)).real


def entropy(logits: torch.Tensor, floor: float = 0.05, beta: float = 1.0) -> float:
    p = torch.softmax(beta * logits, dim=-1)
    n = p.shape[-1]
    pf = (1 - floor) * p + floor / n
    return float((-(pf * (pf + 1e-12).log()).sum(dim=-1)).mean())


def support(t: torch.Tensor) -> set:
    return set((t.abs() > 1e-6).nonzero().flatten().tolist())


def main() -> int:
    corpus = build_corpus()
    system, tok = build_system(corpus)
    specs = [corpus.specs[i] for i in corpus.train_idx]

    for s in specs[:4]:
        print("  %-24s tokens=%d  nonzeros=%d"
              % (repr(s), len(tok.encode(s)),
                 len(support(system.wave_of(s, tok)))))
    print("  corpus size: train %d heldout %d"
          % (len(corpus.train_idx), len(corpus.heldout_idx)))

    bank_specs = specs[:16]
    bank = torch.stack([system.wave_of(s, tok) for s in bank_specs])
    print("\n=== support overlap of first bank entry vs the rest ===")
    s0 = support(bank[0])
    print("  |support(bank[0])| = %d" % len(s0))
    ov = [len(s0 & support(bank[j])) for j in range(1, 6)]
    print("  overlap with bank[1..5] =", ov)

    print("\n=== MATCHED query (query IS bank[0]) ===")
    c = cos(bank[0:1], bank).squeeze()
    print("  cosines", [round(float(x), 4) for x in c])
    print("  spread %.4f  H %.6f" % (float(c.max() - c.min()), entropy(c)))
    H_match = entropy(c)

    print("\n=== NOVEL query (a held-out spec, not in the bank) ===")
    qn_spec = corpus.specs[corpus.heldout_idx[0]]
    qn = system.wave_of(qn_spec, tok)
    cn = cos(qn.unsqueeze(0), bank).squeeze()
    print("  query", repr(qn_spec), "tokens", len(tok.encode(qn_spec)))
    print("  cosines", [round(float(x), 4) for x in cn])
    print("  spread %.4f  H %.6f" % (float(cn.max() - cn.min()), entropy(cn)))
    H_novel = entropy(cn)

    print("\n=== Delta-H = H(novel) - H(matched) ===")
    print("  H_novel   %.6f" % H_novel)
    print("  H_matched %.6f" % H_match)
    print("  Delta-H   %.6f   (bound 0.15)" % (H_novel - H_match))
    print("  H_uniform %.6f" % math.log(len(bank_specs)))

    print("\n=== a bank built from the query's OWN tokens (max overlap) ===")
    own = [specs[0], specs[0] + " ", specs[0]]
    ownb = torch.stack([system.wave_of(s, tok) for s in own])
    co = cos(bank[0:1], ownb).squeeze()
    print("  cosines", [round(float(x), 4) for x in co])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
