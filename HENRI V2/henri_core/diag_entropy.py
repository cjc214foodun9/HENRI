"""One clean probe: why does the entropy metric return H = log(n) for every input?

The previous probe was buggy (bilinear .T instead of Hermitian conj, float() on a
complex tensor). This one uses the correct Hermitian cosine and prints full
precision. It answers three questions:
    1  what dtype and shape does WaveTextGenerator.wave() return?
    2  are the wave vectors EQUAL-arccos from every engram (uniform softmax) or
       merely near-orthogonal?
    3  does the engine's own metric agree with an independent computation?
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from henri_core.daydream import DaydreamConfig, DaydreamEngine  # noqa: E402
from henri_core.m4_generative import (  # noqa: E402
    WaveTextGenerator, build_corpus, build_system,
)

torch.set_printoptions(precision=10)


def flat(t: torch.Tensor) -> torch.Tensor:
    t = t.mean(dim=1) if t.dim() == 3 else t
    return t.reshape(t.shape[0], -1)


def herman(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Correct Hermitian cosine. Returns a REAL tensor."""
    cd = torch.complex128 if (q.is_complex() or k.is_complex()) else torch.float64
    q = q.to(cd)
    k = k.to(cd)
    qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    kn = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return (qn @ kn.conj().transpose(0, 1)).real


def main() -> int:
    corpus = build_corpus()
    system, tok = build_system(corpus)
    cfg = DaydreamConfig()
    eng = DaydreamEngine(system, tok, cfg)
    model = WaveTextGenerator(system, tok, train_body=True)

    bank = model.wave([f"axiom {i}" for i in range(8)])
    a = model.wave(["apply IR to 1234"])
    b = model.wave(["apply IIII to 2468"])

    print("=== 1. dtype and shape ===")
    print("  bank", bank.dtype, tuple(bank.shape))
    print("  a   ", a.dtype, tuple(a.shape))
    if a.is_complex():
        print("  a imag max", float(a.imag.abs().max()))

    fa, fb, fk = flat(a), flat(b), flat(bank)
    print("  flat shapes", tuple(fa.shape), tuple(fb.shape), tuple(fk.shape))
    print("  ||fa|| %.10f  ||fb|| %.10f" % (float(fa.norm()), float(fb.norm())))
    print("  ||fk[i]||", [round(float(x), 6) for x in fk.norm(dim=-1)])

    print("=== 2. Hermitian cosines against the 8-engram bank ===")
    ca = herman(fa, fk).squeeze()
    cb = herman(fb, fk).squeeze()
    print("  cos(a,k)", [round(float(x), 10) for x in ca])
    print("  cos(b,k)", [round(float(x), 10) for x in cb])
    print("  cos(a,b)  %.10f" % float(herman(fa, fb).squeeze()))
    print("  spread a  %.10f (max-min)" % float(ca.max() - ca.min()))
    print("  spread b  %.10f (max-min)" % float(cb.max() - cb.min()))

    g = herman(fk, fk) - torch.eye(8, dtype=torch.float64)
    print("  bank offdiag mean|cos| %.6f  max|cos| %.6f"
          % (float(g.abs().sum() / 56), float(g.abs().max())))

    print("=== 3. entropy, engine vs independent ===")
    for name, q in (("a", fa), ("b", fb)):
        cos = herman(q, fk).squeeze()
        logits = cfg.beta_entropy * cos
        p = torch.softmax(logits, dim=-1)
        pf = (1 - cfg.entropy_floor) * p + cfg.entropy_floor / 8
        H = float((-(pf * (pf + 1e-12).log()).sum()))
        print(f"  {name}: engine H={eng.info_gain_uniform(q, fk):.10f}  "
              f"independent H={H:.10f}  H_uniform={math.log(8):.10f}")
    print("  engine dH(a,b) = %.10f" % eng.info_gain(fa, fb, fk))
    print("  identical-state dH = %.10f" % eng.info_gain(fa, fa.clone(), fk))

    print("=== 4. does the RETRIEVAL temperature saturate? ===")
    cos = herman(fa, fk).squeeze()
    for beta_name, beta in (("beta_entropy=1.0", cfg.beta_entropy),
                            ("beta(retrieval)=26.10", cfg.beta)):
        logits = beta * cos
        p = torch.softmax(logits, dim=-1)
        print(f"  {beta_name:<24} p_max={float(p.max()):.8f} "
              f"p_min={float(p.min()):.10f} H={float(-(p*(p+1e-12).log()).sum()):.8f}")

    print("=== 5. within-spec structure (is a wave rank-1?) ===")
    if a.dim() == 3:
        for name, t in (("a", a), ("bank", bank)):
            x = t.reshape(-1, t.shape[-1]).float()
            s = torch.linalg.svdvals(x - x.mean(0, keepdim=True))
            pr = float((s.sum() ** 2) / (s ** 2).sum())
            print(f"  {name}: participation ratio of the token set = {pr:.3f}")
    else:
        print("  wave() is 2-D; no token axis to test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
