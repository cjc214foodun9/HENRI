"""D4 measurement: does the EXISTING positional flag separate token order?

The ingress loses order: cos('ab','ba') measured 0.9999998808 with the default
(positional=False). zone_a.py already carries a positional mechanism, OFF by
default ("D127: positional algebra. OFF by default so every committed receipt
[stays valid]"). This measures whether turning it ON fixes order WITHOUT
breaking identity, before any default is flipped.

Read-only: builds the system twice, changes no default.
"""
from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, ".")
from henri_core import cli as H_cli
from henri_core.system import TriModelSystem
from henri_core.tokenizer import ByteBPE

torch.set_num_threads(8)
QS = ["ab", "ba", "dog", "god", "cat", "act"]
PAIRS = [("ab", "ba"), ("dog", "god"), ("cat", "act")]
OUT = r"C:/Users/chan/AppData/Local/Temp/d4_result.json"


def cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def arm(label, **kw):
    tok = ByteBPE().train(H_cli.CORPUS, vocab_size=512)
    s = TriModelSystem(vocab=tok.vocab_size, small=True, **kw)
    s.eval()
    with torch.no_grad():
        psi = {q: s.wave_of(q, tok) for q in QS}
        e = {q: s.ingress.encode_text(q, tok) for q in QS}
    d = {
        "label": label,
        "kwargs": {k: v for k, v in kw.items()},
        "ingress_cos_ab_ba": cos(e["ab"], e["ba"]),
        "ingress_cos_dog_god": cos(e["dog"], e["god"]),
        "ingress_cos_ab_dog": cos(e["ab"], e["dog"]),
        "psi_cos_ab_ba": cos(psi["ab"], psi["ba"]),
        "psi_cos_ab_dog": cos(psi["ab"], psi["dog"]),
    }
    s.build_axioms(H_cli.CORPUS, tok)
    ids = {}
    for q in QS:
        r = s.solve(q, tok, use_swarm=True)
        ids[q] = tuple(r["token_ids"])
    d["ids"] = {k: list(v) for k, v in ids.items()}
    d["distinct_ids"] = len(set(ids.values()))
    d["order_pairs_differ"] = sum(1 for a, b in PAIRS if ids[a] != ids[b])
    return d


def main() -> int:
    res = {"arms": []}
    res["arms"].append(arm("positional_OFF (current default)"))
    res["arms"].append(arm("positional_ON", positional=True))
    res["arms"].append(arm("pos_multifreq_ON", positional=True, pos_multifreq=True))

    a0, a1 = res["arms"][0], res["arms"][1]
    res["verdict"] = {
        "order_lost_at_default": a0["ingress_cos_dog_god"] > 0.99,
        "positional_fixes_ingress_order": a1["ingress_cos_dog_god"] < 0.99,
        "positional_keeps_identity": a1["ingress_cos_ab_dog"] < 0.99,
        "distinct_ids": [a["distinct_ids"] for a in res["arms"]],
        "order_pairs_differ": [a["order_pairs_differ"] for a in res["arms"]],
    }
    R = json.dumps(res, indent=1)
    open(OUT, "w", encoding="utf-8").write(R)
    print(R)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
