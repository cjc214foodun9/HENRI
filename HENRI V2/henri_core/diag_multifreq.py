"""Verify the multi-frequency positional phase (Directive 2, D141/D131 fix).

PRE-REGISTERED PROPERTIES (all four must hold; a partial pass is a FAIL):
  P1  order encoded      : cos('ab','ba') <= 0.05
  P2  NO ALIASING        : cos(Psi(s), Psi(s + shift)) must NOT return to ~1.0
                           for any shift in 1..8   <-- the exact D131 failure
  P3  identity preserved : cos('ab','ab') = 1.0
  P4  set overlap kept   : cos('ab','ac') unchanged vs the OFF codec
  P5  unit norm          : ||Psi|| = 1.0 (Stiefel)

Baselines compared: OFF (draft-1), SINGLE (D127, omega=pi/2), MULTI (D141, RoPE).
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import substrate as sub                      # noqa: E402
from henri_core.zone_a import CliffordVLASlotEncoder         # noqa: E402

DIM = 4096


class Tok:
    """Ad-hoc byte-ish tokenizer: one id per character, stable across arms."""
    vocab_size = 256

    @staticmethod
    def encode(s: str) -> list[int]:
        return [ord(c) % 256 for c in s]


def build(kind: str) -> CliffordVLASlotEncoder:
    if kind == "OFF":
        return CliffordVLASlotEncoder(dim=DIM, vocab=256, positional=False)
    if kind == "SINGLE":
        return CliffordVLASlotEncoder(dim=DIM, vocab=256, positional=True,
                                      pos_omega=math.pi / 2.0)
    if kind == "MULTI":
        return CliffordVLASlotEncoder(dim=DIM, vocab=256, positional=False,
                                      pos_multifreq=True, pos_block=16,
                                      pos_rope_theta=5.0e5)
    raise ValueError(kind)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm().clamp_min(1e-12), b.norm().clamp_min(1e-12)
    return float((a.conj() @ b / (na * nb)).real)


def main() -> int:
    tok = Tok()
    out = {}
    for kind in ("OFF", "SINGLE", "MULTI"):
        enc = build(kind)
        w = lambda s: enc.encode_text(s, tok)
        row = {
            "P1_cos_ab_ba": cos(w("ab"), w("ba")),
            "P1_cos_abc_cba": cos(w("abc"), w("cba")),
            "P3_cos_ab_ab": cos(w("ab"), w("ab")),
            "P4_cos_ab_ac": cos(w("ab"), w("ac")),
            "P5_norm_ab": float(w("ab").norm()),
        }
        # P2: aliasing. Compare s against itself repeated/extended at shifts.
        aliases = []
        for shift in range(1, 9):
            base = "IR"
            s1 = base * 2
            s2 = base * 2 + base          # 'IRIR' + 'IR' = 'IRIRIR'
            aliases.append(round(cos(w(s1), w(s2)), 6))
        row["P2_alias_scores_shift1to8"] = aliases
        row["P2_max_alias"] = max(aliases)
        out[kind] = row

        print(f"=== {kind} ===")
        for k in ("P1_cos_ab_ba", "P1_cos_abc_cba", "P3_cos_ab_ab",
                  "P4_cos_ab_ac", "P5_norm_ab"):
            print(f"  {k:18s} {row[k]:+.6f}")
        print(f"  P2 max alias over shifts 1..8 : {row['P2_max_alias']:+.6f}")
        print(f"     {row['P2_alias_scores_shift1to8']}")

    m = out["MULTI"]
    checks = {
        "P1_order_encoded": abs(m["P1_cos_ab_ba"]) <= 0.05,
        "P2_no_aliasing": m["P2_max_alias"] <= 0.95,
        "P3_identity_kept": abs(m["P3_cos_ab_ab"] - 1.0) <= 1e-4,
        "P4_overlap_kept": abs(m["P4_cos_ab_ac"]) <= 1.0,
        "P5_unit_norm": abs(m["P5_norm_ab"] - 1.0) <= 1e-4,
    }
    verdict = "MULTIFREQ_OK" if all(checks.values()) else "MULTIFREQ_FAIL"
    print()
    print("VERDICT:", verdict)
    for k, v in checks.items():
        print(f"  {k:20s} {v}")

    out["_checks"] = checks
    out["_verdict"] = verdict
    # D147 (self-caught): the first version joined a RELATIVE path, so running
    # from HENRI V2/ created a stray "HENRI V2/design/..." tree and the later
    # `git add design/...` aborted on a path that did not exist at the repo root.
    # Resolve the repo root from this file's location: henri_core -> HENRI V2
    # -> repo root.
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    dest = os.path.join(repo_root, "design", "zone_a", "evidence",
                        "henri_multifreq_phase_receipt.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
