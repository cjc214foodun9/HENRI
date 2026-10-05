"""Independent verification of the HENRI tri-model core.

Three checks that do not trust the module docstrings:
  V1 PARAMETER FORMULA -- instantiate a real decoder, count its parameters with
     torch, and compare against decoder_param_formula(). A formula that disagrees
     with a real instantiation cannot gate the full config.
  V2 PURITY -- scan henri_core for any off-the-shelf import or checkpoint load.
  V3 COMPOSITION -- run the closed loop and assert all three models engage.

Run:  python "HENRI V2/henri_core/verify_tri_model.py"
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.model3_decoder import (                 # noqa: E402
    DecoderConfig, HenriDec450M, count_params, decoder_param_formula)
from henri_core.system import (                          # noqa: E402
    MODEL_1_ID, MODEL_2_ID, MODEL_3_ID, TriModelSystem)
from henri_core.tokenizer import ByteBPE                 # noqa: E402

CORPUS = ["entity acts on object", "the axiom holds", "phase clears the port",
          "memory decays over time", "the swarm finds lower energy"]

_FAILS: list[str] = []
_PASSES = 0


def check(name, ok, detail=""):
    global _PASSES
    if ok:
        _PASSES += 1
        print(f"  PASS  {name}  {detail}")
    else:
        _FAILS.append(name)
        print(f"  FAIL  {name}  {detail}")


def main():
    print("=== HENRI tri-model independent verification ===")

    # ---- V1: formula vs real instantiation
    cfg = DecoderConfig(dim=4096, d_model=128, n_layers=2, n_heads=4, n_kv_heads=1,
                        d_ffn=256, n_macro=16, n_invariants=32, vocab=256)
    mod = HenriDec450M(cfg)
    real = count_params(mod)["total"]
    formula = decoder_param_formula(cfg)["total"]
    check("V1_formula_matches_instantiation", real == formula,
          f"real={real:,} formula={formula:,}")
    check("V1_formula_exact", abs(real - formula) == 0,
          f"delta={real - formula}")

    # ---- V1b: full config inside the documented envelope
    full = decoder_param_formula(DecoderConfig())
    in_env = 4.0e8 <= full["total"] <= 5.0e8
    check("V1b_full_config_in_envelope", in_env,
          f"params={full['total']:,} target={full['doc_target']:,} "
          f"delta={full['total'] - full['doc_target']:,}")

    # ---- V2: purity
    # D79 (self-caught): the first scan flagged THIS file, because the banned
    # names appear here as data. A scanner must not treat its own pattern list
    # as a violation. Exclude the scanner itself from the walk.
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    self_name = os.path.basename(os.path.abspath(__file__))
    banned = ("transformers", "huggingface_hub", "sentencepiece", "tiktoken",
              "from_pretrained")
    hits = []
    for fn in sorted(os.listdir(here)):
        if not fn.endswith(".py") or fn == self_name:
            continue
        for ln, line in enumerate(open(os.path.join(here, fn), encoding="utf-8"), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if s.startswith("import ") or s.startswith("from "):
                for t in banned:
                    if t in s:
                        hits.append(f"{fn}:{ln}: {s[:70]}")
    check("V2_no_offtheshelf_import", not hits, f"hits={hits}")

    # ---- V3: composition engages all three models
    tok = ByteBPE().train(CORPUS, vocab_size=256)
    system = TriModelSystem(vocab=tok.vocab_size, small=True)
    system.eval()
    axioms = system.build_axioms(CORPUS[:4], tok)
    res = system.solve("retrieval transfer result", tok, patterns=axioms)
    check("V3_all_three_models_engage",
          res["models_engaged"] == [MODEL_1_ID, MODEL_2_ID, MODEL_3_ID],
          str(res["models_engaged"]))
    check("V3_swarm_reported_energy", res["swarm"]["energy"] is not None,
          f"energy={res['swarm']['energy']}")
    check("V3_memory_reported_offdiag",
          res["memory"]["offdiag_max"] is not None,
          f"offdiag={res['memory']['offdiag_max']:.6f}")
    check("V3_decoder_emitted_token", isinstance(res["decoded"], str) and
          len(res["token_ids"]) >= 1, f"ids={res['token_ids']}")
    check("V3_veto_returned_verdict", "allow" in res["sagnac"],
          f"allow={res['sagnac']['allow']} delta={res['sagnac']['delta']:.4f}")

    # ---- V3b: determinism, the property the prior trainer lacked
    r1 = system.solve("retrieval transfer result", tok, patterns=axioms)
    r2 = system.solve("retrieval transfer result", tok, patterns=axioms)
    check("V3b_deterministic_same_input_same_output",
          r1["token_ids"] == r2["token_ids"],
          f"{r1['token_ids']} vs {r2['token_ids']}")

    print(f"\n=== {_PASSES} passed, {len(_FAILS)} failed ===")
    if _FAILS:
        print("FAILED:", ", ".join(_FAILS))
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
