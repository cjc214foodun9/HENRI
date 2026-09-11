"""D2 DIFFERENTIAL PROBE -- byte-identity evidence for the discrete-egress strip.

WHY THIS EXISTS
  Decision 2 changes live egress code. A diff review is not evidence. This probe
  produces a SHA-256 fingerprint of the discrete-egress surfaces' observable
  behaviour, so the pre-patch and post-patch fingerprints can be compared.

WHAT IT DOES (read-only)
  For each gated class, build an object with small deterministic dimensions on
  CPU and fingerprint:
    * the constructor's observable state (key attribute types and shapes)
    * a forward/decode output tensor, hashed at byte level
  Also asserts the flag semantics in both directions:
    * flag unset  -> all four surfaces construct (STRIP INERT)
    * flag set    -> all four surfaces raise EgressDiscreteStripEnabledError

  The flag-set half is a NEGATIVE CONTROL. Without it, "byte-identity when off"
  could be satisfied by a patch that does nothing at all.

USAGE
  python e6_d2_differential.py pre     > pre.json
  <apply patch>
  python e6_d2_differential.py post    > post.json
  python e6_d2_differential.py compare pre.json post.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HW = Path(__file__).resolve().parent


def _h(t) -> str:
    import torch
    if torch.is_tensor(t):
        return hashlib.sha256(t.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()[:32]
    return hashlib.sha256(repr(t).encode()).hexdigest()[:32]


def fingerprint() -> dict:
    import torch
    torch.manual_seed(0)
    out: dict = {"utc_note": "deterministic seed=0, float32, cpu"}

    from henri_decoder import (HENRINeuralEgressUnbinder, PhaseRingCodebookDecoder,
                               HENRIUnifiedEgressTransducer)
    from henri_ast_grammar_mask import HENRIASTGrammarMask

    D, HID, V = 64, 16, 32

    # 1. HENRINeuralEgressUnbinder -- down_proj + lm_head projection head
    ub = HENRINeuralEgressUnbinder(d_model=D, d_hidden=HID, vocab_size=V, device="cpu")
    x = torch.randn(2, D)
    y = ub(x)
    out["HENRINeuralEgressUnbinder"] = {
        "down_proj_shape": list(ub.down_proj.weight.shape),
        "lm_head_shape": list(ub.lm_head.weight.shape),
        "out_shape": list(y.shape),
        "out_hash": _h(y),
    }

    # 2. PhaseRingCodebookDecoder -- quantize / dequantize / unbind
    cd = PhaseRingCodebookDecoder(d_model=D, device="cpu")
    w = torch.randn(D)
    rings = cd.quantize_phase_ring(w)
    dq = cd.dequantize_phase_ring(rings)
    un = cd.inverse_unbinding(torch.randn(D), torch.randn(D))
    out["PhaseRingCodebookDecoder"] = {
        "k_bins": cd.k_bins,
        "ring_hash": _h(rings),
        "dq_hash": _h(dq),
        "unbound_hash": _h(un),
    }

    # 3. HENRIASTGrammarMask -- vocab + a mask step
    gm = HENRIASTGrammarMask()
    logits = torch.randn(1, 10)
    masked = gm.mask_logits_for_step(logits, [], 0)
    out["HENRIASTGrammarMask"] = {
        "vocab_size": len(gm.code_vocab_map),
        "masked_hash": _h(masked),
    }

    # 4. HENRIUnifiedEgressTransducer -- composition, checkpoint disabled
    tr = HENRIUnifiedEgressTransducer(d_model=D, hidden_dim=HID, vocab_size=V,
                                      device="cpu", checkpoint_policy="disabled")
    out["HENRIUnifiedEgressTransducer"] = {
        "checkpoint_load_status": tr.checkpoint_load_status,
        "has_unbinder": hasattr(tr, "unbinder"),
        "has_codebook": hasattr(tr, "codebook"),
    }
    return out


def negative_control() -> dict:
    """With the flag ON every surface must refuse to construct."""
    from henri_discrete_egress_flag import EgressDiscreteStripEnabledError
    results = {}
    def probe(name, fn):
        try:
            fn()
            results[name] = "CONSTRUCTED (unexpected)"
        except EgressDiscreteStripEnabledError as e:
            results[name] = f"RAISED ({type(e).__name__})"
        except Exception as e:  # noqa: BLE001
            results[name] = f"OTHER ({type(e).__name__}: {str(e)[:60]})"
    import torch
    from henri_decoder import (HENRINeuralEgressUnbinder, PhaseRingCodebookDecoder,
                              HENRIUnifiedEgressTransducer)
    from henri_ast_grammar_mask import HENRIASTGrammarMask
    probe("HENRINeuralEgressUnbinder",
          lambda: HENRINeuralEgressUnbinder(d_model=64, d_hidden=16, vocab_size=32, device="cpu"))
    probe("PhaseRingCodebookDecoder",
          lambda: PhaseRingCodebookDecoder(d_model=64, device="cpu"))
    probe("HENRIASTGrammarMask", lambda: HENRIASTGrammarMask())
    probe("HENRIUnifiedEgressTransducer",
          lambda: HENRIUnifiedEgressTransducer(d_model=64, hidden_dim=16, vocab_size=32,
                                               device="cpu", checkpoint_policy="disabled"))
    return results


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pre"
    if mode == "compare":
        a = json.loads(Path(sys.argv[2]).read_text())
        b = json.loads(Path(sys.argv[3]).read_text())
        fa, fb = a["fingerprint"], b["fingerprint"]
        same = fa == fb
        print(f"pre-fingerprint  sha256={hashlib.sha256(json.dumps(fa,sort_keys=True).encode()).hexdigest()[:16]}")
        print(f"post-fingerprint sha256={hashlib.sha256(json.dumps(fb,sort_keys=True).encode()).hexdigest()[:16]}")
        print(f"BYTE_IDENTITY={same}")
        if not same:
            for k in sorted(set(fa) | set(fb)):
                if fa.get(k) != fb.get(k):
                    print(f"  DIFF {k}: {fa.get(k)} != {fb.get(k)}")
        print(f"flag_set_control_pre ={json.dumps(a['flag_on_control'])}")
        print(f"flag_set_control_post={json.dumps(b['flag_on_control'])}")
        ok = (same
              and all("RAISED" in v for v in b["flag_on_control"].values())
              and all("CONSTRUCTED" in v for v in a["flag_off_control"].values()))
        print(f"VERDICT={'BYTE_IDENTITY_PROVEN_FLAG_GATED' if ok else 'REVIEW'}")
        return

    env_before = os.environ.pop("HENRI_STRIP_DISCRETE_EGRESS", None)
    off = fingerprint()
    # label the ARMS, not the value types: an earlier version labelled by
    # type(fingerprint_value).__name__ and produced nonsense like
    # "CONSTRUCTED (dict)", and wrongly counted the utc_note key as a surface.
    _surfaces = sorted(k for k in off if k != "utc_note")
    off_control = {k: "CONSTRUCTED" for k in _surfaces}
    os.environ["HENRI_STRIP_DISCRETE_EGRESS"] = "1"
    on = negative_control()
    if env_before is not None:
        os.environ["HENRI_STRIP_DISCRETE_EGRESS"] = env_before
    else:
        os.environ.pop("HENRI_STRIP_DISCRETE_EGRESS", None)

    rec = {"mode": mode, "flag_off_control": off_control,
           "flag_on_control": on, "fingerprint": off}
    print(json.dumps(rec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
