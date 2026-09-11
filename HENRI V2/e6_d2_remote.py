#!/usr/bin/env python
"""E6 D2 remote smoke -- byte-identity pre vs post on the SAME remote runtime.

WHY THIS SHAPE
  A fingerprint taken only on patched code proves nothing about byte-identity; it
  needs a comparison. A cross-environment comparison (local CPU torch 2.13 vs
  remote CUDA torch 2.12) can differ for reasons unrelated to the patch. So this
  runner holds the SAME worktree at TWO revisions on the ONE remote runtime:

      pre/   henri_decoder.py, henri_ast_grammar_mask.py  as at git HEAD
      post/  the same two files with the guard calls added
      both   henri_discrete_egress_flag.py, e6_d2_differential.py (identical)

  The ONLY difference between the trees is the four guard calls. A matching
  fingerprint is therefore true byte-identity, and the flag-on control shows the
  guard actually fires.

WHY THE FLAG MODULE IS IN BOTH TREES
  Without it, pre/ raises ModuleNotFoundError instead of constructing, which
  would make the comparison unreadable. In pre/ the module is present but never
  called, so pre/ reproduces the original construction behaviour.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
FILES = ["henri_decoder.py", "henri_ast_grammar_mask.py",
         "henri_discrete_egress_flag.py", "e6_d2_differential.py"]
PY = sys.executable


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_diff(d: Path) -> dict:
    r = subprocess.run([PY, "e6_d2_differential.py", "post"], cwd=str(d),
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        return {"error": f"rc={r.returncode}", "stderr": r.stderr[-1200:]}
    return json.loads(r.stdout)


def import_smoke(d: Path, env_extra: dict) -> dict:
    """Import the egress modules with the flag OFF. CUDA torch must construct."""
    import os
    code = (
        "import torch, henri_decoder as m;"
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available());"
        "print('has_transducer', hasattr(m,'HENRIUnifiedEgressTransducer'));"
        "print('has_unbinder', hasattr(m,'HENRINeuralEgressUnbinder'));"
        "print('has_codebook', hasattr(m,'PhaseRingCodebookDecoder'));"
        "import henri_ast_grammar_mask as g;"
        "print('has_mask', hasattr(g,'HENRIASTGrammarMask'))"
    )
    env = dict(os.environ)
    env.update(env_extra)
    r = subprocess.run([PY, "-c", code], cwd=str(d), capture_output=True,
                       text=True, timeout=600, env=env)
    return {"rc": r.returncode, "out": r.stdout.strip().splitlines()[-6:],
            "err": r.stderr.strip().splitlines()[-3:] if r.returncode else []}


def main() -> None:
    pre, post = BASE / "pre", BASE / "post"
    rec: dict = {"base": str(BASE), "file_shas": {}}
    for label, d in (("pre", pre), ("post", post)):
        rec["file_shas"][label] = {f: (sha(d / f)[:16] if (d / f).exists() else None)
                                   for f in FILES}
    print(json.dumps(rec["file_shas"], indent=2))

    print("\n=== differential pre ===")
    a = run_diff(pre)
    print(json.dumps(a.get("fingerprint", a), indent=2)[:900])
    print("flag_on_control:", json.dumps(a.get("flag_on_control", a), indent=2))

    print("\n=== differential post ===")
    b = run_diff(post)
    print("flag_on_control:", json.dumps(b.get("flag_on_control", b), indent=2))

    fa = a.get("fingerprint")
    fb = b.get("fingerprint")
    same = fa is not None and fa == fb
    on_pre = a.get("flag_on_control", {})
    on_post = b.get("flag_on_control", {})
    guard_fires = bool(on_post) and all("RAISED" in v for v in on_post.values())
    guard_absent_pre = bool(on_pre) and not any("RAISED" in v for v in on_pre.values())

    print("\n=== import smoke (flag OFF, remote CUDA torch) ===")
    sm_off = import_smoke(post, {})
    print(json.dumps(sm_off, indent=2))
    print("\n=== import smoke (flag ON -- modules must still import) ===")
    sm_on = import_smoke(post, {"HENRI_STRIP_DISCRETE_EGRESS": "1"})
    print(json.dumps(sm_on, indent=2))

    verdict = {
        "remote_byte_identity": same,
        "pre_fingerprint_sha": (hashlib.sha256(
            json.dumps(fa, sort_keys=True).encode()).hexdigest()[:16] if fa else None),
        "post_fingerprint_sha": (hashlib.sha256(
            json.dumps(fb, sort_keys=True).encode()).hexdigest()[:16] if fb else None),
        "guard_fires_when_on": guard_fires,
        "guard_absent_when_pre": guard_absent_pre,
        "import_ok_flag_off": sm_off["rc"] == 0,
        "import_ok_flag_on": sm_on["rc"] == 0,
    }
    ok = (same and guard_fires and guard_absent_pre
          and sm_off["rc"] == 0 and sm_on["rc"] == 0)
    verdict["VERDICT"] = ("REMOTE_BYTE_IDENTITY_AND_FLAG_GATE_CONFIRMED"
                          if ok else "REVIEW")
    print("\n" + json.dumps(verdict, indent=2))
    (BASE / "remote_smoke.json").write_text(json.dumps(
        {"verdict": verdict, "pre": a, "post": b,
         "import_off": sm_off, "import_on": sm_on}, indent=2))
    print(f"\nWROTE {BASE / 'remote_smoke.json'}")


if __name__ == "__main__":
    main()
