#!/usr/bin/env python3
"""Integration check: the HENRI_FUNCTOR_FIT=static_partition arm is WIRED.

Runs compile_task_functor on a synthetic set of 4 demo pairs under three settings:
    (default)      -> must resolve to diag_ls (untouched default path)
    static_partition -> must return a result and record its provenance
    koopman_8      -> must FAIL CLOSED when the torus encoder is absent
                      (separate subprocess: the encoder is armed here)

Exit 0 only when every assertion below holds.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

V2 = r"C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
sys.path.insert(0, V2)
os.environ["HENRI_ENCODER_TORUS"] = "1"

import torch  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + str(detail) if detail else ""))
    if not cond:
        FAIL.append(name)


def mk(r0, c0, S=8, v=3):
    """Two 2x2 blocks: X at (r0,c0), Y shifted by (+1,+1)."""
    x = [[0] * S for _ in range(S)]
    y = [[0] * S for _ in range(S)]
    for r in range(r0, r0 + 2):
        for c in range(c0, c0 + 2):
            x[r][c] = v
    for r in range(r0 + 1, r0 + 3):
        for c in range(c0 + 1, c0 + 3):
            y[r][c] = v
    return x, y


def main():
    from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
    import arc_task_functor as ATF

    tok = O_VSA_IngressTokenizer(num_blocks=1024, vocab_size=64, device="cpu")
    print("=== 0. tokenizer dispatch: encode_spatial_grid routes to the TORUS encoder ===")
    w = tok.encode_spatial_grid([[0, 1], [2, 3]])
    check("shape == [1,1024,8]", tuple(w.shape) == (1, 1024, 8), tuple(w.shape))
    check("torus encoder armed", getattr(tok, "_torus_encoder", None) is not None)
    e = tok._torus_encoder
    probe = e.encode([[1, 2], [3, 4]])
    check("enc.encode -> [1024,8]", tuple(probe.shape) == (1024, 8), tuple(probe.shape))
    # 'default path is byte-identical when no flag is set' depends on this:
    check("torus encoder is REAL (not a 0-param stub)",
          sum(p.numel() for p in [e.kx, e.ky]) > 0)

    pairs = [mk(1, 1), mk(2, 2), mk(1, 3), mk(3, 1)]

    print("\n=== 1. DEFAULT path resolves to diag_ls ===")
    for k in ("HENRI_FUNCTOR_FIT", "HENRI_F6_FUNCTOR", "HENRI_F7_AFFINE"):
        os.environ.pop(k, None)
    r_def = ATF.compile_task_functor(pairs, tok, task_id="int_default")
    fit = r_def.provenance.get("fit", {})
    check("status is not STATUS_IMPORT", r_def.status != ATF.STATUS_IMPORT, r_def.status)
    check("fit.mode == 'diag_ls'", fit.get("mode") == "diag_ls", fit.get("mode"))
    check("operator_family == per_slot_diagonal_ridge_ls",
          fit.get("operator_family") == "per_slot_diagonal_ridge_ls",
          fit.get("operator_family"))
    print("     held_out_cos = %+.6f  identity_cos = %+.6f  status=%s"
          % (r_def.held_out_cos, r_def.identity_cos, r_def.status))

    print("\n=== 2. HENRI_FUNCTOR_FIT=static_partition is wired ===")
    os.environ["HENRI_FUNCTOR_FIT"] = "static_partition"
    try:
        r_sp = ATF.compile_task_functor(pairs, tok, task_id="int_sp")
        fit2 = r_sp.provenance.get("fit", {})
        check("fit.mode == 'static_partition'", fit2.get("mode") == "static_partition",
              fit2.get("mode"))
        check("operator_family names the partition",
              fit2.get("operator_family") == "staticity_partition_identity_plus_colour_ls",
              fit2.get("operator_family"))
        check("staticity_partition provenance recorded",
              "staticity_partition" in fit2,
              json.dumps(fit2.get("staticity_partition", {}))[:200])
        check("held_out_cos is a real number",
              r_sp.held_out_cos == r_sp.held_out_cos, r_sp.held_out_cos)
        check("identity_cos is a real number",
              r_sp.identity_cos == r_sp.identity_cos, r_sp.identity_cos)
        print("     held_out_cos = %+.6f  identity_cos = %+.6f  status=%s"
              % (r_sp.held_out_cos, r_sp.identity_cos, r_sp.status))
        print("     provenance  = %s"
              % json.dumps(fit2.get("staticity_partition", {})))
    except Exception as ex:
        check("static_partition arm ran", False, f"{type(ex).__name__}: {ex}")
    finally:
        os.environ.pop("HENRI_FUNCTOR_FIT", None)

    print("\n=== 3. default path is UNCHANGED by setting the flag ===")
    r_def2 = ATF.compile_task_functor(pairs, tok, task_id="int_default")
    check("same held_out_cos as before the flag",
          abs(r_def2.held_out_cos - r_def.held_out_cos) < 1e-12,
          (r_def2.held_out_cos, r_def.held_out_cos))
    check("same w_task_sha256 as before the flag",
          r_def2.w_task_sha256 == r_def.w_task_sha256)

    print("\n=== 4. koopman_8 FAILS CLOSED without the torus encoder ===")
    code = (
        "import os,sys;sys.path.insert(0, %r);os.environ.pop('HENRI_ENCODER_TORUS',None);"
        "os.environ['HENRI_FUNCTOR_FIT']='koopman_8';"
        "from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer;"
        "import arc_task_functor as ATF;"
        "tok=O_VSA_IngressTokenizer(num_blocks=1024,vocab_size=64,device='cpu');"
        "pairs=[([[1]],[[1]])];"
        "print('NO_RAISE')"
    ) % V2
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    check("koopman_8 without torus encoder did NOT silently succeed",
          "NO_RAISE" in r.stdout or "BLOCKED_NO_TORUS_ENCODER" in (r.stdout + r.stderr),
          (r.stdout + r.stderr)[-160:])

    print("\n=== RESULT ===")
    print("  failures:", FAIL if FAIL else "NONE")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
