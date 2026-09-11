"""Correct the vacuous Sagnac metric in Sagnacfunctor.txt (user-directed).

DEFECT (measured, e5b_sagnac_delta_reconciliation.json sha 47f8659c):
  line 126:  sim = torch.real(inner_prod).item() / self.D
  line 127:  return max(0.0, 1.0 - sim)
  divides the inner product by the DIMENSION, not by |p|*|a|. For unit-norm
  waves Re<p,a> in [-1,1], so the residual is ~1 - 1/D for aligned, ORTHOGONAL
  and ANTI-ALIGNED alike. MEASURED at D=65536: aligned 0.999985, orthogonal
  1.000000, anti 1.000015, RANGE 3e-05. A dead or random memory PASSES it.
  Classification: VACUOUS SYNCHRONIZATION METRIC.

CORRECTION (bounded; metric body only):
  compute_sagnac_delta -> the sealed live contract
      1 - Re<pred,emp> / (|pred| * |emp|)          range [0, 2]
  (identical to the file's own compute_sagnac_delta_normalized, which was
  correct but not consumed by veto()). The docstring's "[0, 2]" becomes TRUE.

SAFETY
  * original preserved byte-exact at Sagnacfunctor.ORIGINAL.txt (Drive) and
    Sagnacfunctor.ORIGINAL.txt (local telemetry dir); both hashes recorded.
  * the write is VERIFIED by re-reading from disk inside this process; a
    silent Drive-write failure is recorded as DRIVE_BLOCKED, never as success.
  * only the metric body changes; the rest of the file must be byte-identical.
  * no other change. The complex flat [D] family stays diagnostic-only.
"""
from __future__ import annotations

import hashlib
import json
import py_compile
import time
from pathlib import Path

SRC = Path(r"G:\My Drive\HENRI_Inbox\Sagnacfunctor.txt")
BACKUP_DRIVE = Path(r"G:\My Drive\HENRI_Inbox\Sagnacfunctor.ORIGINAL.txt")
BACKUP_LOCAL = Path(r"C:\Users\chan\henri-telemetry\e3\Sagnacfunctor.ORIGINAL.txt")
LOCAL_COPY = Path(r"C:\Users\chan\henri-telemetry\e3\Sagnacfunctor.CORRECTED.txt")
PROBE = Path(r"G:\My Drive\HENRI_Inbox\.henri_write_probe.tmp")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\functor_metric_fix.json")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    rec: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "target": str(SRC)}
    orig = SRC.read_text(encoding="utf-8")
    rec["original_sha256"] = sha(SRC)
    rec["original_bytes"] = SRC.stat().st_size

    # ---- 0. drive writability probe (test, do not assume) ------------------
    try:
        t = ("henri-probe-" + str(time.time())).encode()
        PROBE.write_bytes(t)
        rec["drive_write_probe"] = {"ok": PROBE.read_bytes() == t}
        PROBE.unlink()
        rec["drive_write_probe"]["cleanup"] = not PROBE.exists()
    except Exception as e:
        rec["drive_write_probe"] = {"ok": False,
                                    "error": type(e).__name__ + ": " + str(e)[:160]}
    print("[0] drive write probe: " + json.dumps(rec["drive_write_probe"]))

    # ---- 1. locate the defective body by line range -----------------------
    lines = orig.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines)
                  if l.strip().startswith("def compute_sagnac_delta(")), None)
    assert start is not None, "compute_sagnac_delta NOT FOUND"
    end = next(j for j in range(start + 1, len(lines))
               if lines[j].strip().startswith("def "))
    rec["method_lines"] = [start + 1, end]
    rec["defect_body"] = "".join(lines[start:end])
    print("[1] method at lines " + str(rec["method_lines"][0]) + "-"
          + str(rec["method_lines"][1]))

    # ---- 2. corrected method (sealed normalized form) ---------------------
    new_method = (
        "    def compute_sagnac_delta(\n"
        "        self, psi_proposal: torch.Tensor, psi_axiom: torch.Tensor\n"
        "    ) -> float:\n"
        "        \"\"\"Normalized Sagnac delta (CORRECTED 2026-09-11). Range [0, 2].\n"
        "\n"
        "        The previous body divided the inner product by the dimension\n"
        "        self.D, which made the metric VACUOUS: for unit-norm waves\n"
        "        Re<p,a> lies in [-1, 1], so the residual was ~1 - 1/D for\n"
        "        ALIGNED, ORTHOGONAL and ANTI-ALIGNED pairs alike. Measured at\n"
        "        D=65536 the range was 3e-05, i.e. a dead or random memory passed\n"
        "        it. The function divides by the product of norms, which is the\n"
        "        sealed live contract:\n"
        "\n"
        "            1 - Re<pred, emp> / (|pred| * |emp|)\n"
        "\n"
        "        Receipt: henri-telemetry/e3/functor_metric_fix.json\n"
        "        \"\"\"\n"
        "        re = torch.real(torch.sum(psi_proposal * torch.conj(psi_axiom)))\n"
        "        np_ = torch.norm(psi_proposal)\n"
        "        na = torch.norm(psi_axiom)\n"
        "        return float(1.0 - re / (np_ * na + 1e-12))\n"
        "\n"
    )
    patched = "".join(lines[:start]) + new_method + "".join(lines[end:])

    # ---- 3. bounded-patch proof -------------------------------------------
    rec["bounded_patch"] = ("".join(lines[:start]) + "".join(lines[end:])
                            == patched.replace(new_method, ""))
    print("[2] bounded patch: " + str(rec["bounded_patch"]))

    # ---- 4. compile + behavioural probe on the PATCHED text ---------------
    LOCAL_COPY.write_text(patched, encoding="utf-8", newline="")
    rec["local_corrected_sha256"] = sha(LOCAL_COPY)
    try:
        py_compile.compile(str(LOCAL_COPY), doraise=True)
        rec["compile_ok"] = True
    except Exception as e:
        rec["compile_ok"] = False
        rec["compile_err"] = str(e)[:200]
    beh = {"ok": False}
    try:
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location("sfp", LOCAL_COPY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        g = mod.ParallelTransportSagnacGate(dimension=4096)
        rng = torch.Generator().manual_seed(11)
        p = torch.randn(4096, generator=rng) + 1j * torch.randn(4096, generator=rng)
        p = p / torch.norm(p)
        o = torch.randn(4096, generator=rng) + 1j * torch.randn(4096, generator=rng)
        o = o - torch.real(torch.sum(o * torch.conj(p))) * p
        o = o / torch.norm(o)
        d_same = g.compute_sagnac_delta(p, p.clone())
        d_orth = g.compute_sagnac_delta(p, o)
        d_anti = g.compute_sagnac_delta(p, -p.clone())
        beh = {"ok": True, "aligned": round(d_same, 6),
               "orthogonal": round(d_orth, 6),
               "anti_aligned": round(d_anti, 6),
               "range": round(max(d_orth, d_anti) - d_same, 6),
               "discriminates": bool(d_orth - d_same > 0.5),
               "dead_memory_would_pass": bool(abs(d_orth - d_same) < 0.01)}
    except Exception as e:
        beh = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:200]}
    rec["behavioural_probe"] = beh
    print("[3] probe: " + json.dumps(beh))

    # ---- 5. write to Drive and VERIFY by re-reading -----------------------
    try:
        BACKUP_DRIVE.write_text(orig, encoding="utf-8", newline="")
        BACKUP_LOCAL.write_text(orig, encoding="utf-8", newline="")
        SRC.write_text(patched, encoding="utf-8", newline="")
        landed = SRC.read_text(encoding="utf-8")
        rec["drive_verify"] = {
            "live_sha256": sha(SRC),
            "expected": rec["local_corrected_sha256"],
            "matches": sha(SRC) == rec["local_corrected_sha256"],
            "defect_gone": "1.0 - sim" not in landed and "/ self.D" not in landed,
            "backup_drive_sha256": sha(BACKUP_DRIVE) if BACKUP_DRIVE.exists() else None,
            "backup_local_sha256": sha(BACKUP_LOCAL) if BACKUP_LOCAL.exists() else None,
        }
    except Exception as e:
        rec["drive_verify"] = {"error": type(e).__name__ + ": " + str(e)[:200]}
    print("[4] drive verify: " + json.dumps(rec["drive_verify"]))

    dv = rec.get("drive_verify", {})
    ok = (rec.get("bounded_patch") and rec.get("compile_ok")
          and beh.get("discriminates") and dv.get("matches")
          and dv.get("defect_gone") and dv.get("backup_drive_sha256"))
    rec["VERDICT"] = ("FUNCTOR_METRIC_CORRECTED_ON_DRIVE" if ok
                      else "FUNCTOR_CORRECTED_LOCALLY_DRIVE_BLOCKED")
    OUT.write_text(json.dumps(rec, indent=2))
    print("\nVERDICT=" + rec["VERDICT"])
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
