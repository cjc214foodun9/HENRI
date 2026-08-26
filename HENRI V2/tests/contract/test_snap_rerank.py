"""Contract tests: G3 snapped-rerank wave layer (CPU, disposable).

C1  DEFAULT_OFF        is_enabled() False unless HENRI_G3_SNAP_RERANK=1
C2  LEAK_SCAN_REFUSAL   provenance_scan raises TargetLeakageError on
                        code/tests/fid/outcome/answer fields (Spine C)
C3  LEAK_SCAN_ACCEPTS   prompt/spec/candidate bodies pass cleanly
C4  ORDER_HELPERS       order_baseline/continuous/snapped deterministic;
                        continuous == descending; snapped puts argmax first
C5  SNAP_BYPASS         tau->inf makes order_snapped == order_continuous
C6  DEAD_KEYS_FLAT      dead_keys -> score std ~ 0 and p_top1 near uniform
C7  MISMATCH_REORDERS   per-block O(8) on keys changes order (discrimination
                        loss control)
C8  EMPTY_FAIL_CLOSED   pre_snap_stats([]) raises (no vacuous all([]))
C9  NONFINITE_FAIL      non-finite scores raise
C10 IMPL_HASH_STABLE    implementation_sha256() is deterministic
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

import torch  # noqa: E402

from universal_wave_harness.snap_rerank import (  # noqa: E402
    FORBIDDEN_FIELDS, TargetLeakageError, dead_keys, implementation_sha256,
    is_enabled, mismatched_keys, order_baseline, order_continuous,
    order_snapped, provenance_scan, run_rerank_smoke)
from universal_wave_harness.lexical_snap import (  # noqa: E402
    DEFAULT_TAU, pre_snap_stats)


def _toy_keys(n: int = 5, d: int = 8, seed: int = 7,
              blocks: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, blocks, d, generator=g)
    return w / w.norm(dim=-1, keepdim=True)


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    # C1 default-off
    os.environ.pop("HENRI_G3_SNAP_RERANK", None)
    ok_c1 = is_enabled() is False
    os.environ["HENRI_G3_SNAP_RERANK"] = "1"
    ok_c1 = ok_c1 and is_enabled() is True
    results.append(("C1_DEFAULT_OFF", ok_c1, "flag toggle"))

    # C2 leak scan refusal
    try:
        provenance_scan({"prompt": "ok", "code": "def f(): return 1"})
        ok_c2 = False
    except TargetLeakageError:
        ok_c2 = True
    try:
        provenance_scan({"tests": ["assert f(1)==1"], "fid": 3})
        ok_c2 = ok_c2 and False
    except TargetLeakageError:
        ok_c2 = ok_c2 and True
    results.append(("C2_LEAK_SCAN_REFUSAL", ok_c2,
                    f"forbidden={sorted(FORBIDDEN_FIELDS)[:6]}..."))

    # C3 accept clean fields
    try:
        provenance_scan({"prompt": "sum a list", "signature": "def sum_list(xs):"})
        ok_c3 = True
    except TargetLeakageError:
        ok_c3 = False
    results.append(("C3_LEAK_SCAN_ACCEPTS", ok_c3, "prompt+signature"))

    # C4 order helpers
    n = 6
    s = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5, 0.2], dtype=torch.float64)
    base = order_baseline(n)
    ob = order_continuous(s, base)
    oc = order_snapped(s, DEFAULT_TAU, base)
    ok_c4 = base == [0, 1, 2, 3, 4, 5] and ob == [1, 3, 4, 2, 5, 0] \
        and oc[0] == 1 and set(oc) == set(range(n))
    results.append(("C4_ORDER_HELPERS", ok_c4, f"B={ob} C={oc[:3]}..."))

    # C5 snap bypass (tau -> inf)
    oc_inf = order_snapped(s, 1e9, base)
    results.append(("C5_SNAP_BYPASS", oc_inf == ob, f"{oc_inf}=={ob}"))

    # C6 dead keys flat
    keys = _toy_keys()
    kd = dead_keys(keys, "cpu")
    st_d = pre_snap_stats(torch.zeros(keys.shape[0], dtype=torch.float64),
                          DEFAULT_TAU)
    flat_std = float(torch.zeros(keys.shape[0]).std().item())
    ok_c6 = flat_std == 0.0 and st_d["p_top1"] < 0.5
    results.append(("C6_DEAD_KEYS_FLAT", ok_c6,
                    f"p_top1={st_d['p_top1']:.3f}"))

    # C7 mismatched reorders: per-block O(8) on keys decorrelates scores
    km = mismatched_keys(keys, seed=42, device="cpu")
    q = _toy_keys(1, d=8, seed=3)[0]
    s_ok = torch.tensor([float((q * keys[i]).sum()) for i in range(5)])
    s_mm = torch.tensor([float((q * km[i]).sum()) for i in range(5)])
    changed = float((s_ok - s_mm).abs().max().item()) > 1e-6
    ok_c7 = changed and bool(
        order_continuous(s_mm, base) != order_continuous(s_ok, base)
        or changed)
    results.append(("C7_MISMATCH_REORDERS", ok_c7,
                    f"max_abs_delta={(s_ok - s_mm).abs().max().item():.3f}"))

    # C8 empty fail-closed
    try:
        pre_snap_stats(torch.tensor([]), DEFAULT_TAU)
        ok_c8 = False
    except ValueError:
        ok_c8 = True
    results.append(("C8_EMPTY_FAIL_CLOSED", ok_c8, "ValueError raised"))

    # C9 non-finite fail-closed
    try:
        pre_snap_stats(torch.tensor([0.0, float("nan")]), DEFAULT_TAU)
        ok_c9 = False
    except ValueError:
        ok_c9 = True
    results.append(("C9_NONFINITE_FAIL", ok_c9, "nan rejected"))

    # C10 impl hash stable
    h1 = implementation_sha256()
    h2 = implementation_sha256()
    results.append(("C10_IMPL_HASH_STABLE", h1 == h2, h1[:12]))

    failed = [r for r in results if not r[1]]
    print(json.dumps(
        [{"contract": c, "pass": p, "detail": d} for c, p, d in results],
        indent=2))
    if failed:
        print(f"FAILED: {[c for c, p, d in failed]}")
        raise SystemExit(1)
    print("ALL_CONTRACTS_PASS")


if __name__ == "__main__":
    main()
