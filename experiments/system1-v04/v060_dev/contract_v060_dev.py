"""
Contract tests: v0.6.0-dev Zone C read-path adapters (CPU, disposable).
=============================================================================
C1  NULL_IDENTITY       beta=0.0 bias ranking == baseline ranking exactly
                        (byte-identical order; matched-control identity).
C2  CACHE_DETERMINISM   hot cache is reproducible (same names+seeds ->
                        identical bytes) and query returns top-k sorted.
C3  FASTWEIGHT_RESET    fast-weight memory is factorized (rank r <= budget),
                        per-task reset, and downweights failed rules only.
C4  PARTITION_COVERAGE  partition ordering covers ALL rules exactly once.
C5  LIVE_IMPORT_CLOSURE bridge imports against live kernel; d_live=384.
C6  VRAM_ACCOUNTING     hot cache FP16 bytes reported; <= 1 MiB at 64x384.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from zone_c_bridge_v060 import (  # noqa: E402
    ZoneCHotCache, NullZoneCAdapter, ZoneCEngramBias, PersistenceStatus,
    FastWeightRuleMemory, PartitionOrder,
)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# C1: null identity — beta=0 must reproduce baseline order exactly
# ---------------------------------------------------------------------------
def test_null_identity():
    torch.manual_seed(0)
    kernel = None
    try:
        from system1_kernel_v041_energy_refactored import (  # noqa: E402
            System1KernelV04, KernelV04Config)
        from system1_kernel_v05_ast_skeleton import System1KernelV05
        backbone = System1KernelV04(cfg=KernelV04Config())
        kernel = System1KernelV05(backbone, num_rules=13)
    except Exception as e:  # noqa: BLE001
        check("C1", False, f"kernel import failed: {e}")
        return
    d = kernel.backbone.cfg.d_slot
    z0 = torch.randn(1, 16, d)   # live eval shape: 16 slots (pad 16)
    sp = torch.randn(1, 16, d)
    task = {"name": "f_closure", "nargs": 1}
    base = kernel.generate_skeleton_candidates(z0, sp, task, top_k=64,
                                               use_energy=False)
    cache = ZoneCHotCache(num_engrams=8, d_live=d, device="cpu", seed=1)
    cache.populate_from_names(
        ["spelke_object_persistence", "spelke_topological_containment",
         "spelke_inertial_continuity", "spelke_affine_invariance",
         "physics_conservation_of_mass", "physics_momentum_conservation",
         "logic_peano_successor", "logic_group_reversibility"])
    bias0 = ZoneCEngramBias(kernel, cache, beta=0.0)
    ranked0 = bias0.ranked_candidates(z0, sp, task, top_k=64,
                                      use_energy=False)
    base_ids = [c["rule_id"] for c in base]
    zero_ids = [c["rule_id"] for c in ranked0]
    check("C1", base_ids == zero_ids,
          f"base={base_ids[:6]}... zero={zero_ids[:6]}...")


# ---------------------------------------------------------------------------
# C2: cache determinism + top-k sorted
# ---------------------------------------------------------------------------
def test_cache_determinism():
    names = ["spelke_object_persistence", "spelke_inertial_continuity"]
    a = ZoneCHotCache(num_engrams=2, d_live=384, seed=99)
    b = ZoneCHotCache(num_engrams=2, d_live=384, seed=99)
    a.populate_from_names(names)
    b.populate_from_names(names)
    check("C2", torch.equal(a.engrams, b.engrams), "same seed -> identical")
    q = F.normalize(torch.randn(3, 384), dim=-1)
    sims, names_out = a.query(q, top_k=2)
    check("C2", sims.shape == (3, 2), f"shape={tuple(sims.shape)}")
    check("C2", bool((sims[:, 0] >= sims[:, 1]).all()), "sorted desc")


# ---------------------------------------------------------------------------
# C5: live import closure
# ---------------------------------------------------------------------------
def test_live_closure():
    from system1_kernel_v041_energy_refactored import (  # noqa: E402
        System1KernelV04, KernelV04Config)
    from system1_kernel_v05_ast_skeleton import System1KernelV05
    k = System1KernelV05(System1KernelV04(cfg=KernelV04Config()),
                         num_rules=13)
    d = k.backbone.cfg.d_slot
    check("C5", d == 384, f"d_slot={d}")


# ---------------------------------------------------------------------------
# C6: VRAM accounting
# ---------------------------------------------------------------------------
def test_vram():
    cache = ZoneCHotCache(num_engrams=64, d_live=384)
    check("C6", cache.vram_mib <= 1.0, f"vram={cache.vram_mib:.4f} MiB")


# ---------------------------------------------------------------------------
# C7: persistence status explicit (never silent)
# ---------------------------------------------------------------------------
def test_persistence():
    ps = PersistenceStatus(timeout=0.5, enforce=False)
    st = ps.probe()
    check("C7", st in ("ok", "degraded"), f"status={st} explicit")
    rec = ps.to_record()
    check("C7", "db_status" in rec, "record has db_status")


# ---------------------------------------------------------------------------
# C8: fast-weight factorized + identity + reset (v0.6.1, default OFF)
# ---------------------------------------------------------------------------
def test_fastweight():
    fw = FastWeightRuleMemory(num_rules=13, rank=8, eta=0.5, lam=0.95)
    check("C8", fw._U.shape == (8, 13), f"factor U={tuple(fw._U.shape)}")
    base = torch.ones(13) / 13.0
    check("C8", torch.equal(fw.adjusted_probs(base), base),
          "no updates -> identity")
    fw.record_failure(rule_id=3)
    after = fw.adjusted_probs(base)
    check("C8", after[3] < after[0], "failed rule downweighted")
    check("C8", abs(float(after.sum()) - 1.0) < 1e-6, "normalized")
    fw.reset()
    check("C8", torch.equal(fw.adjusted_probs(base), base),
          "reset -> identity")


# ---------------------------------------------------------------------------
# C9: partition order covers all rules once + arg rotation closed (v0.6.2)
# ---------------------------------------------------------------------------
def test_partition():
    po = PartitionOrder(num_rules=13, p=3)
    order = po.order()
    check("C9", sorted(order) == list(range(13)),
          f"len={len(order)} unique={len(set(order))}")
    sw = po.sub_swarms()
    check("C9", set(sw.keys()) == {"arity1", "arity2"}, f"swarms={sorted(sw)}")
    for nargs in (1, 2):
        for rid in range(13):
            names = po.arg_rotation(rid, nargs)
            check("C9", all(isinstance(x, str) and x for x in names),
                  f"arg_rotation({rid},{nargs})={names}")


if __name__ == "__main__":
    test_null_identity()
    test_cache_determinism()
    test_live_closure()
    test_vram()
    test_persistence()
    test_fastweight()
    test_partition()
    print("=" * 50)
    print("CONTRACT_V060:", "ALL PASS" if not FAILS else f"FAILS: {FAILS}")
    sys.exit(1 if FAILS else 0)
