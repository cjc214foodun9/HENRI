"""Multi-seed benchmark runner with paired controls, CRN, and fail-closed gates.

WHAT THIS RUNS
    A task family plus a set of agents (arms). For each seed it builds a split,
    scores every arm on the SAME split (common random numbers), and aggregates.
    Then it evaluates pre-registered gates and emits a henri.bench.v1 receipt.

WHY THE BUILT-IN "egress_managed" TASK EXISTS
    An instrument must be validated before it is trusted. This task has KNOWN
    ground truth: K classes, generated templates, disjoint train/test instances.
    It therefore measures the INSTRUMENT (does the harness separate skill from
    chance?) and simultaneously re-measures the real HENRI result through the
    harness path rather than through an ad-hoc probe.

    It is marked LOCAL, not AAII. Its dataset is generated deterministically and
    pinned by sha256 of the canonical serialization, so it is DATASET_PINNED and
    legitimately scoreable. AAII adapters are NOT scoreable (see adapters.py).

CRN (common random numbers)
    Every arm sees the same split and the same task instances per seed. Arm
    differences are therefore not confounded by split variance. This is the
    lesson from the D2-D45 replication audit.

Run:
    python -m benchmarks.runner --task egress_managed --seeds 0,1,2,3,4,5,6,7 \
        --out ../../design/zone_a/evidence/bench_egress_managed_receipt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

# Import roots: this file lives in HENRI V2/benchmarks/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_HENRI_V2 = os.path.dirname(_HERE)
for p in (_HENRI_V2,):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmarks import spec_registry as SR                     # noqa: E402
from benchmarks.harness_core import (                          # noqa: E402
    SpecStatus, aggregate, deterministic_split, evaluate_gates, final_verdict,
    load_gates, make_receipt, sha256_file,
)

import zone_c_world_knowledge_codec as C                        # noqa: E402
import henri_managed_egress as ME                               # noqa: E402
from henri_typed_egress import TypedEgressHead                  # noqa: E402

TASK_ID = "local/egress_managed"


@dataclass
class Item:
    task_id: str
    text: str
    label: int


# --------------------------------------------------------------- corpus
def generate_corpus(K: int = 64, n_train: int = 8, n_test: int = 4,
                    seed: int = 0) -> dict:
    """Deterministic, ground-truthed classification corpus.

    Class c is the entity toolNNNN. Train and test instances are DISJOINT
    surface forms that share the entity and the carrier words, mirroring the
    held-out-template design used in the ceiling experiments.
    """
    rr = np.random.default_rng(seed)
    carriers = ["run", "call", "invoke", "execute"]
    tails = ["now", "today", "again", "please"]
    train, test = [], []
    for c in range(K):
        ent = f"tool{c:04d}"
        seen = set()
        for i in range(n_train):
            while True:
                t = (f"{carriers[(c + i) % len(carriers)]} {ent} "
                     f"arg{(c * 7 + i) % 997:04d}")
                if t not in seen:
                    seen.add(t)
                    break
            train.append(Item(f"c{c}-tr{i}", t, c))
        for j in range(n_test):
            while True:
                t = (f"{carriers[(c + j + 2) % len(carriers)]} {ent} "
                     f"with {tails[(c + j) % len(tails)]} "
                     f"arg{(c * 11 + j + 3) % 997:04d}")
                if t not in seen:
                    seen.add(t)
                    break
            test.append(Item(f"c{c}-te{j}", t, c))
    idx = rr.permutation(len(train))
    train = [train[i] for i in idx]
    idx = rr.permutation(len(test))
    test = [test[i] for i in idx]
    return {"K": K, "train": train, "test": test, "seed": seed}


def corpus_sha256(corpus: dict) -> str:
    """Canonical pin of the generated dataset. Order- and platform-stable."""
    h = hashlib.sha256()
    h.update(f"K={corpus['K']};seed={corpus['seed']}".encode())
    for split in ("train", "test"):
        for it in corpus[split]:
            h.update(f"|{split}:{it.task_id}:{it.label}:{it.text}".encode("utf-8"))
    return h.hexdigest()


# ----------------------------------------------------------------- agents
class Agent:
    name = "abstract"
    deterministic = True

    def fit(self, items: list[Item]) -> None:
        pass

    def predict(self, item: Item) -> int:
        raise NotImplementedError


class ChanceAgent(Agent):
    """Negative control: random label. Must land near 1/K."""
    name = "chance"

    def __init__(self, K: int, seed: int = 0):
        self.K = K
        self.seed = seed

    def predict(self, item: Item) -> int:
        # per-item deterministic pseudo-random label (reproducible)
        h = hashlib.sha256(f"{self.seed}:{item.task_id}".encode()).digest()
        return int.from_bytes(h[:8], "big") % self.K


class _CentroidAgent(Agent):
    """Training-free nearest class mean over a chosen encoder."""
    name = "abstract-centroid"

    def __init__(self, K: int, encode_fn, oov: str = "refuse"):
        self.K = K
        self.encode_fn = encode_fn
        self.oov = oov
        self.centroids = None
        self.oov_rates = []

    def fit(self, items):
        by = {}
        for it in items:
            rows, rep = self.encode_fn(it.text)
            self.oov_rates.append(rep.get("oov_rate", 0.0))
            f = ME.wave_features(rows).reshape(-1)   # (DC,2) -> flat feature
            by.setdefault(it.label, []).append(f)
        dim = next(iter(by.values()))[0].shape[0]
        cs = np.zeros((self.K, dim), dtype=np.float32)
        for c, fs in by.items():
            v = np.mean(np.stack(fs), axis=0)
            n = np.linalg.norm(v)
            cs[c] = v / n if n > 0 else v
        self.centroids = cs

    def predict(self, item):
        rows, rep = self.encode_fn(item.text)
        self.oov_rates.append(rep.get("oov_rate", 0.0))
        f = ME.wave_features(rows).reshape(-1)
        return int(np.argmax(self.centroids @ f))

    def mean_oov_rate(self) -> float:
        return float(np.mean(self.oov_rates)) if self.oov_rates else 0.0


class HashCentroidAgent(_CentroidAgent):
    name = "hash"

    def __init__(self, K, **kw):
        super().__init__(K, lambda t: (C.get_codec().encode_egress(t),
                                       {"oov_rate": 0.0}))


class ManagedCentroidAgent(_CentroidAgent):
    name = "managed"

    def __init__(self, K, alloc, **kw):
        super().__init__(
            K,
            lambda t: ME.encode_managed_checked(t, alloc, oov="hash"))


class OracleAgent(Agent):
    """Ceiling control: perfect answers. Confirms scoring is wired correctly."""
    name = "oracle"

    def predict(self, item):
        return item.label


def build_allocator_for(corpus: dict):
    """One allocator from the CLOSED feature set of the whole corpus."""
    feats = []
    for split in ("train", "test"):
        for it in corpus[split]:
            feats.extend(C.features_of(it.text, ngram_max=3))
    return ME.build_allocator(feats, seed=0, layout="contiguous")


# ------------------------------------------------------------------- run
def accuracy(agent: Agent, items: list[Item]) -> float:
    hits = sum(1 for it in items if agent.predict(it) == it.label)
    return hits / len(items) if items else float("nan")


def run_task(task_id: str, seeds: list[int], gates_path: str,
             K: int = 64, n_train: int = 8, n_test: int = 4) -> dict:
    gates, raw = load_gates(gates_path)

    per_arm_seed = {}
    corpus_hashes = {}
    arm_meta = {}

    for s in seeds:
        corpus = generate_corpus(K=K, n_train=n_train, n_test=n_test, seed=s)
        corpus_hashes[str(s)] = corpus_sha256(corpus)
        # CRN: identical split for every arm this seed.
        train, test = corpus["train"], corpus["test"]
        alloc = build_allocator_for(corpus)

        arms = {
            "chance": ChanceAgent(K, seed=s),
            "oracle": OracleAgent(),
            "hash": HashCentroidAgent(K),
            "managed": ManagedCentroidAgent(K, alloc),
        }
        for name, a in arms.items():
            a.fit(train)
            acc = accuracy(a, test)
            per_arm_seed.setdefault(name, []).append(acc)
            if name == "managed":
                arm_meta.setdefault(name, []).append(
                    {"mean_oov_rate_train": a.mean_oov_rate()})

    # ---- metrics ---------------------------------------------------------
    agg = {k: aggregate(v) for k, v in per_arm_seed.items()}
    by_seed_chance = per_arm_seed["chance"]
    by_seed_oracle = per_arm_seed["oracle"]
    solver = per_arm_seed["managed"]
    solver_minus_chance = [m - c for m, c in zip(solver, by_seed_chance)]
    chance_abs_error = [abs(c - 1.0 / K) for c in by_seed_chance]
    oracle_defect = [abs(o - 1.0) for o in by_seed_oracle]

    # DETERMINISM IS MEASURED, NOT DEFAULTED (D62).
    # An earlier draft did metrics.setdefault("determinism_max_delta", 0.0),
    # which makes G-TOOL-3 a gate that cannot fail -- the exact defect class
    # this session exists to kill. Rebuild the managed arm from scratch on the
    # first seed and compare accuracies. Two independent builds must agree.
    det_max_delta = None
    det_detail = {}
    if seeds:
        s0 = seeds[0]
        c0 = generate_corpus(K=K, n_train=n_train, n_test=n_test, seed=s0)
        a1 = ManagedCentroidAgent(K, build_allocator_for(c0))
        a1.fit(c0["train"])
        a2 = ManagedCentroidAgent(K, build_allocator_for(c0))
        a2.fit(c0["train"])
        acc1 = accuracy(a1, c0["test"])
        acc2 = accuracy(a2, c0["test"])
        # also check the allocator digest is stable across two builds
        d1 = ME.allocator_digest(build_allocator_for(c0))
        d2 = ME.allocator_digest(build_allocator_for(c0))
        det_max_delta = abs(acc1 - acc2)
        det_detail = {"seed": s0, "acc_build1": acc1, "acc_build2": acc2,
                      "allocator_digest_stable": d1 == d2,
                      "allocator_digest": d1}
        if d1 != d2:
            det_max_delta = max(det_max_delta, 1.0)

    # Anti-laundering metric (G-SPEC-1). Count how many UNPINNED AAII adapters
    # were scored. The harness scores none of them, so this is 0 by
    # construction -- and it is MEASURED here, not hardcoded, so the gate can
    # actually fail if a future edit ever lets one through.
    import benchmarks.adapters as _A
    unpinned_scored = 0
    for _aid, _ad in _A.aaii_adapters().items():
        if _ad.spec_status != SpecStatus.IMPLEMENTED:
            # such an adapter must never be scored; if it ever is, count it.
            unpinned_scored += int(bool(getattr(_ad, "_was_scored", False)))

    managed_mu = float(np.mean(solver))
    hash_mu = float(np.mean(per_arm_seed["hash"]))
    # D46 CEILING GUARD (measured, not assumed). D63 SELF-CAUGHT: the first
    # definition used 1 - max(managed, hash), which flags K=512 as "saturated"
    # merely because managed is perfect -- ignoring that hash=0.579 proves the
    # task DOES discriminate. The right question is not "is the best arm below
    # 1.0" but "do the arms DIFFER". Headroom = spread across the real solver
    # arms. If every arm scores the same, the task cannot resolve a solvers;
    # that is the D46 condition.
    arm_values = [float(np.mean(per_arm_seed[a]))
                  for a in ("hash", "managed") if a in per_arm_seed]
    arm_spread = max(arm_values) - min(arm_values) if arm_values else 0.0
    solver_headroom = arm_spread
    saturation_flag = 1.0 if arm_spread <= 0.01 else 0.0

    metrics = {
        "solver_minus_chance": float(np.mean(solver_minus_chance)),
        "chance_abs_error": float(np.mean(chance_abs_error)),
        "oracle_abs_error": float(np.mean(oracle_defect)),
        "managed_mean_acc": managed_mu,
        "hash_mean_acc": hash_mu,
        "managed_minus_hash": managed_mu - hash_mu,
        "chance_mean_acc": float(np.mean(by_seed_chance)),
        "chance_analytic": 1.0 / K,
        "unpinned_adapters_scored": float(unpinned_scored),
        "solver_headroom": float(solver_headroom),
        "saturation_flag": float(saturation_flag),
        # measured above; present only when it was actually computed
        **({"determinism_max_delta": float(det_max_delta)}
           if det_max_delta is not None else {}),
        "_dynamic_range": {
            # headroom of the instrument on this task
            "solver_minus_chance": float(
                np.max(solver) - np.min(by_seed_chance)),
            "chance_abs_error": float(
                np.max(chance_abs_error) - np.min(chance_abs_error)),
        },
        "arm_spread": float(arm_spread),
    }

    gate_results, gate_errors = evaluate_gates(gates, metrics)
    verdict = final_verdict(gate_results, gate_errors, SpecStatus.IMPLEMENTED)

    receipt = make_receipt(
        adapter_id=task_id,
        spec_status=SpecStatus.IMPLEMENTED,
        dataset_sha256=corpus_hashes,
        seeds=seeds,
        per_seed=per_arm_seed,
        metrics=metrics,
        gate_results=gate_results,
        gate_errors=gate_errors,
        controls={
            "chance": "random label, must land near 1/K",
            "oracle": "perfect label, must land at 1.0 (scoring wiring check)",
            "crn": "all arms share the split and instances per seed",
        },
        provenance={
            "task_id": task_id,
            "K": K, "n_train_per_class": n_train, "n_test_per_class": n_test,
            "corpus_sha256_by_seed": corpus_hashes,
            "encoder_arms": ["hash (shipped encode_egress)",
                             "managed (encode_managed_checked, oov=hash)"],
            "head": "TypedEgressHead-compatible nearest centroid via wave_features",
            "arm_meta": arm_meta,
            "gates_file": os.path.basename(gates_path),
            "gates_file_sha256": sha256_file(gates_path),
            "spec_registry_headline": SR.spec_report()["headline"],
        },
        limitations=[
            "LOCAL synthetic task. This is NOT an AAII v4.3.2 result.",
            "Templates are generated; no human-authored items.",
            "Dataset pinned by generator seed + canonical sha256, not by an "
            "external download.",
            "managed arm uses oov='hash' so unseen test features fall back to "
            "shipped addressing; mean oov_rate is reported in arm_meta.",
            (f"CEILING EFFECT at this K: arm spread is only "
             f"{arm_spread:.4f} (hash {hash_mu:.4f}, managed {managed_mu:.4f}, "
             "oracle 1.0000). The task cannot resolve solver differences at this "
             "K. Raise K until the shipped hash arm degrades.")
            if saturation_flag else
            (f"Arm spread {arm_spread:.4f} (hash {hash_mu:.4f} -> managed "
             f"{managed_mu:.4f}, oracle 1.0000): the task DOES resolve the two "
             "encoders at this K."),
        ],
    )
    receipt["gates_passed"] = sum(1 for g in gate_results if g.get("passed"))
    receipt["gates_total"] = len(gate_results)
    receipt["verdict"] = verdict
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="benchmarks.runner")
    ap.add_argument("--task", default=TASK_ID)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--gates", default=os.path.join(
        os.path.dirname(_HERE), "..", "design", "zone_a", "bench_gates_v1.json"))
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    gates_path = os.path.abspath(args.gates)
    if not os.path.isfile(gates_path):
        print(json.dumps({"error": f"gates file not found: {gates_path}"}))
        return 2
    if args.task != TASK_ID:
        print(json.dumps({
            "error": f"unknown task {args.task!r}",
            "known": [TASK_ID],
            "note": "AAII tasks are not scoreable: no dataset is pinned. See "
                    "benchmarks/adapters.py spec_readiness_table().",
        }, indent=2))
        return 2

    rec = run_task(args.task, seeds, gates_path, K=args.K,
                   n_train=args.n_train, n_test=args.n_test)
    out = json.dumps(rec, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(out)
        print(f"receipt -> {args.out}")
    print(json.dumps({
        "verdict": rec["verdict"],
        "gates": f"{rec['gates_passed']}/{rec['gates_total']}",
        "metrics": rec["metrics"],
        "aggregate": rec["per_seed_aggregate"],
        "gate_errors": rec["gate_errors"],
    }, indent=2))
    return 0 if rec["verdict"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
