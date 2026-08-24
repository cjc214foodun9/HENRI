"""
Stage-0c-rev3 FRESH evaluation corpus — real CartPole via verified wrapper.
============================================================================
Reference 3 (gpt-5.6-sol) binding: rev2 evaluated on the 133-record
evaluation split (seeds 505,606,707,808,909). Those records are CONSUMED
for rev3 — a fresh multi-episode corpus with NEW seeds is required.

- One wrapper instance per episode, distinct seeds NOT in {101..1515}.
- Records: complete (obs_t, action, obs_next) from the verified ledger.
- Output: vla_stage0c_rev3_eval_corpus/seed_<ep>.jsonl + manifest.json.
- No-overlap: seeds disjoint by construction; state-overlap vs calib corpus
  reported (obs_t bytes intersection).
"""
import json, hashlib, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vla_stage0_gym_wrapper import Stage0GymWrapper

EPISODE_SEEDS = [2101, 2202, 2303, 2404, 2505, 2606, 2707, 2808, 2909, 3010]
MAX_STEPS = 100
OUT_DIR = pathlib.Path("vla_stage0c_rev3_eval_corpus")
OUT_DIR.mkdir(exist_ok=True)

def policy(rs, step):
    return int(rs.rand() < 0.5)

def main():
    assert all(s not in {101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
                         1111, 1212, 1313, 1414, 1515} for s in EPISODE_SEEDS)
    manifest = {"generator": "vla_stage0c_rev3_eval_corpus_builder.py",
                "wrapper": "Stage0GymWrapper (provenance eb1a061c)",
                "episode_seeds": EPISODE_SEEDS, "max_steps_per_episode": MAX_STEPS,
                "created_utc": "2026-08-24", "files": {}}
    total = 0
    for seed in EPISODE_SEEDS:
        w = Stage0GymWrapper(seed=seed)
        w.reset()
        rs = np.random.RandomState(seed ^ 0xC0FFEE)
        records = []
        for step in range(MAX_STEPS):
            a = policy(rs, step)
            rec = w.step(a)
            records.append(rec)
            if rec["terminated"] or rec["truncated"]:
                break
        path = OUT_DIR / f"seed_{seed}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in w.transitions:
                row = {
                    "episode_id": r["episode_id"], "step_id": r["step_id"],
                    "obs_t": list(r["obs_t"]),
                    "obs_t_hash": r["obs_t_hash"],
                    "action": int(r["action"]), "reward": float(r["reward"]),
                    "obs_next": list(r["obs_next"]),
                    "obs_next_hash": r["obs_next_hash"],
                    "terminated": bool(r["terminated"]),
                    "truncated": bool(r["truncated"]),
                }
                f.write(json.dumps(row) + "\n")
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["files"][path.name] = {"episode_seed": seed, "records": len(records), "sha256": h}
        total += len(records)
        print(f"ep {seed}: {len(records)} recs {h[:16]}")
    mp = OUT_DIR / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("TOTAL", total, "manifest", hashlib.sha256(mp.read_bytes()).hexdigest()[:16])

if __name__ == "__main__":
    main()
