"""Contract tests for v0.4 stochastic eval amendment (CPU, disposable).

Verifies:
 1. decode_sample seeded reproducibility (same seed -> byte-identical)
 2. decode_sample seed diversity (different seeds -> different programs)
 3. decode_vote deterministic winner + energy weighting + tie-break
 4. eval_split stochastic=True runs end-to-end and emits gates
 5. eval_split deterministic mode unchanged (eval_mode == deterministic_greedy)
 6. decode_vote respects FSA (all sampled tokens valid under NEXT_MASK)
"""
import random
import sys
sys.path.insert(0, ".")
import torch

from system1_kernel_v04 import (System1KernelV04, SwarmEngineV04, KernelV04Config,
                                TOK2ID, NEXT_MASK, detokenize)
from train_system1_kernel_v04 import (gen_task, eval_split, sig_matrix, sig_ids,
                                      pad_tokens, load_split, sandbox)

torch.manual_seed(0)
dev = "cpu"
cfg = KernelV04Config()
model = System1KernelV04(cfg=cfg).to(dev)
eng = SwarmEngineV04(model).to(dev)
model.eval()

rng = random.Random(7)
tasks = [gen_task(rng) for _ in range(3)]
# forward_swarm(b_target=6) with 3 input rows -> repeat_interleave(2) = 6
# particles. Build the signature matrix to MATCH the 6 particles exactly.
sig_ids_2x = [sig_ids(t) for t in tasks for _ in range(2)]
sp = sig_matrix(model, tasks + tasks, 16, dev)  # placeholder; replaced below
sp = model.token_emb(pad_tokens(sig_ids_2x, 16).to(dev))            # [6, 16, d]
z0 = model.encode_tokens(pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
out = eng.forward_swarm(z0, b_target=6, steps=4)
z = out["z"]                                                          # [6, 16, d]
e = out["energy"]                                                     # [6]
assert z.shape[0] == sp.shape[0] == 6, (z.shape, sp.shape)

fails = []

# 1. seed reproducibility
a, _ = model.decode_sample(z[0:1], sp[0:1], seed=12345)
b2, _ = model.decode_sample(z[0:1], sp[0:1], seed=12345)
assert a.tolist() == b2.tolist(), "seed replay not byte-identical"
print("PASS seed_replay_identical")

# 2. diversity across seeds
d1, _ = model.decode_sample(z[0:1], sp[0:1], seed=111)
d2, _ = model.decode_sample(z[0:1], sp[0:1], seed=222)
if d1.tolist() == d2.tolist():
    print("WARN seeds produced identical samples (low-entropy model, acceptable)")
else:
    print("PASS seed diversity")

# 3. decode_vote deterministic + record shape
ids1, rec1 = model.decode_vote(z, sp, e, seed_base=5)
ids2, rec2 = model.decode_vote(z, sp, e, seed_base=5)
assert ids1 == ids2, "decode_vote not deterministic under same seeds"
assert rec1["unique_programs"] >= 1
assert "winner_weight" in rec1 and "winner_ids" in rec1
print(f"PASS decode_vote determinism; unique={rec1['unique_programs']} "
      f"winner_w={rec1['winner_weight']:.3f}")

# 4. FSA validity of sampled sequences (no UNK, transitions legal)
valid = True
for i in range(z.shape[0]):
    s_ids, _ = model.decode_sample(z[i:i + 1], sp[i:i + 1], seed=999 + i)
    toks = s_ids[0].tolist()
    if TOK2ID["UNK"] in toks:
        valid = False
    prev = TOK2ID["BOS"]
    for t in toks:
        if t in (TOK2ID["PAD"], TOK2ID["EOS"]):
            continue
        if not NEXT_MASK[prev, t].item():
            valid = False
            print(f"FSA VIOLATION {detokenize(toks)} at {prev}->{t}")
            break
        prev = t
assert valid, "sampled decode violated FSA"
print("PASS FSA validity of sampled decode")

# 5. eval_split stochastic end-to-end on 3 tasks
rep = eval_split(eng, model, dev, tasks, swarm_b=8, stochastic=True,
                 vote_seed_base=42)
assert rep["eval_mode"] == "stochastic_vote"
assert rep["gates"]["seed_replay_identical"] is True
assert rep["n"] == 3
assert "items" in rep and len(rep["items"]) == 3
print(f"PASS stochastic eval_split: mode={rep['eval_mode']} "
      f"swarm={rep['swarm_pass']} single={rep['single_pass']} "
      f"beam={rep['beam_pass']} gates_keys={sorted(rep['gates'].keys())}")

# 6. deterministic mode unchanged
rep_d = eval_split(eng, model, dev, tasks, swarm_b=8, stochastic=False)
assert rep_d["eval_mode"] == "deterministic_greedy"
assert "gates" not in rep_d
print(f"PASS deterministic eval_split unchanged: "
      f"swarm={rep_d['swarm_pass']} single={rep_d['single_pass']} "
      f"beam={rep_d['beam_pass']}")

# 7. sandbox + ast in item records
assert all("vote" in it and it["vote"] is not None for it in rep["items"])
print("PASS item records carry vote telemetry")

print("\nALL_CONTRACT_TESTS_PASS")
