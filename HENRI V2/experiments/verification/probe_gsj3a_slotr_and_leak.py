"""probe_gsj3a_slotr_and_leak.py -- is two_stage's 0.2000 a mechanism or a base rate?

THE RESULT UNDER TEST
    two_stage top1 = 0.2000 on the 217-column dedup bank, against within-group gold
    precision 0.00378 and mean max-group size 188.69 of 217 columns (87% of the bank is
    one tie group). A number 53x the within-group precision must be probed for a defect
    BEFORE it is believed.

THE SHARP CONTROL (added after the first version)
    C3 RANDOM GROUPS: keep the channel, replace the WAVE max group with a random
    same-size subset, then take the max-frequency column inside it. If C3 scores ~= the
    conditional arm, then stage 1 (the wave max group) contributes NOTHING and the score
    is the corpus base rate wearing a two-stage costume.

OTHER CONTROLS
    C1 channel shuffled across columns   -> must NOT reproduce 0.2
    C2 random channel                    -> must collapse to uniform-in-group

slot_r = Pearson r(picked column index, correctness) + permutation drift.
A constant pick vector makes r undefined; that IS the index-bound signature.
"""
import json, os, sys
import torch
sys.path.insert(0, "HENRI V2/henri_core")
import henri_candidate_bank as cb
import wave_readout as wr

K_SEED = 20261010
BANK = os.path.join("design", "zone_a", "evidence", "banks", "candidate_bank_dedup.pt")
OUT = os.path.join("design", "zone_a", "evidence", "henri_gsj3a_slotr_observed.json")

d = torch.load(BANK, weights_only=False)
sim_d = d["sim_d"]; ch = d["ch_dedup"]; cols = d["cols"]
gold_d = [set(g) for g in d["gold_d"]]
Q, K = sim_d.shape
print(f"bank {Q}q x {K}c | channel uniq={int(torch.unique(ch).numel())} | "
      f"tie-group mean={max(len(g) for g in cb.max_groups(sim_d))} (max)")
groups = cb.max_groups(sim_d)
sizes = [len(g) for g in groups]
print(f"mean group size {sum(sizes)/len(sizes):.2f} of {K} columns "
      f"({100*sum(sizes)/len(sizes)/K:.1f}% of the bank)")

def acc(picks):
    return sum(1 for q in range(Q) if int(picks[q]) in gold_d[q]) / Q

def cond_pick(channel):
    out = []
    for q, mem in enumerate(groups):
        vals = sorted((float(channel[q, m]), m) for m in mem)
        out.append(vals[-1][1])          # max channel value; ties -> highest index
    return out

# ---- conditional arm ----
p_cond = wr.select(sim_d, channel=ch, mode="two_stage",
                   generator=torch.Generator().manual_seed(K_SEED))
a_cond = acc(p_cond)

# ---- C3: RANDOM GROUPS, same sizes, same channel (the sharp control) ----
g = torch.Generator().manual_seed(K_SEED + 11)
a_rg = 0.0
NREP = 20
for _ in range(NREP):
    picks = []
    for q in range(Q):
        mem = torch.randperm(K, generator=g)[:len(groups[q])].tolist()
        vals = sorted((float(ch[q, m]), m) for m in mem)
        picks.append(vals[-1][1])
    a_rg += acc(picks) / NREP

# ---- unconditional constant pick ----
freq = ch[0]
gp = int(torch.argmax(freq))
a_uncond = sum(1 for q in range(Q) if gp in gold_d[q]) / Q

# ---- baselines ----
a_uni = acc(wr.select(sim_d, mode="uniform", generator=torch.Generator().manual_seed(1)))
a_arg = acc(wr.select(sim_d, mode="argmax"))

# ---- controls ----
perm = torch.randperm(K, generator=torch.Generator().manual_seed(99))
a_shuf = acc(wr.select(sim_d, channel=ch[:, perm], mode="two_stage",
                       generator=torch.Generator().manual_seed(2)))
rc = torch.rand(Q, K, generator=torch.Generator().manual_seed(7))
a_rand = acc(wr.select(sim_d, channel=rc, mode="two_stage",
                       generator=torch.Generator().manual_seed(3)))

# ---- slot_r + permutation drift ----
idx = torch.tensor([float(p) for p in p_cond])
cor = torch.tensor([1.0 if int(p_cond[q]) in gold_d[q] else 0.0 for q in range(Q)])
const = bool(idx.std() == 0 or cor.std() == 0)
r_slot = float("nan") if const else float(torch.corrcoef(torch.stack([idx, cor]))[0, 1])
order = torch.randperm(K, generator=torch.Generator().manual_seed(K_SEED + 7))
pp = wr.select(sim_d[:, order], channel=ch[:, order], mode="two_stage",
               generator=torch.Generator().manual_seed(K_SEED))
a_perm = sum(1 for q in range(Q) if int(order[pp[q]]) in gold_d[q]) / Q
drift = abs(a_perm - a_cond)

print(f"\n{'arm':<34}{'top1':>9}")
for name, v in (("conditional two_stage", a_cond),
                ("C3 random-group + channel", a_rg),
                ("unconditional most-frequent", a_uncond),
                ("uniform-in-group", a_uni),
                ("argmax (legacy)", a_arg),
                ("C1 shuffled channel", a_shuf),
                ("C2 random channel", a_rand)):
    print(f"  {name:<32}{v:>9.4f}")
print(f"\n  slot_r={r_slot:.5f} const_pick={const} perm_drift={drift:.5f} "
      f"(perm top1={a_perm:.4f})")

stage1_gain = a_cond - a_rg
base_gain = a_cond - a_uncond
if abs(stage1_gain) < 0.05:
    leak = ("BASE_RATE_ARTIFACT -- random groups with the SAME channel score "
            f"{a_rg:.4f} vs {a_cond:.4f}. Stage 1 (the wave max group) adds nothing.")
elif abs(base_gain) < 0.05:
    leak = (f"STAGE1_ADDS_SIGNAL but the pick matches the global most-frequent column "
            f"({a_uncond:.4f}); the score is corpus multiplicity, not wave conditioning.")
else:
    leak = ("MAX_GROUP_ADDS_SIGNAL -- conditional beats both random-group and "
            "unconditional controls.")
print(f"\nLEAK VERDICT: {leak}")

gate = wr.gate_sj3a(a_cond, rand_control=1.0 / K, slot_r=abs(r_slot))
rec = {"bank": BANK, "K": K, "Q": Q, "mean_group_size": sum(sizes) / len(sizes),
       "arm_conditional": a_cond, "control_random_groups": a_rg,
       "arm_unconditional_most_frequent": a_uncond, "arm_uniform": a_uni,
       "arm_argmax": a_arg, "control_shuffled_channel": a_shuf,
       "control_random_channel": a_rand, "slot_r": r_slot, "const_pick": const,
       "perm_drift": drift, "perm_top1": a_perm,
       "gate_sj3a": gate, "leak_verdict": leak,
       "stage1_gain_vs_random_groups": stage1_gain}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(rec, open(OUT, "w", encoding="utf-8"), indent=2)
print(f"\n--- gate ---")
for k, v in gate.items():
    print(f"  {k:26} {v}")
print(f"[receipt] {OUT}")
