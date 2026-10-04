"""Typed egress at the BLUEPRINT'S BOUND (K <= 512), via the SHIPPED module.

Blueprint HENRI-ARCH-2026-SYSTEMIC-EVALUATION-V3 sec 2.4:
    "Codebook size: K <= 512 (domain-specific typed primitives)."

WHY THIS RUN
    typed_egress_end_to_end_demo.py proved the path at 8x4 = 32-way. The blueprint
    names K <= 512. The capacity sweep says the decodable region for a FLAT manifold
    is V <= 128 and falls to 0.5449 at V = 512. So the question is sharp:

        Does TYPED DECOMPOSITION hold at K = 512 where the flat manifold collapses,
        using the SHIPPED module and a training-free readout?

    Note the ingredient control. Flat 512-way over 512 ARBITRARY tokens is 512
    atomic ingredients and measured 0.5449 (capacity sweep). Flat 512-way over
    32 tools x 16 args is 48 ingredients and measured 0.8857 (typed-manifold test).
    BOTH are 512-way manifolds. So this run holds the corpus fixed at 48 ingredients
    and compares:

        ARM-T   typed  : a 32-way head AND a 16-way head, jointly scored
        ARM-F   flat   : ONE 512-way head over the same 32x16 combinations

    Identical corpus, identical readout rule (nearest class mean), identical splits.
    Only the DECISION STRUCTURE differs. That isolates decomposition from manifold
    size and from ingredient count -- the confound that produced the earlier refusal.

PRE-REGISTERED GATES (frozen before the run)
    G-SC-CTRL  every control < 3*chance, else INSTRUMENT_INVALID, nothing read
    G-SC-TYPED typed joint >= 0.80 AND typed - flat >= 0.10
               -> DECOMPOSITION_HELPS_AT_K512
    G-SC-FLAT  flat joint >= 0.80 AND typed - flat < 0.10
               -> FLAT_SUFFICES_AT_48_INGREDIENTS
    G-SC-NULL  otherwise report the table; no mechanism claimed

CONTENT-GROUNDING: typed_vocab() (bijective, sha256-ranked); the codec's own
    encode_egress(); real labels; same-family block-permuted control.

Cost 0. CPU only. No GPU. No store. No checkpoint. No network.
"""
import os
import sys
import time

import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)

import zone_c_world_knowledge_codec as C      # noqa: E402
import henri_typed_egress as TE               # noqa: E402

NB = int(C.NUM_BLOCKS)
codec = C.get_codec()

N_TOOL = int(os.environ.get("SC_TOOLS", "32"))
N_ARG = int(os.environ.get("SC_ARGS", "16"))
JOINT = N_TOOL * N_ARG
CHANCE = 1.0 / JOINT

TOOLS = [f"tool{i:03d}" for i in range(N_TOOL)]
ARGS = [f"arg{i:03d}" for i in range(N_ARG)]
TV = TE.typed_vocab(TOOLS)
AV = TE.typed_vocab(ARGS)
INV_T = {v: k for k, v in TV.items()}
INV_A = {v: k for k, v in AV.items()}

TRAIN_T = ["run {t} on {a}", "{t} then {a}", "please {t} the {a}", "start {t} with {a}"]
TEST_T = ["now {t} for {a}", "go {t} then {a}"]

rng = np.random.default_rng(20261003)

print("=== TYPED EGRESS AT THE BLUEPRINT BOUND (K <= 512) ===")
print(f"   NB={NB}  tools {N_TOOL}-way  args {N_ARG}-way  joint {JOINT}-way")
print(f"   atomic ingredients {N_TOOL + N_ARG}  chance {CHANCE:.5f}")
print(f"   module: SHIPPED HENRI V2/henri_typed_egress.py   readout: training-free centroid")

t0 = time.time()
tr, te = [], []
for t in TOOLS:
    for a in ARGS:
        for tpl in TRAIN_T:
            e = codec.encode_egress(tpl.format(t=t, a=a))
            tr.append((e, {"tool": TV[t], "arg": AV[a], "joint": TV[t] * N_ARG + AV[a]}))
        for tpl in TEST_T:
            e = codec.encode_egress(tpl.format(t=t, a=a))
            te.append((e, {"tool": TV[t], "arg": AV[a], "joint": TV[t] * N_ARG + AV[a]}))
# control: same held-out waves, blocks permuted
te_c = [(e[rng.permutation(NB), :], lab) for (e, lab) in te]
print(f"   corpus: {len(tr)} train, {len(te)} held-out, {len(te_c)} control   "
      f"built in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------- ARM-T typed
head_t = TE.TypedEgressHead({"tool": N_TOOL, "arg": N_ARG}).fit(tr)
s_tr_t = head_t.score(tr)
s_te_t = head_t.score(te)
s_c_t = head_t.score(te_c)

# ---------------------------------------------------------------- ARM-F flat
head_f = TE.TypedEgressHead({"joint": JOINT}).fit(tr)
s_tr_f = head_f.score(tr)
s_te_f = head_f.score(te)
s_c_f = head_f.score(te_c)

print(f"\n   {'arm':<22} {'train':>7} {'held-out':>9} {'control':>8} {'x chance':>9}")
print(f"   {'TYPED tool':<22} {s_tr_t['per_field']['tool']:>7.4f} "
      f"{s_te_t['per_field']['tool']:>9.4f} {s_c_t['per_field']['tool']:>8.4f} "
      f"{s_te_t['per_field']['tool']/CHANCE:>8.1f}x")
print(f"   {'TYPED arg':<22} {s_tr_t['per_field']['arg']:>7.4f} "
      f"{s_te_t['per_field']['arg']:>9.4f} {s_c_t['per_field']['arg']:>8.4f} "
      f"{s_te_t['per_field']['arg']/CHANCE:>8.1f}x")
print(f"   {'TYPED joint':<22} {s_tr_t['joint']:>7.4f} {s_te_t['joint']:>9.4f} "
      f"{s_c_t['joint']:>8.4f} {s_te_t['joint']/CHANCE:>8.1f}x")
print(f"   {'FLAT 512-way':<22} {s_tr_f['joint']:>7.4f} {s_te_f['joint']:>9.4f} "
      f"{s_c_f['joint']:>8.4f} {s_te_f['joint']/CHANCE:>8.1f}x")

print("\n=== VERDICT (pre-registered) ===")
# D23 SELF-CAUGHT DEFECT, found by reading the smoke output against this code.
# The first draft was:
#     tight = min(t_joint_ctrl, t_tool_ctrl, t_arg_ctrl, f_joint_ctrl)
#     ctl_ok = tight < 3 * CHANCE
# Two faults. (1) min() reports the BEST control, so an elevated one is hidden --
# a gate that can only pass. (2) per-field controls were compared against the JOINT
# chance (1/JOINT), but the tool head's chance is 1/N_TOOL and the arg head's is
# 1/N_ARG. At 32x16 the tool threshold would have been 3/512 = 0.0059 against a tool
# chance of 1/32 = 0.031 -- an impossible bar, which min() then masked.
# Fix: every control is compared against 3x ITS OWN field's chance, and the WORST
# ratio is reported.
checks = {
    "typed tool": (s_c_t["per_field"]["tool"], 1.0 / N_TOOL),
    "typed arg":  (s_c_t["per_field"]["arg"], 1.0 / N_ARG),
    "typed joint": (s_c_t["joint"], CHANCE),
    "flat joint": (s_c_f["joint"], CHANCE),
}
ctl_ok = all(v < 3 * c for (v, c) in checks.values())
worst_name, worst_ratio = max(((k, v / c) for k, (v, c) in checks.items()),
                              key=lambda kv: kv[1])
for k, (v, c) in checks.items():
    print(f"   control {k:<12} {v:.5f}  vs chance {c:.5f}  = {v/c:.2f}x"
          f"  {'OK' if v < 3 * c else 'ELEVATED'}")
print(f"   controls_at_chance={ctl_ok}  (worst {worst_name} at {worst_ratio:.2f}x, "
      f"gate < 3.00x)")
typ, fla = s_te_t["joint"], s_te_f["joint"]
print(f"   typed joint {typ:.4f}   flat joint {fla:.4f}   delta {typ-fla:+.4f}")
if not ctl_ok:
    print("   -> INSTRUMENT_INVALID: a control sits above 3x chance; nothing read")
elif typ >= 0.80 and typ - fla >= 0.10:
    print(f"   -> DECOMPOSITION_HELPS_AT_K512: two small typed heads {typ:.4f} vs one")
    print(f"      flat {JOINT}-way head {fla:.4f} on the SAME corpus. Blueprint item 3")
    print("      supported with the manifold size held fixed.")
elif fla >= 0.80 and typ - fla < 0.10:
    print(f"   -> FLAT_SUFFICES_AT_48_INGREDIENTS: flat {fla:.4f} at 48 atomic")
    print("      ingredients. Type the OUTPUT for safety and routing, not for accuracy.")
else:
    print("   -> TABLE_REPORTED: no pre-registered condition met; no claim")

print(f"\n   elapsed {time.time()-t0:.1f}s   cost $0   CPU only")
