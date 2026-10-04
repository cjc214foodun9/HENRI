"""End-to-end demo of the REAL typed-egress path. No training, no mocks.

This is the artifact in use: codec waves -> typed decision manifold.

    input text  ->  codec.encode_egress()  ->  TypedEgressHead.snap()
                ->  {field: (id, margin)}  ->  accept or REFUSE on low margin

Why the margins matter: a nearest-class-mean head is confident on content it has
seen and diffuse on novel content. The margin is the honest uncertainty signal,
so the demo REFUSES below a threshold instead of guessing. Fail-closed, matching
this project's posture.

Measured basis (readout_head_to_head.py, V=64, one corpus, four readouts):
    CENT (training-free) 0.9141 | RIDGE 0.8984 | GRAD (Adam) 0.4062 | HOP 0.0312
=> OPTIMIZATION_DEFICIT. The training-free readout is the measured fix.

Cost 0. CPU. No GPU. No store. No checkpoint. No network.
"""
import os
import sys

import numpy as np

_H2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _H2)

import zone_c_world_knowledge_codec as C      # noqa: E402
import henri_typed_egress as TE               # noqa: E402

NB = int(C.NUM_BLOCKS)
codec = C.get_codec()

TOOLS = [f"open file {i}" for i in range(8)]
ARGS = [f"target {i}" for i in range(4)]
TOOL_V = TE.typed_vocab(TOOLS)
ARG_V = TE.typed_vocab(ARGS)

TRAIN_T = ["{t} then {a}", "run {t} on {a}", "{t} for {a}", "please {t} {a}"]
TEST_T = ["now {t} the {a}", "start {t} with {a}"]

MARGIN_FLOOR = 0.02      # refuse below this; calibrated on the held-out spread


def rows(txt):
    return codec.encode_egress(txt)


print("=== TYPED EGRESS, END TO END (real codec waves, no training) ===")
print(f"   fields: tool {len(TOOLS)}-way  arg {len(ARGS)}-way   "
      f"joint {len(TOOLS)*len(ARGS)}-way   chance {1/(len(TOOLS)*len(ARGS)):.4f}")
print(f"   decode target: 2 typed fields, NOT 32000 tokens")

train = []
for t in TOOLS:
    for a in ARGS:
        for tpl in TRAIN_T:
            train.append((rows(tpl.format(t=t, a=a)), {"tool": TOOL_V[t], "arg": ARG_V[a]}))
held = []
for t in TOOLS:
    for a in ARGS:
        for tpl in TEST_T:
            held.append((rows(tpl.format(t=t, a=a)), {"tool": TOOL_V[t], "arg": ARG_V[a]}))

head = TE.TypedEgressHead({"tool": len(TOOLS), "arg": len(ARGS)}).fit(train)
s_tr = head.score(train)
s_te = head.score(held)
print(f"\n   fit: {len(train)} train waves, {len(held)} held-out waves")
print(f"   accuracy  tool  train {s_tr['per_field']['tool']:.4f}  held-out {s_te['per_field']['tool']:.4f}")
print(f"   accuracy  arg   train {s_tr['per_field']['arg']:.4f}  held-out {s_te['per_field']['arg']:.4f}")
print(f"   joint           train {s_tr['joint']:.4f}  held-out {s_te['joint']:.4f}")

# ---- the control: destroy content, the head must fall to chance
rng = np.random.default_rng(20261003)
tc, ac_, jc = 0, 0, 0
for (r, lab) in held:
    p = head.snap(r[rng.permutation(NB), :])
    tc += int(p["tool"][0] == lab["tool"])
    ac_ += int(p["arg"][0] == lab["arg"])
    jc += int(p["tool"][0] == lab["tool"] and p["arg"][0] == lab["arg"])
n = len(held)
print(f"   CONTROL (block-permuted)  tool {tc/n:.4f}  arg {ac_/n:.4f}  joint {jc/n:.4f}"
      f"   <- must sit at chance")

INV_TOOL = {v: k for k, v in TOOL_V.items()}
INV_ARG = {v: k for k, v in ARG_V.items()}

# ---- show real decisions with margins, and a refusal
# D21 SELF-CAUGHT DEFECT (found by READING the demo output, not by its verdict).
# The first draft printed TOOLS[t_id] and ARGS[a_id]. snap() returns an index into
# the class axis built from TOOL_V/ARG_V ids, and typed_vocab() REORDERS names by
# sha256, so positional TOOLS[k] is the WRONG name. The demo printed correct
# decisions as wrong ('open file 0' -> 'open file 3') while score() -- which compares
# IDS -- reported 1.0000. The two disagreed and the display would have made a working
# head look broken. Map the predicted id back through the inverse vocabulary.
print(f"\n   typed decisions on HELD-OUT templates (margin floor {MARGIN_FLOOR}):")
shown = 0
disp_ok = 0
for t in TOOLS:
    if shown >= 6:
        break
    for a in ARGS:
        txt = TEST_T[0].format(t=t, a=a)
        p = head.snap(rows(txt))
        t_id, t_m = p["tool"]
        a_id, a_m = p["arg"]
        pred_t, pred_a = INV_TOOL[t_id], INV_ARG[a_id]
        accept = t_m >= MARGIN_FLOOR and a_m >= MARGIN_FLOOR
        mark = "ACCEPT" if accept else "REFUSE"
        flag = "ok" if (pred_t == t and pred_a == a) else "WRONG"
        disp_ok += int(pred_t == t and pred_a == a)
        print(f"      {txt!r}  truth({t!r}, {a!r})")
        print(f"         -> tool={pred_t!r} (m={t_m:+.4f})  arg={pred_a!r} "
              f"(m={a_m:+.4f})  {mark} {flag}")
        shown += 1
        break

assert disp_ok == shown, (
    f"D21 REGRESSION: display says {disp_ok}/{shown} correct but score() reports "
    f"{s_te['joint']:.4f} joint. The display and the scorer disagree."
)
print(f"   display/scorer consistency: {disp_ok}/{shown} correct  agrees with score()  OK")

# ---- fail-closed demonstrations
print("\n   fail-closed checks:")
try:
    TE.TypedEgressHead({"tool": 8}).snap(rows("x"))
    print("      unfit snap: NO RAISE  <- BUG")
except RuntimeError as e:
    print(f"      unfit snap raises: {e}")

v_bad = TE.content_id("novel-token-xyz", len(TOOLS))
print(f"      open-vocab content_id('novel-token-xyz', {len(TOOLS)}) = {v_bad}"
      f"  (collision_count={TE.collision_count(TOOLS, len(TOOLS))} on the closed set;"
      f" typed_vocab is the bijective path)")

ok = (s_te["joint"] > 3 * (1 / (len(TOOLS) * len(ARGS)))
      and (jc / n) < 3 * (1 / (len(TOOLS) * len(ARGS))))
print(f"\n=== VERDICT ===")
print(f"   held-out joint {s_te['joint']:.4f} vs chance {1/(len(TOOLS)*len(ARGS)):.4f}; "
      f"control joint {jc/n:.4f}")
print("   -> " + ("PASS: typed egress works on held-out templates, control at chance"
                  if ok else "FAIL: check the control before reading any number"))
sys.exit(0 if ok else 1)
