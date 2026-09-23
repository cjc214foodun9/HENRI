
"""THE DECIDING CONTROL for the pooled domain: does the margin measure
TRANSITION CONTENT, or only SUPPORT OVERLAP?

WHY THIS DECIDES RUN #4. The pooled statistic reads every candidate at the channels
the observation moved. Measured, the true action scores 0.305618 and every other
action scores its do-nothing baseline 0.519882. That separation is consistent with
TWO different mechanisms:
  (M1) the true action's learned transition MATCHES the observed one (content), or
  (M2) the true action merely HAS support at those channels while the others have
       none (support overlap), which is a discrete set-membership property.
M1 is a real capability; M2 is close to a lookup table on WHERE the edit happened.

DISCRIMINATING FIXTURE (the co-scientist "vary where the signal lives while holding
the operator fixed" rule):
  Build an action whose SUPPORT is identical to the true action's but whose
  TRANSITION VALUES differ (same cells edited, different colour mapping), plus an
  action with a DIFFERENT support and the SAME values. Then:
    - if same-support/different-content scores ~0.3 -> SUPPORT OVERLAP (M2)
    - if same-support/different-content scores ~0.5 -> CONTENT (M1)
This is free and local. If M2 holds, run #4 would produce a margin that proves
nothing about the mechanism, and the domain needs content, not just pooling.
"""
import sys
import numpy as np, torch
SB = r"C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
sys.path.insert(0, SB)
import henri_external_outcome_refactor_module as M
import uhr02_exteroceptive_gate as G
from chromodynamic_grounding import GELL_MANN_BASIS, encode_su3_color_field

SIDE, NCH, N_ACT, N_UPD, AID = 16, 256, 7, 16, 2
TRUE_CELLS = [(4, j) for j in range(SIDE)]
PICK_TRUE  = list(np.random.default_rng(100).choice(len(TRUE_CELLS), size=8, replace=False))
OTHER_CELLS = [(10, j) for j in range(SIDE)]
PICK_OTHER  = list(np.random.default_rng(200).choice(len(OTHER_CELLS), size=8, replace=False))

base = np.random.default_rng(7).integers(0, 4, size=(SIDE, SIDE), dtype=np.int64)
enc  = lambda g: encode_su3_color_field(
    torch.tensor(g, dtype=torch.int64).unsqueeze(0)).reshape(-1, 3, 3)
basis = GELL_MANN_BASIS.to(torch.complex64)


def edit(g, cells, pick, delta):
    out = g.copy()
    for p in pick:
        i, j = cells[int(p) % len(cells)]
        out[i, j] = (out[i, j] + delta) % 4
    return out


# --- The TRUE action (aid=2): edits TRUE_CELLS by +1 (+color) ------------------
def grid_true(color):
    return edit(base, TRUE_CELLS, PICK_TRUE, 1 + color)


# --- Arm S: SAME SUPPORT (TRUE_CELLS), DIFFERENT CONTENT (edits by +2) ---------
def grid_same_support(color):
    return edit(base, TRUE_CELLS, PICK_TRUE, 2 + color)


# --- Arm D: DIFFERENT SUPPORT (OTHER_CELLS), SAME content (+1) -----------------
def grid_diff_support(color):
    return edit(base, OTHER_CELLS, PICK_OTHER, 1 + color)


store = M.ActionOutcomeGeneratorStore(num_actions=N_ACT, num_channels=NCH, lr=0.1)
# action 2 learns the TRUE transition; actions 3 and 4 learn the two controls
for k in range(N_UPD):
    store.update_generator(enc(grid_true(2*k)), 2, enc(grid_true(2*k+1)), basis)
    store.update_generator(enc(grid_same_support(2*k)), 3,
                           enc(grid_same_support(2*k+1)), basis)
    store.update_generator(enc(grid_diff_support(2*k)), 4,
                           enc(grid_diff_support(2*k+1)), basis)

print("=" * 80); print("SUPPORT MAP (theta_a) "); print("=" * 80)
for a in range(N_ACT):
    s = (store.theta_a[a].norm(dim=-1) > 1e-5).nonzero().flatten().tolist()
    print("  action %d: |support|=%2d  %s" % (a, len(s), s[:10]))

# Held-out TRUE transition (same cells, unseen colours)
U_t, U_n = enc(grid_true(200)), enc(grid_true(201))
disp = G.relative_displacement(U_n, U_t).detach()
chs = G.observed_change_channels(disp)
print("\n  observed-change channels for the TRUE transition: n=%d %s" % (len(chs), chs))

K = 8192
nz = lambda x: torch.nn.functional.normalize(x, p=2, dim=-1)
roles = nz(torch.randn(K, 8, generator=torch.Generator().manual_seed(21)))

print()
print("=" * 80)
print("POOLED RESIDUAL PER ACTION (the deciding numbers)")
print("=" * 80)
truth = {c: G.generators_from_displacement(disp, channel=c) for c in chs}
labels = {2: "TRUE action      (true support, true content)",
          3: "SAME SUPPORT, DIFF CONTENT  <- control",
          4: "DIFF SUPPORT, SAME CONTENT  <- control"}
res = {}
for a in range(N_ACT):
    tot = 0.0
    for c in chs:
        cand = store.lie_element(a, basis)[c]
        tot += float(G.exteroceptive_residual_vs_recorded(
            roles, [cand], truth[c], basis).delta_pred)
    res[a] = tot / len(chs)
    tag = labels.get(a, "(empty / unrelated action)")
    print("  action %d: pooled=%.6f   %s" % (a, res[a], tag))

print()
band = G.sampling_band(K)
print("  band = %.6f" % band)
print("  TRUE             = %.6f" % res[2])
print("  SAME SUPPORT     = %.6f  (support overlap would put this near TRUE)" % res[3])
print("  DIFF SUPPORT     = %.6f  (content-only would put this near TRUE)" % res[4])
print("  empty actions    = %s" % sorted(round(res[a], 6) for a in (0, 1, 5, 6)))
print()
d_same = res[3] - res[2]
d_diff = res[4] - res[2]
print("  SAME SUPPORT - TRUE = %+.6f  (small => SUPPORT OVERLAP drives the margin)" % d_same)
print("  DIFF SUPPORT - TRUE = %+.6f  (small => CONTENT drives the margin)" % d_diff)
print()
if abs(d_same) < abs(d_diff):
    print("  VERDICT: M2 -- the margin is dominated by SUPPORT OVERLAP.")
    print("           The pooled statistic detects WHERE the edit happened, not WHAT it was.")
    print("           Run #4 would emit a margin that does not evidence transition content.")
else:
    print("  VERDICT: M1 -- the margin tracks transition CONTENT.")

print()
print("=" * 80); print("FULL POPULATION + C1/C2 THROUGH THE SHIPPED HELPER"); print("=" * 80)
info = G.pooled_domain_statistic(roles, store, disp, basis, 2)
print("  status=%s n_channels=%s" % (info.get("status"), info.get("n_channels")))
print("  delta_all   = %s" % info.get("delta_pooled_all"))
print("  own_is_min=%s  margin=%s  margin_above_band=%s"
      % (info.get("own_is_min"), info.get("margin"), info.get("margin_above_band")))
