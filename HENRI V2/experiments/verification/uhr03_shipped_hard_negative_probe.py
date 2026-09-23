
"""Does the SHIPPED helper do what v3's reimplementation did?

v3 proved the ALGORITHM with its own pooling code. This runs the SAME hard-comparator
fixture through `G.pooled_domain_statistic` -- the function the live runner calls --
because a verified algorithm and a verified function are different claims.

PRE-REGISTERED (before the numbers): PASS iff own_is_min AND margin > band, in >=4/7
seeds, with the hard comparator (same cells, DIFFERENT delta) present.
"""
import sys
import numpy as np, torch
SB = r"C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
sys.path.insert(0, SB)
import henri_external_outcome_refactor_module as M
import uhr02_exteroceptive_gate as G
from chromodynamic_grounding import GELL_MANN_BASIS, encode_su3_color_field

SIDE, NCH, N_UPD, N_ACT, AID, HARD, EASY = 16, 256, 16, 7, 2, 3, 4
LR, K = 0.1, 8192
TRUE_R, EASY_R = (4, 6), (10, 12)
basis = GELL_MANN_BASIS.to(torch.complex64)
base  = np.random.default_rng(7).integers(0, 4, size=(SIDE, SIDE), dtype=np.int64)
enc   = lambda g: encode_su3_color_field(
    torch.tensor(g, dtype=torch.int64).unsqueeze(0)).reshape(-1, 3, 3)
PK_T = list(np.random.default_rng(100).choice((TRUE_R[1]-TRUE_R[0])*SIDE, 8, replace=False))
PK_E = list(np.random.default_rng(200).choice((EASY_R[1]-EASY_R[0])*SIDE, 8, replace=False))

def edit(g, rows, pick, d):
    o = g.copy(); r0, r1 = rows
    cells = [(i, j) for i in range(r0, r1) for j in range(SIDE)]
    for p in pick:
        i, j = cells[int(p) % len(cells)]
        o[i, j] = (o[i, j] + d) % 4
    return o

g2 = lambda d: edit(base, TRUE_R, PK_T, d)
g3 = lambda d: edit(base, EASY_R, PK_E, d)
nz = lambda x: torch.nn.functional.normalize(x, p=2, dim=-1)

print("  fixtured: action2 delta +1 (true), action3 delta +2 (HARD, same cells),")
print("            action4 disjoint rows (easy anchor); store num_channels = %d" % NCH)

for stimulus in ("STABLE", "VARIED"):
    print()
    print("  --- %s ---" % stimulus)
    print("  seed | status   n_ch  own_is_min  margin    >band | verdict")
    npass = 0
    for seed in (21, 22, 23, 24, 25, 26, 27):
        store = M.ActionOutcomeGeneratorStore(num_actions=N_ACT, num_channels=NCH, lr=LR)
        for k in range(N_UPD):
            td = 1 if stimulus == "STABLE" else 1 + (k % 3)
            store.update_generator(enc(g2(td)), AID, enc(g2(td + 1)), basis)   # +1
            store.update_generator(enc(g2(td)), HARD, enc(g2(td + 2)), basis)  # +2 SAME cells
            store.update_generator(enc(g3(1)), EASY, enc(g3(2)), basis)        # disjoint
        U_t, U_n = enc(g2(1)), enc(g2(2))
        disp = G.relative_displacement(U_n, U_t).detach()
        roles = nz(torch.randn(K, 8, generator=torch.Generator().manual_seed(seed)))
        info = G.pooled_domain_statistic(roles, store, disp, basis, AID)
        ok = bool(info.get("own_is_min")) and bool(info.get("margin_above_band"))
        npass += ok
        print("  %-4d | %-8s %-5s %-11s %+8.6f %-6s | %s"
              % (seed, info.get("status"), info.get("n_channels"),
                 info.get("own_is_min"), info.get("margin"),
                 info.get("margin_above_band"), "PASS" if ok else "fail"))
    print("  -> %s PASS %d/7  (rule: >=4/7)" % (stimulus, npass))
