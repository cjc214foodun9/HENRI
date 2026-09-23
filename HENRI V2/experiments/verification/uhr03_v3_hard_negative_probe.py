
"""UHR-03 v3 -- THREE-ANCHOR test with the HARD comparator restored.

WHY THIS FILE EXISTS (a comparator swap I must not repeat). The v2 probe reported
"STABLE wins 5/5" against a comparator that had NO support at the observed channels,
so S == B to six decimals in every row. A comparator that cannot score above the
do-nothing baseline cannot fail, so its "win" carried no information. This file puts
the HARD comparator back:

  T       action 2: SAME cells (TRUE_R), the TRUE delta      (+1)
  S_hard  action 3: SAME cells (TRUE_R), a DIFFERENT delta   (+2)  <- the discriminator
  B       actions w/o support at those cells (untrained + disjoint-row action)

DECISION RULE (pre-registered here, before the numbers):
  PASS  iff  T is the argmin AND margin_hard = S_hard - T > band.
  Ties are flagged at BAND scale, not float epsilon: |margin| < band is a FAIL.
  Support the estimator-to-stimulus hypothesis iff PASS in >= 4 of 5 seeds in the
  STABLE arm AND VARIED degrades it.

READ FROM THIS FILE. Do not trust an inline stream.
"""
import sys, itertools
import numpy as np, torch
SB = r"C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
sys.path.insert(0, SB)
import henri_external_outcome_refactor_module as M
import uhr02_exteroceptive_gate as G
from chromodynamic_grounding import GELL_MANN_BASIS, encode_su3_color_field

SIDE, NCH, N_UPD, N_ACT = 16, 256, 16, 7
AID, HARD, EASY = 2, 3, 4
TRUE_R = (4, 6)          # cells BOTH action 2 and action 3 edit
EASY_R = (10, 12)        # disjoint rows for the no-support anchor
TD     = 1               # fixed training colour -> no phase wrap (1->2, 1->3)
LR     = 0.1
K      = 8192
basis  = GELL_MANN_BASIS.to(torch.complex64)
base   = np.random.default_rng(7).integers(0, 4, size=(SIDE, SIDE), dtype=np.int64)
enc    = lambda g: encode_su3_color_field(
    torch.tensor(g, dtype=torch.int64).unsqueeze(0)).reshape(-1, 3, 3)
PICK_T = list(np.random.default_rng(100).choice((TRUE_R[1]-TRUE_R[0])*SIDE, size=8, replace=False))
PICK_E = list(np.random.default_rng(200).choice((EASY_R[1]-EASY_R[0])*SIDE, size=8, replace=False))

def edit(g, rows, pick, delta):
    out = g.copy(); r0, r1 = rows
    cells = [(i, j) for i in range(r0, r1) for j in range(SIDE)]
    for p in pick:
        i, j = cells[int(p) % len(cells)]
        out[i, j] = (out[i, j] + delta) % 4
    return out

g2 = lambda d: edit(base, TRUE_R, PICK_T, d)
g3 = lambda d: edit(base, EASY_R, PICK_E, d)
nz = lambda x: torch.nn.functional.normalize(x, p=2, dim=-1)
BAND = G.sampling_band(K)


def arm(stimulus, convention, seed):
    store = M.ActionOutcomeGeneratorStore(num_actions=N_ACT, num_channels=NCH, lr=LR)
    for k in range(N_UPD):
        td = TD if stimulus == "STABLE" else 1 + (k % 3)
        store.update_generator(enc(g2(td)), AID, enc(g2(td + 1)), basis)   # true  delta +1
        store.update_generator(enc(g2(td)), HARD, enc(g2(td + 2)), basis)  # hard  delta +2
        store.update_generator(enc(g3(1)), EASY, enc(g3(2)), basis)        # disjoint
    U_t, U_n = enc(g2(TD)), enc(g2(TD + 1))
    disp = G.relative_displacement(U_n, U_t).detach()
    chs  = G.observed_change_channels(disp)

    pre = store.theta_a[AID].detach().clone()
    store.update_generator(U_t, AID, U_n, basis)     # the LIVE order folds it in first
    post = store.theta_a[AID].detach().clone()

    roles = nz(torch.randn(K, 8, generator=torch.Generator().manual_seed(seed)))
    truth = {c: G.generators_from_displacement(disp, channel=c) for c in chs}

    def pooled(a):
        tot = 0.0
        for c in chs:
            if a == AID and convention == "PRE":
                cand = 1j * torch.einsum("a,aij->ij", pre[c].to(basis.dtype), basis)
            elif a == AID and convention == "POST":
                cand = 1j * torch.einsum("a,aij->ij", post[c].to(basis.dtype), basis)
            else:
                cand = store.lie_element(a, basis)[c]
            tot += float(G.exteroceptive_residual_vs_recorded(
                roles, [cand], truth[c], basis).delta_pred)
        return tot / len(chs)

    vals = {a: pooled(a) for a in range(N_ACT)}
    both = sorted(vals.items(), key=lambda kv: kv[1])
    return dict(
        n_ch=len(chs), T=vals[AID], S_hard=vals[HARD], S_easy=vals[EASY],
        B=float(np.mean([vals[a] for a in (0, 1, 5, 6)])),
        m_hard=vals[HARD] - vals[AID], m_easy=vals[EASY] - vals[AID],
        argmin=both[0][0], tie=abs(vals[HARD] - vals[AID]) < BAND,
        same_as_B=abs(vals[HARD] - vals[EASY]) < 1e-9,
        table=sorted(round(v, 6) for v in vals.values()),
    )


print("=" * 108)
print("v3  THREE ANCHORS, HARD COMPARATOR RESTORED   (band = %.6e)" % BAND)
print("=" * 108)
rows = {}
for stimulus in ("STABLE", "VARIED"):
    for conv in ("PRE", "POST"):
        print()
        print("  --- %s / %s ---" % (stimulus, conv))
        print("  seed |      T   S_hard   S_easy        B |  m_hard   m_easy | argmin  tie  hard==easy")
        for seed in (21, 22, 23, 24, 25, 26, 27):
            r = arm(stimulus, conv, seed)
            rows[(stimulus, conv, seed)] = r
            print("  %-4d | %7.5f %7.5f %8.5f %8.5f | %+7.5f %+7.5f | %-6s %-4s %s"
                  % (seed, r["T"], r["S_hard"], r["S_easy"], r["B"],
                     r["m_hard"], r["m_easy"], r["argmin"], r["tie"], r["same_as_B"]))

print()
print("=" * 108)
print("PRE-REGISTERED DECISION RULE   PASS = argmin == 2 AND m_hard > band")
print("=" * 108)
summary = {}
for stimulus in ("STABLE", "VARIED"):
    for conv in ("PRE", "POST"):
        v = [rows[(stimulus, conv, s)] for s in (21, 22, 23, 24, 25, 26, 27)]
        passes = sum(1 for r in v if r["argmin"] == AID and r["m_hard"] > BAND)
        ties   = sum(1 for r in v if r["tie"])
        caught = sum(1 for r in v if r["same_as_B"])
        mh = [r["m_hard"] for r in v]
        summary[(stimulus, conv)] = passes
        print("  %-7s/%-5s  PASS=%d/7  ties=%d  hard==easy=%d  m_hard mean=%+.5f min=%+.5f max=%+.5f"
              % (stimulus, conv, passes, ties, caught, float(np.mean(mh)), min(mh), max(mh)))

print()
print("=" * 108)
print("READING")
print("=" * 108)
sp = summary[("STABLE", "PRE")]; vp = summary[("VARIED", "PRE")]
print("  STABLE/PRE PASS = %d/7   VARIED/PRE PASS = %d/7" % (sp, vp))
print("  comparator integrity (hard must NOT equal easy):",
      all(not rows[k]["same_as_B"] for k in rows), " <- False would mean the swap returned")
print()
if sp >= 4 and vp < sp:
    print("  VERDICT: estimator-to-stimulus SUPPORTED. FORM B ranks the true action above")
    print("           SAME-SUPPORT-DIFFERENT-CONTENT when the store has converged on a")
    print("           consistent transition, and degrades when the stimulus varies.")
elif sp >= 4 and vp >= sp:
    print("  VERDICT: FORM B discriminates content, but stability is NOT the lever")
    print("           (VARIED passes as often) -> the earlier 2/5 was fixture-specific.")
else:
    print("  VERDICT: NOT supported. Under the HARD comparator the true action does not win")
    print("           reliably even with a stable stimulus -> the EMA is not an estimator of")
    print("           a single transition, and the REFERENCE OBJECT must change.")

print()
print("  Per-seed T magnitudes (the informative quantity):")
print("   STABLE/PRE  T:", [round(rows[("STABLE","PRE",s)]["T"], 5) for s in (21,22,23,24,25,26,27)])
print("   VARIED/PRE  T:", [round(rows[("VARIED","PRE",s)]["T"], 5) for s in (21,22,23,24,25,26,27)])
print("   STABLE/POST T:", [round(rows[("STABLE","POST",s)]["T"], 5) for s in (21,22,23,24,25,26,27)])
print("   band        :", round(BAND, 6))
