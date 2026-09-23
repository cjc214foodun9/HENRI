"""FRESH-PROCESS verification of the channel-semantics fix (None-safe).

WHY FRESH: the execute_code kernel caches imported modules, so an in-kernel
`import uhr02_exteroceptive_gate` after a patch tests the OLD code. This script
prints the module source hash as provenance.

DEFECT FIXED IN THIS SCRIPT (mine): the first version indexed
`recorded_transition_generators(..., channel=0)[0]` unconditionally. Channel 0 on
this fixture is BELOW the 1e-5 floor, so the function correctly returns `None`
and the script died with `TypeError: 'NoneType' object is not subscriptable` —
a script bug that briefly looked like a code failure. Every call is now
None-checked and the None case is PRINTED as the calibrated behaviour it is.

DECISIVE CHECK: before the fix the read path returned channel 0 (one grid cell),
norm ~4e-08, and `relative_group_element` was 3.0 (= |Tr(I)|, both operands
identity). After the fix the read path must return the argmax-norm channel.
"""
import sys, hashlib, numpy as np, torch, pathlib

SRC = r"C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
sys.path.insert(0, SRC)

src = pathlib.Path(SRC, "uhr02_exteroceptive_gate.py").read_bytes()
print("module sha256 =", hashlib.sha256(src).hexdigest()[:32])
print("has transition_channel:", b"def transition_channel" in src)
print("has channel param      :", b"channel: int | None = None" in src)

from chromodynamic_grounding import encode_su3_color_field, GELL_MANN_BASIS
from henri_external_outcome_refactor_module import ActionOutcomeGeneratorStore
import uhr02_exteroceptive_gate as G

NB = 8192
basis = GELL_MANN_BASIS.to(torch.complex64)
f2 = lambda X: float(torch.linalg.matrix_norm(X))


def pad(f, nb=NB):
    k = f.shape[0]
    return f[:nb] if k >= nb else torch.cat(
        [f, torch.eye(3, dtype=f.dtype).unsqueeze(0).repeat(nb - k, 1, 1)], 0)


def enc(g):
    return pad(encode_su3_color_field(
        torch.tensor(g, dtype=torch.int64).unsqueeze(0)).reshape(-1, 3, 3))


# Grid whose cell (0,0) NEVER moves but cells elsewhere DO -- reproduces the live
# ft09 condition in which channel 0 (grid cell (0,0)) was dead.
store = ActionOutcomeGeneratorStore(num_actions=16, num_channels=NB, lr=0.15)
prev = enc(np.zeros((16, 16), dtype=np.int64))
for t in range(20):
    g = np.zeros((16, 16), dtype=np.int64)
    g[10, 3] = t % 5
    g[10, 4] = t % 5
    g[12, 9] = (t + 1) % 5
    g[1, 1] = t % 5
    nxt = enc(g)
    store.update_generator(prev, 2, nxt, basis)
    prev = nxt

th = store.theta_a[2]
norms = th.norm(dim=-1)
print("\n=== STORE STATE (action 2) ===")
print(f"  ||theta_a[2]|| aggregate    = {float(th.norm()):.6f}")
print(f"  ||theta_a[2][0]|| channel 0 = {float(th[0].norm()):.4e}   <- OLD read path")
print(f"  argmax channel              = {int(norms.argmax())}  norm={float(norms.max()):.6f}")
print(f"  channels > 1e-5             = {int((norms > 1e-5).sum())}/{NB}")

c = G.transition_channel(store, 2)
print("\n=== transition_channel (NEW helper) ===")
print(f"  transition_channel(store,2) = {c}")

gen_new = G.recorded_transition_generators(store, 2, basis)
gen_ch0 = G.recorded_transition_generators(store, 2, basis, channel=0)
print("\n=== recorded_transition_generators ===")
print(f"  fixed  (argmax ch {c}): {'None (below floor)' if gen_new is None else f'||H||={f2(gen_new[0]):.6e}'}")
print(f"  legacy (channel 0)    : {'None (below floor) <- CORRECT' if gen_ch0 is None else f'||H||={f2(gen_ch0[0]):.6e}'}")
print(f"  identity control      : ||H||={f2(torch.zeros(3,3,dtype=torch.complex64)):.6e}")

print("\n=== DOES |Tr(Uc^dag Ut)| LEAVE 3.0? (3.0 == both operands identity) ===")
if gen_new is not None:
    ident = [torch.zeros(3, 3, dtype=torch.complex64)]
    print(f"  |Tr(argmax, argmax)| = {G.relative_group_element(gen_new, gen_new):.6f}")
    print(f"  |Tr(identity, argmax)| = {G.relative_group_element(ident, gen_new):.6f}   <- must be < 3")
    other = [gen_new[0] * 1.0]
else:
    print("  CANNOT EVALUATE: argmax channel below floor on this fixture")
    other = None

ok = (c is not None and c != 0 and gen_new is not None and f2(gen_new[0]) > 1e-5)
print("\n=== VERDICT ===")
print(f"  channel fix effective: {ok}")
print(f"  (channel={c} (not 0); ||H||={None if gen_new is None else f'{f2(gen_new[0]):.3e}'} > 1e-5)")
