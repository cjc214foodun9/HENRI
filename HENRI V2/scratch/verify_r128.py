"""
Ad-hoc verification of r=128 field channel: EDMD identifiability + manifold.

Per architecture-skill caveats:
- Synthetic ground truth must be LOW-RANK (rank < r): a full-rank random
  target is structurally unrepresentable by the rank-r field channel.
- Target amplitude must be O(1)+ (per-block renorm flattens sub-threshold
  field components).
- Post-fit batch loss should descend far below the pre-fit level.
- Manifold constraints: field_V semi-unitary? NO — after EDMD, field_V
  holds right singular vectors (orthonormal) and field_W holds X^T U S
  (amplitudes). The QR retraction on field_V is skipped in residual refit
  (residual_only=True) precisely to preserve solved amplitudes.
"""
import torch
from efe_planner import EFEPlanner, UnitaryWaveTransition

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
NB, R = 64, 128          # reduced blocks, production rank
D = NB * 8

planner = EFEPlanner(num_blocks=NB, d_model=D).to(device)
t = planner.transition
assert t.rank == 128, f"rank={t.rank}"

def mk_wave(seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(NB, 8, generator=g)
    return (w / torch.norm(w, p=2, dim=-1, keepdim=True)).to(device)

# Build a LOW-RANK synthetic ground truth operator (true rank 32 < 128):
# next = Vt (Wt^T fused) + local(identity-ish), O(1) amplitude.
Vt = torch.randn(D, 32, device=device)
Vt, _ = torch.linalg.qr(Vt, mode="reduced")
Wt = torch.randn(2 * D, 32, device=device)

N = 40  # transitions; N < rank exercises the k=min(rank,N) path
states = torch.stack([mk_wave(100 + i) for i in range(N)])
actions = torch.stack([mk_wave(200 + i) for i in range(N)])
nexts = []
for i in range(N):
    fused = t.bind(states[i], actions[i])               # [NB,8] complex
    fflat = torch.cat([fused.real.reshape(-1), fused.imag.reshape(-1)])
    field = (Vt @ (Wt.T @ fflat)).view(NB, 8)
    nxt = fused.real + 0.5 * field                      # O(1) amplitude
    nxt = nxt / (torch.norm(nxt, p=2, dim=-1, keepdim=True) + 1e-9)
    nexts.append(nxt)
observed = torch.stack(nexts)

pre = planner._batch_sagnac_loss(states, actions, observed).item()
post = planner.train_transition_batch(states, actions, observed)
# recompute post-fit loss
post_fit = planner._batch_sagnac_loss(states, actions, observed).item()

# Cross-block coupling check (block-diagonal regression guard)
s1, a = mk_wave(40), mk_wave(41)
s2 = s1.clone(); s2[0] = torch.randn(8, device=device); s2[0] /= s2[0].norm()
o1, o2 = t(s1, a), t(s2, a)
xblock = (o1[5] - o2[5]).abs().max().item()

# field_V column orthonormality (right singular vectors from thin SVD)
gram = t.field_V.T @ t.field_V
gram_err = (gram - torch.eye(t.rank, device=device)).abs().max().item()

# Per-block residual unitarity
prod = torch.matmul(t.block_residual, t.block_residual.mH)
unit_err = (prod - torch.eye(8, dtype=torch.complex64, device=device)).abs().max().item()

print(f"device={device}  NB={NB}  rank={t.rank}  N={N}")
print(f"pre-fit batch Sagnac loss : {pre:.4f}")
print(f"train_transition_batch ret: {post:.4f}  (pre-fit mean, by API)")
print(f"post-fit batch Sagnac loss: {post_fit:.4f}")
print(f"cross-block coupling      : {xblock:.2e}  (>1e-6 required)")
print(f"field_V orthonorm err     : {gram_err:.2e}")
print(f"residual unitarity err    : {unit_err:.2e}")

ok = (post_fit < 0.5 * pre) and xblock > 1e-6 and unit_err < 1e-3
print("VERDICT:", "PASS" if ok else "FAIL")
