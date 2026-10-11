
import sys, math, torch
sys.path.insert(0, "HENRI V2/henri_core")
from henri_heterosynaptic import DualStreamNetwork
DIMS=(32,64,64,32); G,EW,WU=0.02,0.002,40
net=DualStreamNetwork(DIMS,gamma=G,eta_w=EW,seed=0)
g=torch.Generator().manual_seed(0)
xs=[torch.randn(32,generator=g) for _ in range(4)]; tg=[torch.randn(32,generator=g) for _ in range(4)]
for t in range(WU):
    i=t%4
    with torch.no_grad(): net.sal_step(xs[i],2.0*(net.activations(xs[i])[-1]-tg[i]))
print(f"{'l':>2} {'|V.dT|':>10} {'|dV.T|':>10} {'resid':>10} {'resid-V.dT':>12}")
for l in range(net.n):
    with torch.no_grad():
        h0=net.activations(xs[0]); ab0=net.feedback(h0,2.0*(h0[-1]-tg[0]))
        Du=(h0[l]>0).to(net.dtype)
        T0=Du[:,None]*net.W[l].t(); W0=net.W[l].clone(); V0=net.V[l].clone()
        F0=V0@T0
        net.sal_step(xs[0],2.0*(h0[-1]-tg[0]),retract=False)
        h1=net.activations(xs[0]); ab1=net.feedback(h1,2.0*(h1[-1]-tg[0]))
        Du1=(h1[l]>0).to(net.dtype); T1=Du1[:,None]*net.W[l].t()
        F1=net.V[l]@T1
        dVT=(net.V[l]-V0)@T0; VdT=V0@(T1-T0)
        resid=(F1-F0)-dVT
    print(f"{l:>2} {float(VdT.norm()):>10.4f} {float(dVT.norm()):>10.4f} {float(resid.norm()):>10.4f} {float((resid-VdT).norm()):>12.4e}")
print("If |resid - V.dT| ~ 0, the omitted V@dT term fully explains the C1 residual.")
