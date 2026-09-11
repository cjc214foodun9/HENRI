import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

def frame_arr(r):
    try:
        return np.array(r.frame[0].tolist())
    except Exception:
        return None

g = Arcade().make("lf52")
r = g.reset()
print("RESET state:", getattr(r, "state", None))
print("RESET available_actions:", [a.name for a in g.action_space])
f0 = frame_arr(r)
print("grid shape:", None if f0 is None else f0.shape)

a = g.step(GameAction.ACTION1)
f1 = frame_arr(a)
print("ACTION1(no-data) state:", getattr(a, "state", None))
print("ACTION1(no-data) available:", [x.name for x in g.action_space])
if f0 is not None and f1 is not None:
    print("frame pixels changed after ACTION1:", int(np.sum(f0 != f1)), "/", f0.size)

a2 = g.step(GameAction.ACTION2)
f2 = frame_arr(a2)
if f1 is not None and f2 is not None:
    print("frame pixels changed after ACTION2:", int(np.sum(f1 != f2)), "/", f1.size)

try:
    a3 = g.step(GameAction.ACTION6, data={"x": 0, "y": 0})
    print("ACTION6(data={x:0,y:0}) state:", getattr(a3, "state", None))
    f3 = frame_arr(a3)
    if f2 is not None and f3 is not None:
        print("frame pixels changed after ACTION6+data:", int(np.sum(f2 != f3)), "/", f2.size)
except Exception as e:
    print("ACTION6 data err:", type(e).__name__, str(e)[:120])

try:
    a4 = g.step(GameAction.ACTION6)
    print("ACTION6(no-data) state:", getattr(a4, "state", None))
except Exception as e:
    print("ACTION6 no-data err:", type(e).__name__, str(e)[:120])
