import inspect
import arc_agi
env = arc_agi.Arcade().make("ls20")
print(inspect.signature(env.step))
