"""
Compare transition-loss trajectories: run 6 (r=64, phase-locked) vs the
r=128 rerun. Falsification target of the spectral-leakage hypothesis:
  - If rank-64 truncation was the divergence driver, r=128 should show
    descending/stable transition loss in bp35 and cd82 (run 6 diverged),
    and preserve ar25's descent.
  - If loss still diverges at r=128, rank capacity is falsified as the
    sole bottleneck and the boundary-axiom / binding-algebra hypotheses
    move to the front.
"""
import json
import sys
import collections


def load_series(path):
    out = collections.defaultdict(list)
    with open(path) as fp:
        for line in fp:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("transition_loss") is not None and "env" in r and "step" in r:
                out[r["env"]].append((r["step"], r["transition_loss"]))
            # L2 EDMD batch losses
            if r.get("edmd_L2_loss") is not None:
                out[r["env"] + " ::edmd_L2"].append((r["step"], r["edmd_L2_loss"]))
    for k in out:
        out[k].sort()
    return out


def summarize(series):
    n = len(series)
    if n == 0:
        return "n=0"
    vals = [v for _, v in series]
    first = vals[:3]
    last = vals[-3:]
    # linear slope (least squares) over the whole series
    xs = [s for s, _ in series]
    xm = sum(xs) / n
    ym = sum(vals) / n
    denom = sum((x - xm) ** 2 for x in xs) or 1.0
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, vals)) / denom
    return (f"n={n} first3={[round(v,3) for v in first]} "
            f"last3={[round(v,3) for v in last]} "
            f"mean={ym:.3f} slope={slope:+.4f}/step")


def main():
    run6_path, r128_path = sys.argv[1], sys.argv[2]
    run6 = load_series(run6_path)
    r128 = load_series(r128_path)
    envs = sorted({e.split(" ::")[0] for e in list(run6) + list(r128)})
    print(f"{'env':<22} {'run':<10} summary")
    print("-" * 100)
    for env in envs:
        s6 = run6.get(env, [])
        s7 = r128.get(env, [])
        print(f"{env:<22} {'r64-run6':<10} {summarize(s6)}")
        print(f"{'':<22} {'r128':<10} {summarize(s7)}")
        if s6 and s7:
            d6 = sum(v for _, v in s6[-5:]) / min(5, len(s6))
            d7 = sum(v for _, v in s7[-5:]) / min(5, len(s7))
            verdict = "IMPROVED" if d7 < d6 - 0.05 else ("WORSE" if d7 > d6 + 0.05 else "UNCHANGED")
            print(f"{'':<22} {'verdict':<10} tail-mean r64={d6:.3f} vs r128={d7:.3f} -> {verdict}")
        for tag in (" ::edmd_L2",):
            s6e, s7e = run6.get(env + tag, []), r128.get(env + tag, [])
            if s6e or s7e:
                print(f"{'':<22} {'L2 r64':<10} {summarize(s6e)}")
                print(f"{'':<22} {'L2 r128':<10} {summarize(s7e)}")
        print()


if __name__ == "__main__":
    main()
