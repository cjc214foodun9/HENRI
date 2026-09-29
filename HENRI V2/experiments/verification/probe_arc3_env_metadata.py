"""Probe: does ARC-AGI-3 expose a LEGITIMATE env->source-task provenance link?

Reads PUBLIC metadata only (env id / title / tags). Never reads cached
environment_files task grids (FORBIDDEN_LEAKAGE_SOURCE).
"""
import json
import sys


def main() -> int:
    import arc_agi
    arcade = arc_agi.Arcade()
    out = {"schema": "probe.env-metadata.v1", "envs": []}
    try:
        envs = arcade.get_environments()
    except Exception as exc:
        print(json.dumps({"error": f"get_environments: {type(exc).__name__}: {exc}"}))
        return 2
    print("n_environments =", len(envs))
    for info in envs[:4]:
        fields = {}
        for attr in dir(info):
            if attr.startswith("_"):
                continue
            try:
                v = getattr(info, attr)
            except Exception:
                continue
            if callable(v):
                continue
            if isinstance(v, (str, int, float, bool, list, tuple, dict, type(None))):
                fields[attr] = v if not isinstance(v, str) or len(v) < 400 else v[:400] + "..."
        out["envs"].append(fields)
    print(json.dumps(out, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
