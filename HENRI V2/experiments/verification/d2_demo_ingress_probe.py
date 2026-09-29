"""Directive 2 evidence-preserving probe: demonstration ingress.

WHY THIS FILE EXISTS
====================
`arc_demo_preflight.py` handles its own logging badly: `arc_agi` writes INFO
lines to stdout, so piping its output straight into a JSON parser fails
(measured 2026-09-28: JSONDecodeError at char 0). This probe captures stdout,
extracts the JSON object, and ALSO records the raw bytes so the result is
auditable rather than a claim.

RESULT RECORDED (measured 2026-09-28, arc_agi 0.9.9)
  envs_probed 16, demos_present 0, blocked_no_demos 16, env_errors 0
  provenance = public_api; detail = "examples=None / demonstrations=None"

That is the honest state. Directive 2's stated verification criterion is
"demo_pair_count > 0 across all evaluation tasks". The public environment API
exposes NO demonstration channel, and the env .py files contain no
examples/demonstrations keys. A demo_pair_count > 0 could only be produced by
FABRICATION, so the criterion is recorded BLOCKED rather than satisfied.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main() -> int:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sys.argv = ["arc_demo_preflight.py"]
        try:
            import runpy
            runpy.run_path(os.path.join(HERE, "arc_demo_preflight.py"),
                           run_name="__main__")
        except SystemExit:
            pass
    raw = buf.getvalue()
    i = raw.find("{")
    if i < 0:
        print("NO_JSON_IN_OUTPUT", file=sys.stderr)
        return 3
    d = json.loads(raw[i:])

    present = [r for r in d["results"] if r["status"] == "DEMOS_PRESENT"]
    blocked = [r for r in d["results"] if r["status"] == "BLOCKED_NO_DEMONSTRATIONS"]
    errors = [r for r in d["results"] if r["status"] == "ENV_ERROR"]
    R = {
        "schema": "henri.d2-demo-ingress-probe.v1",
        "directive": 2,
        "envs_probed": d["envs_probed"],
        "demos_present": len(present),
        "blocked_no_demos": len(blocked),
        "env_errors": len(errors),
        "per_env": [{"env": r["env"], "status": r["status"],
                     "demo_pair_count": r["demo_pair_count"],
                     "provenance": r.get("provenance")} for r in d["results"]],
        "criterion_from_document": "demo_pair_count > 0 across all evaluation tasks",
        "criterion_status": ("SATISFIED" if present else
                             "BLOCKED__NO_PUBLIC_DEMO_CHANNEL"),
        "why_not_fixable_by_editing_the_ingress_schema": (
            "the field the document says to 'update' does not exist on the public "
            "API surface: game.examples and game.demonstrations are both absent, "
            "and the downloaded env .py files contain no examples/demonstrations/"
            "train/test keys at all. There is nothing to load into the in-context "
            "buffer. Writing pairs from elsewhere would fabricate demonstrations."),
    }
    out = os.path.join(HERE, "d2_demo_ingress_probe.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print(json.dumps({k: v for k, v in R.items() if k != "per_env"}, indent=2))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
