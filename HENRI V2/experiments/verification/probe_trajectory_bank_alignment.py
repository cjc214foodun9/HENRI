"""Deterministic trajectory-bank alignment probe (Carrier P1, step 1).

Verifies that rows in the live trajectory-bank npz map 1:1 to lines in its
companion jsonl under the schema consumed by the G-series gauntlets
(arc_f15_trajectory_engine.resolve_trajectory_goal, arc_g4_aligned_engine):

  npz keys:  psi [N, D], next_wave [N, D], actions_onehot [N, A] (uint8,
             one-hot), action_names [A] (str, e.g. ACTION1..ACTION7)
  jsonl:     one JSON record per line {env, step, action_name, t}, N lines,
             per-env rows contiguous.

Assertions:
  1. row_count_equal  : npz row count == jsonl line count
  2. onehot_valid     : every actions_onehot row sums to 1
  3. action_aligned   : for every row i, npz action name (from onehot argmax)
                        == jsonl[i].action_name
  4. env_contiguous   : each env's jsonl rows form one contiguous index block
  5. terminal_ok      : for each env, its last jsonl row index is in range,
                        belongs to that env, and psi[row] is finite (this row
                        is the engine's goal source)

Emits a JSON receipt (default: <jsonl>.alignment_receipt.json). Exit 0 only
when every assertion passes; exit 1 on misalignment; exit 2 on schema error.
"""

import argparse
import json
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, help="path to bank npz")
    ap.add_argument("--jsonl", required=True, help="path to bank jsonl")
    ap.add_argument("--out", default=None, help="receipt output path")
    args = ap.parse_args()

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"verdict": "NUMPY_UNAVAILABLE", "error": repr(exc)}))
        return 2

    npz_path = pathlib.Path(args.npz)
    jsonl_path = pathlib.Path(args.jsonl)

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        print(json.dumps({"verdict": "NPZ_LOAD_FAILED", "error": repr(exc)}))
        return 2

    for key in ("psi", "actions_onehot", "action_names"):
        if key not in data.files:
            print(json.dumps({"verdict": "SCHEMA_MISSING_KEY", "key": key}))
            return 2

    psi = np.asarray(data["psi"])
    oh = np.asarray(data["actions_onehot"])
    names = [str(x) for x in np.asarray(data["action_names"])]
    n_npz = int(psi.shape[0])

    if psi.ndim != 2:
        print(json.dumps({"verdict": "SCHEMA_PSI_RANK", "ndim": int(psi.ndim)}))
        return 2
    if oh.shape[0] != n_npz:
        print(json.dumps({"verdict": "SCHEMA_ROW_MISMATCH",
                          "psi_rows": n_npz, "onehot_rows": int(oh.shape[0])}))
        return 2

    recs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                recs.append(None)
    n_json = len(recs)

    errors = []
    action_mismatch = []

    if n_json != n_npz:
        errors.append({"kind": "ROW_COUNT", "npz": n_npz, "jsonl": n_json})

    sums = oh.sum(axis=1)
    bad_oh = int((sums != 1).sum())
    if bad_oh:
        errors.append({"kind": "ONEHOT_INVALID", "count": bad_oh})

    valid_rows = min(n_npz, n_json)
    for i in range(valid_rows):
        rec = recs[i]
        if rec is None:
            continue
        name_json = rec.get("action_name")
        if name_json is None:
            continue
        a = int(oh[i].argmax()) if sums[i] == 1 else -1
        name_npz = names[a] if 0 <= a < len(names) else None
        if name_npz != name_json:
            action_mismatch.append({"row": i, "npz": name_npz, "jsonl": name_json})

    env_blocks = {}
    order = []
    for i, rec in enumerate(recs):
        if rec is None:
            continue
        env = rec.get("env")
        if env is None:
            continue
        if env not in env_blocks:
            env_blocks[env] = [i, i]
            order.append(env)
        else:
            b = env_blocks[env]
            if i != b[1] + 1:
                errors.append({"kind": "ENV_NON_CONTIGUOUS", "env": env,
                               "expected_next": b[1] + 1, "actual": i})
            b[1] = i

    terminals = {}
    for env in order:
        first, last = env_blocks[env]
        terminals[env] = last
        if last >= n_npz:
            errors.append({"kind": "TERMINAL_OOR", "env": env, "terminal": last})
            continue
        if not bool(np.isfinite(psi[last]).all()):
            errors.append({"kind": "TERMINAL_NONFINITE", "env": env,
                           "terminal": last})
        rec = recs[last]
        if rec is None or rec.get("env") != env:
            errors.append({"kind": "TERMINAL_ENV_MISMATCH", "env": env,
                           "terminal": last})

    aligned = (
        len(action_mismatch) == 0
        and len(errors) == 0
        and n_json == n_npz
        and bad_oh == 0
    )

    receipt = {
        "schema": "henri.arc-trajectory-bank.v1",
        "probe": "probe_trajectory_bank_alignment",
        "npz": str(npz_path),
        "jsonl": str(jsonl_path),
        "npz_rows": n_npz,
        "jsonl_lines": n_json,
        "env_count": len(env_blocks),
        "env_order": order,
        "terminal_indices": terminals,
        "misalignments_count": len(action_mismatch),
        "misalignments": action_mismatch[:20],
        "terminal_discrepancies_count": len(errors),
        "terminal_discrepancies": errors[:20],
        "aligned": bool(aligned),
    }

    out_path = args.out or (str(jsonl_path) + ".alignment_receipt.json")
    pathlib.Path(out_path).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0 if aligned else 1


if __name__ == "__main__":
    sys.exit(main())
