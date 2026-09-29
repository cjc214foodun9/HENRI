# ARC-AGI-3 demonstration provenance — BLOCKED (measured)

**Date:** 2026-09-28 · **Branch:** `carrier/zone-a-selfplay` @ `85c834b`
**Status:** `BLOCKED` — do not build a demonstration manifest.
**Probe:** `experiments/verification/probe_arc3_env_metadata.py` (public API only)

## 1. The binding constraint for the 0% external score

`experiments/verification/arc_demo_preflight.py` returns
`BLOCKED_NO_DEMONSTRATIONS` for **every** environment it probes:

```
status: BLOCKED_NO_DEMONSTRATIONS
demo_pair_count: 0
provenance: public_api
detail: examples=None / demonstrations=None on public API
```

Independently confirmed on all 17 environments present locally:
`ENVS_WITH_DEMOS 0 of 17`. This is the honest reason external scores are 0.0 —
it is an **ingress** constraint, not a decoder defect. No decoder change can
score a task whose demonstrations never arrive.

## 2. The environments ARE reachable — earlier claims were wrong

Two claims made earlier in this session are `FALSIFIED` by probe:

| Claim | Probe result |
|---|---|
| "`environment_files/` is empty" | **FALSE** — populated, 17 game dirs / 40 files, created 2026-09-28 18:02. Grew further during probing (e.g. `wa30/ee6fef47/`). |
| "the ARC-AGI-3 API is unreachable" | **FALSE** — "Successfully fetched 25 environment(s) from API" on every preflight run. |

So the boundary is narrow and precise: the API works, the games load, and the
games expose **no demonstrations**.

## 3. Provenance check — why an ARC-1 -> ARC-3 manifest would be a category error

The local corpus at `C:/Users/chan/henri_data/ARC-AGI/data` is **ARC-AGI-1**:
400 `training/` + 400 `evaluation/` JSON files, static input/output grid pairs.

`references/henri_phase10_3_staticity_partition.md:241` records how ARC-1 is
consumed internally: "on ARC-AGI-1 tasks (test[0] held out, first 3 train pairs
as demos)". ARC-1 demos are therefore *static grid pairs*.

The ARC-AGI-3 public environment metadata exposes these fields only:

```
game_id, title, default_fps, tags, private_tags, level_tags,
baseline_actions, date_downloaded, class_name, local_dir
```

Measured example (`wa30-ee6fef47`):
`baseline_actions = [71, 119, 183, 98, 368, 68, 79, 442, 415]`.

Two observations decide the question:

1. **There is NO field that maps an ARC-AGI-3 environment to an ARC-AGI-1 task
   id.** No `source_task`, no `task_ref`, no parent link. The provenance link
   required to justify a mapping does not exist in the public schema.
2. `baseline_actions` is a per-level action BUDGET consumed by the RHAE scoring
   formula — it is a **scoring input**, not a demonstration source.

ARC-AGI-3 is interactive (Arcade / OperationMode / ACTION6 coordinates); ARC-AGI-1
is static grid pairs. There is no measured evidence of an adapter, and the
metadata carries no key to build one. **Constructing a manifest by pairing the two
would fabricate a provenance link and contaminate any future score.** Decision:
`BLOCKED`. No manifest is built.

## 4. What would unblock this (pre-registered)

Any ONE of these, each independently verifiable:

1. An official ARC-AGI-3 demonstration channel (a public `examples` /
   `demonstrations` attribute that is non-empty, or a documented download).
2. A documented, citable ARC-AGI-1 -> ARC-AGI-3 provenance mapping from the
   benchmark authors.
3. A published ARC-AGI-3 interaction trace set (state, action, next_state) usable
   as demonstration pairs.

Until one exists, the correct output is this record. Per the operating standard:
a negative is a governance win, not a loop iteration.

## 5. Evidence class

- `OBSERVED`: preflight status per env; 0-of-17 demo count; 25 envs fetched;
  `environment_files` contents; ARC-1 corpus file counts; the metadata field list
  and the `baseline_actions` value.
- `FALSIFIED`: "environment_files is empty"; "no ingress manifest exists
  anywhere" (the *manifest* is absent, but the *environments* are present, which
  is a different and weaker claim); the earlier reading that the API was
  unreachable.
- `BLOCKED`: remote CUDA verification. `torch.cuda.is_available()` is False on
  this host; Vast is exited with negative balance.

## 6. Next falsification

Re-run `arc_demo_preflight.py` after any `arc_agi` package update or any change
to the three.arcprize.org environment set. The gate is a single command and the
verdict is binary. If any env ever reports a non-zero `demo_pair_count`, the
whole blocker is lifted and Path B (continuous egress) becomes testable.
