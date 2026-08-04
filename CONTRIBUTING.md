# Contributing to HENRI V2

## Before you change code

1. Read the relevant architecture contract and the live caller and consumer.
2. Record the current branch, commit, and both Git status boundaries.
3. Keep unrelated dirty files out of the change. Use an external clean worktree for release work.
4. State the mechanism, assumptions, resource limits, failure mode, cheapest kill experiment, and acceptance criteria before load-bearing changes.

## Change boundaries

- Make one bounded change at a time.
- Preserve the default path unless the experiment explicitly changes it.
- Put experimental behavior behind a named flag and default it to off.
- Do not delete deprecated HENRI code. Archive it under `HENRI V2/_archive/` unless deletion is explicitly approved.
- Keep flat imports until an approved package migration is complete.
- Do not commit credentials, checkpoints, generated telemetry, local databases, or benchmark task data.

## Evidence rules

A test name, function name, log line, or hash does not prove the claimed mechanism. Map material claims to live symbols, execution commands, and artifacts. Separate:

- software/code-health tests;
- mathematical or CUDA component checks;
- external task outcomes;
- design targets and hypotheses.

A passing component test cannot substitute for a failed release suite. A successful process exit cannot substitute for an external task result.

## Required checks

For source changes:

```bash
python -m compileall -q "HENRI V2"
python -m pytest
```

For release candidates, also run:

```bash
git diff --check
git status --porcelain=v1 -uall
```

The complete status must contain zero lines before a candidate worktree is removed or promoted.

CUDA checks run on the approved remote target or canonical CI. Do not report local CPU checks as CUDA verification.

## Checkpoints and evaluation

Decoder checkpoints are architecture-bound. Validate file hash, state-dict hash, metadata, and exact tensor shapes before `load_state_dict(strict=True)`.

- `required`: production inference, API serving, and score-bearing evaluation.
- `disabled`: reduced tests and intentional untrained ablations.
- `auto`: development convenience only.

Missing or incompatible required checkpoints produce a typed compatibility error and a `BLOCKED` result. Never use `strict=False` to hide a mismatch.

## Pull requests

A pull request should include:

- purpose and scope;
- changed paths;
- caller and consumer trace;
- tensor shapes and device boundaries when applicable;
- test commands and real return codes;
- artifact paths and SHA-256 values for material evidence;
- known failures and uncertainty;
- rollback or revert procedure.

Do not include raw logs in the pull request. Store them externally and provide compact hashes and bounded summaries.

## Review standard

Reviewers must reject circular validation, simulated outcomes presented as live results, causal leakage, unsupported physical equivalence, dead configuration, duplicate wrappers, and benchmark graders that reproduce the expected answer.
