# HENRI V2 installation and verification

## Scope

This document covers the HENRI V2 source tree. TrustGraph deployment files are not part of the HENRI V2 production release.

## Requirements

- Python 3.12 or newer.
- A local CPU environment for syntax and software tests.
- An approved CUDA host for GPU and CUDA component verification.
- PostgreSQL/TimescaleDB only for Zone C tests or services that explicitly require it.

Do not place credentials, database DSNs, checkpoints, or private host details in Git.

## Local environment

Create an environment outside the repository, then install the declared dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, use the equivalent activation command for the selected shell. The repository tests use flat imports from `HENRI V2`; the root `pyproject.toml` supplies the test paths.

## Local checks

```bash
python -m compileall -q "HENRI V2"
python -m pytest
```

Local CPU execution verifies software behavior only. It does not verify CUDA kernels, remote deployment, benchmark capability, or a live service.

## CUDA checks

Run approved CUDA checks only in a clean detached worktree on the approved remote target. Before execution, record:

- exact commit SHA;
- Python, PyTorch, CUDA, and GPU identity;
- complete worktree status digest;
- checkpoint policy and compatibility result;
- suite commands and return codes.

Production and score-bearing decoder paths require `checkpoint_policy="required"`. If the exact compatible checkpoint is absent, classify the path as `BLOCKED`; do not copy an unverified checkpoint, use `strict=False`, resize weights, or weaken the policy.

## Zone C

Zone C uses environment-only DSNs. Keep production DSNs outside Git and outside chat. Development schema and bootstrap files are under `HENRI V2/migrations/` and the repository's explicit development compose configuration.

## Release verification

A release candidate is not a release until the local gates, remote CUDA gates, repository-professionalism gates, promotion approval, and clean deployment reconciliation all pass.
