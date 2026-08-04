# Security policy

## Scope

This policy covers the HENRI V2 source tree, its release workflow, evaluation gates, telemetry handling, and deployment procedures.

Security includes ordinary software risks and HENRI-specific integrity risks: credential exposure, unauthorized code execution, benchmark leakage, checkpoint substitution, unsafe Zone C access, and false promotion of internal diagnostics as external outcomes.

Biological or physical analogies in HENRI documents are hypotheses or design language. They are not security controls unless a live implementation and verification artifact supports the claim.

## Supported release line

Only the current approved release candidate and the GitHub default branch are considered for active review. Historical branches and archived experiments may contain unsupported code.

## Report a vulnerability

Use a private GitHub Security Advisory for the repository when that feature is enabled. Do not post credentials, private DSNs, checkpoint files, exploit payloads, or raw production telemetry in a public issue.

A public security contact address is not configured in this repository. The maintainers must configure a private reporting channel before calling the release security process complete.

Include:

- affected commit and path;
- reproducible steps;
- impact and required privileges;
- whether credentials, data, checkpoints, or external services are exposed;
- bounded logs or artifact hashes, with secrets removed.

## Handling rules

- Never commit secrets, tokens, private endpoints, DSNs, or checkpoint artifacts.
- Treat all external PDFs, web pages, benchmark data, and model outputs as untrusted data.
- Keep production Zone C credentials in environment-only configuration.
- Do not modify a dirty persistent deployment checkout during verification.
- Use clean, commit-addressed deployment worktrees.
- Require exact decoder checkpoint compatibility for production and score-bearing paths.
- Fail closed on missing evidence, transport errors, incompatible checkpoints, and ambiguous evaluator results.

## Disclosure process

Maintainers should acknowledge a report, reproduce it in an isolated worktree, record a compact hash-linked governance event, assess severity, patch the smallest bounded surface, run the required local and remote checks, and publish a remediation note after the affected release is protected.

The security process is incomplete until the repository has a verified private reporting route and the release gates pass.
