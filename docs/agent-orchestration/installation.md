# Install boundary and reproducibility

This overlay is for the existing default-profile HENRI stack. It is not a standalone distribution. All support references and frozen contract files must exist before apply. Do not merge overlay files into a different profile or change a running session's prefix.

## Draw.io upstream

Install with Hermes' native skill installer, not an invented wrapper:

```bash
python -m hermes_cli.main skills install Agents365-ai/drawio-skill/skills/drawio-skill --category creative --yes
```

The observed community scanner gave CAUTION. The user explicitly approved the override for commit `9d9e19a65f94862fd68d2cb825c49d3bb4f5c8a8`; do not infer blanket future approval. The install omitted 23 supporting files. Use `drawio-repair.json` and a checkout at that exact upstream commit to restore only missing files; preserve license and verify hashes. Do not duplicate the entire vendor repository into HENRI GitHub. Do not register an MCP or install the GUI binary unless separately approved. `python` is the native Windows interpreter; `python3` is absent there.

## OpenShell pilot

Published CLI, prover, gateway archives for v0.1.2:

- openshell-x86_64-unknown-linux-musl.tar.gz: `7eb6917285331a09e3300266a0558616481a5e9927cae2612ea07c4045b6dd6f`
- openshell-prover-x86_64-unknown-linux-musl.tar.gz: `e1c9db66ae459850cab43765d7ad569effe2ac0a338c902a1b895d0cc361fbb0`
- openshell-gateway-x86_64-unknown-linux-gnu.tar.gz: `218d887845b3a020ab7535c9985eb9c666d6938f144044957f8b82b42892aadb`

These were checked against official release checksum files. Install into an approved WSL user directory; do not execute a native Windows shell installer. Match gateway/sandbox/prover versions. WSL + Docker Desktop is experimental. Docker WSL integration and host networking are required; never silently disable Enhanced Container Isolation.

The approved local pilot uses an mTLS gateway at `https://127.0.0.1:17670`, user systemd unit `henri-openshell-pilot`, gateway alias `henri-pilot`, and named sandbox `henri-guarded-pilot`. Its transient service is not reboot-persistent. TLS private keys and raw logs stay outside this repository. Do not copy local TLS materials to another host. Generate fresh certificates with the real gateway generate-certs command and run config preflight. User-approved limits: no host secret/data mounts, no attached provider credentials, one local sandbox, no GPU or Vast claim.

`profile-overlay/henri-openshell.json` is a host-specific pilot config. A new host needs an approved, verified gateway/sandbox/boundary hash and executable allowlist; never inherit identifiers as proof of availability. The guard resolves a complete effective policy before the coverage-limited prover. Any nonzero verdict blocks. Concurrency with operator policy mutation remains an explicit limit.

## Hermes plugin

The plugin adds two unique tools and one tail-only pre_llm_call hook. It has no built-in tool override grant. Native `plugins doctor` passed; enabling was read back and a fresh session exercised the real hook. Existing conversations do not hot-load it. Start a new Hermes process before using the tools. Keep normal approvals; a Jev choice cannot grant them. The current parent terminal backend remains local.

## Verify

Run the real SOUL and bundle builders, frozen-contract validator, plugin doctor, API receipt checks, prover positive/negative/error/unsupported controls, real sandbox allowed/denied probes, and exact overlay verifier. Each result proves only its scope. A green document check does not establish HENRI CUDA/model behavior or remote sandbox security.
