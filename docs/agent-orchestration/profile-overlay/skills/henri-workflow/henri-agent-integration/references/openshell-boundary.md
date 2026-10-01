# OpenShell boundary: approved local WSL pilot

Cache first: keep policy/rubric/schema versions static; pass only command and source/receipt deltas. Security/proof/kernel checks are deterministic and outrank model advice. Jev cannot grant permission. No failed-proof schema fallback.

## Live interface

OpenShell v0.1.2 CLI and prover were installed in Ubuntu WSL from official SHA-256 verified release archives. Native Windows is not supported; WSL2 + Docker Desktop is experimental. Docker integration and host networking are prerequisites; Enhanced Container Isolation must not be disabled silently. Keep the active Hermes backend unchanged during the session.

Policy schema: integer version: 1; filesystem_policy.read_only/read_write, landlock.compatibility, process.run_as_user/run_as_group, network_policies, network_middlewares. Gateway TOML starts [openshell] version = 2, [openshell.gateway], [openshell.drivers.docker]. The attached blueprint fields are not this schema.

Prover: openshell-prover check candidate.yaml --boundary boundary.yaml --output json. Only within_boundary/RC0 passes, within reported coverage. RC1 exceeds; RC2 error; RC3 unsupported/inconclusive. Prover does not apply policy or prove runtime enforcement. Unsupported MCP/GraphQL/WebSocket/JSON-RPC policy shapes must block a declared boundary proof, not be skipped.

## Data and enforcement path

The active-profile henri_openshell.py caller obtains the sandbox's complete effective policy, compares it with the operator-owned boundary, checks coverage, then executes only in the named sandbox. It returns real CLI commands, proof result, exit code, stdout/stderr and hashes. Any missing/failed proof or gateway blocks. No unsandboxed fallback. No auto policy expansion or Jev approval.

Local pilot grants no host mounts, no real provider credentials, no production data, and no Docker socket to the workload. Gateway control of Docker is trusted administrative access; it is not workload authority. Require kernel Landlock ABI3+, nested seccomp notification, and task-memory qualification. landlock hard_requirement prevents degraded filesystem enforcement.

## Verification

Checksum/version, config preflight, within-boundary positive, exceeds-boundary negative, invalid-policy rejection, real admitted command, forbidden-file write, denied network, and audit evidence. A gateway status or CLI install alone cannot establish containment. Confirm exact sandbox/effective policy after writes. Preserve raw receipts outside Git; append compact governance after chain verification.

## Limits

Observed local controls: mTLS-connected v0.1.2 gateway; sandbox 2bfca82d-7bd5-4baa-8c19-04ff1dde31fa Ready; Landlock ABI3 with 14 applied and 0 skipped paths; permitted id command; denied /dev/shm write despite mode1777; denied 1.1.1.1:443 with OCSF reason transparent_tcp_policy_denied; actual gate refused a narrower boundary without execution. Prover verdict controls returned RC0/1/2/3 as defined. The gateway is a transient user systemd pilot; do not claim reboot persistence.

No Vast production deployment is authorized by this pilot. Standard Vast containers do not imply a host Docker socket, CDI, kernel privileges, or OpenShell containment. Do not install Docker-in-Docker or claim remote protection from a local boundary. GPU/model tests remain remote CUDA/CI. Sagnac/Kuramoto are not software authorization, external outcomes, or universal success metrics.

Sources: https://docs.nvidia.com/openshell/latest/about/support-matrix ; https://docs.nvidia.com/openshell/latest/how-it-works/policies/prover ; https://docs.nvidia.com/openshell/latest/how-it-works/policies/schema ; https://docs.nvidia.com/openshell/latest/how-it-works/gateways/configuration . Verified release v0.1.2; revalidate before upgrades.
