# OpenShell–HENRI Interface Audit — Evidence Note

**Mode:** read-only. No install, no clone, no GPU provision, no policy execution.
**Revision pins (2026-10-01):** repo main HEAD = `348a1fc62566892ba5530ac9f35acc983e4d9324` (`git ls-remote`); latest stable tag `v0.1.2` = `6648bd0c290efbc41ba131ee9831ee45cd431f94` (= docs `latest`, per `llms.txt`); newest pre `v0.1.3-pre.2` = `021400be…`.
**Sources read:** docs `about/*`, `how-it-works/policies/*`, `how-it-works/sandboxes/{overview,runtimes}`, `how-it-works/gateways/configuration`, `security/best-practices`; repo `AGENTS.md`, `install.sh`; issue #2028.
**Local state (measured):** `openshell`/`osc` absent from PATH. Docker CLI 29.5.2 → daemon `OSType=linux`, name `docker-desktop`; CDI dirs `/etc/cdi`, `/var/run/cdi` exist; `nvidia` runtime present. No OpenShell config/gateway/policy on disk.

## 1. Installation target — verdict: Linux/macOS only; native Windows unsupported

Support matrix: Linux Debian/Ubuntu x86_64 + aarch64 **Supported**; macOS Apple Silicon **Supported**; Windows (WSL 2 + Docker Desktop) x86_64 **Experimental**. Published artifacts have no Windows target: gateway `linux-{x86_64,aarch64}-gnu` + `aarch64-apple-darwin`; sandbox `linux-*-musl`; prover `linux-*-musl` + `aarch64-apple-darwin`.

`install.sh` (1497 lines, main) hard-errors on native Windows: `error "unsupported OS: $(uname -s); this installer supports Linux and macOS"`; its prerelease matrix (lines 485–499) admits only `linux:x86_64|amd64`, `linux:aarch64|arm64`, `darwin:arm64`. Git Bash is **not** an install target. Issue #2028 (`HOME is not set`; no `USERPROFILE` handling) was closed **not planned**; triage kept "native Windows support and validation … out of scope".

**Exact HENRI target:** WSL 2 Ubuntu 24.04 (glibc ≥ 2.28), kernel ≥ 6.2, with Docker Desktop. This Git Bash already drives the *Linux* `docker-desktop` daemon, so a **WSL-side** CLI + gateway is the compatible placement.

## 2. CLI and policy schema — blueprint is invalid against the real schema

Real CLI: `openshell sandbox create|get|connect|exec|start|stop|delete|logs`, `gateway add|select`, `policy get|list|update|set`, `openshell-gateway config preflight --path <toml>`.

Real policy: YAML, `version: 1` (**integer**), ≤4 MiB, ≤256 paths, `read_write` may not contain `/`. Exactly six top-level keys: `version`, `filesystem_policy`, `landlock`, `process`, `network_policies`, `network_middlewares`. **Unknown fields and duplicate keys are rejected.** Filesystem is only `read_only`/`read_write` lists — there is **no `deny:` list**; unlisted paths are already inaccessible.

Blueprint findings (`project_henri_ml_r_d_infrastructure_blueprint_hermes_openshell_and_vast_ai.md`):
- Policy (lines 226–282): `version: "1.0"` (string), top-level `filesystem:`, `network:`, `process: {no_new_privs, seccomp_profile, blocked_syscalls, rlimits}`, `filesystem.deny:`. All undeclared → the file **fails validation** — non-loadable, not merely narrow.
- Gateway TOML (195–222): `[server] listen_addr = "0.0.0.0:50051"`, `[authentication]`, `[storage]`, `[telemetry]`. Real schema is rooted at `[openshell] version = 2`; gateway keys under `[openshell.gateway]` (default `bind_address = 127.0.0.1:17670`); per-driver under `[openshell.drivers.<name>]`. `database_url` is env-only and **rejected in-file**; schema v1 is rejected at startup; unknown/misplaced driver keys **fail startup**.
- `enforce_l7_inspection`/`event_buffer_capacity` do not exist; L7 is per-endpoint (`protocol: rest`, `enforcement: enforce`, `rules`/`deny_rules`).

## 3. Prover — binary exists; blueprint invocation is wrong

`openshell-prover` is real (`crates/openshell-prover-cli`; standalone musl/darwin archives; **not** in the snap; no Windows build). Correct form:
```
openshell-prover check candidate.yaml --boundary boundary.yaml
openshell-prover check cand.yaml --boundary b.yaml --output json --timeout 30
```
Exit codes: `0` within_boundary, `1` exceeds_boundary, `2` error, `3` unsupported/inconclusive. It is a local *containment* check (no gateway, does not apply policy). Limits: >1024 network rules or >4096 endpoints → `inconclusive`; GraphQL/MCP/WebSocket/JSON-RPC and audit-mode REST → `unsupported`. Z3 is a genuine dependency (`.cargo` z3-sys bindgen fix), so "Z3 SMT" is accurate.

Blueprint lines 452–465 run **`openshell-prover verify --boundary … --contract …`**: no `verify` subcommand, no `--contract` flag. Worse, `except (CalledProcessError, FileNotFoundError)` treats a real exit-1 *violation* as "prover not detected" and returns `True` from a structural key check — **fail-open on a policy breach**, the opposite of OpenShell's contract (only `within_boundary` passes).

## 4. GPU and Vast nested-container limits

- `--gpu` / `--gpu N`: Docker/Podman → NVIDIA **CDI** devices (pin via `driver-config-json {"docker":{"cdi_devices":["nvidia.com/gpu=0"]}}`); Kubernetes → `nvidia.com/gpu` limit; microVM → **exactly one** GPU (`gpu_device_ids`). `--driver-config-json` is **disabled by default** (`allow_driver_config = true`). CDI must be configured **before** gateway start; on WSL2-only runtimes the default may fall back to `nvidia.com/gpu=all`, counted as **one** device. GPU sandboxes gain read-write `/dev/nvidia*`, `/dev/dxg`, `/proc`.
- Kernel boundary required **"including when it runs inside a container or microVM"**: Landlock ABI v3+ (Linux 6.2), nested seccomp user-notification with `SECCOMP_IOCTL_NOTIF_ADDFD` usable "under the runtime's existing seccomp profile", and same-UID `process_vm_readv`/`/proc/<pid>/mem`. All are probed before admission and **fail closed**. Kernels <5.19 (RHEL 9.x/RHCOS) run **legacy read-only**, returning `EOPNOTSUPP` for `getpeername`/addressed `accept`/`sendmmsg`.
- **Vast.ai nesting (inference, flagged):** no documented pattern runs the gateway inside a third-party container. The Docker driver needs a Docker/Podman socket on the gateway host; Docker Desktop must allow host networking and cannot use Enhanced Container Isolation. Gateway-in-Vast therefore implies a nested/privileged Docker daemon plus passing Landlock/seccomp/`/proc` probes inside Vast's runtime profile — unsupported and unqualified. Treat Vast as: workload GPU via Docker, OpenShell policy **not** covering it; OpenShell does not provision or bill remote GPUs.

## 5. Fail-closed deny controls (verified defaults)

Network: deny-by-default egress; match on host+port+binary. Loopback, link-local, unspecified and `169.254.169.254` never authorized; K8s/etcd ports 2379/2380/6443/10250/10255 blocked. Filesystem: Landlock; `landlock.compatibility: hard_requirement` = sandbox **fails to start**; `/.openshell` baseline mandatory even under `best_effort`. Process: `run_as_user/group` must be non-root (root rejected); seccomp + privilege drop. Policy: `policy_validation_failure_mode = "fail_closed"` (default). Middleware: `on_error` defaults to `fail_closed`. Binary identity: SHA256 trust-on-first-use; digest mismatch **denies immediately**, missing evidence fails closed. Prover: any non-zero exit = failure.

## 6. Acceptance and kill tests

Acceptance (WSL/Linux): `openshell --version`; `openshell gateway add http://127.0.0.1:17670 --local --name local`; `openshell-gateway config preflight --path ~/.config/openshell/gateway.toml`; `openshell sandbox create --name t --policy candidate.yaml -- curl -s https://api.vast.ai` must be **denied**; `openshell-prover check candidate.yaml --boundary boundary.yaml; echo $?` → **0 only**.
Kill tests: (K1) `install.sh` under Git Bash → `unsupported OS`. (K2) blueprint policy load → schema rejection on `version: "1.0"`/`filesystem`/`deny`. (K3) `openshell-prover verify …` → unknown subcommand. (K4) candidate writing `/tmp` vs `/usr`-only boundary → exit **1**. (K5) unallowlisted egress → deny + OCSF finding. (K6) kernel <6.2 or blocked nested seccomp → sandbox fails to start, not degrade-open.

**Verdict:** OpenShell is a real Linux/macOS product with a documented CLI, a `version: 1` policy schema, and a real `openshell-prover`. The HENRI blueprint's integration is **not runnable**: wrong install target, two invalid schemas, a non-existent prover subcommand, and a fail-open verification fallback. Confirmed main-graph gaps: no production caller of `openshell-prover`; no gateway/sandbox lifecycle wiring.

**Unresolved (host-dependent, not doc-answerable):** whether this exact Windows/`docker-desktop` VM satisfies the nested-user-notif seccomp and `/proc/<pid>/mem` probes; whether a Vast container exposes CDI.
