# STE Framework Source Audit (read-only)

Audited: `asd_ste100_integration_framework_for_project_henri_hermes_orchestration (1).md` (line refs below) against ASD-STE100 Issue 9 (2025-01-15; `asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf`), the STEMG FAQ (`asd-ste100.org/STE_faq.html`), and NVIDIA OpenShell supervisor-middleware docs (`/extensibility/supervisor-middleware/{operations,configure}.md`, `/extensibility/overview.md`). Read-only: no install, config, credential, or remote change. Sandbox = local WSL; Honcho offline; no runs claimed.

## 1. Direct primary support
- Rule 5.1: "Use a maximum of 20 words in each sentence"; "Warnings, cautions, and other safety instructions must also obey this rule."
- Rule 6.3: "Use a maximum of 25 words in each sentence."
- Rule 2.1: "Write multi-word nouns of no more than three words."
- Rule 3.5: use the "-ing" form "only as a technical noun or as a modifier in a technical noun." Rule 3.6: "Use the active voice. In descriptive writing, you can use the passive voice only if the agent is unknown."
- ~900 approved words: FAQ "approximately 900 approved words"; Rule 1.3 (approved meaning).
- Rule 7.1 establishes the WARNING/CAUTION markers.
Caveat: the framework targets "Issue 8"; Issue 9 "fully replaces all other issues."

## 2. Safety-marker conflict
Line 31 gives the primary definitions, but lines 146-147/255-256 override them: WARNING = "system host damage, kernel crash, or data loss"; CAUTION = "thermal throttling, metric distortion, or degraded convergence." Issue 9 Section 7: "A warning tells the reader that there is a risk of injury or death. A caution tells the reader that there is a risk of damage to objects." Rule 7.1: injury/death uses "warning"; damage to machines, tools, or equipment uses "caution." Host damage/data loss is not "injury or death"; thermal throttling is not "damage to objects." Rules 7.2 (command/condition first) and 7.3 (explain the risk) are omitted.

## 3. Lexicon: approximately/about is inverted
Framework lines 93 and 174 say "Do not use 'approximately'. Use 'about'." FAQ: "about is approved only with the meaning concerned with. You cannot use it to mean approximately or around (these words are also approved, with their own specific definitions)." Issue 9 dictionary: APPROXIMATELY (adv), "Almost correct or accurate", STE example "DRAIN APPROXIMATELY 2 LITERS OF FUEL"; paired Non-STE example "Drain about 2 liters of fuel from the tank." The substitution is backwards.

## 4. OpenShell middleware: protocol, auth, match
- Protocol: the service "is a gRPC server that implements the supervisor middleware protocol" (configure.md).
- Operations: HTTP_REQUEST (PRE_CREDENTIALS), HTTP_RESPONSE (PRE_RETURN), WEBSOCKET_MESSAGE (PRE_CREDENTIALS); declared as bindings in the `Describe` manifest.
- Auth: short-lived JWT bearer, `typ`=`openshell-ext+jwt`, `alg`=EdDSA, `aud`=registration audience (default `urn:openshell:extension:middleware:<name>`), `iss`=`openshell-gateway:<gateway_id>`, `caller_kind`; `https://` only (or `tls_ca_cert_path`); negotiation via `openshell.extension.v1.PeerMetadata`.
- Match/attach: gateway TOML `[[openshell.supervisor.middleware]]` (`name`, `grpc_endpoint`, `max_payload_bytes` <=4 MiB, `timeout`, `tls_ca_cert_path`, `audience`, `allow_insecure_transport`); policy `network_middlewares` (`middleware`, `endpoints.include`/`exclude`, `order`, `on_error`, `config`).
- Mismatch: the framework claims an "L7 Regex & Grammar Interceptor" that "intercepts shell calls and syscalls" (lines 59, 251). The documented surface is HTTP bodies and WebSocket text messages only; syscalls, shell calls, and binary messages are not inspected. Reachability: gateway and sandbox supervisors, no Unix sockets. "OpenShell never forwards free-form text from your service to the sandbox or to logs" - results carry only a validated reason code (1-64 bytes, lowercase) and finding labels.

## 5. No regex as compliance, AST proof, or OCSF severity
- Lines 167-181 are pure `re` matching (8 entries + 3 patterns), yet line 153 calls it "a deterministic AST and token validator"; no AST parser exists. Regex cannot apply Rules 1.1-1.3 (approved meaning, one part of speech, technical nouns/verbs) or look up ~900 words.
- FAQ: "no tool can replace the standard itself"; ASD/STEMG "DO NOT endorse or certify" tools "claimed to be 'fully compliant'."
- OCSF: lines 264-268 invent WARNING -> `severity_id=4`/`status_id=2` and CAUTION -> `severity_id=3`/`status_id=1 ("Success with Caveat")`. OCSF `status_id` = 0 Unknown, 1 Success, 2 Failure, 99 Other; there is no "Success with Caveat". ASD-STE100 defines no severity or disposition codes. OpenShell logs middleware as OCSF detection findings with reason codes, not STE markers.
