# Jev System 1: current OpenRouter integration

Cache first: batch independent typed questions with the same state; pin model/rubric and exact source hashes. Local exact-request memoization is application caching, not provider KV caching. Do not expect an OpenRouter prompt-cache discount for Jev; the observed provider does not advertise implicit caching.

## Current interface

OBSERVED 2026-10-01: typesafe/jev-1.13 works on POST https://openrouter.ai/api/alpha/decisions. The response identifies typesafe/jev-1.13-20260917 and provider TypeSafe. Alternative API: /api/v1/systemone for the TypeSafe SDK. /api/v1/chat/completions is not the typed interface. The chat catalog lists typesafe/jev-router, a separate model-selection product; absence of the typed model there is not absence of Decisions support.

Request: model, state (string/object/array), questions mapping. Primitives: choice with named criteria; noul with yes/no criteria; score with an ordered rubric. Response: answers, dated model, provider/generation ID, usage.input_tokens/output_tokens/cost. session_id groups observations; it is not a documented Jev KV cache key.

## Live caller and scope

Active-profile scripts/henri_system1.py uses Hermes OpenRouter credential resolution and the real endpoint. Rubrics cover owner, escalation, failure, evidence, store, policy_risk. User requested all eligible typed model judgments use Jev first. The acting root consumes advice and still applies deterministic policy and approval. No change to MoA reference roster.

Never use Jev for arithmetic, dates, hashes, schema/proof verdicts, authorization, filesystem permissions, or final task/capability acceptance. Never use it as a generative MoA slot. Finite choices can classify novel input; this does not prove novel generative reasoning or semantic correctness. The older claim that finite choices cannot handle any out-of-distribution input was too broad.

Threshold 0.85 is a pilot confidence rule, not a calibrated error bound. Unclear/review/deny/malformed/missing/stale results escalate without execution. Every receipt says authorization=false and execution_permitted=false. Approval cannot be cached from a model decision.

## Observed probes and falsified claims

First real probe returned HTTP 200 and ARCHITECTURE at confidence 0.99 in about 401 ms. The installed caller returned ARCHITECTURE in about 583 ms, then reused the exact request locally without a new generation. A prohibited-secret-mount advisory returned DENY in about 599 ms and did not authorize execution. These observations contradict the attached <50 ms end-to-end requirement for this path, not the possibility of faster server computation. They do not measure HENRI calibration or security false negatives.

Vendor calibration, type-safe output, and model confidence do not imply zero semantic errors. OpenRouter tutorial describes confidence as concentration of the alternatives. HENRI auto-routing needs a separately approved labeled calibration with held-out data, observed latency/cost, and rejection rules. Keep honest uncertainty; no universal deterministic or zero-false-negative claim.

## Verification

Inspect exact generation IDs, request/response hashes, returned fields and usage. Repeat identical state: application hit with source generation retained. Change state/source/rubric/model: cache miss. Invalid response, unknown choice, missing distribution/confidence, or nonzero API failure blocks. Use actual independent labeled cases before automatic delegation or deployment.

Sources: https://openrouter.ai/docs/guides/community/jev ; https://openrouter.ai/docs/guides/community/jev-tutorial ; https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request ; https://docs.typesafe.ai/confidence . Raw probes live outside Git under henri-telemetry/cache-openshell-jev. Never save credentials.
