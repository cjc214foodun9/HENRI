# Jev on OpenRouter — independent audit (evidence note)

Status: READ-ONLY AUDIT. No live inference. No credentials used. No configuration changed.
Date: 2026-10-01. Host: this Windows machine.
Scope: official model ID, request/response schema, confidence semantics, cache
eligibility, latency claims, failure controls, supported task/answer space.

Evidence classes used below:
- OBSERVED — read from a primary source (OpenRouter catalog API, OpenRouter docs,
  TypeSafe docs) on 2026-10-01.
- HYPOTHESIS — vendor marketing or unverified claim. Not measured on this host.

A prior proposal exists at
`hermes/skills/henri-workflow/henri-moa-routing/references/jev-systemone-integration.md`.
It is marked PROPOSAL / NOT WIRED. It cited a `defapi.org` gateway and the endpoint
`POST /v1/systemone`. Sections 9 and 10 give the corrections.

## 1. Official model IDs (OBSERVED)

TypeSafe publishes three entries on OpenRouter. They are not the same product.

| OpenRouter ID | Name | Output | Context | Price | Role |
|---|---|---|---|---|---|
| `typesafe/jev-1.13` | TypeSafe: Jev 1.13 | typed decisions | 32,000 | $0.042/M in, $0/M out | The System One model. Use this for System 1 tasks. |
| `~typesafe/jev-latest` | TypeSafe: Jev Latest | typed decisions | 32,000 | $0.042/M in, $0/M out | Alias. It always points to the newest Jev release. It points to Jev 1.13 today. |
| `typesafe/jev-router` | TypeSafe: Jev Router | text | 1,000,000 | dynamic (`-1` in the catalog) | A chat router. It picks a model for each request. It is NOT the typed model. |

Upstream TypeSafe model IDs: `jev-1.13.0` (versioned) and `jev-latest`, `jev-preview`
(aliases). All aliases point to `jev-1.13.0` today.

Notes:
- The versioned response field reports the dated snapshot, for example
  `typesafe/jev-1.13-20260917`. This is expected.
- The OpenRouter `/api/v1/models` catalog (464 entries, OBSERVED 2026-10-01) lists
  `typesafe/jev-router` only. It does NOT list `typesafe/jev-1.13` or
  `~typesafe/jev-latest`. An app that discovers Jev from the model list will find
  only the chat router. Pin the model ID by hand.
- `typesafe/jev-router` returns text. It is a free-form-text surface. It is not a
  typed decision surface. Do not confuse the two.

## 2. Endpoints (OBSERVED)

Two OpenRouter surfaces expose the same System One model. Use one.

| Surface | Endpoint | Use |
|---|---|---|
| Decisions API | `POST https://openrouter.ai/api/alpha/decisions` | Plain HTTP from any language, or the OpenRouter TS/Python/Go SDK. |
| System One API | `POST https://openrouter.ai/api/v1/systemone` | The TypeSafe JS/Python SDK. Set the base URL to `https://openrouter.ai/api`. |

TypeSafe upstream endpoint: `POST https://api.typesafe.ai/v1/systemone`.
Auth for OpenRouter: header `Authorization: Bearer <OPENROUTER_API_KEY>`.
No separate TypeSafe account or key is needed.

The Decisions path is outside the `/api/v1` prefix. A client that keeps the default
base URL gets a 404 on that call.

## 3. Request schema (OBSERVED)

Body: `{ "model": string, "state": <value>, "questions": <map>, ...optional }`.

Required:
- `model` — string. Send `typesafe/jev-1.13` or `~typesafe/jev-latest`.
- `state` — string, object, or array. The content to evaluate. Text only.
- `questions` — map<string, Question>. You choose each key. Answers come back under
  the same keys. The key is not sent to the model.

Optional (Decisions API): `provider`, `session_id` (max 256 chars, for grouping),
`trace` (observability), `user` (max 256 chars).

Question object. All three types carry `type` and `instructions`.
- `type` — `"noul"`, `"choice"`, or `"score"`.
- `instructions` — string, object, or array. Object/array form holds the question in
  one field and the referenced data in others. Reference a `state` path in backticks.
- `criteria`:
  - noul — optional object `{ "true": ..., "false": ... }`.
  - choice — required map<option, string|object|array|null>. Maximum 255 options.
  - score — required ordered array of level descriptions. Minimum 2 levels.
    Maximum 10 levels.

Answer space limits (hard):
- Choice: 2..255 options.
- Score: 2..10 levels.
- Input type: text only. No image, audio, or video. Convert them to text first.
- Context: OpenRouter lists 32,000 tokens. TypeSafe docs state 64k per request
  (state + all questions combined), and 32k for state + the single longest question.
  For a safe design, keep state + questions under 32k.

## 4. Response schema (OBSERVED)

```json
{
  "id": "gen-dec-...",
  "model": "typesafe/jev-1.13-20260917",
  "provider": "TypeSafe",
  "answers": {
    "is_bug":   { "type": "noul",   "noul": 0.96 },
    "team":     { "type": "choice", "choice": "payments", "confidence": 0.75,
                  "probabilities": { "payments": 0.84, "frontend": 0.16, "account": 0 } },
    "urgency":  { "type": "score",  "score": 1.99, "confidence": 0.99,
                  "legend": { "0": "Can wait", "1": "This week", "2": "Blocking revenue" },
                  "probabilities": { "0": 0, "1": 0.01, "2": 0.99 } }
  },
  "usage": { "input_tokens": 476, "output_tokens": 70, "cost": 0.000019992 }
}
```

Field notes:
- `answers[id].type` always matches the question type.
- noul answer: `noul` only. It is a probability in [0, 1]. There is no `confidence`.
- choice answer: `choice` (highest-probability option), `probabilities` (map, floats
  that sum to 1), `confidence` (0..1).
- score answer: `score` (probability-weighted position; it can fall between levels),
  `legend` (level index -> description), `probabilities` (map), `confidence` (0..1).
- `usage.cost` is in USD and is the cost of that call.
- The Decisions schema marks `confidence` and `probabilities` as OPTIONAL.
  TypeSafe's own API reference marks them as required. Guard for a missing field.

## 5. Confidence semantics (OBSERVED)

- Noul is NOT a degree. `noul` is the probability that the answer is yes. A value near
  0.5 means yes and no are about equally likely. It does not mean a medium degree.
- Choice and Score return a full `probabilities` distribution plus one `confidence`.
- `confidence` is a statistic computed from that distribution. A concentrated
  distribution gives a high value. A flat distribution gives a low value. All the
  probability on one option gives 1.0.
- TypeSafe publishes the full `probabilities`. You are not bound to their `confidence`
  formula. You can compute your own statistic from `probabilities`.
- Calibration is a group property. TypeSafe states: "Calibration is measured across
  groups of predictions; it does not guarantee that an individual answer is correct."
- OpenRouter states: "Confidence describes the distribution of the alternatives, not
  whether the workflow is safe to run."
- The 0.95-conviction claim in the prior proposal stays HYPOTHESIS. You must measure
  ECE and BSS on domain-labeled cases before you trust any threshold.
- Vendor guidance for thresholds: three bands (high = act, medium = review, low = do
  not act). Set the band edges from the cost of each error, not from a round number.

## 6. Cache eligibility (OBSERVED)

- OpenRouter Response Caching supports only these endpoints:
  `/api/v1/chat/completions`, `/api/v1/responses`, `/api/v1/messages`,
  `/api/v1/embeddings`.
- The Decisions endpoint (`/api/alpha/decisions`) and the System One endpoint
  (`/api/v1/systemone`) are NOT in that list. They are NOT eligible for OpenRouter
  response caching. The `X-OpenRouter-Cache` header does not apply to them.
- The provider endpoint for `typesafe/jev-1.13` reports
  `"supports_implicit_caching": false`. Provider-side prompt caching is off.
- The Jev 1.13 model page reports a cache hit rate of 0.00%.
- Conclusion: Jev decisions are not cacheable at the OpenRouter layer today. Any
  caching must be application-side (your own key on the request). Do not price Jev on
  an expected cache discount.

## 7. Latency claims (OBSERVED)

OpenRouter provider performance data for `typesafe/jev-1.13` (week ending 2026-10-01):
- P50 0.18 s; average 0.21 s.
- P75 0.26 s; P90 0.31 s; P95 0.35 s; P99 0.76 s.
- End-to-end latency percentiles match the latency percentiles above.
- Provider uptime 100.00% (3 days); availability 99.99%.

TypeSafe docs: "Most queries complete in about 100 ms."
OpenRouter cookbook (gate tool calls): each Decisions request "returned in under 600 ms"
and cost `usage.cost` between 0.000030 and 0.000036 USD.

Assessment of the prior "70-500 ms" claim: partly consistent, not verified on this host.
The 0.18 s P50 and 0.35 s P95 fit the range. The 70 ms low end is not supported by any
primary source. Treat "70-500 ms" as HYPOTHESIS. Design for p95 <= 0.35 s and p99 <= 0.76 s.

## 8. Failure controls (OBSERVED)

HTTP status set for the Decisions API: 400, 401, 402, 403, 404, 413, 429, 500, 502,
503, 524, 529. TypeSafe's own API documents 401 (auth), 422 (validation), 429 (rate
limit), 529 (overloaded). OpenRouter uses 400 for a validation error.

Recommended fail-closed pattern (from the OpenRouter gate cookbook):
- On a non-2xx response, throw. Do not treat an error as an approval or a review.
- On a missing answer for a requested key, throw.
- On a probability below 0 or above 1, throw.
- On a missing `confidence` or `probabilities`, default to a value that fails the
  acceptance threshold (for example 0). Never default to a pass.

Rate limits (TypeSafe, OBSERVED): 100K tokens/second and 40 requests/second. The
limits change dynamically. The TypeSafe SDK retries with exponential backoff and
honors `retry-after`. On the raw HTTP path, add your own backoff for 429 and 529.

Do not let Jev accept or reject a capability claim, and do not let it replace a
deterministic check. Jev routes, triages, scores, and recommends.

## 9. Discrepancies and contradictions found (OBSERVED)

1. Pricing on `typesafe/jev-router`: the model page FAQ says the price is zero, but
   the catalog API returns `pricing.prompt = "-1"` and `pricing.completion = "-1"`
   (dynamic). The endpoint list for this router is empty. Do not rely on the FAQ text.
   This applies to the router only, not to Jev 1.13.
2. Context length: OpenRouter lists 32,000 for Jev 1.13. TypeSafe docs state 64k per
   request. OpenRouter's value is the conservative one. Keep under 32k.
3. Missing-field shape: OpenRouter's Decisions schema marks `confidence` and
   `probabilities` optional. TypeSafe's API reference marks them required.
4. Validation status code: OpenRouter returns 400. TypeSafe returns 422.
5. Model discovery: the OpenRouter model catalog omits `typesafe/jev-1.13` and
   `~typesafe/jev-latest`. Only `typesafe/jev-router` appears.

## 10. Corrections to the prior proposal (OBSERVED)

- Endpoint: the prior note used `POST /v1/systemone` on a `defapi.org` gateway. The
  current official OpenRouter paths are `/api/v1/systemone` and
  `/api/alpha/decisions`. Use the official OpenRouter paths.
- Price: the prior note listed a conflict between $0.042 and $0.084 per Mtok. The
  official OpenRouter price for Jev 1.13 is $0.042/M in and $0/M out. Output is free.
- Model ID: the prior note did not name an OpenRouter model ID. Use
  `typesafe/jev-1.13`, or the `~typesafe/jev-latest` alias.
- The prior note's core interface finding still holds. Jev returns typed values, not
  text. It is not an MoA advisor slot.
- The prior note's endpoint sentence is corrected above. Its standards S1..S9 stay
  valid. This audit does not change the references (Ref C and others stay unchanged).

## 11. Supported task and answer space (OBSERVED)

Supported: routing, classification, ranking, verification, extraction checks, and
guardrails. Any decision with a pre-enumerated answer space.

Not supported: free-form text, code, explanations, summaries, rationales, reasoning
traces, and image/audio/video input. Jev is not a chat model and is not a drop-in
replacement for one.

## 12. Sources

- OpenRouter catalog API: `GET https://openrouter.ai/api/v1/models` (2026-10-01).
- OpenRouter endpoints API: `GET https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints`.
- OpenRouter model pages: `/typesafe/jev-1.13`, `/~typesafe/jev-latest`, `/typesafe/jev-router`, `/typesafe`.
- OpenRouter docs: Jev hub, Jev tutorial, TypeSafe SDK guide, Decisions API reference,
  Response Caching, Jev Router routing guide, gate-tool-calls and verified-cascade cookbooks.
- TypeSafe docs (docs.typesafe.ai): System One, How to build with System One,
  Confidence, API reference, Models, Noul.

## 13. What this audit did NOT do

- No live inference call. No tokens spent.
- No credential used. No configuration changed.
- No production run reported. All latency and calibration numbers are OBSERVED from
  primary sources, or marked HYPOTHESIS. They are not measured on this host.
- No security approval was requested or granted. This is an evidence note only.
