# AAII v4.2 — Is Pure Continuous Test-Time Learning Sufficient to Dominate?

Author: HENRI arbiter (2026-09-05, direct evidence). Branch: carrier/g4-egress-calibration @ main (f220469).

## 0. Scope and method

Question (user): with NO demo source for ARC-AGI-3, "pure test time learning should be
sufficient with henri algorithm" — audit whether continuous test-time learning alone is
sufficient to dominate the AAII v4.2 benchmark.

Method: map every AAII v4.2 member to the four capability channels HENRI can or cannot
supply, using only live-verified evidence (this session + sealed prior receipts). No score
fabrication: AAII is API-only; HENRI is NOT_EVALUATED on all 10 members.

## 1. AAII v4.2 composition (pinned 2026-09-05, official byte hashes in
   henri-research references/aaii-v42-composite-and-exposure-audit.md)

| Member | Weight | Category | Needed capability |
|---|---|---|---|
| AA-Briefcase | 15% | Agents | Hosted agent endpoint + tool execution + calibrated text egress |
| GDPval-AA v2 | 10% | Agents | Hosted endpoint + document understanding + text egress |
| τ³-Banking | 5% | Agents | Tool/agent loop + numeric accuracy + text egress |
| Terminal-Bench v2.1 | 10% | Coding | Shell harness (Harbor) + code egress + container runtime |
| SciCode | 10% | Coding | Python REPL + math/science knowledge + code egress |
| HLE | 10% | Sci Reasoning | World knowledge + long-form reasoning + text egress (gated dataset) |
| CritPt | 10% | Sci Reasoning | Claim-level critical reasoning + text egress |
| AA-Omniscience Acc | 10% | General | Broad factual knowledge + MCQ egress |
| AA-Omniscience Non-Hal | 5% | General | Quiet/abstain discipline (measurable) |
| GDP.pdf | 10% | General | Document reasoning + text egress |
| AA-LCR v1.1 | 5% | General | Legal citation retrieval + structured egress |

## 2. Channel audit (live-verified)

| Channel | State | Evidence |
|---|---|---|
| Demo-pair task compilation (W_task) | EXISTS, online-only, zero-pretrain | main physics; compiled 100% test-time from (X_i,Y_i) |
| ARC-AGI-3 public demos | 0/16, 0/25 — `BLOCKED_NO_DEMONSTRATIONS` | g1_arcade_demo_audit.json (2026-09-05): provenance=public_api, examples=None |
| World knowledge | Bounded, NOT complete: 27 sources / 3,433 chunks / 14 domains; world_claims=0 | Zone C corpus_chunks + domain_source_manifest (OBSERVED live) |
| Text egress | `DIAGNOSTIC_ONLY` — K2/U2 BLOCKED_SEMANTIC_CAPACITY (sealed 2026-08-25); trained linear unbinder is basis-dependent | henri_decoder.py HENRINeuralEgressUnbinder; checkpoint overlay sha 7557238908… |
| Retrieval egress | K5 compositional codec: identical 0.9999 / 1-edit 0.905 / reversal 0.476 / random 0.032; grounded chunk hit_rank ≤ 4/5 probes, sims 0.07–0.12 (chunk-level dilution) | g4_retrieval_probe3 (OBSERVED live pgvector) |
| Hosted endpoint (AAII evaluates hosted models via API) | ABSENT. No public HENRI endpoint, no submission pipeline | AAII methodology live page; repo is not a submittable unit |
| Tool/shell execution | Terminal-Bench: no container runtime on Vast (no Docker CLI); Harbor pin verified but not deployable yet | g3 scaffold receipt; OBSERVED `which docker` = absent |
| ARC internal scoring | Possible via arc engine + CEGIS; NO demos → goal source starved | G1 A/B: arms 0.0/30; GOAL_EDMD_NO_DEMOS |

## 3. Verdict

**Pure continuous test-time learning is NOT sufficient to dominate AAII v4.2.**

Reasons (each is a blocking channel, not a tuning gap):

1. **Egress capacity (structural):** the only trained text egress is the legacy linear
   unbinder (K2/U2 BLOCKED_SEMANTIC_CAPACITY). Test-time adaptation (SGLD) can align it to
   seen exemplar pairs; it cannot compose novel free text (measured MBPP 0/500 at that
   boundary). 8 of 10 members need free-text or code output. This is a capacity boundary,
   not a lack of demos.
2. **Knowledge coverage (bounded):** zero-pretraining + K5 corpus (3,433 chunks) covers
   ~14 domains at shallow depth. AA-Omniscience/HLE/GDP.pdf are open-domain. Test-time
   learning compiles operators from *presented* pairs; it cannot add world facts absent
   from Zone C. Abstain discipline can maximize Non-Hal (5%) but caps Accuracy.
3. **Hosted endpoint (infrastructure):** AAII v4.2 evaluates hosted models via API.
   No endpoint exists. This alone blocks all 10 members regardless of algorithm.
4. **Tool execution (Terminal-Bench/τ³/AA-Briefcase):** need sanctioned container/shell
   runtime + calibrated tool egress; currently BLOCKED (no Docker CLI, schema only).
5. **Demo starvation compounds:** ARC-AGI-3 (the only domain with demo-parity
   architecture) exposes 0 public demos, so even the internal pure-test-time path has
   nothing to compile from; G1 measured MECHANISM_NOT_EXERCISED.

## 4. What IS sufficient (honest boundary)

- Test-time learning is sufficient for: task compilation WHEN demos exist (ARC with an
  authorized public-ingress manifest), and non-hallucination (abstain) discipline.
- To score on AAII v4.2 at all, HENRI requires, in order: (a) calibrated semantic egress
  (G4-class; currently DIAGNOSTIC_ONLY), (b) a hosted endpoint, (c) an authorized
  knowledge boundary (K5 corpus is a start, not completion), (d) tool-execution
  environments, (e) a demo/ingress source for agentic tasks.
- Expected honest behavior TODAY on AAII-style prompts: high abstention (Non-Hal OK),
  near-zero accuracy, `NOT_EVALUATED` on the official composite.

## 5. Kill criteria / falsification

- FALSIFIED if any of: calibrated egress with novel-text composition (≥20% completion on
  an unseen heldout of short answers), hosted endpoint live, K5 corpus covering ≥80% of a
  sampled AAII-constituent question distribution with retrieval hit_rank ≤ 3.
- Any score claim requires the Section 6A evidence chain (dataset hash + evaluator +
  item results + checkpoint lineage + endpoint), which none of the current carriers
  produce. This audit is analysis, NOT an evaluation.
