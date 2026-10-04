"""AAII v4.3.2 specification registry -- pinned from the operator's five PDFs.

WHAT IS PINNED, AND WHERE FROM
    All facts below come from HENRI V2/benchmarks/spec/*.pdf.txt, extracted by
    design/zone_a/extract_aaii_spec.py (EXIT=0). Doc sha256 values load from
    MANIFEST.json so this file carries no unverifiable numbers.

THE NAME
    "AAII" = Artificial Analysis Intelligence Index. The acronym NEVER appears
    literally in any document -- only the expansion. A naive grep for "AAII"
    returns 0 and would wrongly report the spec absent. AGENT_benchmarking.pdf
    line 1: "Artificial Analysis Intelligence Index Evaluations". Line 7:
    "Included in Artificial Analysis Intelligence Index v4.3.2 at 15% weighting."

WHAT THE ATTACHED DOCUMENTS ARE, AND ARE NOT
    They are the METHODOLOGY, prompts, scoring rules, and source links.
    They are NOT the datasets and NOT the executable harnesses. Every dataset and
    harness is referenced by URL and lives elsewhere. So every entry below is
    METHODOLOGY_ONLY until its dataset is fetched and pinned by sha256, and
    spec_status must be read before any score is reported.

WHY WEIGHTS ARE PARTLY NULL
    The documents state AA-Omniscience's 15% split (Accuracy 10%, Non-Hallucination
    5%) and AA-Briefcase's 15%. They do NOT state the full 100% composition.
    Unstated weight is recorded as None. A None never becomes a number here.
"""
from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(_HERE, "spec")

INDEX_VERSION = "4.3.2"
INDEX_VERSIONS_ALIAS = ("v4.3", "4.3.2")   # both appear, once each, in AGENT doc
INDEX_NAME = "Artificial Analysis Intelligence Index"
INDEX_ACRONYM = "AAII"

# spec_status vocabulary. Read before reporting any score.
STATUS_METHODOLOGY_ONLY = "METHODOLOGY_ONLY"      # method pinned; dataset absent
STATUS_DATASET_PINNED = "DATASET_PINNED"          # fetched + sha256 recorded
STATUS_IMPLEMENTED = "IMPLEMENTED"                # adapter runs end to end
STATUS_UNAVAILABLE = "UNAVAILABLE"                # needs external access (API)

_T = "HENRI V2/benchmarks/spec"

# One record per evaluation named in the five documents. Facts are quoted or
# transcribed from the extracted text; nothing is inferred.
EVALS = [
    # ---------------------------------------------------------------- agents
    dict(
        id="aa-briefcase", track="agent", name="AA-Briefcase", version="1.1",
        index_weight=0.15,
        dataset_url="https://huggingface.co/datasets/ArtificialAnalysis/AA-Briefcase-Lite",
        harness_url="https://github.com/ArtificialAnalysis/Stirrup",
        evidence=f"{_T}/AGENT_benchmarking.pdf.txt:5-16,110-113",
        spec_status=STATUS_DATASET_PINNED,
        pin_record="design/zone_a/evidence/aaii_dataset_pin_aa-briefcase.json",
        scoreable_official=False, scoreable_local_proxy=False,
        scoreability_note=("AA-Briefcase-Lite is a PUBLIC EXAMPLE SCENARIO and its README states "
                           "verbatim it 'is not part of the scored leaderboard'; the four scored "
                           "scenarios stay private. Pinning this repo buys HARNESS VALIDATION "
                           "(rubric + evidence chains + 6 frontier submissions), never a score."),
        metric="Elo (Crowd-BT); aggregates analytical Elo, presentation Elo, rubric pass rate",
        grading="rubric binary (panel of 3) + pairwise (analytical, presentation)",
        runs_per_task=1, turns=500, sandbox="E2B, week-scoped, no internet",
        notes=("Multi-week knowledge-work scenarios, 2-5 tasks/week. Tasks run "
               "INDEPENDENTLY (no carry-over of prior submissions). Panel "
               "anti-bias sampling. Normalized clamp((Elo-500)/2000); anchor "
               "GPT-5.5 (medium) at 1000. Elo frozen at model addition."),
    ),
    dict(
        id="gdpval-aa", track="agent", name="GDPval-AA", version="2.1",
        index_weight=None,
        dataset_url="https://huggingface.co/datasets/openai/gdpval",
        harness_url="https://github.com/ArtificialAnalysis/Stirrup",
        evidence=f"{_T}/AGENT_benchmarking.pdf.txt:345-541",
        spec_status=STATUS_METHODOLOGY_ONLY,
        metric="Elo (Bradley-Terry MLE, sandwich CI); anchored DeepSeek V4.1 Flash (max)=1600",
        grading="pairwise, judge sampled from panel of 3, submissions anonymized A/B",
        runs_per_task=1, turns=250, sandbox="E2B, 419 py + 762 system packages",
        notes=("44 occupations. Two stages: Task Submission then Pairwise "
               "Grading. Context-overflow unwind at >70% window. Start of "
               "turn budget announced near the limit."),
    ),
    dict(
        id="automationbench-aa", track="agent", name="AutomationBench-AA",
        version="1.0.6",
        index_weight=None,
        dataset_url="https://github.com/zapier/AutomationBench",
        harness_url="https://github.com/ArtificialAnalysis/Stirrup",
        evidence=f"{_T}/AGENT_benchmarking.pdf.txt:657-706",
        spec_status=STATUS_METHODOLOGY_ONLY,
        metric=("0 if any guardrail violated; else percent of objectives "
                "completed. NO LLM judge -- programmatic on final env state."),
        grading="programmatic assertions: objectives (must make true) + guardrails (must not break)",
        runs_per_task=1, turns=50, sandbox="multi-turn simulated SaaS apps via REST",
        notes=("PRIVATE held-out split, 657 tasks. Six domains: Finance, HR, "
               "Marketing, Operations, Sales, Support. Apps: Gmail, Sheets, "
               "Slack, Salesforce, Zendesk, Jira, HubSpot. Domain cuts are "
               "mutually exclusive; app cuts are NOT."),
    ),
    # --------------------------------------------------------------- general
    dict(
        id="aa-omniscience", track="general", name="AA-Omniscience", version=None,
        index_weight=0.15,
        dataset_url="https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public",
        harness_url=None,
        evidence=f"{_T}/GENERAL_Benchmarking.pdf.txt:2-31",
        spec_status=STATUS_DATASET_PINNED,
        pin_record="design/zone_a/evidence/aaii_dataset_pin_aa-omniscience.json",
        scoreable_official=False, scoreable_local_proxy=True,
        scoreability_note=("600 public questions with short pinned answers. The official Index needs a "
                           "GRADING MODEL plus abstention behaviour; a deterministic answer match is "
                           "ACCURACY ONLY and is not the Omniscience Index."),
        metric=("AA-Omniscience Index: +correct, -hallucinated, abstain neutral. "
                "Index contributes Accuracy (10%) + Non-Hallucination Rate (5%)"),
        grading="each answer CORRECT | INCORRECT | PARTIAL_ANSWER | NOT_ATTEMPTED; judge GPT-5.6 Luna (medium)",
        runs_per_task=1, turns=None, sandbox=None,
        notes=("6,000 questions, 42 topics. Rewards abstention over guessing. "
               "Paper arXiv:2511.13029."),
    ),
    dict(
        id="gdp-pdf", track="general", name="GDP.pdf (Surge AI impl)", version=None,
        index_weight=None,
        dataset_url="surgeai/GDP.pdf (HF)",
        harness_url=None,
        evidence=f"{_T}/GENERAL_Benchmarking.pdf.txt:32-100",
        spec_status=STATUS_METHODOLOGY_ONLY,
        metric="All-pass (headline) + Mean Pass (secondary); both over 500 attempts",
        grading="criteria independent, judge GPT-5.6 Luna Medium; all criteria must have a verdict",
        runs_per_task=5, turns=1, sandbox="none; single turn, no browsing/tools",
        notes=("100 tasks, 10 domains, 4,592 PDF pages, 1,275 criteria. Fixed "
               "denominator 500 attempts. LiteParse 2.5.0 + English OCR; page "
               "images 150 DPI (72 min). AA and Surge scores NOT comparable "
               "(doc: input and judge differ). Errors/missing = 0."),
    ),
    dict(
        id="aa-lcr", track="general", name="AA-LCR", version="1.1",
        index_weight=None,
        dataset_url="https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR",
        harness_url=None,
        evidence=f"{_T}/GENERAL_Benchmarking.pdf.txt:101-126",
        spec_status=STATUS_DATASET_PINNED,
        pin_record="design/zone_a/evidence/aaii_dataset_pin_aa-lcr.json",
        scoreable_official=False, scoreable_local_proxy=False,
        scoreability_note=("100 questions pinned, but each needs its ~100k-token source document set "
                           "from the extraction zip AND an equality-checker LLM. Neither is present, "
                           "so this is answerable=False and scoreable=False."),
        metric="pass@1",
        grading="equality-checker LLM GPT-5.6 Luna (medium)",
        runs_per_task=1, turns=1, sandbox=None,
        notes=("100 questions, 7 doc categories. ~100k tokens input per question "
               "(cl100k_base); ~230 docs, ~3M unique input tokens; needs >=128K "
               "context. v1.1 scores NOT comparable with v1.0."),
    ),
    # ---------------------------------------------------------------- coding
    dict(
        id="terminal-bench", track="coding", name="Terminal-Bench", version="4.0",
        index_weight=None,
        dataset_url="https://github.com/harbor-framework/terminal-bench",
        harness_url="https://github.com/SWE-agent/mini-swe-agent",
        evidence=f"{_T}/CODING_Benchmarking.pdf.txt:2-31,53",
        spec_status=STATUS_METHODOLOGY_ONLY,
        metric="pass@1 averaged over 3 repeats",
        grading="each task's own verification suite; ALL tests must pass; verifier timeout = failure",
        runs_per_task=3, turns=500, sandbox="per-task verifier container, isolated from agent",
        notes=("66 tasks. mini-swe-agent defaults: native bash tool, NO context "
               "compaction, full transcript always visible. Task timeouts and "
               "resources follow upstream definitions."),
    ),
    dict(
        id="scicode", track="coding", name="SciCode", version="1.0.1",
        index_weight=None,
        dataset_url="https://scicode-bench.github.io/",
        harness_url=None,
        evidence=f"{_T}/CODING_Benchmarking.pdf.txt:32-52",
        spec_status=STATUS_METHODOLOGY_ONLY,
        metric="pass@1 at SUB-PROBLEM level",
        grading="isolated executor, 300-second timeout",
        runs_per_task=1, turns=1, sandbox="isolated executor",
        notes=("Scientist-annotated background included in prompt. Paper "
               "arXiv:2407.13168."),
    ),
    # ---------------------------------------------------- scientific reasoning
    dict(
        id="hle", track="scientific_reasoning", name="Humanity's Last Exam",
        version="May-2025 revision",
        index_weight=None,
        dataset_url="https://huggingface.co/datasets/cais/hle",
        harness_url=None,
        evidence=f"{_T}/SCIENTIFIC_REASONING_Benchmarking.pdf.txt:2-25",
        spec_status=STATUS_DATASET_PINNED,
        pin_record="design/zone_a/evidence/aaii_dataset_pin_hle.json",
        scoreable_official=False, scoreable_local_proxy=True,
        scoreability_note=("2,500 parquet rows pinned. Official AAII grading uses an equality-checker "
                           "LLM; a normalized exact match is an APPROXIMATION and must never be "
                           "reported as the AAII score."),
        metric="pass@1",
        grading="equality checker GPT-5.6 Luna (medium), prompt from Hendrycks et al.",
        runs_per_task=1, turns=1, sandbox=None,
        notes=("2,158 text-only questions (2,500 total in revision). Dataset "
               "curated ADVERSARIALLY against GPT-4o, Gemini 1.5 Pro, Claude 3.5 "
               "Sonnet, o1 family -- direct comparison with non-curation models "
               "is DISCOURAGED by the authors. Reproduce that caveat in reports."),
    ),
    dict(
        id="critpt", track="scientific_reasoning", name="CritPt", version=None,
        index_weight=None,
        dataset_url="https://huggingface.co/datasets/CritPt-Benchmark/CritPt",
        harness_url="https://github.com/CritPt-Benchmark/CritPt",
        evidence=f"{_T}/SCIENTIFIC_REASONING_Benchmarking.pdf.txt:26-60",
        spec_status=STATUS_UNAVAILABLE,
        metric="pass@1 over 5 repeats",
        grading=("OFFICIAL CritPt grading server. API access is granted case by "
                 "case to approved labs (critpt@artificialanalysis.ai)"),
        runs_per_task=5, turns=2, sandbox=None,
        notes=("70 challenge-level test items (example excluded). Two-step "
               "parsing: reason, then format for grading. Answers: numerics, "
               "SymPy expressions, Python functions (test cases). BLOCKED on "
               "grading-API authorization -- cannot be self-scored."),
    ),
]

# Index composition is only PARTIALLY disclosed. Record what is stated.
INDEX_COMPOSITION = {
    "stated": {
        "AA-Omniscience": {"total": 0.15, "accuracy": 0.10, "non_hallucination": 0.05},
        "AA-Briefcase v1.1": {"total": 0.15},
    },
    "unstated": ("The full 100% composition of Intelligence Index v4.3.2 is NOT "
                 "in the attached documents. Do not assume or reconstruct it."),
}


def doc_hashes() -> dict:
    """PDF sha256 per source doc, read from the extraction MANIFEST."""
    p = os.path.join(SPEC_DIR, "MANIFEST.json")
    with open(p, encoding="utf-8") as fh:
        man = json.load(fh)
    return {r["file"]: r.get("pdf_sha256") for r in man if r.get("pdf_sha256")}


def by_id(eval_id: str) -> dict:
    for e in EVALS:
        if e["id"] == eval_id:
            return dict(e)
    raise KeyError(f"unknown eval {eval_id!r}; known: {[e['id'] for e in EVALS]}")


def index_weights(only_stated: bool = True) -> dict:
    """Return {eval_id: weight}. Unstated weights are omitted, never guessed."""
    return {e["id"]: e["index_weight"] for e in EVALS
            if e["index_weight"] is not None or not only_stated}


def spec_report() -> dict:
    """Machine-readable status of the whole spec. Print this before any run."""
    n_by = {}
    for e in EVALS:
        n_by[e["spec_status"]] = n_by.get(e["spec_status"], 0) + 1
    return {
        "index_name": INDEX_NAME,
        "index_acronym": INDEX_ACRONYM,
        "index_version": INDEX_VERSION,
        "acronym_appears_literally_in_docs": False,
        "acronym_note": ("'AAII' never appears literally; the expansion does. "
                         "A grep for AAII returns 0 and is NOT evidence of absence."),
        "n_evals": len(EVALS),
        "status_counts": n_by,
        "evals_without_dataset_pinned": [e["id"] for e in EVALS
                                        if e["spec_status"] == STATUS_METHODOLOGY_ONLY],
        "evals_blocked_external": [e["id"] for e in EVALS
                                   if e["spec_status"] == STATUS_UNAVAILABLE],
        "index_composition": INDEX_COMPOSITION,
        "doc_sha256": doc_hashes(),
        "headline": ("METHODOLOGY PINNED. DATASETS AND HARNESSES ARE NOT IN THE "
                     "ATTACHED DOCUMENTS. No AAII v4.3.2 score can be reported "
                     "until each dataset is fetched and pinned by sha256."),
    }


if __name__ == "__main__":
    print(json.dumps(spec_report(), indent=2))
