#!/usr/bin/env python
"""CLASS51 P3(a) EFFICACY MATRIX -- 264-item paired Arm A/B (repaired grader).

Repairs over backbone_p3_pilot.py (commit 83724b5, pilot RATIFIED A1):
  1. HumanEval grading now EXECUTES check(entry_point). The pilot only
     defined check() and never called it (INVALID_EVALUATOR, 2026-08-22).
  2. MBPP arm added: sanitized-mbpp.json first 100 items (canonical blob
     a999d25d... / local sha256 ca95deaa...), graded by executing
     test_imports + test_list against the generated code. Reference code
     is NEVER appended.
  3. Arm pairing fixed: single canonical base prompt; Arm B = Arm A with
     the retrieval block inserted at a fixed sentinel; a byte-identity
     verifier proves B minus block == A for all 264 items.
  4. Full 264-item prompt/test surface registered in the v3.2
     contamination scan (fresh receipt, not carried from the pilot).
  5. Pre-registered evaluator self-test runs BEFORE any model load.

R2 (RATIFIED 2026-08-22): graders use the official prompt + completion +
tests form (canonical prompt prepended; MBPP signature from the dataset
`code` field; re-emitted def signatures stripped deterministically).
Self-test extended to 10 canonical cases (5 HE + 5 MBPP).
R3 (RATIFIED 2026-08-22): MBPP graded as completion-as-generated. The MBPP
prompt is prose and carries no signature; the model invents its own argument
names, so substituting the dataset signature renames arguments the body
already binds -> NameError (OBSERVED under R2: lst, tuples_list, list1, arr,
s, n, a, tup). Leading import/from lines are hoisted to module scope
(OBSERVED IndentationError when a completion opened with 'import math').
Body-only completions fall back to the canonical dataset signature. Self-test
extended to 16 canonical cases (5 HE + 11 MBPP, incl. renamed-function alias
binding). Raw completions are stored in
run receipts (deterministic regrading / replay gate). Final scorecard is also
delivered to <repo>/telemetry/scorecard_final_264.json.
A2 (RATIFIED 2026-08-22): itertools.rst Recipes section excised (HE/31
is_prime overlap); detector v3.2 (bare-import + prose-keyword exclusions);
new corpus aggregate 2ced6f6f1afefb4d54129b9fff74d4edf08a91efe38c4b639ef8301144786f04.
R4 (RATIFIED 2026-08-23): deterministic Markdown-fence extraction; MBPP
canonical-name alias binding when the model renames the entry point (exactly
one top-level def; fail closed on zero/multiple); self-test 16 cases. Kill
rule unchanged (delta < +0.010).

Pre-registered kill: accuracy_B - accuracy_A < 0.010 (fired => no promotion).
Any infrastructure failure -> BLOCKED_INFRASTRUCTURE, no efficacy verdict.
Sagnac veto: not_applicable (no Sagnac veto machinery in this causal path).
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import pathlib
import random
import re
import subprocess
import sys
import time

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # <wt> root (script at experiments/verification/aaiv41_grounding_20260822/)
for _p in (str(SCRIPT_DIR), str(REPO_ROOT / "HENRI V2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from henri_backbone_adapter import QwenBackboneAdapter  # noqa: E402
from henri_backbone_retrieval import (  # noqa: E402
    BackboneRetrieval,
    RetrievalBlockedError,
    add_contamination_shingles,
)

# --------------------------------------------------------------------------
# Immutable dataset digests (full SHA-256; packet CLASS51 P3 / digest json)
# --------------------------------------------------------------------------
HE_GZ_SHA = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
HE_RAW_SHA = "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2"
MBPP_LOCAL_SHA = "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9"
MBPP_BLOB_SHA1 = "a999d25df7e489e4c183cadfa5c0ecf557a702b6"
MBPP_N = 100
HE_N = 164

REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 384

# Prompt pairing contract (canonical base; B = base + block at sentinel)
BASE_HEADER = "You are solving a programming task. Provide only the final Python solution.\n\n"
TASK_MARKER = "### Task\n"
RETRIEVAL_MARKER = "### Retrieved reference material (provenance-tagged; not task data)\n"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def _repo_root() -> pathlib.Path:
    """Worktree root found by upward .git search (worktrees carry a .git FILE)."""
    p = SCRIPT_DIR
    for _ in range(8):
        if (p / ".git").exists():
            return p
        p = p.parent
    return SCRIPT_DIR


# --------------------------------------------------------------------------
# Graders: result class in {PASSED, FAILED, EXECUTION_ERROR}
# --------------------------------------------------------------------------
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _classify(proc: subprocess.CompletedProcess) -> tuple[str, str]:
    if proc.returncode == 0:
        return "PASSED", ""
    err = (proc.stderr or "") + (proc.stdout or "")
    if "AssertionError" in err:
        return "FAILED", err[-300:]
    return "EXECUTION_ERROR", err[-300:]


def _mbpp_signature(code: str) -> str:
    """Canonical open def signature from the dataset reference code."""
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("def "):
            return s
    return ""


def _hoist_leading_imports(code: str) -> str:
    """Move leading import/from lines (and comments) to module scope.

    R3: a completion that opens with 'import math' and then a def would put
    the import inside the function after signature concatenation; hoisting
    prevents the OBSERVED IndentationError. Only LEADING lines are hoisted;
    imports inside the function body are left untouched.
    """
    lines = code.splitlines()
    head: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith(("#", "import ", "from ")):
            head.append(s)
            i += 1
            continue
        break
    if not head or i >= len(lines):
        return code
    return "\n".join(head) + "\n\n" + "\n".join(lines[i:]).strip("\n") + "\n"


def _has_def_head(code: str) -> bool:
    """True if the first non-blank, non-import, non-comment line is 'def '."""
    for ln in code.splitlines():
        s = ln.strip()
        if not s or s.startswith(("#", "import ", "from ")):
            continue
        return s.startswith("def ")
    return False


def _extract_fenced_code(code: str) -> str:
    """Deterministic Markdown-fence extraction (R4).

    If the text contains fenced code blocks, return the content of the LAST
    complete block (```lang ... ```). Unterminated fences are closed at EOF.
    No fences -> return the input unchanged.
    """
    lines = code.splitlines()
    fences: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("```"):
            start = i
            j = i + 1
            while j < len(lines) and lines[j].strip() != "```":
                j += 1
            fences.append((start, j))  # j == len(lines) means unterminated
            i = j + 1
        else:
            i += 1
    if not fences:
        return code
    start, close = fences[-1]
    return "\n".join(lines[start + 1:close]).strip()


def _canonical_mbpp_name(test_list: list[str], signature: str = "") -> str | None:
    """Canonical entry point: signature def name FIRST (dataset truth).

    Falls back to the test surface ONLY when no signature is available.
    The assert-based fallback must NOT run when a signature exists: the
    first identifier after 'assert ' is frequently a BUILTIN (e.g.
    'assert set(similar_elements(...))' -> 'set'), which would poison the
    test namespace when used as an alias target (OBSERVED R4 preflight).
    """
    if signature:
        m = re.match(r"\s*def\s+([A-Za-z_]\w*)", signature)
        if m:
            return m.group(1)
    tl = "\n".join(test_list or [])
    m = re.search(r"\bassert\s+([A-Za-z_]\w*)\s*\(", tl)
    if m:
        return m.group(1)
    m = re.search(r"\bcheck\(\s*([A-Za-z_]\w*)\s*\)", tl)
    return m.group(1) if m else None


def _alias_binding_source(src: str, canon: str) -> str:
    """Alias `canon` to the UNIQUE top-level generated def; fail closed otherwise.

    Uses ast.parse: exactly one top-level FunctionDef/AsyncFunctionDef in the
    module body. Zero, multiple, or parse errors -> no alias (the tests will
    then fail if the canonical name is genuinely absent).
    """
    if not canon:
        return ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    defs = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(defs) == 1 and defs[0].name != canon:
        return f"{canon} = {defs[0].name}\n"
    return ""


def grade_humaneval(code: str, entry_point: str, tests: str,
                    imports: list[str], prompt: str, timeout: int = 15) -> tuple[str, str]:
    if not _IDENT_RE.fullmatch(entry_point):
        return "EXECUTION_ERROR", f"invalid entry_point identifier: {entry_point!r}"
    imp = "\n".join(imports) if imports else ""
    # R2: official prompt + completion + tests form (canonical prompt prepended)
    # R4: deterministic fence extraction before grading.
    body = f"{imp}\n\n{prompt}\n\n{_extract_fenced_code(code)}\n\n{tests}\n\ncheck({entry_point})\n"
    try:
        proc = subprocess.run([sys.executable, "-c", body],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "EXECUTION_ERROR", "TIMEOUT"
    return _classify(proc)


def grade_mbpp(code: str, test_imports: list[str], test_list: list[str],
               prompt: str, signature: str, timeout: int = 15) -> tuple[str, str]:
    imp = "\n".join(test_imports) if test_imports else ""
    tests = "\n".join(test_list) if test_list else ""
    # MBPP prompt is natural-language prose (NOT Python) and must NOT be
    # concatenated into the graded source (SyntaxError, OBSERVED preflight).
    # R3: grade the completion AS GENERATED (its own def head binds the
    # model's argument names). Leading imports hoisted to module scope.
    # Body-only completions (no def head) fall back to the canonical
    # dataset signature.
    # R4: fence extraction -> import hoist -> own-def binding -> canonical alias.
    completion = _hoist_leading_imports(_extract_fenced_code(code))
    if not _has_def_head(completion) and signature:
        completion = f"{signature}\n{completion}"
    canon = _canonical_mbpp_name(test_list, signature)
    alias = _alias_binding_source(completion, canon)
    body = f"{imp}\n\n{completion}\n\n{alias}\n\n{tests}\n"
    try:
        proc = subprocess.run([sys.executable, "-c", body],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "EXECUTION_ERROR", "TIMEOUT"
    return _classify(proc)


# --------------------------------------------------------------------------
# Dataset loaders (digest-asserted)
# --------------------------------------------------------------------------
def load_humaneval(gz_path: pathlib.Path) -> list[dict]:
    raw_gz = gz_path.read_bytes()
    assert sha256_bytes(raw_gz) == HE_GZ_SHA, "HumanEval gz digest mismatch"
    raw = gzip.decompress(raw_gz)
    assert sha256_bytes(raw) == HE_RAW_SHA, "HumanEval decompressed digest mismatch"
    items = []
    for line in raw.decode("utf-8").splitlines():
        obj = json.loads(line)
        test_field = obj["test"]
        test = test_field if isinstance(test_field, str) else "\n".join(test_field)
        items.append({
            "task_id": obj["task_id"],
            "prompt": obj["prompt"],
            "entry_point": obj["entry_point"],
            "test": test,
            "imports": obj.get("imports", []),
            "dataset": "humaneval",
        })
    assert len(items) == HE_N, f"HumanEval item count {len(items)} != {HE_N}"
    return items


def load_mbpp(json_path: pathlib.Path) -> list[dict]:
    data = json_path.read_bytes()
    assert sha256_bytes(data) == MBPP_LOCAL_SHA, "MBPP local digest mismatch"
    assert git_blob_sha1(data) == MBPP_BLOB_SHA1, "MBPP git blob id mismatch"
    full = json.loads(data)
    assert len(full) >= MBPP_N, f"MBPP has {len(full)} items < {MBPP_N}"
    items = []
    for obj in full[:MBPP_N]:
        items.append({
            "task_id": f"mbpp-{obj['task_id']}",
            "prompt": obj["prompt"],
            "code": obj["code"],
            "signature": _mbpp_signature(obj["code"]),
            "test_imports": obj.get("test_imports", []),
            "test_list": obj.get("test_list", []),
            "source_file": obj.get("source_file", ""),
            "dataset": "mbpp",
        })
    ids = [it["task_id"] for it in items]
    assert len(set(ids)) == MBPP_N, "MBPP task_ids are not unique"
    items[0]["_ordered_digest"] = sha256_bytes("\n".join(ids).encode())
    return items


# --------------------------------------------------------------------------
# Prompt pairing (single canonical base; B = A + retrieval block)
# --------------------------------------------------------------------------
def build_arm_a_prompt(task_prompt: str) -> str:
    return BASE_HEADER + TASK_MARKER + task_prompt


def build_arm_b_prompt(task_prompt: str, snippets: list[dict]) -> tuple[str, str]:
    parts = [RETRIEVAL_MARKER]
    for s in snippets:
        parts.append(
            f"[source: {s['source_file']} sha256:{s['sha256'][:12]} "
            f"score:{s['bm25_score']}]\n{s['snippet']}"
        )
    block = "\n\n".join(parts)
    return BASE_HEADER + block + "\n\n" + TASK_MARKER + task_prompt, block


def strip_block(prompt_b: str) -> str | None:
    i = prompt_b.find(RETRIEVAL_MARKER)
    j = prompt_b.find(TASK_MARKER, i if i >= 0 else 0)
    if i < 0 or j < 0 or j <= i:
        return None
    return prompt_b[:i] + prompt_b[j:]


# --------------------------------------------------------------------------
# Self-test (pre-registered; runs BEFORE model load; any mismatch => exit 2)
# --------------------------------------------------------------------------
def run_self_test(he_gz: pathlib.Path, mbpp_json: pathlib.Path) -> dict:
    he = load_humaneval(he_gz)
    mbpp = load_mbpp(mbpp_json)
    cases = []

    # --- HumanEval: HE/2 truncate_number (R2: canonical prompt passed) ---
    it2 = he[2]
    good_he = "def truncate_number(number):\n    return number - int(number)\n"
    wrong_he = "def truncate_number(number):\n    return number + int(number)\n"
    cases.append(("HE known-good", "PASSED",
                  grade_humaneval(good_he, "truncate_number", it2["test"], it2["imports"], it2["prompt"])))
    cases.append(("HE deliberate-wrong", "FAILED",
                  grade_humaneval(wrong_he, "truncate_number", it2["test"], it2["imports"], it2["prompt"])))
    cases.append(("HE wrong-entry", "EXECUTION_ERROR",
                  grade_humaneval(good_he, "truncate_nmbr", it2["test"], it2["imports"], it2["prompt"])))
    cases.append(("HE timeout", "EXECUTION_ERROR",
                  grade_humaneval(
                      "def truncate_number(number):\n    while True:\n        pass\n",
                      "truncate_number", it2["test"], it2["imports"], it2["prompt"], timeout=2)))
    cases.append(("HE body-only-with-prefix", "PASSED",
                  grade_humaneval("    return number - int(number)\n", "truncate_number",
                                  it2["test"], it2["imports"], it2["prompt"])))

    # --- MBPP: item 0 similar_elements (R2: prompt + canonical signature) ---
    m0 = mbpp[0]
    sig0 = _mbpp_signature(m0["code"])
    good_mb = ("def similar_elements(test_tup1, test_tup2):\n"
               "    return tuple(set(test_tup1) & set(test_tup2))\n")
    wrong_mb = ("def similar_elements(test_tup1, test_tup2):\n"
                "    return tuple(set(test_tup1) | set(test_tup2))\n")
    cases.append(("MBPP known-good", "PASSED",
                  grade_mbpp(good_mb, m0["test_imports"], m0["test_list"], m0["prompt"], sig0)))
    cases.append(("MBPP deliberate-wrong", "FAILED",
                  grade_mbpp(wrong_mb, m0["test_imports"], m0["test_list"], m0["prompt"], sig0)))
    cases.append(("MBPP syntax-error-body", "EXECUTION_ERROR",
                  grade_mbpp("def similar_elements(a, b):\n    if True print(1)\n",
                             m0["test_imports"], m0["test_list"], m0["prompt"], sig0)))
    cases.append(("MBPP timeout", "EXECUTION_ERROR",
                  grade_mbpp("def similar_elements(a, b):\n    while True:\n        pass\n",
                             m0["test_imports"], m0["test_list"], m0["prompt"], sig0, timeout=2)))
    cases.append(("MBPP body-only-with-prefix", "PASSED",
                  grade_mbpp("    return tuple(set(test_tup1) & set(test_tup2))\n",
                             m0["test_imports"], m0["test_list"], m0["prompt"], sig0)))
    cases.append(("MBPP leading-import completion", "PASSED",
                  grade_mbpp("import math\n\ndef similar_elements(test_tup1, test_tup2):\n"
                             "    return tuple(set(test_tup1) & set(test_tup2))\n",
                             m0["test_imports"], m0["test_list"], m0["prompt"], sig0)))
    cases.append(("MBPP descriptive-args own-def", "PASSED",
                  grade_mbpp("def similar_elements(lst1, lst2):\n"
                             "    return tuple(set(lst1) & set(lst2))\n",
                             m0["test_imports"], m0["test_list"], m0["prompt"], sig0)))
    cases.append(("MBPP renamed-function alias", "PASSED",
                  grade_mbpp("def find_shared_elements(lst1, lst2):\n"
                             "    return tuple(set(lst1) & set(lst2))\n",
                             m0["test_imports"], m0["test_list"], m0["prompt"], sig0)))
    cases.append(("MBPP renamed-function wrong-body", "FAILED",
                  grade_mbpp("def find_shared_elements(lst1, lst2):\n"
                             "    return tuple(set(lst1) | set(lst2))\n",
                             m0["test_imports"], m0["test_list"], m0["prompt"], sig0)))
    cases.append(("MBPP fenced completion", "PASSED",
                  grade_mbpp("```python\ndef similar_elements(test_tup1, test_tup2):\n"
                             "    return tuple(set(test_tup1) & set(test_tup2))\n```",
                             m0["test_imports"], m0["test_list"], m0["prompt"], sig0)))
    cases.append(("MBPP fenced leading-import", "PASSED",
                  grade_mbpp("```python\nimport math\n\ndef similar_elements(test_tup1, test_tup2):\n"
                             "    return tuple(set(test_tup1) & set(test_tup2))\n```",
                             m0["test_imports"], m0["test_list"], m0["prompt"], sig0)))

    results = []
    ok = True
    for name, expected, (actual, detail) in cases:
        results.append({"case": name, "expected": expected, "actual": actual,
                        "pass": actual == expected,
                        "detail": (detail[:120] if actual != "PASSED" else "")})
        ok = ok and actual == expected
    return {"pass": ok, "cases": results,
            "evaluator_source_sha256": sha256_bytes(
                pathlib.Path(__file__).read_bytes()),
            "template_sha256": sha256_bytes(
                (BASE_HEADER + TASK_MARKER + RETRIEVAL_MARKER).encode())}


# --------------------------------------------------------------------------
# Contamination (fresh, full 264 surface)
# --------------------------------------------------------------------------
def contamination_scan(items: list[dict], corpus_dir: pathlib.Path,
                       out: pathlib.Path, run_id: str, detector_commit: str) -> list[str]:
    for it in items:
        add_contamination_shingles(it["prompt"])
        if it["dataset"] == "humaneval":
            add_contamination_shingles(it["test"])
        else:
            add_contamination_shingles("\n".join(it["test_list"]))
            add_contamination_shingles("\n".join(it["test_imports"]))
    retr = BackboneRetrieval(corpus_dir, enabled=True)
    hits = retr.scan_contamination()
    detector = {
        "version": "v3.2-code-dominant",
        "rule": ("code-bearing line (underscore | code-dominant keyword | '>>>' | '=') "
                 "or ident4 shingle containing an underscore token or code-dominant keyword; "
                 "v3.2 excludes bare import lines and prose-common single-keyword signals"),
        "commit": detector_commit,
        "amendment": "A2-RATIFIED",
        "surface_items": len(items),
    }
    (out / f"contamination_receipt_{run_id}.json").write_text(json.dumps({
        "schema_id": "henri.contamination-receipt.v1",
        "status": "CLEAN" if not hits else "CONTAMINATION_BLOCKED",
        "hits": hits,
        "corpus_aggregate": "2ced6f6f1afefb4d54129b9fff74d4edf08a91efe38c4b639ef8301144786f04",
        "detector": detector,
        "run_id": run_id,
    }, indent=2))
    return hits


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
def mcnemar_exact_one_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    total = 0.0
    for k in range(b, n + 1):
        total += math.comb(n, k) * (0.5 ** n)
    return min(1.0, total)


def bootstrap_ci(deltas: list[int], n_iter: int = 10000,
                 seed: int = 20260822) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_iter):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return round(means[int(0.025 * n_iter)], 4), round(means[int(0.975 * n_iter)], 4)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["preflight", "run"], default="preflight")
    ap.add_argument("--arm", choices=["AB", "A"], default="AB",
                    help="AB: paired matrix (default). A: backbone-baseline only "
                         "(retrieval corpus unused -> contamination gate N/A).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--humaneval-gz", required=True)
    ap.add_argument("--mbpp-json", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--commit", default=os_environ("MATRIX_COMMIT", "unknown"))
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    he = load_humaneval(pathlib.Path(args.humaneval_gz))
    mbpp = load_mbpp(pathlib.Path(args.mbpp_json))
    items = he + mbpp
    assert len(items) == 264, f"items {len(items)} != 264"

    # pairing verifier (B minus block == A, all 264); N/A in A-only mode
    retr = None
    pairing_ok = None
    if args.arm == "AB":
        retr = BackboneRetrieval(pathlib.Path(args.corpus_dir), enabled=True)
        pairing_ok = True
        for it in items:
            a = build_arm_a_prompt(it["prompt"])
            snips = retr.retrieve(it["prompt"])
            if not snips:
                pairing_ok = False
                continue
            b, block = build_arm_b_prompt(it["prompt"], snips)
            stripped = strip_block(b)
            if stripped != a:
                pairing_ok = False
        print(f"[matrix] pairing_verifier={pairing_ok} items={len(items)}")
    else:
        print("[matrix] arm=A-only: retrieval corpus unused; pairing N/A")

    # contamination (full surface, fresh); N/A for A-only (corpus unused)
    hits = []
    if args.arm == "AB":
        hits = contamination_scan(items, pathlib.Path(args.corpus_dir), out, run_id,
                                  args.commit)
        print(f"[matrix] contamination hits={hits}")
    else:
        (out / f"contamination_receipt_{run_id}.json").write_text(json.dumps({
            "schema_id": "henri.contamination-receipt.v1",
            "status": "NOT_APPLICABLE_ARM_A_ONLY",
            "hits": [],
            "corpus_aggregate": "2ced6f6f1afefb4d54129b9fff74d4edf08a91efe38c4b639ef8301144786f04",
            "detector": {"version": "v3.2-code-dominant", "amendment": "A2-RATIFIED",
                         "note": "retrieval corpus unused in arm A-only mode",
                         "surface_items": len(items)},
            "run_id": run_id}, indent=2))
        print("[matrix] arm=A-only: contamination gate NOT_APPLICABLE (corpus unused)")

    # self-test
    st = run_self_test(pathlib.Path(args.humaneval_gz),
                        pathlib.Path(args.mbpp_json))
    print(f"[matrix] self_test={st['pass']}")
    (out / f"self_test_{run_id}.json").write_text(json.dumps(st, indent=2))
    if not st["pass"]:
        print("[matrix] SELF_TEST_FAILED")
        return 2

    # manifest seal
    manifest = {
        "schema_id": "henri.class51-p3-matrix-manifest.v1",
        "run_id": run_id,
        "commit": args.commit,
        "datasets": {
            "humaneval": {"items": HE_N, "gz_sha256": HE_GZ_SHA,
                          "decompressed_sha256": HE_RAW_SHA,
                          "source": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"},
            "mbpp": {"items": MBPP_N, "local_sha256": MBPP_LOCAL_SHA,
                     "git_blob_sha1": MBPP_BLOB_SHA1,
                     "source": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
                     "task_ids": [it["task_id"] for it in mbpp],
                     "ordered_digest": mbpp[0]["_ordered_digest"]},
        },
        "arm_mode": args.arm,
        "pairing": {"ok": pairing_ok, "template_sha256": st["template_sha256"]},
        "self_test": st["pass"],
        "contamination": (("CLEAN" if not hits else "CONTAMINATION_BLOCKED")
                          if args.arm == "AB" else "NOT_APPLICABLE_ARM_A_ONLY"),
        "model": {"id": MODEL_ID, "revision": REVISION,
                  "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False},
    }
    (out / "matrix264_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[matrix] manifest sealed")

    if args.mode == "preflight":
        if hits:
            print("[matrix] PREFLIGHT_BLOCKED (contamination)")
            return 1
        print("[matrix] PREFLIGHT_PASS")
        return 0

    # ---- run mode: requires model -------------------------------------
    if hits:
        print("[matrix] CONTAMINATION_BLOCKED, abort")
        return 1
    if args.arm == "AB" and not pairing_ok:
        print("[matrix] PAIRING_FAILED, abort")
        return 1
    if args.model_dir is None or args.manifest is None:
        print("[matrix] run mode requires --model-dir and --manifest")
        return 2

    print("[matrix] loading model ...")
    t0 = time.time()
    adapter = QwenBackboneAdapter(
        model_dir=args.model_dir, manifest_path=args.manifest,
        revision=REVISION, verify_shards=True, max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False, temperature=None,
    ).load()
    print(f"[matrix] model loaded {time.time() - t0:.1f}s; "
          f"frozen={adapter.telemetry.trainable_params == 0}")
    assert adapter.telemetry.trainable_params == 0, "backbone not frozen"

    # pre-build arm prompts once (same order, same items)
    arm_a_prompts: list[str] = [build_arm_a_prompt(it["prompt"]) for it in items]
    arm_b_prompts: list[str] = []
    b_tel: list[dict] = []
    if args.arm == "AB":
        for it in items:
            snips = retr.retrieve(it["prompt"])
            if not snips:
                print(f"[matrix] RETRIEVAL_BLOCKED at {it['task_id']}")
                return 1
            b, block = build_arm_b_prompt(it["prompt"], snips)
            arm_b_prompts.append(b)
            b_tel.append({"retrieval_engaged": True,
                          "snippets": [s["source_file"] for s in snips],
                          "prompt_sha256": sha256_bytes(b.encode()),
                          "retrieval_block_bytes": len(block)})

    def run_arm(arm: str, prompts: list[str]) -> dict:
        rows = []
        passed = failed = exec_errors = 0
        gen_times = []
        for idx, (it, prompt) in enumerate(zip(items, prompts)):
            t0 = time.time()
            try:
                response, _ = adapter.generate_text(prompt)
            except Exception as exc:
                rows.append({"task_id": it["task_id"], "class": "EXECUTION_ERROR",
                             "error": f"GENERATION_ERROR: {exc}"[:300]})
                exec_errors += 1
                continue
            gen_s = time.time() - t0
            gen_times.append(gen_s)
            if it["dataset"] == "humaneval":
                cls, detail = grade_humaneval(normalize_answer(response),
                                              it["entry_point"], it["test"], it["imports"],
                                              it["prompt"])
            else:
                cls, detail = grade_mbpp(normalize_answer(response),
                                         it["test_imports"], it["test_list"],
                                         it["prompt"], it.get("signature", ""))
            rows.append({"task_id": it["task_id"], "dataset": it["dataset"],
                         "class": cls, "error": detail if cls != "PASSED" else None,
                         "gen_s": round(gen_s, 2),
                         "completion": response,
                         "prompt_sha256": sha256_bytes(prompt.encode())})
            if cls == "PASSED":
                passed += 1
            elif cls == "FAILED":
                failed += 1
            else:
                exec_errors += 1
        gt = sorted(gen_times)
        median = gt[len(gt) // 2] if gt else None
        p95 = gt[int(0.95 * len(gt))] if gt else None
        return {
            "arm": arm, "schema_id": "henri.run-evidence.v1",
            "kind": "diagnostic-efficacy", "not_official_aaii": True,
            "held_out_status": "CONDITIONAL",
            "metrics": {"item_count": len(prompts),
                        "passed": passed, "failed": failed,
                        "execution_errors": exec_errors, "vetoes": 0,
                        "attempted": passed + failed,
                        "accuracy": round(passed / len(prompts), 4),
                        "median_gen_s": round(median, 2) if median else None,
                        "p95_gen_s": round(p95, 2) if p95 else None,
                        "total_gen_s": round(sum(gen_times), 2)},
            "items": rows,
            "telemetry": {"model_id": MODEL_ID, "revision": REVISION,
                          "device": "cuda:0",
                          "frozen": adapter.telemetry.trainable_params == 0},
        }

    print("[matrix] running Arm A (frozen backbone, no retrieval) ...")
    rec_a = run_arm("A", arm_a_prompts)
    (out / f"matrix_arm_a_receipt_{run_id}.json").write_text(json.dumps(rec_a, indent=2))
    print(f"[matrix] Arm A: {rec_a['metrics']['passed']}/{rec_a['metrics']['attempted']} "
          f"failed={rec_a['metrics']['failed']} exec={rec_a['metrics']['execution_errors']}")

    # ---- Arm A-only baseline (corpus unused; B gated on A2 ratification) ----
    if args.arm == "A":
        m = rec_a["metrics"]
        arith_ok = (
            m["item_count"] == 264
            and m["passed"] + m["failed"] == m["attempted"]
            and m["attempted"] + m["execution_errors"] + m["vetoes"] == 264
        )
        print(f"[matrix] arithmetic_reconcile={arith_ok}")
        baseline = {
            "schema_id": "henri.class51-p3-backbone-baseline.v1",
            "run_id": run_id, "commit": args.commit, "arm": "A",
            "status": ("BACKBONE_BASELINE_COMPLETE"
                       if (arith_ok and m["execution_errors"] == 0 and st["pass"])
                       else "BLOCKED_INFRASTRUCTURE"),
            "metrics": m,
            "self_test": st["pass"],
            "datasets": manifest["datasets"],
            "arm_b": "BLOCKED pending A2 ratification (full-surface contamination gate)",
        }
        (out / "baseline_arm_a_verdict.json").write_text(json.dumps(baseline, indent=2))
        scorecard_a = {
            "schema_id": "henri.scorecard-arm-a-264.v1",
            "run_id": run_id, "commit": args.commit,
            "verdict": baseline, "items": rec_a["items"],
        }
        (out / "scorecard_arm_a_264.json").write_text(json.dumps(scorecard_a, indent=2))
        if baseline["status"] == "BACKBONE_BASELINE_COMPLETE":
            (out / "ARM_A_COMPLETE").write_text(json.dumps(baseline, indent=2))
            print("[matrix] ARM_A_COMPLETE written")
            return 0
        print("[matrix] BLOCKED_INFRASTRUCTURE (no ARM_A_COMPLETE)")
        return 1

    print("[matrix] running Arm B (frozen backbone + retrieval) ...")
    rec_b = run_arm("B", arm_b_prompts)
    (out / f"matrix_arm_b_receipt_{run_id}.json").write_text(json.dumps(rec_b, indent=2))
    print(f"[matrix] Arm B: {rec_b['metrics']['passed']}/{rec_b['metrics']['attempted']} "
          f"failed={rec_b['metrics']['failed']} exec={rec_b['metrics']['execution_errors']}")

    # arithmetic reconciliation (run-evidence.v1: passed+failed==attempted;
    # attempted+execution_errors+vetoes==item_count)
    m_a = rec_a["metrics"]
    m_b = rec_b["metrics"]
    arith_ok = (
        m_a["item_count"] == 264 and m_b["item_count"] == 264
        and m_a["passed"] + m_a["failed"] == m_a["attempted"]
        and m_b["passed"] + m_b["failed"] == m_b["attempted"]
        and m_a["attempted"] + m_a["execution_errors"] + m_a["vetoes"] == 264
        and m_b["attempted"] + m_b["execution_errors"] + m_b["vetoes"] == 264
    )
    print(f"[matrix] arithmetic_reconcile={arith_ok}")

    # paired deltas
    deltas = []
    rows = []
    for ra, rb in zip(rec_a["items"], rec_b["items"]):
        d = (1 if rb["class"] == "PASSED" else 0) - (1 if ra["class"] == "PASSED" else 0)
        deltas.append(d)
        rows.append({
            "task_id": ra["task_id"], "dataset": ra["dataset"],
            "arm_a": {"class": ra["class"], "gen_s": ra.get("gen_s"),
                      "error": ra.get("error")},
            "arm_b": {"class": rb["class"], "gen_s": rb.get("gen_s"),
                      "error": rb.get("error")},
            "delta": d,
            "retrieval": b_tel[len(rows)] if len(rows) < len(b_tel) else None,
        })
    b_over_a = sum(1 for d in deltas if d == 1)
    a_over_b = sum(1 for d in deltas if d == -1)
    delta_acc = m_b["accuracy"] - m_a["accuracy"]
    mcp = mcnemar_exact_one_sided(b_over_a, a_over_b)
    ci = bootstrap_ci(deltas)
    engaged_a = sum(1 for r in rows if r["arm_a"]["class"] != "EXECUTION_ERROR")
    engaged_b = sum(1 for r in rows if r["arm_b"]["class"] != "EXECUTION_ERROR")

    latency_kill = (m_b["total_gen_s"] > 1.5 * m_a["total_gen_s"]) if m_a["total_gen_s"] else None

    verdict = {
        "schema_id": "henri.class51-p3-matrix.v1",
        "run_id": run_id, "commit": args.commit,
        "status": "COMPLETE" if (arith_ok and m_a["execution_errors"] == 0
                                 and m_b["execution_errors"] == 0 and hits == []
                                 and pairing_ok and st["pass"]) else "BLOCKED_INFRASTRUCTURE",
        "datasets": manifest["datasets"],
        "arm_a": m_a, "arm_b": m_b,
        "paired": {"delta_b_minus_a": round(delta_acc, 4),
                   "mcnemar_exact_one_sided_p": round(mcp, 6),
                   "bootstrap_95_ci": ci,
                   "n_b_pass_a_fail": b_over_a, "n_a_pass_b_fail": a_over_b,
                   "n_264": len(deltas)},
        "gates": {"contamination": "CLEAN" if not hits else "CONTAMINATION_BLOCKED",
                  "pairing": pairing_ok, "self_test": st["pass"],
                  "arithmetic_reconcile": arith_ok,
                  "latency_ratio_1_5_kill": latency_kill,
                  "retrieval_engaged_b": f"{sum(1 for t in b_tel if t['retrieval_engaged'])}/264"},
        "kill_criterion": {"rule": "accuracy_B - accuracy_A < 0.010",
                           "value": round(delta_acc, 4),
                           "fired": delta_acc < 0.010},
        "sagnac_veto": "not_applicable",
        "scorecard_final": "telemetry/scorecard_final_264.json",
    }
    (out / "matrix_verdict.json").write_text(json.dumps(verdict, indent=2))

    scorecard = {
        "schema_id": "henri.scorecard-final-264.v1",
        "run_id": run_id, "commit": args.commit,
        "verdict": verdict,
        "items": rows,
    }
    (out / "scorecard_final_264.json").write_text(json.dumps(scorecard, indent=2))
    try:
        tel = _repo_root() / "telemetry"
        tel.mkdir(exist_ok=True)
        (tel / "scorecard_final_264.json").write_text(json.dumps(scorecard, indent=2))
        print(f"[matrix] scorecard delivered to {tel / 'scorecard_final_264.json'}")
    except Exception as exc:  # delivery must never mask the verdict
        print(f"[matrix] WARNING scorecard telemetry delivery failed: {exc}")
    print(json.dumps({k: verdict[k] for k in ("status", "arm_a", "arm_b", "paired",
                                              "kill_criterion", "gates")}, indent=2))

    if verdict["status"] == "COMPLETE":
        (out / "DONE").write_text(json.dumps(verdict, indent=2))
        print("[matrix] DONE written")
        return 0
    print("[matrix] BLOCKED_INFRASTRUCTURE (no DONE)")
    return 1


def normalize_answer(code: str) -> str:
    return _extract_fenced_code((code or "").strip())


def os_environ(k: str, dflt: str = "") -> str:
    import os
    return os.environ.get(k, dflt)


if __name__ == "__main__":
    sys.exit(main())
