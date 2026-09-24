"""Real candidate sources for the SciCode scaffold arm.

WHY THIS MODULE EXISTS (measured 2026-09-24, own calls)
=======================================================
`s1_scicode_scaffold_runner.py:813` instantiates `NullCandidateSource()` unconditionally:
`produces_code=False`, `produce()` returns `""`. A whole-tree scan found `CandidateSource`
in only two files, BOTH harness. So the SciCode arm has NEVER scored a generator, and its
`pass@1 = 0/2` is the floor for an ABSENT generator -- not a measurement of anything.

That leaves the arm with no POSITIVE control. Every instrument defect this project has
found (M1 gate unsatisfiable, S1 never past the pin, the candidate off-by-one) shared one
signature: an instrument that could not return success. A scorer that has only ever
emitted 0.0 has not been shown capable of emitting anything else.

`BackboneCandidateSource` closes that. It drives a LOCAL pretrained backbone through the
existing generation stack and emits real code, so a run can show a non-zero numerator or
a clean `FALSIFIED_AT_SCAFFOLD`. It is a BACKBONE BASELINE, not a HENRI capability claim --
HENRI's own wave->text egress is a separate, still-unrepaired channel.

INFORMATION-FLOW CONTRACT (the load-bearing part)
-------------------------------------------------
The runner hands the source the WHOLE problem dict, which contains `ground_truth_code`,
`general_solution` and `test_cases` -- the ANSWER. A generator that reads those scores
against the answer key, and the G2 guard only catches >=60-character literal overlap, so
G2 is a backstop and not a defence. `sanitize_problem_view` is the defence: it returns a
whitelisted view and RAISES if any forbidden key survives into it. A source MUST take its
input through that function.

Self-test mode proves the contract without a model:
    python scicode_candidate_sources.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Keys that carry the reference answer or the grading oracle. A candidate view must
# never contain these. Kept as a module constant so a test can iterate it.
FORBIDDEN_VIEW_KEYS: Tuple[str, ...] = (
    "ground_truth_code",
    "general_solution",
    "test_cases",
    "solution",
    "reference_solution",
    "answer",
)

# Keys a candidate source legitimately needs. Everything else is dropped.
#
# MEASURED against the real corpus (problems_dev.jsonl, 15 problems / 50 sub-steps):
# the ACTUAL sub-step keys are
#   ['function_header', 'ground_truth_code', 'return_line', 'step_background',
#    'step_description_prompt', 'step_number', 'test_cases']
# An earlier version of this list named `step_description_special_comment` and
# `return_type` -- PHANTOM KEYS that occur in ZERO sub-steps, so two whitelist entries
# could never match while two real context fields were silently dropped. The whitelist
# is now checked against the corpus, not against the runner's prose.
ALLOWED_SUBSTEP_KEYS: Tuple[str, ...] = (
    "function_header",
    "step_description_prompt",
    "step_background",
    "return_line",
)

MODEL_ID_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS_DEFAULT = 512


class CandidateSourceUnavailable(RuntimeError):
    """Raised when the source cannot run. Callers must treat this as BLOCKED."""


def resolve_local_snapshot(model_id: str, cache_root: Optional[str] = None) -> Optional[pathlib.Path]:
    """Locate a usable local HF snapshot for `model_id`.

    HF hub stores a snapshot as SYMLINKS into `blobs/`, so a size or existence check that
    excludes symlinks reports a false negative -- a mistake this project already made once
    when sizing local weights. `_is_file` therefore resolves through `stat()`.
    """
    roots = [cache_root] if cache_root else [
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "huggingface", "hub"),
        os.path.join(os.path.expanduser("~"), "henri_data", "hf"),
    ]
    dirname = "models--" + model_id.replace("/", "--")
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        snaps = os.path.join(root, dirname, "snapshots")
        if not os.path.isdir(snaps):
            continue
        for rev in sorted(os.listdir(snaps), reverse=True):
            d = pathlib.Path(snaps) / rev
            if not d.is_dir():
                continue
            has_cfg = _is_file(d / "config.json")
            has_w = any(_is_file(p) for p in d.glob("*.safetensors"))
            has_w = has_w or any(_is_file(p) for p in d.glob("*.bin"))
            if has_cfg and has_w:
                return d
    return None


def _is_file(p: pathlib.Path) -> bool:
    """Exist-as-a-file, following symlinks (HF snapshots are symlinks into blobs/)."""
    try:
        return p.is_file()
    except OSError:
        return False


def feature_dependencies(problem: Dict[str, Any]) -> str:
    """Sanitized dependency reads. Empty string when absent (never invent an import)."""
    raw = problem.get("required_dependencies") or ""
    return str(raw).strip()


def sanitize_problem_view(problem: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Whitelisted view of one sub-step. Raises if any forbidden key survives.

    This runs on EVERY call, not once at construction, because the problem dict is a
    plain dict that could be mutated between steps.
    """
    if not isinstance(problem, dict):
        raise CandidateSourceUnavailable(f"problem must be a dict; got {type(problem).__name__}")
    sub_steps = problem.get("sub_steps") or []
    if idx < 0 or idx >= len(sub_steps):
        raise CandidateSourceUnavailable(f"sub-step index {idx} out of range (n={len(sub_steps)})")
    case = sub_steps[idx] or {}

    view: Dict[str, Any] = {
        "required_dependencies": feature_dependencies(problem),
        "function_header": str(case.get("function_header") or "").strip(),
        "step_description_prompt": str(case.get("step_description_prompt") or "").strip(),
        # Real corpus fields. `step_background` is legitimate problem context. `return_line`
        # is DELIBERATELY ABSENT: measured 50/50 sub-steps carry it VERBATIM inside
        # `ground_truth_code`, i.e. it IS a fragment of the answer. Prompting it would hand
        # the candidate the solution body it is supposed to write.
        "step_background": str(case.get("step_background") or "").strip(),
    }

    # The contract check: nothing forbidden may reach the model.
    leaked = [k for k in FORBIDDEN_VIEW_KEYS if k in view]
    if leaked:
        raise CandidateSourceUnavailable(f"sanitizer leaked forbidden key(s): {leaked}")
    for k in FORBIDDEN_VIEW_KEYS:
        if k in problem:
            # Present in the raw dict is expected and allowed -- it must simply not
            # travel. Assert we did not copy it.
            if k in view:
                raise CandidateSourceUnavailable(f"view carries {k!r}")
    return view


def build_prompt(view: Dict[str, Any], prior_outputs: List[str]) -> str:
    """Deterministic prompt from the sanitized view only."""
    parts: List[str] = []
    deps = view.get("required_dependencies") or ""
    if deps:
        parts.append(deps)
    prior = [p for p in (prior_outputs or []) if p and p.strip()]
    if prior:
        parts.append("# Already-implemented sub-steps\n" + "\n\n".join(prior))
    desc = view.get("step_description_prompt") or ""
    bg = view.get("step_background") or ""
    header = view.get("function_header") or ""
    if bg:
        parts.append("### Background\n" + bg)
    parts.append("### Task\n" + (desc or "Complete the function below."))
    if header:
        parts.append("### Function to complete\n```python\n" + header + "\n```")
    parts.append(
        "Return ONLY the complete Python function, in one ```python fenced block. "
        "Do not include tests, examples, or explanation."
    )
    return "\n\n".join(parts)


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_DEF = re.compile(r"^[ \t]*(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)


def _defines(block: str, name: str) -> bool:
    return bool(re.search(rf"^[ \t]*(?:async\s+)?def\s+{re.escape(name)}\s*\(", block, re.MULTILINE))


def extract_code(text: str, expect_name: Optional[str] = None) -> str:
    """Pull the generated function out of a model response.

    PREFERENCE ORDER (each step exists because the naive version got it wrong):
      1. the fenced block that defines `expect_name`        -- the requested function
      2. the first fenced block that defines ANY function   -- wrong-name output is
         RETURNED, not blanked. The generator really did emit a function; reporting it
         lets the harness fail as `CANDIDATE_UNDEFINED_SYMBOL:<expected>` and the receipt
         record `defines_expect=False`. Substituting "" would hide the generator's actual
         behaviour behind a different failure mode -- both score 0, but only one
         taxonomy tells us what to fix.
      3. the unfenced `def` in the raw text               -- a response whose only
         definition sits outside the fence (measured: an example fence `f(1)` ahead of a
         bare `def g(...)` must not win over the definition)
      4. ""                                               -- nothing usable

    THERE IS DELIBERATELY NO "first fenced block" FALLBACK. Measured defect: an earlier
    version returned the first fence unconditionally, so a shell block, a JSON blob or a
    prose fence would have been passed into the sandbox as the candidate payload. A
    payload that defines no function can only fail, and `""` is the honest return --
    the caller reports it as `no_code_emitted`, which is the accurate taxonomy for "the
    model produced no function" (distinct from "it produced the wrong function", step 2).
    """
    if not text:
        return ""
    blocks = [b.strip() for b in _FENCE.findall(text)]

    if expect_name:
        for b in blocks:
            if _defines(b, expect_name):
                return _strip_trailing_prose(b)

    for b in blocks:
        if _DEF.search(b):
            return _strip_trailing_prose(b)

    m = _DEF.search(text)
    if m:
        return _strip_trailing_prose(text[m.start():].strip())

    return ""


def _strip_trailing_prose(block: str) -> str:
    lines = block.splitlines()
    out: List[str] = []
    for ln in lines:
        stripped = ln.strip()
        # Drop prose that follows the code, but keep blank/comment/code lines.
        if stripped and not ln[:1].isspace() and not stripped.startswith(("#", "def ", "class ", "import ", "from ", "@")):
            if any(ch in stripped for ch in ("。",)) or stripped.endswith(".") and " " in stripped:
                break
        out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


class BackboneCandidateSource:  # structural match for CandidateSource
    """A LOCAL pretrained backbone as the candidate generator.

    BACKBONE BASELINE, NOT A HENRI CAPABILITY CLAIM. Its purpose is to prove the SciCode
    arm can score a real generator at all -- the positive control the arm never had.

    Deterministic by construction: `do_sample=False`, fixed `max_new_tokens`, no seed
    dependence. Two runs on the same item must emit identical bytes; the receipt records
    the model directory and revision so the number is attributable.
    """

    name = "BackboneCandidateSource"
    description = ("local pretrained backbone (transformers) generates each SciCode sub-step "
                   "from a sanitized view; BACKBONE BASELINE, not a HENRI capability claim")
    produces_code = True

    def __init__(self, model_id: str = MODEL_ID_DEFAULT, model_dir: Optional[str] = None,
                 max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT, device: Optional[str] = None,
                 verbose: bool = True) -> None:
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.verbose = verbose
        self._dir = pathlib.Path(model_dir) if model_dir else None
        self._model = None
        self._tok = None
        self._device = device
        self.emitted: List[Dict[str, Any]] = []
        self.load_error: Optional[str] = None
        # Deterministic memo. WHY IT IS NEEDED (measured from the runner's call pattern):
        # `prior_candidate = [source.produce((problem, k), []) for k in range(idx + 1)]`
        # regenerates steps 0..idx for EVERY window item, so a 3-step problem contributes
        # 1+2+3 = 6 calls for 3 distinct outputs. Generation is `do_sample=False`, so the
        # same (problem_id, idx) MUST yield identical bytes -- serving from cache is
        # output-identical and cuts the wall clock proportionally. Keyed on problem_id
        # with an object-identity fallback for dicts that lack it.
        self._cache: Dict[Tuple[str, int], str] = {}
        self.cache_hits = 0

    # -- provenance ---------------------------------------------------------
    def resolved_dir(self) -> pathlib.Path:
        if self._dir is not None and _is_file(self._dir / "config.json"):
            return self._dir
        found = resolve_local_snapshot(self.model_id)
        if found is None:
            raise CandidateSourceUnavailable(
                f"no local snapshot for {self.model_id!r}; set HENRI_SCICODE_BACKBONE_DIR. "
                f"HF cache probed at ~/.cache/huggingface/hub and %LOCALAPPDATA%/huggingface/hub")
        return found

    def revision(self) -> str:
        try:
            return self.resolved_dir().name
        except CandidateSourceUnavailable:
            return ""

    # -- model --------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise CandidateSourceUnavailable(
                f"generation stack unavailable ({type(exc).__name__}: {exc}); "
                f"validate on the CUDA target") from exc
        d = self.resolved_dir()
        # CPU needs float32: half precision on CPU is slow and partly unsupported.
        dtype = "auto"
        try:
            import torch
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            if self._device == "cpu":
                dtype = torch.float32
        except Exception:
            self._device = self._device or "cpu"
            dtype = "auto"
        try:
            self._tok = AutoTokenizer.from_pretrained(str(d), trust_remote_code=False)
            self._model = AutoModelForCausalLM.from_pretrained(
                str(d), dtype=dtype, trust_remote_code=False)
            self._model.to(self._device)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - environment dependent
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise CandidateSourceUnavailable(f"model load failed from {d}: {self.load_error}") from exc
        if self.verbose:
            print(f"  [backbone] loaded {self.model_id} from {d} device={self._device} "
                  f"max_new_tokens={self.max_new_tokens}", flush=True)

    # -- generation ---------------------------------------------------------
    def produce(self, item: Any, prior_outputs: List[str]) -> str:
        """`item` is `(problem, idx)` as the runner passes it."""
        try:
            problem, idx = item
        except Exception as exc:
            raise CandidateSourceUnavailable(f"item must be (problem, idx); got {item!r}") from exc

        view = sanitize_problem_view(problem, idx)

        ckey = (str(problem.get("problem_id") or id(problem)), int(idx))
        if ckey in self._cache:
            self.cache_hits += 1
            self.emitted.append({"idx": idx, "status": "CACHE_HIT",
                                 "chars": len(self._cache[ckey])})
            return self._cache[ckey]

        prompt = build_prompt(view, prior_outputs)
        header = view.get("function_header") or ""
        m = _DEF.search(header)
        expect = m.group(1) if m else None

        self.load()
        assert self._tok is not None and self._model is not None
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt
        try:
            import torch
            inputs = self._tok(text, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,          # deterministic
                    num_beams=1,
                    pad_token_id=(self._tok.pad_token_id or self._tok.eos_token_id),
                )
            gen = out[:, inputs["input_ids"].shape[1]:]
            gen_tokens = int(gen.shape[1])
            raw = self._tok.decode(gen[0], skip_special_tokens=True)
        except Exception as exc:
            self.emitted.append({"idx": idx, "status": "GENERATION_ERROR",
                                 "error": f"{type(exc).__name__}: {exc}"})
            if self.verbose:
                print(f"  [backbone] generation failed at idx={idx}: {exc}", flush=True)
            return ""

        # TRUNCATION DISCRIMINATOR. MEASURED DEFECT this exists to expose: item
        # SciCode-78-78.2 emitted 1307 chars and scored sub=SYNTAX_ERROR with
        # `'(' was never closed` at the last line -- the emission was CUT MID-EXPRESSION by
        # max_new_tokens=384. Without this flag the receipt cannot separate "the model
        # stopped" from "the budget ran out", so a truncated emission would be misreported
        # as a generator capability limit. `gen_tokens == max_new_tokens` means the model was
        # still generating when the cap hit.
        truncated = gen_tokens >= self.max_new_tokens
        code = extract_code(raw, expect_name=expect)
        self._cache[ckey] = code
        self.emitted.append({
            "idx": idx,
            "status": "EMITTED" if code.strip() else "NO_CODE_EMITTED",
            "chars": len(code),
            "raw_chars": len(raw),
            "gen_tokens": gen_tokens,
            "max_new_tokens": self.max_new_tokens,
            "truncated": truncated,
            "expect_name": expect,
            "defines_expect": bool(expect and re.search(
                rf"^[ \t]*(?:async\s+)?def\s+{re.escape(expect)}\s*\(", code, re.MULTILINE)),
        })
        if self.verbose:
            print(f"  [backbone] idx={idx} {self.emitted[-1]['status']} "
                  f"chars={len(code)} tokens={gen_tokens}"
                  f"{' TRUNCATED' if truncated else ''} "
                  f"defines_{expect}={self.emitted[-1]['defines_expect']}",
                  flush=True)
        return code

    # -- receipt ------------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        emitted = [e for e in self.emitted if e.get("status") == "EMITTED"]
        return {
            "name": self.name,
            "produces_code": self.produces_code,
            "model_id": self.model_id,
            "model_dir": str(self._dir) if self._dir else None,
            "resolved_dir": (str(self.resolved_dir()) if self._model is not None else None),
            "revision": self.revision(),
            "device": self._device,
            "max_new_tokens": self.max_new_tokens,
            "deterministic": True,
            "calls": len(self.emitted),
            "cache_hits": self.cache_hits,
            "generations": len(self.emitted) - self.cache_hits,
            "emitted": len(emitted),
            "truncated": sum(1 for e in self.emitted if e.get("truncated")),
            "no_code_emitted": sum(1 for e in self.emitted if e.get("status") == "NO_CODE_EMITTED"),
            "defines_expected": sum(1 for e in emitted if e.get("defines_expect")),
            "per_item": self.emitted[-24:],
            "evidence_class": "BACKBONE_BASELINE_NOT_HENRI_CAPABILITY",
            "load_error": self.load_error,
        }


def select_source(name: Optional[str] = None, verbose: bool = True):
    """Factory used by the runner. Default stays the Null source: the default path is
    byte-identical unless a caller explicitly opts in."""
    kind = (name if name is not None else os.environ.get("HENRI_SCICODE_CANDIDATE_SOURCE", "null"))
    kind = str(kind).strip().lower()
    if kind in ("", "null", "none", "absent", "0"):
        return None  # runner keeps NullCandidateSource
    if kind in ("backbone", "hf", "qwen"):
        model_id = os.environ.get("HENRI_SCICODE_BACKBONE_MODEL", MODEL_ID_DEFAULT)
        model_dir = os.environ.get("HENRI_SCICODE_BACKBONE_DIR") or None
        mnt = int(os.environ.get("HENRI_SCICODE_MAX_NEW_TOKENS", MAX_NEW_TOKENS_DEFAULT))
        return BackboneCandidateSource(model_id=model_id, model_dir=model_dir,
                                       max_new_tokens=mnt, verbose=verbose)
    raise CandidateSourceUnavailable(
        f"unknown HENRI_SCICODE_CANDIDATE_SOURCE={kind!r}; known: null, backbone")


def selftest() -> int:
    """Prove the leak contract without loading a model."""
    print("== selftest: sanitization contract ==")
    problem = {
        "required_dependencies": "import numpy as np",
        "ground_truth_code": "def f(): return 42  # THE ANSWER",
        "general_solution": "def f(): return 42",
        "sub_steps": [
            {"function_header": "def f(x):",
             "step_description_prompt": "Return x.",
             "test_cases": ["assert f(1) == 1"],
             "ground_truth_code": "def f(x): return x"},
        ],
    }
    view = sanitize_problem_view(problem, 0)
    print(f"  view keys           : {sorted(view)}")
    leaked = [k for k in FORBIDDEN_VIEW_KEYS if k in view]
    print(f"  forbidden keys leaked: {leaked or 'NONE'}")
    assert not leaked, leaked

    prompt = build_prompt(view, [])
    for k in FORBIDDEN_VIEW_KEYS:
        blob = str(problem.get(k, "")) + " " + str(problem["sub_steps"][0].get(k, ""))
        for frag in (frag for frag in re.findall(r"return \d+", blob)):
            assert frag not in prompt, f"prompt leaked {frag!r}"
    print(f"  prompt chars        : {len(prompt)}")
    print(f"  prompt leaks answer : False")

    print()
    print("== selftest: extraction ==")
    # (raw, expect_name, want_nonempty, want_defines_expect, why)
    cases = [
        ("prose\n```python\ndef g(a):\n    return a\n```\ntrailing", "g", True, True,
         "matching fence wins"),
        ("no fence\ndef g(a):\n    return a\n", "g", True, True,
         "unfenced def is accepted"),
        # MEASURED DEFECT this case pins: the first version returned "" here. A
        # wrong-name function is the generator's REAL output and must be returned so the
        # harness fails as CANDIDATE_UNDEFINED_SYMBOL:<expected> -- blanking it would
        # hide the behaviour behind a different failure mode.
        ("```python\ndef other():\n    pass\n```", "g", True, False,
         "wrong-name output is RETURNED, defines_expect=False"),
        # An example fence ahead of a bare definition must not win.
        ("```python\nf(1)\n```\ndef g(a):\n    return a\n", "g", True, True,
         "example fence loses to the real definition"),
        ("I cannot help with that.", "g", False, False, "prose only -> empty"),
        ("", "g", False, False, "empty input -> empty"),
        ("```python\nimport os\n```", "g", False, False, "no definition -> empty"),
    ]
    for raw, expect, want_nonempty, want_defines, why in cases:
        got = extract_code(raw, expect_name=expect)
        nonempty = bool(got.strip())
        defines = _defines(got, expect) if nonempty else False
        ok = (nonempty == want_nonempty) and (defines == want_defines)
        print(f"  raw={raw[:30]!r:34s} -> chars={len(got):3d} "
              f"defines_{expect}={defines!s:5s} {'OK' if ok else 'FAIL'}  ({why})")
        assert ok, (raw, got, want_nonempty, want_defines)
    print()
    print("SELFTEST_PASS")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SciCode candidate sources")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--resolve", metavar="MODEL_ID", nargs="?", const=MODEL_ID_DEFAULT)
    ap.add_argument("--generate", metavar="PROMPT", default=None,
                    help="smoke-generation one prompt through the backbone")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.resolve:
        d = resolve_local_snapshot(args.resolve)
        print(f"  model_id = {args.resolve}")
        print(f"  snapshot = {d}")
        if d:
            tot = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            print(f"  bytes    = {tot}  ({tot/1e9:.2f} GB, symlink-resolved)")
        return 0 if d else 1
    if args.generate is not None:
        src = BackboneCandidateSource()
        src.load()
        item = ({"required_dependencies": "", "sub_steps": [
            {"function_header": "def f(x):", "step_description_prompt": args.generate}]}, 0)
        code = src.produce(item, [])
        print("--- generated ---")
        print(code)
        print("--- report ---")
        import json
        print(json.dumps(src.report(), indent=1)[:1200])
        return 0 if code.strip() else 2
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
