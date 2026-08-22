"""Contract tests for the CLASS51 P3(a) retrieval layer.

CPU-only. Cover: default-OFF, provenance load, deterministic retrieval,
contamination gate, fail-closed paths, arm-prompt construction.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

import henri_backbone_retrieval as hbr
from henri_backbone_retrieval import (
    BackboneRetrieval,
    RetrievalBlockedError,
    add_contamination_shingles,
    build_arm_a_prompt,
)

REPO = pathlib.Path(__file__).resolve().parents[3]
CORPUS = REPO / "data" / "backbone_retrieval_corpus"


@pytest.fixture()
def retrieval():
    return BackboneRetrieval(CORPUS, enabled=True)


@pytest.fixture(autouse=True)
def clean_contamination():
    yield
    hbr._CONTAMINATION_LINES.clear()
    hbr._CONTAMINATION_IDENT4.clear()


def test_corpus_manifest_exists_and_valid():
    assert (CORPUS / "manifest.json").exists()
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "henri.corpus-manifest.v1"
    assert len(manifest["files"]) >= 13
    # per-file hash verification happens inside _load_corpus; exercise it
    r = BackboneRetrieval(CORPUS, enabled=True)
    assert r.enabled


def test_default_off_no_corpus_access():
    r = BackboneRetrieval(CORPUS, enabled=False)
    assert not r.enabled
    with pytest.raises(RetrievalBlockedError):
        r.build_prompt("x")


def test_deterministic_retrieval(retrieval):
    q = "sort items by key with functools cmp_to_key"
    a = retrieval.retrieve(q)
    b = retrieval.retrieve(q)
    assert a == b
    assert a, "expected at least one snippet"
    assert all("sha256" in s and "source_file" in s and "snippet" in s for s in a)


def test_provenance_tags_are_source_tracked(retrieval):
    q = "compute median of a list"
    snippets = retrieval.retrieve(q)
    assert snippets
    for s in snippets:
        src = CORPUS / s["source_file"]
        assert src.exists(), f"source file {s['source_file']} not in corpus"
        raw = src.read_bytes().replace(b"\r\n", b"\n")
        assert hashlib_sha256(raw).startswith(s["sha256"][:12])


def hashlib_sha256(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


def test_contamination_gate_fires(retrieval):
    # negative control: clean corpus must not fire
    assert retrieval.scan_contamination() == []
    # positive control: register a real code-bearing line from the corpus
    corpus_text = (CORPUS / "itertools.rst").read_text(encoding="utf-8")
    doctest_line = next(
        line.strip() for line in corpus_text.splitlines()
        if "list(factor(8))" in line
    )
    add_contamination_shingles(doctest_line)
    assert retrieval.scan_contamination(), "gate must detect corpus overlap"
    with pytest.raises(RetrievalBlockedError):
        retrieval.build_prompt("use itertools")


def test_bare_literals_do_not_fire_gate(retrieval):
    """Vacuous-detector regression: bare numeric/string literals are not
    contamination (classification evidence: '[2, 2, 2]' was a doctest OUTPUT
    in itertools.rst coinciding with a task data literal; verbatim-line total
    was 1 and benign)."""
    add_contamination_shingles("[2, 2, 2]")
    add_contamination_shingles("1 2 3 4 5")
    add_contamination_shingles("a b c d e")
    assert retrieval.scan_contamination() == []
    # retrieval still works
    snippets = retrieval.retrieve("sort items with bisect")
    assert snippets


def test_fail_closed_missing_manifest(tmp_path):
    with pytest.raises(RetrievalBlockedError):
        BackboneRetrieval(tmp_path, enabled=True)


def test_arm_a_prompt_no_retrieval():
    prompt, tel = build_arm_a_prompt("do the thing")
    assert tel["retrieval_engaged"] is False
    assert "Retrieved reference material" not in prompt


def test_arm_b_prompt_contains_provenance_block(retrieval):
    prompt, tel = retrieval.build_prompt("sort a list with bisect insort")
    assert tel["retrieval_engaged"] is True
    assert "### Retrieved reference material" in prompt
    assert "[source:" in prompt
    assert "### Task" in prompt
    assert tel["prompt_sha256"]


def test_arm_matching_prompt_delta(retrieval):
    """Arm A and B prompts must be byte-identical except the retrieval block."""
    task = "implement binary search with bisect"
    a_prompt, _ = build_arm_a_prompt(task)
    b_prompt, tel = retrieval.build_prompt(task)
    assert a_prompt != b_prompt
    # task section identical
    assert a_prompt.split("### Task")[1] == b_prompt.split("### Task")[1]
    # retrieval block present only in B
    assert "Retrieved reference material" not in a_prompt
    assert "Retrieved reference material" in b_prompt
