"""Tests for the henri_mvp.py --managed flag (DEFAULT OFF).

Every test here can fail. The load-bearing ones:
  * default path is UNCHANGED -- encoding=="hash", no allocator, oov_rate 0.
  * --managed produces a DIFFERENT wave than hash on the same corpus.
  * --managed with oov=refuse FAILS CLOSED on a query outside the vocabulary.
  * --managed with oov=hash REPORTS oov_rate > 0 instead of hiding it.
  * the allocator digest is deterministic across two processes (D59: feature
    order changes accuracy at fixed geometry, so order is part of the identity).

Run:  python -u "HENRI V2/tests/unit/test_henri_mvp_managed.py"
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HENRI_V2 = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(HENRI_V2, ".."))
CORPUS = os.path.join(ROOT, "design", "zone_a", "evidence", "ak4_corpus")
MVP = os.path.join(HENRI_V2, "henri_mvp.py")

# restates fact_02 ("The block count for a full HENRI wave is 8192.")
IN_VOCAB_QUERY = "The block count for a full HENRI wave is 8192."
# shares no content word with any of the six facts.
OOV_QUERY = "The zorblax quux calibrates forty wibbles of spline vorticity."

FAILS = []


def _run(query, *extra):
    cmd = [sys.executable, "-u", MVP, "snap",
           "--corpus-dir", CORPUS, "--query", query]
    cmd.extend(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=HENRI_V2)
    try:
        return p.returncode, json.loads(p.stdout), p.stderr
    except json.JSONDecodeError:
        return p.returncode, None, (p.stdout or "") + (p.stderr or "")


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def test_default_path_is_unchanged():
    rc, d, err = _run(IN_VOCAB_QUERY)
    check("default_rc0", rc == 0, f"rc={rc} err={err[:200]}")
    check("default_encoding_hash", d and d.get("encoding") == "hash", f"{d}")
    check("default_no_allocator", d and d.get("allocator") is None, f"{d.get('allocator')}")
    check("default_oov_rate_zero", d and d.get("query_oov_rate") == 0.0,
          f"{d.get('query_oov_rate')}")
    check("default_accept_in_vocab", d and d.get("status") == "ACCEPT",
          f"status={d.get('status')} margin={d.get('margin')}")


def test_managed_changes_the_wave():
    _, dh, _ = _run(IN_VOCAB_QUERY)
    rc, dm, err = _run(IN_VOCAB_QUERY, "--managed")
    check("managed_rc0", rc == 0, f"rc={rc} err={err[:200]}")
    check("managed_encoding", dm and dm.get("encoding") == "managed", f"{dm}")
    a = (dm or {}).get("allocator") or {}
    check("managed_allocator_present", bool(a), f"{dm}")
    check("managed_collision_free", a.get("zero_collision_frac") == 1.0,
          f"zero_collision_frac={a.get('zero_collision_frac')}")
    check("managed_digest_present", bool(a.get("digest")), f"{a.get('digest')}")
    check("managed_layout_contiguous", a.get("layout") == "contiguous", f"{a.get('layout')}")
    # The whole point: the addressing changed. Margins must not be identical.
    if dh and dm and "margin" in dh and "margin" in dm:
        check("managed_margin_differs_from_hash",
              abs(dh["margin"] - dm["margin"]) > 1e-9,
              f"hash={dh['margin']} managed={dm['margin']}")


def test_managed_fails_closed_on_oov():
    rc, d, err = _run(OOV_QUERY, "--managed")
    check("oov_refuse_rc0", rc == 0, f"rc={rc}")
    check("oov_refuse_status", d and d.get("status") == "REFUSE", f"{d}")
    check("oov_refuse_encoding_managed", d and d.get("encoding") == "managed", f"{d}")
    check("oov_refuse_no_action", d and "typed_decision" not in d, f"{d}")


def test_managed_hash_escape_hatch_reports_oov():
    rc, d, err = _run(OOV_QUERY, "--managed", "--managed-oov", "hash")
    check("oov_hash_rc0", rc == 0, f"rc={rc} err={err[:200]}")
    check("oov_hash_reports_rate", d and d.get("query_oov_rate", 0) > 0.0,
          f"oov_rate={d.get('query_oov_rate') if d else None}")
    check("oov_hash_policy_label", d and d.get("query_oov_policy") == "hash",
          f"{d.get('query_oov_policy') if d else None}")


def test_allocator_digest_is_deterministic_across_processes():
    _, d1, _ = _run(IN_VOCAB_QUERY, "--managed")
    _, d2, _ = _run(IN_VOCAB_QUERY, "--managed")
    g1 = ((d1 or {}).get("allocator") or {}).get("digest")
    g2 = ((d2 or {}).get("allocator") or {}).get("digest")
    check("digest_deterministic", g1 and g1 == g2, f"{g1} vs {g2}")


def test_codec_is_byte_unchanged():
    """encode()/encode_egress() must not be touched by this wire-in."""
    p = subprocess.run(["git", "diff", "--stat", "--",
                        "HENRI V2/zone_c_world_knowledge_codec.py"],
                       capture_output=True, text=True, cwd=ROOT)
    check("codec_not_modified", p.stdout.strip() == "", f"diff={p.stdout.strip()}")


if __name__ == "__main__":
    for fn in (test_default_path_is_unchanged,
               test_managed_changes_the_wave,
               test_managed_fails_closed_on_oov,
               test_managed_hash_escape_hatch_reports_oov,
               test_allocator_digest_is_deterministic_across_processes,
               test_codec_is_byte_unchanged):
        fn()
    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {', '.join(FAILS)}")
        sys.exit(1)
    print("all managed-egress MVP tests passed")
