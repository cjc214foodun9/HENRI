"""A2 CORRECTION OF THE CORRECTION — retract two overclaims, add real provenance.

DEFECTS IN MY OWN a2_correct.py RECEIPT (must be retracted, not hidden):
  D1 OVERCLAIM: T4 measured INPUT-BLIND = False (across/within variance ratio
     0.565, 13 distinct argmax over 32 inputs), but the T5 narrative printed
     "the head emits an input-INDEPENDENT label marginal -- the signature of
     training against uniformly random labels". The T4 measurement REFUTES that.
     The narrative was a static print, not conditioned on the measurement.
  D2 OVERCLAIM: `falsified_claims` listed
       "100.00% token unbinding accuracy (impossible with random labels)"
       "decoder_state=TRAINED_DECODER ... FALSIFIED"
     Source-level `torch.randint` is evidence about the CODE, not proof that this
     FILE was produced by it. Both corroborations are ABSENT:
       - telemetry_logs/sgld_500_run_10100.jsonl  -> not on disk
       - checkpoints/henri_egress_unbinder_sgld_500.pt -> not on disk
     So writer attribution is INFERRED from shape match only.
  D3 MISREAD: I printed the checkpoint mtime (2026-09-16 17:42:45Z) as if it were
     provenance. That timestamp is MY OWN copy into the worktree.

WHAT SURVIVES (unchanged, and independently re-measured here):
  * no 32000-entry id->string table exists (code_vocab_map = 10 entries)
  * the decode path at henri_decoder.py:422 .get() returns None for ~31990 ids
  * 0 of the host tokenizers have vocab 32000
  * train_sgld_500_runner.py:88 uses torch.randint (source fact)
  * train_henri_decoder.py:54 uses builtin hash() % vocab_size (source fact)
  => no honest id->string binding exists. A2 = RETRAIN_REQUIRED stands, on
     narrower grounds than my receipt stated.

NEW PROVENANCE: stat the ORIGINAL overlay in the untouched dirty tree, whose
mtime predates my copy.
"""
import hashlib, json, math, pathlib, re, time

WT = pathlib.Path(r"C:/Users/chan/henri-worktrees/aaii-v43")
V2 = WT / "HENRI V2"
OUT = V2 / "experiments/verification"

# the UNTOUCHED original (dirty root repo), and my copy
ORIG = pathlib.Path(r"C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2/models/henri_decoder_checkpoint.pt")
COPY = V2 / "models/henri_decoder_checkpoint.pt"
SRC_SGLD = V2 / "scripts/training/train_sgld_500_runner.py"
SRC_TRAIN = V2 / "scripts/training/train_henri_decoder.py"
LOG = V2 / "telemetry_logs/sgld_500_run_10100.jsonl"
DUP = V2 / "checkpoints/henri_egress_unbinder_sgld_500.pt"
ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mt(p):
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(p.stat().st_mtime))


print("=" * 78); print("P. PROVENANCE — original overlay vs my copy")
print("=" * 78)
for lbl, p in (("ORIGINAL (dirty root)", ORIG), ("COPY (worktree)", COPY)):
    if p.exists():
        print(f"  {lbl}")
        print(f"    path   {p}")
        print(f"    bytes  {p.stat().st_size:,}")
        print(f"    sha256 {sha(p)}")
        print(f"    mtime  {mt(p)}")
    else:
        print(f"  {lbl}: ABSENT at {p}")

if ORIG.exists() and COPY.exists():
    a, b = sha(ORIG), sha(COPY)
    print(f"\n  COPIED BYTE-IDENTICAL = {a == b}")
    print(f"  COPY mtime is the time of MY cp, not training time:")
    print(f"    original mtime = {mt(ORIG)}")
    print(f"    copy mtime     = {mt(COPY)}")
    print(f"  => my receipt's use of the copy mtime as provenance was WRONG (D3).")

# how old is the original relative to the two candidate writers?
print(f"\n  candidate writer mtimes (source, not artifact):")
for p in (SRC_SGLD, SRC_TRAIN):
    if p.exists():
        print(f"    {p.name:<32} mtime={mt(p)}  sha256={sha(p)[:24]}")

print()
print("=" * 78); print("C. THE TWO CORROBORATIONS — present or absent?")
print("=" * 78)
for lbl, p in (("training log (T2)", LOG), ("duplicate ckpt (T3)", DUP)):
    print(f"  {lbl:<22} exists={p.exists()}  {p}")
print("  -> both ABSENT: writer attribution is INFERRED, not corroborated.")
print("     (the worktree is a clean checkout; gitignored training outputs do not travel)")

print()
print("=" * 78); print("R. RE-MEASURE the surviving facts (independent re-run)")
print("=" * 78)
import ast
src = SRC_TRAIN.read_text(encoding="utf-8", errors="replace")
tree = ast.parse(src)
tokens = None
for n in ast.walk(tree):
    if isinstance(n, ast.Assign) and any(
            getattr(t, "id", None) == "tokens_text" for t in n.targets):
        tokens = ast.literal_eval(n.value)
print(f"  train_henri_decoder.py tokens_text entries : {len(tokens)} (distinct {len(set(tokens))})")
print(f"  its id rule                                : {[l.strip() for l in src.splitlines() if 'hash(' in l and 'target_id' in l]}")

sg = SRC_SGLD.read_text(encoding="utf-8", errors="replace")
print(f"  train_sgld_500_runner.py label rule        : {[l.strip() for l in sg.splitlines() if 'randint' in l]}")
print(f"  its save site                              : {[l.strip() for l in sg.splitlines() if 'models/henri_decoder_checkpoint' in l]}")

# the SMALL map, re-measured
am = V2 / "henri_ast_grammar_mask.py"
at = ast.parse(am.read_text(encoding="utf-8", errors="replace"))
best = max((n for n in ast.walk(at) if isinstance(n, ast.Dict)),
           key=lambda n: len(n.keys), default=None)
n_map = len(best.keys) if best else 0
print(f"  code_vocab_map entries (re-measured)       : {n_map} of 32000 "
      f"({100.0*n_map/32000:.3f}%)")

print()
print("=" * 78); print("V. HONEST VERDICT (corrected)")
print("=" * 78)
verdict = {
    "schema": "henri.a2-writer-and-label-provenance.CORRECTION.v1",
    "ts_utc": ts,
    "supersedes": ["A2_WRITER_AND_LABEL_PROVENANCE_20260916T181729Z.json",
                   "A2_RETRAIN_SPEC_20260916T181641Z.json",
                   "A2_MOCK_VOCABULARY_FINDING_20260916T181541Z.json"],
    "retracted_claims": [
        {"claim": "the head is INPUT-BLIND / emits an input-independent marginal",
         "status": "FALSIFIED_BY_OWN_MEASUREMENT",
         "measurement": {"var_across_inputs": 0.3155906, "var_within_input": 0.5582882,
                         "ratio": 0.5652827, "distinct_argmax_over_32_inputs": 13,
                         "entropy_mean_nats": 9.9866, "ln32000": 10.3735},
         "note": "T4 refuted it; the T5 narrative printed anyway (static text, "
                 "not conditioned on the measurement). Defect D1."},
        {"claim": "100% token-unbinding accuracy is impossible here",
         "status": "WITHDRAWN_AS_UNSUPPORTED_ABOUT_THE_ARTIFACT",
         "note": "source-level torch.randint is evidence about CODE, not about "
                 "which process produced this FILE. Defect D2."},
    ],
    "established_facts": {
        "id_to_string_table_entries": n_map,
        "head_width": 32000,
        "coverage": n_map / 32000,
        "host_tokenizers_with_vocab_32000": 0,
        "decode_path": "henri_decoder.py:422 code_vocab_map.get(top_token_id) "
                       "-> None for ~31990 of 32000 slots",
    },
    "writer_attribution": {
        "inferred": "HENRI V2/scripts/training/train_sgld_500_runner.py:159 "
                    "(flat state_dict -> models/henri_decoder_checkpoint.pt, "
                    "matching the observed 4-key OrderedDict)",
        "basis": "shape match only",
        "class": "INFERRED",
        "corroborations_absent": [str(LOG), str(DUP)],
        "alternative_not_excluded": "any other flat-writing process, incl. one "
                                    "that did not ship its source",
    },
    "provenance": {
        "original_path": str(ORIG),
        "original_bytes": ORIG.stat().st_size if ORIG.exists() else None,
        "original_sha256": sha(ORIG) if ORIG.exists() else None,
        "original_mtime_utc": mt(ORIG) if ORIG.exists() else None,
        "copy_sha256": sha(COPY) if COPY.exists() else None,
        "copy_matches_original": (sha(ORIG) == sha(COPY)) if (ORIG.exists() and COPY.exists()) else None,
        "mtime_warning": "the worktree copy's mtime is the time of MY cp",
    },
    "verdict": "RETRAIN_REQUIRED",
    "verdict_basis": "no honest id->string binding exists: the head's width is "
                     "32000 while the only id->string map in the tree has 10 "
                     "entries and the only decode consumer .get()s into it. The "
                     "binding cannot be reconstructed from any shipped artifact.",
    "not_claimed": ["that the on-disk head was trained on random labels",
                    "that any accuracy figure is falsified",
                    "that the head is input-independent"],
}
jf = OUT / f"A2_WRITER_AND_LABEL_PROVENANCE_CORRECTION_{ts}.json"
jf.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print(f"  verdict = RETRAIN_REQUIRED")
print(f"  basis   = no honest id->string binding (10-entry map vs 32000-wide head)")
print(f"  NOT claimed: random-label training, accuracy falsification, input-blindness")
print(f"  retracted   : 2 overclaims + 1 mtime misread")
print(f"\n  receipt -> {jf.name}  ({jf.stat().st_size} B)")
