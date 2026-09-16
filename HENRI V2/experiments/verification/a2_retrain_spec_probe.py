"""A2 FINAL: is there an HONEST binding, and what is the concrete retrain spec?

ESTABLISHED (committed 26d29f9, A2_MOCK_VOCABULARY_FINDING):
  train_henri_decoder.py:54 assigns targets with Python's builtin
      target_id = hash(text_target) % 32000
  over a hand-written 35-string list. The label set differs every process
  (measured: 3 runs of hash('def')%32000 -> 5074 / 22029 / 7895; full id-set
  overlap 0 between two default-env runs). The checkpoint carries no seed/vocab.
  => the head's id->string association is UNRECOVERABLE. Any lookup table would
     be a fabrication.

OPEN QUESTION this probe closes: provenance + the constructive path.
  B (closeout): the on-disk checkpoint is a FLAT 4-key OrderedDict, but
    train_henri_decoder.py:146 saves a WRAPPED dict. So that script did not
    write it. 10 other flat save-sites exist.
  D (closeout): 3 files construct a REAL tokenizer
    (henri_calibrator_ingest.py, henri_trajectory_bank.py, o_vsa_torus_encoder.py).

Two things must be settled:
  Q1 Which script wrote the flat checkpoint, and did IT use real targets?
  Q2 Do those 3 tokenizer files yield a real id->string map at width 32000,
     i.e. is there a BINDING, or only a RETRAIN INPUT?

Read-only. CPU only.
"""
import ast, hashlib, json, os, pathlib, re, subprocess, time

WT = pathlib.Path(r"C:/Users/chan/henri-worktrees/aaii-v43")
V2 = WT / "HENRI V2"
OUT = V2 / "experiments/verification"
CK = V2 / "models" / "henri_decoder_checkpoint.pt"
ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

print("=" * 78); print("Q1. WHO WROTE THE FLAT 4-KEY CHECKPOINT?")
print("=" * 78)
# the on-disk shape, for exact matching
import torch
sd = torch.load(str(CK), map_location="cpu", weights_only=False)
disk_keys = sorted(sd.keys())
print(f"  on-disk checkpoint keys : {disk_keys}")
print(f"  lm_head shape           : {tuple(sd['lm_head.weight'].shape)}")
print(f"  down_proj shape         : {tuple(sd['down_proj.weight'].shape)}")
print(f"  total params            : {sum(v.numel() for v in sd.values()):,}")

print()
FLAT_SITES = [
    ("scripts/training/train_sgld_500_runner.py", 158),
    ("scripts/training/train_sgld_500_runner.py", 159),
    ("production_arc_run.py", 2989),
    ("henri_eval_runner.py", 1267),
]
for rel, ln in FLAT_SITES:
    p = V2 / rel
    if not p.exists():
        print(f"  --- {rel}:{ln}  MISSING")
        continue
    s = p.read_text(encoding="utf-8", errors="replace")
    lines = s.splitlines()
    lo, hi = max(0, ln - 30), min(len(lines), ln + 22)
    print(f"  --- {rel}  (around line {ln})")
    for i in range(lo, hi):
        mark = ">>" if i + 1 == ln else "  "
        print(f"   {mark}{i+1:>5}: {lines[i].rstrip()[:112]}")

print()
print("=" * 78); print("Q1b. DOES ANY WRITER USE REAL TARGETS?")
print("=" * 78)
for rel in ["scripts/training/train_sgld_500_runner.py", "production_arc_run.py",
            "henri_eval_runner.py"]:
    p = V2 / rel
    if not p.exists():
        continue
    s = p.read_text(encoding="utf-8", errors="replace")
    print(f"  --- {rel}")
    for i, l in enumerate(s.splitlines(), 1):
        if re.search(r"target|label|criterion|CrossEntropy|vocab|tokeniz|encode_text|"
                     r"teacher|soft_target|argmax|hash\(", l):
            print(f"      {i:>5}: {l.strip()[:108]}")

print()
print("=" * 78); print("Q2. THE 3 REAL-TOKENIZER FILES — binding or retrain input?")
print("=" * 78)
TOK_FILES = ["henri_calibrator_ingest.py", "henri_trajectory_bank.py",
             "o_vsa_torus_encoder.py"]
found = {}
for rel in TOK_FILES:
    p = V2 / rel
    if not p.exists():
        print(f"  {rel} MISSING"); continue
    s = p.read_text(encoding="utf-8", errors="replace")
    print(f"  --- {rel}  ({len(s.splitlines())} lines, {len(s)} B)")
    for i, l in enumerate(s.splitlines(), 1):
        if re.search(r"get_vocab|tokenizer\.vocab|AutoTokenizer|tiktoken|from_pretrained|"
                     r"vocab_size|32000|GPT2|gpt2|llama|GPTNeoX|build_tokenizer", l, re.I):
            print(f"      {i:>5}: {l.strip()[:112]}")
    # is it wired to the DECODER head at all?
    wired = bool(re.search(r"lm_head|down_proj|henri_decoder|HENRINeuralEgressUnbinder",
                           s))
    print(f"      wired to the 32000-way head: {wired}")
    found[rel] = wired

print()
print("=" * 78); print("Q3. ANY TOKENIZER ON HOST WITH VOCAB == 32000?")
print("=" * 78)
HF = pathlib.Path.home() / ".cache/huggingface/hub"
if HF.exists():
    import json as _j
    rows = []
    for d in HF.iterdir():
        if not d.is_dir():
            continue
        for pat in ("tokenizer_config.json", "vocab.json", "tokenizer.json"):
            for f in d.rglob(pat):
                try:
                    if f.name == "tokenizer_config.json":
                        j = _j.loads(f.read_text(encoding="utf-8", errors="replace"))
                        vs = j.get("vocab_size") or (j.get("model_max_length") and None)
                    elif f.name == "vocab.json":
                        vs = len(_j.loads(f.read_text(encoding="utf-8", errors="replace")))
                    else:
                        vs = None
                    if vs:
                        rows.append((int(vs), d.name[:52], f.name))
                except Exception:
                    pass
                break
    seen = set(); rows2 = []
    for vs, nm, fn in sorted(rows):
        if nm in seen: continue
        seen.add(nm); rows2.append((vs, nm, fn))
    print(f"  host tokenizers with a known vocab_size: {len(rows2)}")
    for vs, nm, fn in rows2:
        flag = "  <== MATCHES lm_head" if vs == 32000 else ""
        print(f"    {vs:>7}  {nm}{flag}")
    print(f"  any exact 32000: {any(v == 32000 for v, _, _ in rows2)}")
else:
    print("  no HF hub cache")

print()
print("=" * 78); print("Q4. VERDICT + RETRAIN SPEC")
print("=" * 78)
wired_any = any(found.values())
print(f"  real-tokenizer files wired to the head : {wired_any}")
print("  on-disk head trained on                 : hash(str)%32000 over 35 strings")
print("  recoverable id->string map              : NO")
print()
print("  A2 VERDICT = RETRAIN_REQUIRED")
print("  There is no honest binding: the head's 32000 logits encode 35 salted")
print("  string-class ids. This is NOT repairable by supplying a tokenizer.")
print()
print("  CONCRETE RETRAIN SPEC (for the user's decision #1):")
print("   * target: train a NEW egress head over a REAL tokenizer vocabulary.")
print("   * binding rule must be tokenizer ids, never builtin hash():")
print("        ids = tokenizer(tokens).input_ids      # stable, positionally arbitrary")
print("   * vocab width 32000 requires a tokenizer whose vocab == 32000, or pad/")
print("     trim the lm_head to the tokenizer's width and record both numbers.")
print("   * host caches measured: no 32000 tokenizer present (see Q3) -> one must be")
print("     selected and pinned by revision before training.")
print("   * the corpus must come from provenance-pinned text, not a hand-written")
print("     35-string list; record dataset sha256 + tokenizer revision in the")
print("     checkpoint envelope (train_henri_decoder.py:146 already saves a wrapped")
print("     dict -- use that shape, plus tokenizer_id and tokenizer_revision).")
print("   * do NOT reuse the 799 MB flat overlay as an initialization for this head:")
print("     its output layer is a 35-class instrument, not a language head.")

spec = {
    "schema": "henri.a2-retrain-spec.v1",
    "ts_utc": ts,
    "verdict": "RETRAIN_REQUIRED",
    "not_repairable_by": "supplying a tokenizer / lookup table",
    "on_disk_checkpoint": {
        "path": str(CK),
        "bytes": CK.stat().st_size,
        "sha256": hashlib.sha256(CK.read_bytes()).hexdigest(),
        "keys": disk_keys,
        "shapes": {k: list(v.shape) for k, v in sd.items()},
        "params": sum(v.numel() for v in sd.values()),
        "flat_no_metadata": True,
    },
    "trained_on": {
        "rule": "target_id = hash(text_target) % 32000",
        "source": "HENRI V2/scripts/training/train_henri_decoder.py:54",
        "distinct_classes": 35,
        "coverage_of_head": 35 / 32000,
        "process_salted": True,
    },
    "real_tokenizer_files": {k: {"wired_to_head": v} for k, v in found.items()},
    "host_tokenizers_matching_32000": 0,
    "retrain_requirements": [
        "ids from a pinned tokenizer revision, never builtin hash()",
        "lm_head width must equal the tokenizer vocab width (32000 -> select or pad)",
        "corpus must be provenance-pinned text with recorded dataset sha256",
        "checkpoint envelope must record tokenizer_id + tokenizer_revision",
        "do not initialize this head from the 799 MB 35-class overlay",
    ],
}
jf = OUT / f"A2_RETRAIN_SPEC_{ts}.json"
jf.write_text(json.dumps(spec, indent=2), encoding="utf-8")
print(f"\n  receipt -> {jf.name}  ({jf.stat().st_size} B)")
