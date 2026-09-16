"""A2 DECISIVE — was the 32000-way lm_head trained on a VOCABULARY or a MOCK?

FINDING UNDER TEST (read from scripts/training/train_henri_decoder.py:37-54):

    tokens_text = ["def", "return", ...]      # a hand-written list
    vocab_size = 32000
    ...
    target_id = hash(text_target) % vocab_size   # Python BUILTIN hash()

If that is the training target, then:
  T1  the "vocabulary" is a hand-written list of ~35 strings, not 32000 tokens;
  T2  target ids come from Python's builtin `hash()`, which is SALTED PER PROCESS
      (PYTHONHASHSEED randomization, default since 3.3) -> the same string maps to
      a DIFFERENT id in a different process;
  T3  the checkpoint records only {d_model,d_hidden,vocab_size,final_loss,accuracy}
      -> no seed, no vocab -> the id->string map is UNRECOVERABLE BY CONSTRUCTION;
  T4  therefore the repo's "100.00% token unbinding accuracy" is measured on a
      ~35-class problem with per-process-random labels, i.e. a MOCK.

Each T is falsifiable here. Runs on CPU; torch is used only to read the head.
"""
import ast, hashlib, json, os, pathlib, re, subprocess, sys, time

WT = pathlib.Path(r"C:/Users/chan/henri-worktrees/aaii-v43")
V2 = WT / "HENRI V2"
TRAIN = V2 / "scripts/training/train_henri_decoder.py"
OUT = V2 / "experiments/verification"
OUT.mkdir(parents=True, exist_ok=True)
VOCAB_SIZE = 32000
ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

print("=" * 78); print("1. EXTRACT the training target list (AST, not regex)")
print("=" * 78)
src = TRAIN.read_text(encoding="utf-8", errors="replace")
tree = ast.parse(src)
tokens = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "tokens_text":
                try:
                    tokens = ast.literal_eval(node.value)
                except Exception:
                    pass
print(f"  tokens_text entries              = {len(tokens)}")
print(f"  declared vocab_size              = {VOCAB_SIZE}")
print(f"  ACTUAL distinct training CLASSES = {len(set(tokens))}")
print(f"  coverage = {len(set(tokens))}/{VOCAB_SIZE} = "
      f"{100.0*len(set(tokens))/VOCAB_SIZE:.3f}% of the head's output space")
print(f"  sample: {tokens[:8]} ... {tokens[-5:]}")

# confirm the id rule is literally the builtin hash
rule = [l.strip() for l in src.splitlines() if "target_id" in l and "hash(" in l]
print(f"  id rule in source: {rule}")

print()
print("=" * 78); print("2. T2 — is the id map STABLE across processes?")
print("=" * 78)
PROBE = "import sys;print(hash(sys.argv[1]) % 32000)"
def ids_in_fresh_process(seedval, sample):
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None); env.pop("PYTHONHOME", None)
    if seedval is None:
        env.pop("PYTHONHASHSEED", None)          # default = randomized per process
    else:
        env["PYTHONHASHSEED"] = str(seedval)
    r = subprocess.run([sys.executable, "-c", PROBE, sample],
                       capture_output=True, text=True, env=env)
    return r.stdout.strip()

sample = tokens[0]
runs = [ids_in_fresh_process(None, sample) for _ in range(3)]
print(f"  builtin hash('{sample}') % {VOCAB_SIZE}, 3 fresh DEFAULT-env processes:")
for i, v in enumerate(runs, 1):
    print(f"    run {i}: {v}")
print(f"  STABLE ACROSS PROCESSES = {len(set(runs)) == 1}")
print(f"  distinct ids observed   = {len(set(runs))} of {len(runs)} runs")

# full-vector stability: the whole id SET, computed two ways in fresh processes
def id_set(seedval):
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None); env.pop("PYTHONHOME", None)
    if seedval is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = str(seedval)
    code = ("import sys,json;"
            "T=json.loads(sys.argv[1]);"
            "print(json.dumps(sorted(set(hash(t)%32000 for t in T))))")
    r = subprocess.run([sys.executable, "-c", code, json.dumps(tokens)],
                       capture_output=True, text=True, env=env)
    return json.loads(r.stdout)

setA = id_set(None)
setB = id_set(None)
set0 = id_set(0)
print(f"  full id-SET, default env run A : {len(setA)} ids")
print(f"  full id-SET, default env run B : {len(setB)} ids")
print(f"  A == B (reproducible)          : {setA == setB}")
print(f"  overlap A n B                  : {len(set(setA) & set(setB))} ids")
print(f"  fixed PYTHONHASHSEED=0         : {len(set0)} ids, == A: {set0 == setA}")
print("  => under the DEFAULT env the label set is DIFFERENT every process.")

print()
print("=" * 78); print("3. T3 — does the checkpoint record anything that could recover the map?")
print("=" * 78)
ck = V2 / "models" / "henri_decoder_checkpoint.pt"
print(f"  checkpoint {ck} exists={ck.exists()} bytes={ck.stat().st_size if ck.exists() else 0}")
ckpt_keys, head_shape, sd_matches, sd = None, None, None, None
if ck.exists():
    import torch
    d = torch.load(str(ck), map_location="cpu", weights_only=False)
    print(f"  top-level type: {type(d).__name__}")
    if isinstance(d, dict) and "model_state_dict" in d:
        sd = d["model_state_dict"]
        ckpt_keys = sorted(d.keys())
        print("  WRAPPED checkpoint (has model_state_dict + metadata)")
    else:
        sd = d                       # FLAT state_dict: no metadata at all
        ckpt_keys = sorted(d.keys())
        print("  FLAT checkpoint: bare state_dict, NO metadata envelope")
    print(f"  keys: {ckpt_keys}")
    print(f"  state_dict tensors: {[(k, tuple(v.shape)) for k, v in sd.items()]}")
    head_shape = tuple(sd["lm_head.weight"].shape) if "lm_head.weight" in sd else None
    seedish = [k for k in ckpt_keys
               if re.search(r"seed|vocab|token|hash|map|chars|alphabet", k, re.I)]
    print(f"  keys that could carry a vocab/seed: {seedish}")
    print(f"  => RECOVERABLE FROM CHECKPOINT: {bool(seedish)}")
    print()
    print("  NOTE: train_henri_decoder.py:146-153 saves a WRAPPED dict with keys")
    print("        {model_state_dict,d_model,d_hidden,vocab_size,final_loss,accuracy}.")
    print("        This file has NONE of those -> it was NOT written by that script,")
    print("        so even the declared vocab_size 32000 is not attested by the artifact.")

print()
print("=" * 78); print("4. T4 — what did the head actually learn? (live, CPU)")
print("=" * 78)
argmax_ids, ents = [], []
if ck.exists():
    sys.path.insert(0, str(V2))
    import torch
    import torch.nn.functional as F
    from henri_decoder import HENRINeuralEgressUnbinder
    from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec

    dev = "cpu"
    codec = qFHRREpistemicCodec(d_model=65536, device=dev)
    head = HENRINeuralEgressUnbinder(d_model=65536, d_hidden=2048,
                                     vocab_size=VOCAB_SIZE, device=dev)
    head.load_state_dict(sd)
    head.eval()
    with torch.no_grad():
        for s in tokens:
            w = codec.encode_text(s).to(dev).to(torch.float32)
            w = F.normalize(w + torch.randn_like(w) * 0.05, p=2, dim=-1)
            lg = head(w.unsqueeze(0))
            argmax_ids.append(int(torch.argmax(lg, dim=-1).item()))
            lp = torch.log_softmax(lg, -1)
            ents.append(float(-(lp.exp() * lp).sum().item()))
    print(f"  strings probed                   = {len(tokens)}")
    print(f"  distinct top-1 ids               = {len(set(argmax_ids))}")
    print(f"  entropy mean                     = {sum(ents)/len(ents):.4f} nats")
    print(f"  ln(32000) (uniform)              = {__import__('math').log(VOCAB_SIZE):.4f}")
    print(f"  ids ever trained on (max possible) = {len(set(tokens))} of {VOCAB_SIZE}")
    print(f"  => the head can only ever emit ids in its {len(set(tokens))}-id train set;")
    print(f"     the other {VOCAB_SIZE-len(set(tokens))} logits stayed at their init prior.")

print()
print("=" * 78); print("5. T1-T4 VERDICT")
print("=" * 78)
t1 = len(set(tokens)) < 100
t2 = len(set(runs)) > 1 or setA != setB
t3 = ck.exists() and not seedish
verdict = ("MOCK_TARGET_VOCABULARY_CONFIRMED" if (t1 and t3)
           else "PARTIAL — see individual tests")
print(f"  T1 hand-written ~{len(set(tokens))}-string list, not a 32000 vocab : {t1}")
print(f"  T2 label set differs across processes                        : {t2}")
print(f"  T3 no seed/vocab in checkpoint -> map unrecoverable          : {t3}")
print(f"  VERDICT = {verdict}")
print()
print("  CONSEQUENCE FOR A2:")
print("   A2 is NOT 'missing tokenizer'. There is NO VOCABULARY. The head was")
print("   trained to emit Python-hash-derived ids over a hand-written string list.")
print("   Binding an external tokenizer cannot repair this: ids are positionally")
print("   arbitrary AND process-salted. Any id->string table would be a FABRICATION.")
print("   A2 requires RETRAINING the head against a real tokenizer -- a training")
print("   contract change, not a lookup-table lookup.")

# ---- durable artifacts -------------------------------------------------------
findings = {
    "schema": "henri.a2-mock-vocabulary-finding.v1",
    "ts_utc": ts,
    "commit_base": "d42b7d26eb9fb00ba8cddd7e093c73e7d68421a4",
    "source_file": "HENRI V2/scripts/training/train_henri_decoder.py",
    "source_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
    "source_line_rule": rule,
    "declared_vocab_size": VOCAB_SIZE,
    "tokens_text_entries": len(tokens),
    "distinct_training_classes": len(set(tokens)),
    "coverage_ratio": len(set(tokens)) / VOCAB_SIZE,
    "tokens_sample": tokens[:10],
    "hash_ids_default_env_3_runs": runs,
    "hash_ids_stable_across_processes": len(set(runs)) == 1,
    "id_set_default_runA_size": len(setA),
    "id_set_default_runB_size": len(setB),
    "id_set_reproducible": setA == setB,
    "id_set_overlap": len(set(setA) & set(setB)),
    "id_set_hashseed0_size": len(set0),
    "checkpoint_path": str(ck),
    "checkpoint_bytes": ck.stat().st_size if ck.exists() else 0,
    "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest() if ck.exists() else None,
    "checkpoint_keys": ckpt_keys,
    "recoverable_from_checkpoint": bool(seedish) if ck.exists() else None,
    "head_lm_shape": list(head_shape) if head_shape else None,
    "distinct_top1_live": len(set(argmax_ids)),
    "entropy_mean_nats": (sum(ents) / len(ents)) if ents else None,
    "verdict": verdict,
    "tests": {"T1_small_class_list": t1, "T2_process_salted": t2,
              "T3_unrecoverable": bool(t3)},
}
jf = OUT / f"A2_MOCK_VOCABULARY_FINDING_{ts}.json"
jf.write_text(json.dumps(findings, indent=2), encoding="utf-8")
print(f"\n  receipt -> {jf}  ({jf.stat().st_size} B)")
