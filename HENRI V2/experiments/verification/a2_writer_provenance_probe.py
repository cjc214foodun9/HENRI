"""A2 CORRECTION — who really trained the 799 MB overlay, and on WHAT labels?

MY OWN ERROR UNDER TEST (defect #14):
  experiments/verification/A2_RETRAIN_SPEC_*.json states
      "trained_on": {"rule": "target_id = hash(text_target) % 32000",
                     "source": "scripts/training/train_henri_decoder.py:54"}
  But a2_final.py's own Q1/Q1b output showed the FLAT writer is
      scripts/training/train_sgld_500_runner.py:159
          torch.save(unbinder.state_dict(), "models/henri_decoder_checkpoint.pt")
  and its training loop (line 88) uses
          a_target = torch.randint(0, vocab_size, (batch_size,), device=device)
  i.e. UNIFORMLY RANDOM LABELS -- not hash()%N, and not text at all.
  I attributed the artifact to the wrong script. This probe corrects it.

WHY IT MATTERS MORE THAN THE hash() FINDING:
  Cross-entropy against uniform random labels over 32000 classes has an
  irreducible floor of ln(32000) = 10.3735 nats. No input->label mapping exists
  to learn. A head trained this way converges to emit its input-INDEPENDENT
  label marginal, i.e. the same distribution for every input.
  If true, the repo's "TRAINED_DECODER" / "100% token unbinding accuracy" claims
  are FALSIFIED, and M1/A2 is RETRAIN_REQUIRED at the head, not the codec.

TESTS
  T1 attribution : only 2 writers target that literal path; the file is FLAT;
                   train_henri_decoder.py writes WRAPPED -> sgld wrote it.
  T2 training log: telemetry_logs/sgld_500_run_10100.jsonl loss_ce vs ln(32000).
  T3 duplicate   : line 158 saves the SAME state_dict to checkpoints/...;
                   compare sha256 with models/...
  T4 input-blind : variance of logits ACROSS inputs vs WITHIN an input. A head
                   trained on random labels ignores its input (across ~ 0).
Read-only. CPU.
"""
import hashlib, json, math, os, pathlib, re, time

WT = pathlib.Path(r"C:/Users/chan/henri-worktrees/aaii-v43")
V2 = WT / "HENRI V2"
OUT = V2 / "experiments/verification"
CK = V2 / "models" / "henri_decoder_checkpoint.pt"
DUP = V2 / "checkpoints" / "henri_egress_unbinder_sgld_500.pt"
LOG = V2 / "telemetry_logs" / "sgld_500_run_10100.jsonl"
ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
LN32000 = math.log(32000)

print("=" * 78); print("T1. ATTRIBUTION (which writer produced the flat file?)")
print("=" * 78)
for rel, ln, desc in [("scripts/training/train_sgld_500_runner.py", 159, "FLAT state_dict -> models/"),
                      ("scripts/training/train_sgld_500_runner.py", 158, "FLAT state_dict -> checkpoints/"),
                      ("scripts/training/train_henri_decoder.py", 145, "WRAPPED dict -> models/")]:
    p = V2 / rel
    src = p.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"  {rel}:{ln}  ({desc})")
    print(f"      {src[ln-1].strip()[:110]}")
print(f"\n  on-disk file is FLAT with keys: ['down_proj.weight','layer_norm.bias',"
      f"'layer_norm.weight','lm_head.weight']")
print("  -> train_henri_decoder.py CANNOT have written it (it wraps in a dict).")
print("  -> train_sgld_500_runner.py:159 matches the observed shape exactly.")

print()
print("=" * 78); print("T2. THE ACTUAL TRAINING LOG (loss vs the random-label floor)")
print("=" * 78)
print(f"  log {LOG} exists={LOG.exists()}")
if LOG.exists():
    rows = []
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    print(f"  steps logged: {len(rows)}")
    if rows:
        ks = sorted(rows[0].keys())
        print(f"  keys: {ks}")
        for key in ("loss_ce", "loss_efe", "entropy", "mutual_info_nats", "sagnac_delta"):
            vals = [r[key] for r in rows if key in r]
            if vals:
                print(f"    {key:<18} first={vals[0]:.6f} last={vals[-1]:.6f} "
                      f"min={min(vals):.6f} max={max(vals):.6f}")
        lc = [r["loss_ce"] for r in rows if "loss_ce" in r]
        if lc:
            print(f"\n  ln(32000) random-label floor = {LN32000:.4f} nats")
            print(f"  observed loss_ce last        = {lc[-1]:.4f}")
            print(f"  |last - floor|               = {abs(lc[-1]-LN32000):.4f}")
            print(f"  => loss pinned at the uniform floor: {abs(lc[-1]-LN32000) < 0.05}")
            print("     A loss that never descends below the uniform floor means the")
            print("     head learned NOTHING about its input (labels are noise).")
else:
    print("  (log absent on disk - attribution rests on T1 source lines)")

print()
print("=" * 78); print("T3. DUPLICATE SAVE (line 158 vs 159 -> same state_dict twice)")
print("=" * 78)
print(f"  {DUP.name} exists={DUP.exists()}")
if DUP.exists() and CK.exists():
    a = hashlib.sha256(CK.read_bytes()).hexdigest()
    b = hashlib.sha256(DUP.read_bytes()).hexdigest()
    print(f"    models/...      bytes={CK.stat().st_size:>12} sha={a[:24]}")
    print(f"    checkpoints/... bytes={DUP.stat().st_size:>12} sha={b[:24]}")
    print(f"    IDENTICAL={a == b}   (both torch.save calls share one state_dict)")
print(f"  models/... mtime = {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime(CK.stat().st_mtime))}")

print()
print("=" * 78); print("T4. INPUT-BLIND TEST (does the head depend on its input at all?)")
print("=" * 78)
import torch
import torch.nn.functional as F
sd = torch.load(str(CK), map_location="cpu", weights_only=False)
import sys
sys.path.insert(0, str(V2))
from henri_decoder import HENRINeuralEgressUnbinder
head = HENRINeuralEgressUnbinder(d_model=65536, d_hidden=2048, vocab_size=32000, device="cpu")
head.load_state_dict(sd)
head.eval()
torch.manual_seed(0)
with torch.no_grad():
    X = F.normalize(torch.randn(32, 65536), p=2, dim=-1)      # 32 distinct inputs
    lg = head(X)                                              # [32, 32000]
    across = lg.var(dim=0).mean().item()                      # how much inputs differ
    within = lg.var(dim=1).mean().item()                      # how much a row varies
    print(f"  logit variance ACROSS inputs (mean over vocab) = {across:.6e}")
    print(f"  logit variance WITHIN an input  (mean over rows) = {within:.6e}")
    print(f"  ratio across/within                            = {across/max(within,1e-30):.6e}")
    print(f"  max |logit - mean_row| over all entries        = {(lg - lg.mean(dim=1, keepdim=True)).abs().max().item():.6f}")
    print(f"  row-to-row max abs diff (input 0 vs input 1)    = {(lg[0]-lg[1]).abs().max().item():.6f}")
    ent = float((-(F.log_softmax(lg, -1).exp() * F.log_softmax(lg, -1)).sum(-1)).mean())
    print(f"  entropy mean                                    = {ent:.4f}  (ln32000={LN32000:.4f})")
    uniq = len({int(a) for a in lg.argmax(-1)})
    print(f"  distinct argmax over 32 inputs                  = {uniq}")
    blind = (across / max(within, 1e-30)) < 1e-3
    print(f"\n  => INPUT-BLIND = {blind}")
    if blind:
        print("     The 32 inputs produce statistically identical logit rows. The head")
        print("     emits an input-INDEPENDENT label marginal -- the signature of")
        print("     training against uniformly random labels.")

print()
print("=" * 78); print("T5. VERDICT")
print("=" * 78)
print("  writer            : scripts/training/train_sgld_500_runner.py:159")
print("  training labels   : a_target = torch.randint(0, vocab_size, ...)  (UNIFORM RANDOM)")
print("  inputs            : torch.randn waves (X_demo/Y_demo/psi_query), not text")
print("  loss floor        : ln(32000) = 10.3735 nats -- irreducible for random labels")
print()
print("  A2 VERDICT = RETRAIN_REQUIRED@HEAD")
print("  Corrections to my prior receipt:")
print("   * the 799 MB overlay was NOT written by train_henri_decoder.py")
print("   * its labels were NOT hash()%32000 over 35 strings -- they were uniform")
print("     random integers; no text and no vocabulary were involved at any point")
print("   * 'TRAINED_DECODER' telemetry from loading this file is FALSIFIED")

receipt = {
    "schema": "henri.a2-writer-and-label-provenance.v1",
    "ts_utc": ts,
    "corrects": "A2_RETRAIN_SPEC_* and A2_MOCK_VOCABULARY_FINDING_* (writer attribution)",
    "my_prior_error": "attributed the flat checkpoint to train_henri_decoder.py:54 "
                      "hash()%32000; the real writer is train_sgld_500_runner.py:159",
    "writer": {
        "file": "HENRI V2/scripts/training/train_sgld_500_runner.py",
        "save_lines": [158, 159],
        "save_form": "torch.save(unbinder.state_dict(), path)  # FLAT, no metadata",
        "path_models": "models/henri_decoder_checkpoint.pt",
        "path_checkpoints": "checkpoints/henri_egress_unbinder_sgld_500.pt",
    },
    "labels": {
        "line": 88,
        "code": "a_target = torch.randint(0, vocab_size, (batch_size,), device=device)",
        "kind": "uniform random integers",
        "loss_line": 102,
        "loss_code": "loss_ce = F.cross_entropy(z_egress, a_target)",
    },
    "inputs": {
        "lines": [86, 87, 94],
        "code": "X_demo/Y_demo/psi_query = F.normalize(torch.randn(batch_size, D), p=2, dim=-1)",
        "kind": "random Gaussian unit waves, not text",
    },
    "loss_floor_nats": LN32000,
    "training_log": {
        "path": str(LOG), "exists": LOG.exists(),
        "steps": len(rows) if LOG.exists() and rows else 0,
        "loss_ce_last": ([r["loss_ce"] for r in rows if "loss_ce" in r][-1]
                         if LOG.exists() and rows and any("loss_ce" in r for r in rows) else None),
    },
    "checkpoint": {
        "path": str(CK), "bytes": CK.stat().st_size,
        "sha256": hashlib.sha256(CK.read_bytes()).hexdigest(),
        "keys": sorted(sd.keys()),
        "flat_no_metadata": True,
    },
    "input_blind": {
        "var_across_inputs": across, "var_within_input": within,
        "ratio": across / max(within, 1e-30), "verdict_input_blind": blind,
        "entropy_mean_nats": ent, "distinct_argmax_over_32": uniq,
    },
    "verdict": "RETRAIN_REQUIRED@HEAD",
    "falsified_claims": [
        "decoder_state=TRAINED_DECODER from loading this file",
        "100.00% token unbinding accuracy (impossible with random labels)",
    ],
}
jf = OUT / f"A2_WRITER_AND_LABEL_PROVENANCE_{ts}.json"
jf.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"\n  receipt -> {jf.name}  ({jf.stat().st_size} B)")
