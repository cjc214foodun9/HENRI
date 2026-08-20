"""
Phase 8.39-V3: Execution-grounded correctness head trainer.

Trains a linear correctness probe over the WaveASTDecoder relative-direction
feature (the SAME feature used for candidate ranking):

    v_rel(body, prompt) = normalize( wave(body) - cos * wave(prompt) )
    score = <w, v_rel>

Labels are NOT hash-based or synthetic: they come from real sandbox
execution of MBPP items (a DIFFERENT benchmark than HumanEval, keeping the
evaluated set unseen). Positives = gold bodies that pass their test_list.
Negatives = grammar/perturbation candidates that FAIL all tests.

Pre-registered gates (kills, checked after training):
  Gate A (oracle, CPU/GPU): correct HumanEval/23 + /35 bodies rank <= 12
      under the trained head on the full 71-candidate set.
  Gate B (production): runner --trained-rank scores >= 3/50 on HumanEval.
Either gate fails -> FALSIFIED; head stays default-OFF.

Checkpoint contract: {w: [D] float32, provenance: {...}, train_acc, val_acc,
counts, sha256 of mbpp bytes, commit}.
"""
import argparse, hashlib, json, os, re, sys, time

import torch
import torch.nn.functional as F

repo = os.path.dirname(os.path.abspath(__file__))
for p in [repo, os.path.dirname(repo)]:
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, os.path.join(os.path.dirname(repo), "HENRI V2"))

from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec
from wave_ast_decoder import WaveASTDecoder
from mbpp_secure_executor import SecurePythonSandbox

SIG = re.compile(r"^def\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE)


def parse_sig(code: str):
    m = SIG.search(code)
    if not m:
        return None, []
    entry, args_raw = m.group(1), m.group(2).strip()
    args = [p.split(":")[0].split("=")[0].strip()
            for p in args_raw.split(",") if p.strip()]
    return entry, args


def passes_tests(sandbox, code: str, tests: list[str], setup: str = "") -> bool:
    if not tests:
        return False
    src = code + "\n\n" + (setup + "\n" if setup else "") + "\n".join(tests)
    try:
        res = sandbox.execute(src)
        return res.status == "PASS"
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to mbpp.jsonl")
    ap.add_argument("--d", type=int, default=65536)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-negatives", type=int, default=8)
    ap.add_argument("--max-items", type=int, default=0, help="bound items for kill-gate run (0=all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    raw = open(args.data, "rb").read()
    data_sha = hashlib.sha256(raw).hexdigest()
    items = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    print(f"[DATA] items={len(items)} sha256={data_sha[:16]} device={device}")

    codec = qFHRREpistemicCodec(d_model=args.d, device=device)
    dec = WaveASTDecoder(codec, device=device)
    sandbox = None
    for mode in ("namespace", "container-rlimit"):
        try:
            c = SecurePythonSandbox(timeout_sec=8.0, mode=mode)
            if c.execute("x = 41 + 1\nassert x == 42\n").status == "PASS":
                sandbox = c
                print(f"[SANDBOX] {mode} PASS")
                break
        except Exception as e:
            print(f"[SANDBOX] {mode} unavailable: {str(e)[:100]}")
    if sandbox is None:
        raise SystemExit("BLOCKED: no sandbox")

    # Collect execution-labeled pairs.
    xs, ys = [], []
    n_pos = n_neg = n_skip = 0
    t0 = time.perf_counter()
    for it in items:
        if args.max_items and n_pos >= args.max_items:
            break
        entry, args_ = parse_sig(it["code"])
        if entry is None or not args_ or not it.get("test_list"):
            n_skip += 1
            continue
        tests = it["test_list"]
        setup = it.get("test_setup_code") or ""
        gold_ok = passes_tests(sandbox, it["code"], tests, setup)
        if not gold_ok:
            n_skip += 1
            continue  # only verified-passing gold code is a positive
        prompt = it["text"]
        pw = dec._wave(prompt)
        gold_wave = dec._wave(it["code"])
        vg = gold_wave - pw * torch.dot(gold_wave, pw).clamp(min=0.0)
        vg = F.normalize(vg, p=2, dim=0)
        xs.append(vg); ys.append(1.0); n_pos += 1

        # Negatives: grammar candidates + perturbations that FAIL all tests.
        neg = 0
        for body in dec._instantiate(entry, args_):
            if neg >= args.max_negatives:
                break
            cand = f"def {entry}({', '.join(args_)}):\n{body}"
            try:
                import ast as _ast
                _ast.parse(cand)
            except SyntaxError:
                continue
            if passes_tests(sandbox, cand, tests, setup):
                continue  # actually-correct candidate: not a negative
            cw = dec._wave(cand)
            vc = cw - pw * torch.dot(cw, pw).clamp(min=0.0)
            vc = F.normalize(vc, p=2, dim=0)
            xs.append(vc); ys.append(0.0); neg += 1; n_neg += 1
        for perturb in ("return 0", "return None", "return a0", "return len(a0)"):
            if neg >= args.max_negatives:
                break
            body = f"    {perturb}" if perturb.startswith("return") else perturb
            cand = f"def {entry}({', '.join(args_)}):\n{body}"
            try:
                import ast as _ast
                _ast.parse(cand)
            except SyntaxError:
                continue
            if passes_tests(sandbox, cand, tests, setup):
                continue
            cw = dec._wave(cand)
            vc = cw - pw * torch.dot(cw, pw).clamp(min=0.0)
            vc = F.normalize(vc, p=2, dim=0)
            xs.append(vc); ys.append(0.0); neg += 1; n_neg += 1

    if n_pos < 50 or n_neg < 50:
        raise SystemExit(f"BLOCKED: insufficient labels pos={n_pos} neg={n_neg}")
    X = torch.stack(xs).to(device)
    Y = torch.tensor(ys, device=device).unsqueeze(1)
    print(f"[LABELS] pos={n_pos} neg={n_neg} skip={n_skip} "
          f"collect_sec={time.perf_counter()-t0:.1f} X={tuple(X.shape)}")

    # Split (stratified shuffle).
    perm = torch.randperm(X.shape[0], device=device)
    n_val = max(1, X.shape[0] // 5)
    val_idx, trn_idx = perm[:n_val], perm[n_val:]

    w = torch.zeros(args.d, 1, device=device, requires_grad=True)
    opt = torch.optim.AdamW([w], lr=args.lr, weight_decay=1e-4)
    for ep in range(1, args.epochs + 1):
        opt.zero_grad()
        logits = X[trn_idx] @ w
        loss = F.binary_cross_entropy_with_logits(logits, Y[trn_idx])
        loss.backward()
        opt.step()
        with torch.no_grad():
            trn_acc = ((logits > 0).float() == Y[trn_idx]).float().mean().item()
            vl = X[val_idx] @ w
            val_acc = ((vl > 0).float() == Y[val_idx]).float().mean().item()
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"[EP {ep:02d}] loss={loss.item():.4f} trn_acc={trn_acc:.4f} "
                  f"val_acc={val_acc:.4f} w_norm={w.norm().item():.3f}")

    commit = "unknown"
    try:
        commit = os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or "unknown"
    except Exception:
        pass
    ckpt = {
        "w": w.detach().cpu().float().squeeze(0),
        "provenance": {
            "source": "mbpp.jsonl (google-research)",
            "data_sha256": data_sha,
            "d": args.d, "epochs": args.epochs, "lr": args.lr,
            "n_pos": n_pos, "n_neg": n_neg, "n_val": n_val,
            "commit": commit, "labels": "execution-verified sandbox PASS/FAIL",
            "evaluated_benchmark_unseen": "HumanEval",
        },
        "train_acc": trn_acc, "val_acc": val_acc,
        "feature": "v_rel = normalize(wave(body) - cos * wave(prompt))",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[CKPT] saved {args.out} w={tuple(ckpt['w'].shape)} "
          f"val_acc={val_acc:.4f} sha256={hashlib.sha256(open(args.out,'rb').read()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
