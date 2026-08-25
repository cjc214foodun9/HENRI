"""DeltaMem-1 evaluation (C4/C5/C8/C9/C10 on the sealed stream).

Contract: deltamem1_prereg.md (cc6d7f59, A1-A10).
- Two arms on the SAME stream: delta (gamma=0.985, eta=0.1, error-correction)
  vs Hebbian (gamma=1, eta=0.1, no error-correction, M += eta v k^T).
- Stream: sealed split task codes -> canonical embeddings (deltamem_embed,
  A8), cyclic 2 epochs = 1,040 steps; k_t = psi[t mod 520], v_t = psi[(t+1) mod 520].
- C4: 1 - mean(||raw_e_delta||2)/mean(||raw_e_hebbian||2) >= 0.40 (A4, A11(b)):
  raw = pre-update, PRE-VETO residual ||psi_{t+1} - M_{t-1} psi_t|| (A10).
  The veto-zeroed write error is C10 telemetry only (never C4).
- C5 (A10): mean over steps 901-1000 of S_t = 1 - Delta_t (A6 Sagnac) >= 0.90;
  dead-memory negative control must FAIL C5 (A11(e)), else C5 vacuous.
- C8 (A3): SSR_eval = ||(Yhat-Y)V||_F / ||Y V||_F, V = top-r right singular
  vectors of calibration-only first 40% of stream (r=8); eval over remaining
  60%. Gate <= 0.35.
- C9 (A3): SSR_rollout5 = mean_t ||phat_{t+5} - psi_{t+5}|| / ||psi_{t+5}||,
  5-step open-loop from frozen state at t, t in [920, 1010]. Gate <= 0.70.
- C10: veto_count telemetry; zeroed-error assertion on veto steps (A6).
- C6/C7 measured by deltamem_verify.py (remote CUDA); pass via --receipt.
- Incremental JSONL per row BEFORE aggregation; --expect-sha/--expect-count
  refusal before loading tasks; --smoke uses a disposable synthetic stream.
- Verdict precedence (C12): C6 or C5 fail -> DELTA_MEM_REJECTED (named);
  any C4/C8/C9/C10/C11 fail -> DELTA_MEM_REJECTED with list; else ACCEPTED.
  A smoke run NEVER emits a verdict (plumbing only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from delta_qfhrr_associative_memory import (  # noqa: E402
    DeltaQFHRRAssociativeMemory, DEFAULT_GAMMA, DEFAULT_ETA, DEFAULT_R,
)
from deltamem_embed import DeltaMemEmbedder  # noqa: E402

D = 4096
N_EPOCHS = 2
STREAM_LEN = 520
TOTAL_STEPS = N_EPOCHS * STREAM_LEN          # 1040
C5_LO, C5_HI = 901, 1000                      # A10 window (1-indexed rows)
C8_CALIB_FRAC = 0.40                          # A3
ROLLOUT_HORIZON = 5
ROLLOUT_T0, ROLLOUT_T1 = 920, 1010            # inclusive window
V_SEED = 20260824                             # A9


def _sagnac_delta_real(pred: torch.Tensor, obs: torch.Tensor) -> float:
    """A6: S = 0.5*(1+<a,b>/(||a||||b||)), delta = 1 - S in [0,1]."""
    a = pred.reshape(-1).float()
    b = obs.reshape(-1).float()
    na, nb = a.norm(), b.norm()
    if na.item() == 0.0 or nb.item() == 0.0:
        return 0.5
    s = 0.5 * (1.0 + torch.dot(a, b) / (na * nb))
    return float((1.0 - s).clamp(0.0, 1.0).item())


class HebbianMemory:
    """Hebbian arm (C4 baseline, A11(a) ratified 2026-08-25): SAME frozen
    seeded rank-8 V (seed 20260824), same-subspace factorized comparator;
    gamma=1, eta=0.1, NO error correction (M += eta * v k^T within span(V)).
    Dense DxD Hebbian is infeasible under the mandatory rank-r factorization
    and is NOT the comparator."""

    def __init__(self, d: int = D, r: int = DEFAULT_R, eta: float = DEFAULT_ETA,
                 v_seed: int = V_SEED, enabled: bool = True):
        self.d, self.r, self.eta = d, r, eta
        self.enabled = enabled
        self.U = torch.zeros(d, r, dtype=torch.float32)
        self.V = torch.zeros(d, r, dtype=torch.float32)
        if v_seed is not None:
            g = torch.Generator().manual_seed(v_seed)
            Vr = torch.randn(d, r, generator=g, dtype=torch.float32)
            Q, _ = torch.linalg.qr(Vr)
            sgn = torch.sign(Q[0]); sgn[sgn == 0] = 1.0
            self.V.copy_(Q * sgn)
        self.step = 0
        self.veto_count = 0

    def reset(self) -> None:
        self.U.zero_()
        self.step = 0
        self.veto_count = 0

    def step_once(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(v.reshape(-1))
        k = k.reshape(-1).float()
        v = v.reshape(-1).float()
        phat = self.U @ (self.V.T @ k)
        e = v - phat
        s = self.V.T @ k
        g = self.V.T @ self.V
        a = torch.linalg.solve(g, s)
        self.U.add_(self.eta * torch.outer(v, a))   # gamma=1, no correction
        self.step += 1
        return e

    def readout(self, k: torch.Tensor) -> torch.Tensor:
        return self.U @ (self.V.T @ k.reshape(-1).float())


class DeadMemory:
    """C5 non-vacuity control (A11(e)): readout is always 0 (no learning).
    A dead memory on the same stream must FAIL C5 (R_sync ~ 0.5 < 0.90).
    If the control passes, C5 is vacuous on this stream -> fail closed."""

    def __init__(self):
        self.step = 0
        self.veto_count = 0

    def reset(self) -> None:
        self.step = 0
        self.veto_count = 0

    def readout(self, k: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(k.reshape(-1).float())

    def step_once(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self.step += 1
        return torch.zeros_like(v.reshape(-1).float())


def readout_for(mem, k: torch.Tensor) -> torch.Tensor:
    if hasattr(mem, "readout"):
        return mem.readout(k)
    return mem.U @ (mem.V.T @ k.reshape(-1).float())


def run_arm(mem, stream: list[torch.Tensor], run_id: str, arm: str,
            out_fh, total: int) -> dict:
    """Run one arm over the cyclic stream; write per-step JSONL immediately."""
    rows = []
    n = len(stream)
    for t in range(total):
        k = stream[t % n]
        v = stream[(t + 1) % n]
        # A10/A11(b): Delta_t and the C4 error use the PRE-update, PRE-veto
        # readout M_{t-1} k_t. The veto-zeroed write error is C10 telemetry only.
        phat = readout_for(mem, k)
        delta = _sagnac_delta_real(phat, v)
        raw = v.reshape(-1).float() - phat.reshape(-1).float()
        raw_norm = float(raw.norm().item())
        prev_veto = int(mem.veto_count)
        e = mem.step_once(k, v)
        veto_fired = int(mem.veto_count) > prev_veto
        row = {"run_id": run_id, "arm": arm, "t": t,
               "raw_error_norm": raw_norm,          # A11(b): C4 signal
               "write_error_norm": float(e.reshape(-1).norm().item()),  # C10
               "sagnac_delta": delta,
               "veto_fired": int(veto_fired),
               "veto_count": int(mem.veto_count),
               "step": int(mem.step)}
        rows.append(row)
        out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()
    return {"rows": rows, "final_veto": int(mem.veto_count),
            "final_step": int(mem.step)}


def compute_metrics(delta_rows, hebb_rows, dead_rows, stream) -> dict:
    """C4/C5/C8/C9/C10 from telemetry + a fresh reference prediction pass."""
    d_means = torch.tensor([r["raw_error_norm"] for r in delta_rows])
    h_means = torch.tensor([r["raw_error_norm"] for r in hebb_rows])
    c4 = 1.0 - float(d_means.mean()) / float(h_means.mean())

    c5_lo = C5_LO - 1
    c5_hi = C5_HI - 1
    if len(delta_rows) < C5_HI:          # disposable smoke: last 100 steps
        c5_lo = max(0, len(delta_rows) - 100)
        c5_hi = len(delta_rows) - 1
    c5_window = [r for r in delta_rows if c5_lo <= r["t"] <= c5_hi]
    c5 = float(torch.tensor([1.0 - r["sagnac_delta"]
                             for r in c5_window]).mean()) if c5_window else float("nan")

    # C8 (A3): top-r basis from calibration-only first 40%; eval on rest.
    m2 = DeltaQFHRRAssociativeMemory(d=D, r=DEFAULT_R, gamma=DEFAULT_GAMMA,
                                     eta=DEFAULT_ETA, enabled=True)
    n = len(stream)
    total = len(delta_rows)
    Yhats, Ys = [], []
    for t in range(total):
        k = stream[t % n]
        v = stream[(t + 1) % n]
        phat = m2.readout(k)
        m2.update(k, v)
        Yhats.append(phat.clone())
        Ys.append(v.clone())
    Y = torch.stack(Ys)
    Yhat = torch.stack(Yhats)
    calib_n = max(1, int(C8_CALIB_FRAC * total))
    if calib_n >= total:
        calib_n = total // 2
    _, _, Vh = torch.linalg.svd(Y[:calib_n], full_matrices=False)
    r_basis = min(DEFAULT_R, Vh.shape[0])
    V = Vh[:r_basis].T                       # [D, r]
    eval_slice = slice(calib_n, total)
    Ye, Yhe = Y[eval_slice], Yhat[eval_slice]
    num = torch.linalg.norm((Yhe - Ye) @ V, ord="fro")
    den = torch.linalg.norm(Ye @ V, ord="fro")
    c8 = float(num / (den + 1e-12))

    # C9 (A3): 5-step open-loop from frozen state at t in [T0, T1].
    m3 = DeltaQFHRRAssociativeMemory(d=D, r=DEFAULT_R, gamma=DEFAULT_GAMMA,
                                     eta=DEFAULT_ETA, enabled=True)
    n = len(stream)
    for t in range(0, ROLLOUT_T0):
        m3.update(stream[t % n], stream[(t + 1) % n])
    errs = []
    for t in range(ROLLOUT_T0, ROLLOUT_T1 + 1):
        U_frozen = m3.U.clone()
        cur = stream[t % n]
        phat = None
        for _ in range(ROLLOUT_HORIZON):
            phat = U_frozen @ (m3.V.T @ cur.reshape(-1).float())
            cur = phat
        truth = stream[(t + ROLLOUT_HORIZON) % n]
        tn = truth.norm().item()
        errs.append(float((phat - truth).norm().item() / (tn + 1e-12)))
        m3.update(stream[t % n], stream[(t + 1) % n])
    c9 = float(torch.tensor(errs).mean()) if errs else float("nan")

    # C5 (A10): non-vacuity control — a DEAD memory (U=0, readout=0) on the
    # same stream must FAIL C5 (R_sync ~ 0.5). If the control passes, C5 is
    # vacuous on this stream -> C5 fails closed (A11(e)).
    c5_dead_window = [r for r in dead_rows if c5_lo <= r["t"] <= c5_hi]
    c5_dead = float(torch.tensor([1.0 - r["sagnac_delta"]
                                  for r in c5_dead_window]).mean()) \
        if c5_dead_window else float("nan")
    c5_dead_control = (c5_dead < 0.90) if dead_rows else None

    # C10: veto zeroing assertion (A6: e_t := 0 on veto). Raw error must be
    # nonzero (fixture) and the write error must be zeroed on every veto.
    veto_rows = [r for r in delta_rows if r["veto_fired"]]
    c10_count = len(veto_rows)
    veto_zeroed = all(r["write_error_norm"] < 1e-9 for r in veto_rows)
    veto_nonzero_raw = all(r["raw_error_norm"] > 1e-9 for r in veto_rows)

    return {"c4_error_reduction": c4, "c5_r_sync": c5,
            "c5_dead": c5_dead, "c5_dead_control": c5_dead_control,
            "c8_ssr_eval": c8, "c9_ssr_rollout5": c9,
            "c10_veto_count": c10_count,
            "c10_veto_zeroed": bool(veto_zeroed),
            "c10_veto_nonzero_raw": bool(veto_nonzero_raw),
            "n_steps": total, "n_stream": n}


def verdict(m: dict, latency_us, footprint) -> dict:
    """C12 precedence: C6/C5 fail -> REJECTED named; else list failures."""
    fails = []
    if latency_us is not None and latency_us >= 15.0:
        fails.append("C6")
    if m["c5_r_sync"] < 0.90 or m.get("c5_dead_control") is False:
        fails.append("C5")
    if m["c4_error_reduction"] < 0.40:
        fails.append("C4")
    if m["c8_ssr_eval"] > 0.35:
        fails.append("C8")
    if m["c9_ssr_rollout5"] > 0.70:
        fails.append("C9")
    if m["c10_veto_count"] > 0 and not m["c10_veto_zeroed"]:
        fails.append("C10")
    if footprint is not None and footprint > 128 * 1024:
        fails.append("C7")
    status = "DELTA_MEM_ACCEPTED"
    if fails:
        status = "DELTA_MEM_REJECTED"
        if any(c in ("C5", "C6") for c in fails):
            status = "DELTA_MEM_REJECTED"   # C12 named-failure clause
    return {"status": status, "failed_contracts": fails}


def load_stream(split_path: str, expect_sha: str, expect_count: int):
    """Refuse to load unless sha + count match (before any execution)."""
    p = pathlib.Path(split_path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expect_sha and sha != expect_sha:
        raise SystemExit(f"REFUSE: split sha {sha[:16]} != expected {expect_sha[:16]}")
    tasks = json.loads(raw)
    if len(tasks) != expect_count:
        raise SystemExit(f"REFUSE: split count {len(tasks)} != expected {expect_count}")
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="")
    ap.add_argument("--expect-sha", type=str, default="")
    ap.add_argument("--expect-count", type=int, default=520)
    ap.add_argument("--smoke", action="store_true",
                    help="disposable synthetic stream (plumbing only)")
    ap.add_argument("--smoke-n", type=int, default=32)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--receipt", type=str, default="",
                    help="deltamem_verify.py receipt JSON (C6/C7)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing --out file")
    ap.add_argument("--arms", type=str, default="delta,hebbian,dead")
    args = ap.parse_args()

    out_p = pathlib.Path(args.out)
    if out_p.exists() and not args.force:
        raise SystemExit(f"REFUSE: output exists (no --force): {out_p}")

    if args.smoke:
        g = torch.Generator().manual_seed(1234)
        stream = [torch.randn(D, generator=g) for _ in range(args.smoke_n)]
        total = min(TOTAL_STEPS, args.smoke_n * 2)
        run_id = f"smoke-{int(time.time())}"
    else:
        if not args.split:
            raise SystemExit("REFUSE: --split required unless --smoke")
        tasks = load_stream(args.split, args.expect_sha, args.expect_count)
        from system1_kernel_v041_energy_refactored import tokenize_code, ID2TOK
        emb = DeltaMemEmbedder(tokenize_code, ID2TOK)
        stream = emb.embed_stream([t["code"] for t in tasks])
        if len(stream) != STREAM_LEN:
            raise SystemExit(f"REFUSE: stream {len(stream)} != {STREAM_LEN}")
        total = TOTAL_STEPS
        run_id = f"deltamem1-{int(time.time())}"

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results = {}
    with open(out_p, "w", encoding="utf-8") as fh:
        for arm in arms:
            if arm == "delta":
                mem = DeltaQFHRRAssociativeMemory(d=D, r=DEFAULT_R,
                                                  gamma=DEFAULT_GAMMA,
                                                  eta=DEFAULT_ETA,
                                                  enabled=True)
            elif arm == "hebbian":
                mem = HebbianMemory()
            elif arm == "dead":
                mem = DeadMemory()
            else:
                raise SystemExit(f"REFUSE: unknown arm {arm}")
            results[arm] = run_arm(mem, stream, run_id, arm, fh, total)

    delta_rows = results.get("delta", {}).get("rows", [])
    hebb_rows = results.get("hebbian", {}).get("rows", [])
    if not delta_rows or not hebb_rows:
        print(json.dumps({"status": "PLUMBING_PARTIAL", "arms": sorted(results),
                          "rows": {a: len(results[a]["rows"]) for a in results}}))
        return 0

    metrics = compute_metrics(delta_rows, hebb_rows,
                              results.get("dead", {}).get("rows", []), stream)

    latency_us = footprint = None
    if args.receipt:
        rec = json.loads(pathlib.Path(args.receipt).read_text(encoding="utf-8"))
        c6 = rec.get("c6", {})
        c7 = rec.get("c7", {})
        c11 = rec.get("c11", {})
        # Sealed C6 binding = host-visible mean (C6: "on host GPU");
        # CUDA-event mean stays diagnostic (both reported by the verifier).
        latency_us = c6.get("host_visible_mean_us")
        footprint = c7.get("bf16_bytes")
        # Fail closed: the receipt must carry the sealed gates AND pass them.
        rec_fail = []
        if latency_us is None:
            rec_fail.append("receipt missing c6.host_visible_mean_us")
        if footprint is None:
            rec_fail.append("receipt missing c7.bf16_bytes")
        if not c7.get("bf16_pass"):
            rec_fail.append("receipt C7 gate not passed")
        if not c11.get("equal"):
            rec_fail.append("receipt C11 determinism not passed")
        eq = rec.get("equiv", {})
        cases = eq.get("cases", [])
        if cases:
            for c in cases:
                insts = [v for k, v in c.items() if k.startswith("inst")]
                if not all(i.get("fp32_pass") for i in insts):
                    rec_fail.append("receipt fp32 equivalence not all-pass")
        gpu = rec.get("gpu", "")
        if "5090" not in gpu:
            rec_fail.append(f"receipt GPU mismatch: {gpu}")
        if rec_fail:
            raise SystemExit("REFUSE: verification receipt not pass-closed: "
                             + "; ".join(rec_fail))

    full_window = (len(delta_rows) == TOTAL_STEPS and len(stream) == STREAM_LEN
                   and not args.smoke)
    v = verdict(metrics, latency_us, footprint) if full_window else \
        {"status": "PLUMBING_OK", "failed_contracts": []}

    report = {"run_id": run_id, "smoke": bool(args.smoke), "arms": arms,
              "metrics": metrics, "verdict": v,
              "c6_latency_us": latency_us, "c7_footprint_bytes": footprint,
              "out": str(out_p)}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
