"""R1: Zone C novel-question transfer test. dim = 1024 complex (num_blocks=256).

Implements design/zone_a/NOVEL-QUESTION-TRANSFER-DESIGN.md. CPU only. $0.

PRE-REGISTERED (frozen before the first run; no post-hoc change):
  tau            = 0.90 cosine correlation with the true operator (criterion)
  "sample"       = one gradient presentation of a 64-example batch
  mechanism      = retrieved Zone C solutions initialize the novel task's
                   learner:  m_init = unit((1-BLEND)*m_hat + BLEND*mean(retrieved))
                   BLEND = 0.3, matching production_arc_run.py:1843 (0.7/0.3)
  query signal   = m_hat, estimated from 4 DEMONSTRATION pairs only.
                   m_true is NEVER used to build a query. No leakage.
  seeds          = 20261002, 20261003, 20261004
  top_k = 4, budget = 120 steps, LR = 0.05
  delta K1 = 0.15 * median(A0);  delta K2 = 0.075 * median(A0)

KILLS:
  K1: median(A1) < median(A0) - delta_K1
  K2: median(A1) < median(A2) - delta_K2      <-- load-bearing
  K3: median(A3) > median(A0)                 [baseline validity; report only]
  K4: mean_final_corr(A1) >= mean_final_corr(A2)
  T1 dies if K1 or K2 fails.

CONDITIONS (the boundary test):
  structured   = 8 shared prototypes. Tasks cluster near a prototype, so a
                 retrieved neighbour carries task-relevant content.
  unstructured = EVERY task gets a fresh random base. No two tasks share
                 structure, so retrieval can carry no useful content.
                 A1 must NOT beat A0/A2 here. This is the negative control.

Encoding note (measured 2026-10-03): wave_to_bytes casts the imaginary part
away, silently (complex64 [4,8] = 256 B -> float32 = 128 B). The complex
operator is therefore stored as real re/im pairs via view_as_real, losslessly.

Isolation: rows are written with domain_family="ast" (a namespace with 0
pre-existing rows; the store holds general|5, action|4). Each condition uses
its own run_id and deletes exactly its rows afterwards. Count readback before
and after. Attribution is live (HENRI_FREEZE_LEARNING is NOT set).
"""
import os
import sys
import json
import hashlib
import statistics
import subprocess

HERE = os.path.abspath(__file__)
HENRI_V2 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
REPO = os.path.dirname(HENRI_V2)
sys.path.insert(0, HENRI_V2)

import torch
import zone_c_segment_cache as zsc
from zone_c_env import resolve_zone_c_dsn

# R1 = 256 (dim 1024 complex).  R3 = 8192 (dim 32768 complex), the stored
# engram width the reader validates: 8192 * 8 * 4 = 262144 bytes.
NUM_BLOCKS = int(os.environ.get("XFER_NUM_BLOCKS", "256"))
SLOTS = 4
D_C = NUM_BLOCKS * SLOTS          # 1024 complex parameters
FAMILY = "ast"

N_PROTO = 8
N_FIT_PER = int(os.environ.get("XFER_FIT_PER", "8"))     # 64 fit engrams/condition
N_NOVEL_PER = int(os.environ.get("XFER_NOVEL_PER", "8"))  # 64 novel tasks/condition
N_DEMO = int(os.environ.get("XFER_N_DEMO", "4"))
SIGMA = 0.30
TAU = float(os.environ.get("XFER_TAU", "0.90"))
BUDGET = int(os.environ.get("XFER_BUDGET", "120"))
# Calibrated 2026-10-03 by zone_c_transfer_r1_calibrate.py, BEFORE this run:
#   lr=0.05 fit-from-random-init never reached tau in 400 steps (0/3 seeds).
#   lr=0.15 reached it in 54/53/61 steps (3/3 seeds).  Using 0.15.
#   obs_noise=0.0 gives corr(m_hat,m_true)=1.0000 exactly -> all arms start
#   at tau -> vacuous.  obs_noise=3.0 gives corr 0.5174 and A0 cold-start
#   steps 16/11/17, a real dynamic range with a baseline that can fail.
LR = float(os.environ.get("XFER_LR", "0.15"))
OBS_NOISE = float(os.environ.get("XFER_OBS_NOISE", "3.0"))
TOP_K = 4
BLEND = 0.3
SEEDS = [20261002, 20261003, 20261004]
TWO_PI = 6.283185307179586

COMMIT = subprocess.check_output(
    ["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()


def encode(m):
    """[D_C] complex -> [NUM_BLOCKS, 8] real, lossless (re/im interleaved)."""
    return torch.view_as_real(
        m.reshape(NUM_BLOCKS, SLOTS)).reshape(NUM_BLOCKS, 8).contiguous()


def decode(w):
    """[NUM_BLOCKS, 8] real -> [D_C] complex."""
    return torch.view_as_complex(
        w.reshape(NUM_BLOCKS, SLOTS, 2).contiguous()).reshape(-1)


def unit(m):
    return m / (m.abs() + 1e-12)


def make_task(base, g, noise=SIGMA):
    """Unit-modulus diagonal operator, phases = base + bounded noise."""
    return torch.exp(1j * (base + noise * torch.randn(D_C, generator=g)))


def estimate_from_demos(m_true, g, n_demo=N_DEMO, obs_noise=OBS_NOISE):
    """Non-leaky query signal: least-squares phases from demos only.

    Observation noise is REQUIRED.  For a diagonal operator the noise-free
    estimator sum(y*conj(x))/sum|x|^2 recovers m exactly (calibration:
    corr=1.0000), which would start every arm at tau and make the test
    vacuous.  obs_noise makes m_hat a genuine proxy for the unseen operator.
    """
    x = torch.randn(n_demo, D_C, generator=g, dtype=torch.cfloat)
    y = x * m_true + obs_noise * torch.randn(
        n_demo, D_C, generator=g, dtype=torch.cfloat)
    num = (y * x.conj()).sum(0)
    den = (x.abs() ** 2).sum(0) + 1e-9
    return unit(num / den)


def fit_steps(m_true, m_init, g, budget=BUDGET, tau=TAU):
    """Steps until cosine(x*m, y) >= tau, else budget. Returns (steps, corr)."""
    m = m_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([m], lr=LR)
    x = torch.randn(64, D_C, generator=g, dtype=torch.cfloat)
    y = x * m_true
    yn = torch.linalg.vector_norm(y)
    corr = 0.0
    for step in range(1, budget + 1):
        opt.zero_grad()
        loss = ((x * m - y).abs() ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            m.data = unit(m.data)
            xm = x * m
            corr = float((xm * y.conj()).sum().abs()
                         / (torch.linalg.vector_norm(xm) * yn + 1e-9))
            if corr >= tau:
                return step, corr
    return budget, corr


def run_condition(store, structured, run_id):
    """Return {arm: ([steps], [corr])}, ingested_count, retrieval integrity."""
    allres = {a: [] for a in ("A0", "A1", "A2", "A3")}
    allacc = {a: [] for a in ("A0", "A1", "A2", "A3")}
    n_ing = 0
    ids_in_family_ok = None

    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed + (0 if structured else 7_000_000))

        if structured:
            protos = [torch.rand(D_C, generator=g) * TWO_PI for _ in range(N_PROTO)]
            fit_bases = [protos[p] for p in range(N_PROTO) for _ in range(N_FIT_PER)]
            novel_bases = [protos[p] for p in range(N_PROTO)
                           for _ in range(N_NOVEL_PER)]
        else:
            # No shared structure: every task draws its own base.
            fit_bases = [torch.rand(D_C, generator=g) * TWO_PI
                         for _ in range(N_PROTO * N_FIT_PER)]
            novel_bases = [torch.rand(D_C, generator=g) * TWO_PI
                           for _ in range(N_PROTO * N_NOVEL_PER)]

        # ---- ingest: solve fit tasks, store converged solutions ----
        for base in fit_bases:
            m = make_task(base, g)
            steps, corr = fit_steps(
                m, unit(torch.randn(D_C, generator=g, dtype=torch.cfloat)), g)
            if corr < TAU:
                continue
            store.write_engram(encode(m), "ast", 0.0, run_id=run_id,
                               arm_id=run_id, commit_sha=COMMIT,
                               domain_family=FAMILY)
            n_ing += 1

        # ---- integrity: every row in this family belongs to this run ----
        with store._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT run_id FROM phylogenetic_engrams_65536 "
                            "WHERE domain_family = %s", (FAMILY,))
                seen = {r[0] for r in cur.fetchall()}
        ids_in_family_ok = seen <= {run_id}

        # ---- novel tasks: measure steps per arm ----
        for base in novel_bases:
            m_true = make_task(base, g)
            m_hat = estimate_from_demos(m_true, g)

            # A1: retrieve nearest to the novel task's own demo estimate
            hits = store.query_engrams(encode(m_hat), TOP_K, 8760.0, FAMILY)
            got = [decode(h[0].to(torch.float32)) for h in hits]

            # A2 (shuffled): retrieve for a DIFFERENT task's estimate
            other = make_task(torch.rand(D_C, generator=g) * TWO_PI, g)
            hits_w = store.query_engrams(
                encode(estimate_from_demos(other, g)), TOP_K, 8760.0, FAMILY)
            got_w = [decode(h[0].to(torch.float32)) for h in hits_w]

            # A3 (random engram): a random stored row, no query
            rand_w = None
            with store._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT engram_wave_bytes FROM phylogenetic_engrams_65536 "
                        "WHERE run_id = %s ORDER BY random() LIMIT 1", (run_id,))
                    row = cur.fetchone()
            if row:
                rand_w = zsc.bytes_to_wave(bytes(row[0]), NUM_BLOCKS)

            def init_from(extra):
                if extra is None:
                    return m_hat
                return unit((1 - BLEND) * m_hat + BLEND * extra)

            arms = {
                "A0": m_hat,
                "A1": init_from(torch.stack(got).mean(0) if got else None),
                "A2": init_from(torch.stack(got_w).mean(0) if got_w else None),
                "A3": init_from(decode(rand_w.to(torch.float32))
                                if rand_w is not None else None),
            }
            for arm, m0 in arms.items():
                s, c = fit_steps(m_true, m0, g)
                allres[arm].append(s)
                allacc[arm].append(c)

    return allres, allacc, n_ing, ids_in_family_ok


def summarize(res, acc):
    med = {a: statistics.median(v) for a, v in res.items()}
    macc = {a: statistics.mean(v) for a, v in acc.items()}
    d0 = med["A0"]
    return {
        "median_steps": med,
        "mean_final_corr": macc,
        "n_per_arm": {a: len(v) for a, v in res.items()},
        "K1": med["A1"] < d0 - 0.15 * d0,
        "K2": med["A1"] < med["A2"] - 0.075 * d0,
        "K3": med["A3"] > d0,
        "K4": macc["A1"] >= macc["A2"],
    }


def main():
    store = zsc.TimescaleZoneCStore(resolve_zone_c_dsn(), NUM_BLOCKS)
    with store._connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM phylogenetic_engrams_65536")
            n_before = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM phylogenetic_engrams_65536 "
                        "WHERE domain_family = %s", (FAMILY,))
            fam_before = int(cur.fetchone()[0])
    print("store_before=%d family_ast_before=%d commit=%s"
          % (n_before, fam_before, COMMIT[:7]))

    results = {}
    for name, structured, run_id in (("structured", True, "xfer-r1-s"),
                                     ("unstructured", False, "xfer-r1-u")):
        res, acc, n_ing, ok = run_condition(store, structured, run_id)
        summ = summarize(res, acc)
        results[name] = {"summary": summ, "ingested": n_ing,
                         "n_novel": len(res["A0"]),
                         "retrieval_integrity": ok,
                         "nonvacuous": n_ing > 0}
        # Clean up exactly this condition's rows.
        with store._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM phylogenetic_engrams_65536 "
                            "WHERE run_id = %s", (run_id,))
                deleted = cur.rowcount
            conn.commit()
        results[name]["deleted"] = deleted
        print("%s: ingested=%d novel=%d deleted=%d K1=%s K2=%s K3=%s K4=%s"
              % (name, n_ing, len(res["A0"]), deleted,
                 summ["K1"], summ["K2"], summ["K3"], summ["K4"]))

    with store._connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM phylogenetic_engrams_65536")
            n_after = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM phylogenetic_engrams_65536 "
                        "WHERE domain_family = %s", (FAMILY,))
            fam_after = int(cur.fetchone()[0])

    s = results["structured"]["summary"]
    u = results["unstructured"]["summary"]
    # Vacuity guard: a run that ingested nothing cannot report a verdict.
    vac_s = results["structured"]["ingested"] == 0
    vac_u = results["unstructured"]["ingested"] == 0
    verdict_s = ("VACUOUS_NO_INGESTION" if vac_s else
                 ("T1_SUPPORTED" if (s["K1"] and s["K2"] and s["K4"])
                  else "T1_FALSIFIED"))
    verdict_u = ("VACUOUS_NO_INGESTION" if vac_u else
                 ("T1_SUPPORTED" if (u["K1"] and u["K2"] and u["K4"])
                  else "T1_FALSIFIED"))

    out = {
        "experiment": "zone_c_transfer_R1",
        "commit": COMMIT,
        "dim_complex": D_C, "num_blocks": NUM_BLOCKS, "family": FAMILY,
        "preregistered": {
            "tau": TAU, "blend": BLEND, "top_k": TOP_K, "budget": BUDGET,
            "n_demo": N_DEMO, "sigma": SIGMA, "lr": LR,
            "obs_noise": OBS_NOISE,
            "calibration_ref": "zone_c_transfer_r1_calibrate.py (run before freeze)",
            "calibration_lr_choice": "0.05 fails 0/3 seeds; 0.15 converges 3/3",
            "calibration_noise_choice": "0.0 is vacuous (corr 1.0000); 3.0 gives A0 steps 16/11/17",
            "delta_k1_frac": 0.15, "delta_k2_frac": 0.075, "seeds": SEEDS,
            "sample_definition": "one gradient presentation of a 64-example batch",
            "query_signal": "demos only; m_true never used to build a query",
            "mechanism": "m_init = unit((1-0.3)*m_hat + 0.3*mean(retrieved))",
        },
        "conditions": results,
        "store_rows_before": n_before, "family_ast_before": fam_before,
        "store_rows_after": n_after, "family_ast_after": fam_after,
        "verdict_structured": verdict_s,
        "verdict_unstructured": verdict_u,
        "interpretation": (
            "A structured pass with an unstructured fail shows the mechanism "
            "exploits shared task structure: the cache transfers content, not "
            "a generic warm start. Both passes or both fails are weaker."
        ),
        "limits": [
            "Synthetic clustered diagonal-operator family; not the full HENRI model.",
            "CPU only. No latency claim, no benchmark score, no AAII result.",
            "R1 is dim 1024. A pass at R1 is NOT a pass at R3 (dim 65536).",
        ],
    }
    body = json.dumps(out, sort_keys=True).encode()
    out["receipt_sha256"] = hashlib.sha256(body).hexdigest()
    outp = os.path.join(os.environ.get("TEMP", "/tmp"),
                        "zonec_transfer_r1_receipt.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(json.dumps({"structured": s, "unstructured": u}, indent=2))
    print("VERDICT_STRUCTURED=" + verdict_s)
    print("VERDICT_UNSTRUCTURED=" + verdict_u)
    print("RECEIPT_SHA256=" + out["receipt_sha256"])
    print("STORE_AFTER=%d FAMILY_AFTER=%d" % (n_after, fam_after))


if __name__ == "__main__":
    main()
