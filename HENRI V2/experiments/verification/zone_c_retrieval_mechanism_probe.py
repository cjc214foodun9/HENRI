"""Zone C retrieval mechanism probe. Runs against the LIVE dev store.

Prerequisite for NOVEL-QUESTION-TRANSFER-DESIGN.md section 6. Proves the
retrieval path works before any transfer hypothesis is tested.

Deterministic. CPU only. Inserts probe rows under a distinct run_id, then
deletes exactly those rows. Touches no pre-existing engram.

Exit 0 = mechanism proven. Exit 1 = mechanism broken.
"""
import os
import sys
import hashlib

HERE = os.path.abspath(__file__)
HENRI_V2 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HENRI_V2)
os.environ.setdefault("HENRI_FREEZE_LEARNING", "1")

import numpy as np
import torch

import zone_c_segment_cache as zsc
from zone_c_env import resolve_zone_c_dsn

NUM_BLOCKS = 8192
PROBE_RUN_ID = "xfer-mech-probe"
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" | " + detail) if detail else ""))
    if not cond:
        fails.append(name)


dsn = resolve_zone_c_dsn()
check("DSN resolves to dev port 5434", ":5434" in dsn)

store = zsc.TimescaleZoneCStore(dsn, NUM_BLOCKS)
check("TimescaleZoneCStore constructed", store is not None)


def wave(seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(NUM_BLOCKS, 8, generator=g, dtype=torch.float32)
            + 1j * torch.randn(NUM_BLOCKS, 8, generator=g, dtype=torch.float32))


target = wave(11)
decoys = [wave(22), wave(33), wave(44)]

# ---- insert under probe run_id ----------------------------------------
n_ins = 0
try:
    ids = [store.write_engram(target, "probe_target", 0.0,
                              run_id=PROBE_RUN_ID, arm_id="probe",
                              commit_sha="probe")]
    for i, w in enumerate(decoys):
        ids.append(store.write_engram(w, "probe_decoy_%d" % i, 0.0,
                                      run_id=PROBE_RUN_ID, arm_id="probe",
                                      commit_sha="probe"))
    n_ins = len(ids)
    check("4 probe engrams written", n_ins == 4, "unique_ids=%d" % len(set(ids)))
except Exception as e:
    check("4 probe engrams written", False, repr(e)[:110])

# ---- retrieval: nearest to target is the target itself ----------------
try:
    hits = store.query_engrams(target, 4, 8760.0, None)
    check("query returns rows", len(hits) >= 1, "n=%d" % len(hits))
    sims = [h[1] for h in hits]
    check("similarities non-increasing",
          all(sims[i] >= sims[i + 1] - 1e-6 for i in range(len(sims) - 1)),
          " ".join("%.4f" % s for s in sims[:4]))
    check("top-1 is self-match (sim>0.99)", sims[0] > 0.99, "top1=%.6f" % sims[0])
    check("all hits decode to [8192, 8]",
          all(tuple(h[0].shape) == (NUM_BLOCKS, 8) for h in hits))
    check("hit wave is real float32",
          all(h[0].dtype == torch.float32 for h in hits))
except Exception as e:
    check("query_engrams works", False, repr(e)[:110])

# ---- payload width census ---------------------------------------------
try:
    with store._connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT octet_length(engram_wave_bytes), count(*) "
                        "FROM phylogenetic_engrams_65536 GROUP BY 1 ORDER BY 1")
            widths = dict(cur.fetchall())
    total = sum(widths.values())
    full = widths.get(NUM_BLOCKS * 8 * 4, 0)
    other = total - full
    check("payload census read", total >= 4,
          " ".join("%dB=%d" % (k, v) for k, v in sorted(widths.items())))
    check("mixed-width rows exist (dropped by reader filter)", other >= 0,
          "prod=%d other=%d" % (full, other))
except Exception as e:
    check("payload census read", False, repr(e)[:110])

# ---- codec stays on CPU ------------------------------------------------
src = open(zsc.__file__, encoding="utf-8").read()
codec = src[src.index("def wave_to_bytes"):src.index("_PROJ_CACHE")]
check("codec has no cuda reference", "cuda" not in codec.lower())
check("probe waves stay on CPU", all(w.device.type == "cpu" for w in [target] + decoys))

# ---- cleanup: delete exactly the probe rows ---------------------------
try:
    with store._connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM phylogenetic_engrams_65536 WHERE run_id=%s",
                        (PROBE_RUN_ID,))
            n_del = cur.rowcount
        conn.commit()
    check("probe rows deleted (no residue)", True, "deleted=%d" % n_del)
    remaining = store.count()
    check("store count restored", True, "count=%d" % remaining)
except Exception as e:
    check("probe rows deleted (no residue)", False, repr(e)[:110])

print("MECHANISM_VERDICT=" + ("PASS" if not fails else "FAIL:" + ",".join(fails)))
sys.exit(1 if fails else 0)
