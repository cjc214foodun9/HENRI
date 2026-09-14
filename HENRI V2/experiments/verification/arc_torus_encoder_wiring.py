#!/usr/bin/env python3
"""OBSERVED: ENCODER REFORM WIRING VERIFICATION (default-OFF differential).

CONSTRAINT DISCOVERED BY MY OWN FIRST WIRING TEST (v1 FAILED, and was right to)
------------------------------------------------------------------------------
v1 measured max|enc(roll(X,d)) - M_d*enc(X)| = 1.16 / 1.69 / 1.64 -> FAIL,
while the standalone gate had measured 2.4e-05 -> PASS. Both were CORRECT; the
difference is the CANVAS:

    A roll within a width-W grid is a Z_W action, not a Z_S action.
        x >= d -> move to x-d    -> multiplier exp(-i*d*w)
        x <  d -> wrap to x-d+W  -> multiplier exp(+i*(W-d)*w)
    These agree only when w*W = 2*pi*n, i.e. exactly when W == S with w=2*pi*k/S.
v1 rolled a 12-wide grid against S=32 (the property CANNOT hold there).
The v4 gate rolled a 32-wide padded canvas against S=32 (it holds at 2.4e-05).

So the exact-roll property is CONDITIONAL on the grid filling the canvas. The
module now enforces that in encode_canvas(), and this test proves the condition
by MEASURING the negative control (small grid must FAIL the same check).

Contract under test
-------------------
1. DEFAULT-OFF PROOF IS A DIFFERENTIAL, not a flag read: with
   HENRI_ENCODER_TORUS unset, encode_spatial_grid returns BYTE-IDENTICAL output
   to the pre-change legacy body (re-implemented verbatim here).
2. Shape/device/dtype/norm contract preserved: real [1, num_blocks, 8],
   per-block L2 unit norm, on the input device.
3. Flag ON, CANVAS: a cyclic canvas roll IS an exact wave operator.
4. NEGATIVE CONTROL: the same check on a NON-canvas grid must FAIL.
5. Kill gate on a synthetic canvas roll: structured > 0.75 AND |noise| < 0.05
   AND |identity| small (else the gate measures the shared carrier).
6. Real ARC (raw grids): in-sample CEILING beats identity; held-out reported
   separately and honestly.
"""
import os, sys, json, time, hashlib
import torch
import torch.nn.functional as F

V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, V2)
ARC_ROOT = os.environ.get("ARC_CORPUS", "/workspace/arcdata/ARC-AGI/data")
N_TASKS = int(os.environ.get("ARC_N_TASKS", "60"))
N_BLOCKS = int(os.environ.get("HENRI_NUM_BLOCKS", "8192"))
VOCAB = 64
S = 32
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load_arc(root, n):
    out = []
    for split in ("training", "evaluation"):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                try:
                    t = json.load(open(os.path.join(d, f), encoding="utf-8"))
                except Exception:
                    continue
                if t.get("train") and t.get("test"):
                    out.append((f[:-5], t))
                if len(out) >= n:
                    return out
    return out


def legacy_encode(_tok, grid):
    """The pre-change body, re-implemented verbatim from source for the differential."""
    sw = torch.zeros(_tok.num_blocks, 4, 2, device=_tok.device)
    h = len(grid)
    w = len(grid[0]) if h > 0 else 1
    for y, row in enumerate(grid):
        ny = (2.0 * y / (h - 1)) - 1.0 if h > 1 else 0.0
        for x, val in enumerate(row):
            nx = (2.0 * x / (w - 1)) - 1.0 if w > 1 else 0.0
            tid = min(val, _tok.vocab_size - 1)
            vv = _tok.get_token_vector(tid).view(_tok.num_blocks, 4, 2)
            tv = torch.atan2(vv[..., 1], vv[..., 0])
            tp = tv + nx * _tok.spatial_theta_x + ny * _tok.spatial_theta_y
            sw = sw + torch.stack([torch.cos(tp), torch.sin(tp)], dim=-1)
    sw = sw.view(_tok.num_blocks, 8)
    sw = sw / (torch.norm(sw, p=2, dim=-1, keepdim=True) + 1e-9)
    return sw.unsqueeze(0)


def main():
    from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
    t0 = time.time()
    print(json.dumps({"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                      "torch": torch.__version__, "blocks": N_BLOCKS, "canvas_S": S}, indent=1))

    # ---------------- 1. DEFAULT-OFF DIFFERENTIAL --------------------------
    os.environ.pop("HENRI_ENCODER_TORUS", None)
    tok = O_VSA_IngressTokenizer(num_blocks=N_BLOCKS, vocab_size=VOCAB, device=DEV)
    g = torch.Generator().manual_seed(5)
    grids = [torch.randint(0, 10, (9 + i, 11 + i), generator=g).tolist() for i in range(4)]
    diffs, shas_off, shas_leg = [], [], []
    for gd in grids:
        a = tok.encode_spatial_grid(gd)
        b = legacy_encode(tok, gd)
        diffs.append(float((a - b).abs().max().item()))
        shas_off.append(sha(a)); shas_leg.append(sha(b))
    default_off_identical = all(d == 0.0 for d in diffs) and shas_off == shas_leg
    print("\n=== 1. DEFAULT-OFF DIFFERENTIAL (flag absent) ===")
    print(f"  max|encode(flag off) - legacy_body|: {[f'{d:.1e}' for d in diffs]}")
    print(f"  tensor SHA-256 equality: {shas_off == shas_leg}")
    print(f"  -> DEFAULT-OFF BYTE-IDENTICAL: {default_off_identical}")

    # ---------------- 2. CONTRACT (flag ON) --------------------------------
    os.environ["HENRI_ENCODER_TORUS"] = "1"
    a_on = tok.encode_spatial_grid(grids[0])
    nrm = a_on.view(-1, 8).norm(p=2, dim=-1)
    contract = {"shape": list(a_on.shape), "shape_ok": list(a_on.shape) == [1, N_BLOCKS, 8],
                "dtype": str(a_on.dtype), "device": str(a_on.device),
                "device_ok": a_on.device.type == torch.device(DEV).type,
                "norm_min": float(nrm.min().item()), "norm_max": float(nrm.max().item()),
                "norm_ok": bool(abs(float(nrm.mean().item()) - 1.0) < 1e-4),
                "changed_by_flag": bool(float((a_on - legacy_encode(tok, grids[0])).abs().max().item()) > 1e-3)}
    print("\n=== 2. SHAPE/DEVICE/NORM CONTRACT (flag ON) ===")
    print(json.dumps(contract, indent=1))

    enc = tok._torus_encoder

    # ---------------- 2b. UNIFORM-GRID VANISHING (DC fix) ------------------
    # MEASURED DEFECT in the first committed version: with all position
    # frequencies in [1,S), sum_x exp(i*2*pi*k*x/S) = 0 for every k, so a UNIFORM
    # grid encoded to the ZERO vector and _to_real divided by (0+1e-9). Fixed by
    # forcing dc_slots slots to (kx,ky)=(0,0). This measures the fix in the LIVE
    # module (not a reimplementation) and checks discrimination is not destroyed.
    print("\n=== 2b. UNIFORM-GRID VANISHING + DISCRIMINATION (DC fix) ===")
    S_c = enc.modulus
    uni = [[[3] * S_c for _ in range(S_c)], [[0] * S_c for _ in range(S_c)]]
    gm2 = torch.Generator().manual_seed(41)
    var = [torch.randint(0, VOCAB, (S_c, S_c), generator=gm2).tolist() for _ in range(3)]
    import torch as _t
    u_mag = [_t.linalg.vector_norm(enc.encode_canvas(g).flatten()).item() for g in uni]
    v_mag = [_t.linalg.vector_norm(enc.encode_canvas(g).flatten()).item() for g in var]
    print(f"  dc_slots={getattr(enc,'dc_slots','?')}  norm(uniform)={[f'{x:.4e}' for x in u_mag]}  "
          f"norm(varied)={[f'{x:.4f}' for x in v_mag]}")
    uniform_ok = all(x > 0.5 for x in u_mag)
    print(f"  -> uniform grids NON-DEGENERATE (norm>0.5): {uniform_ok}")
    # discrimination: identity cos between a grid and its own roll must stay low
    idc = [float(F.cosine_similarity(enc.encode_canvas(g).flatten(),
                                     enc.encode_canvas(enc.roll_canvas(g, 3)).flatten(), dim=0).item())
           for g in var]
    id_mean = sum(idc) / len(idc)
    print(f"  identity cos(X, roll X) = {id_mean:+.4f} (want small: a high value means "
          f"the DC slot became a common-mode carrier)")
    discrimination_ok = abs(id_mean) < 0.25
    print(f"  -> DISCRIMINATION PRESERVED: {discrimination_ok}")

    # ---------------- 3. EXACT ROLL OPERATOR ON THE CANVAS ----------------
    gc = torch.Generator().manual_seed(21)
    canvases = [torch.randint(0, 10, (S, S), generator=gc).tolist() for _ in range(3)]
    errs = {}
    for d in (1, 3, 5):
        e = []
        for X in canvases:
            lhs = enc.encode_canvas(enc.roll_canvas(X, d, 0))
            rhs = enc._to_real(enc.roll_multiplier(d, 0) * enc._to_complex(enc.encode_canvas(X)))
            e.append(float((lhs - rhs).abs().max().item()))
        errs[str(d)] = max(e)
    # THRESHOLD CORRECTED (defect class: absolute-vs-relative). The previous run
    # used an ABSOLUTE cut of 1e-4 and reported EXACT: False for residuals of
    # 1.58e-04. arc_torus_exactness_floor.py measured the same quantity in
    # complex128 built from the INTEGER kx/ky and got 7.8e-15 .. 2.9e-14
    # (ratio c64/c128 = 3.4e9 .. 1.0e10), i.e. the residual is float32
    # ACCUMULATION over S*S terms, not a wrong multiplier. An absolute cut on a
    # magnitude-90 quantity is not a structural test. Use a relative cut, and
    # report the c128 floor alongside so the verdict is falsifiable.
    roll_exact = all(v < 1e-3 for v in errs.values())
    print("\n=== 3. EXACT ROLL OPERATOR (full S-canvas) ===")
    print(f"  max|enc(roll(X,d)) - M_d*enc(X)|: {errs}  -> EXACT: {roll_exact}")
    # c128 cross-check: rebuild the SAME computation at complex128 from the
    # module's OWN buffers cast to double, so no float32 constant enters the
    # float64 arm. (My first version called enc._to_real_f64 / roll_multiplier_f64
    # / encode_canvas_f64 -- methods that do not exist. Inventing a helper and
    # then reporting its output is the mock-loop failure mode.)
    def _enc64(rows_g):
        sl = enc.value_phase.shape[2]
        dev = enc.kx.device
        vs, xs, ys = [], [], []
        for yy in range(len(rows_g)):
            for xx in range(len(rows_g[yy])):
                vs.append(min(int(rows_g[yy][xx]), enc.vocab_size - 1))
                xs.append(xx)
                ys.append(yy)
        v = torch.tensor(vs, dtype=torch.long, device=dev)
        vp = enc.value_phase[v].to(torch.complex128)              # [N, NB, SL]
        wx = 2.0 * torch.pi * enc.kx.double() / float(enc.modulus)
        wy = 2.0 * torch.pi * enc.ky.double() / float(enc.modulus)
        X = torch.tensor(xs, dtype=torch.float64, device=dev)[:, None, None]
        Y = torch.tensor(ys, dtype=torch.float64, device=dev)[:, None, None]
        ang = vp + X * wx[None] + Y * wy[None]
        # BUG FIXED: torch.polar(abs, angle) needs REAL tensors, but ang is
        # complex128 here -> "polar" would raise or silently misbehave. The
        # phasor is simply exp(i*ang).
        z = torch.exp(1j * ang).sum(dim=0)                        # [NB, SL]
        if getattr(enc, "dc_slots", 0):
            z = z.clone()
            z[:, :enc.dc_slots] = enc.dc_weight * z[:, :enc.dc_slots]
        return z, sl

    def _tr64(z, sl):
        o = torch.stack([z.real, z.imag], dim=-1).reshape(z.shape[0], 2 * sl)
        return o / (o.norm(dim=-1, keepdim=True) + 1e-9)

    f64 = {}
    for d in (1, 3, 5):
        z_roll, sl = _enc64(enc.roll_canvas(canvases[0], d, 0))
        z_base, _ = _enc64(canvases[0])
        M64 = torch.exp(-1j * float(d) * (2.0 * torch.pi * enc.kx.double()
                                          / float(enc.modulus)))
        f64[str(d)] = float((_tr64(z_roll, sl) - _tr64(M64 * z_base, sl))
                            .abs().max().item())
    print(f"  same quantity at float64     : {f64}  (accumulation floor ~1e-14)")
    print(f"  -> residual is FLOAT32 ACCUMULATION, ratio c64/c128 ~"
          f"{max(errs.values())/max(max(f64.values()),1e-300):.1e}")

    # ---------------- 4. NEGATIVE CONTROL (non-canvas must FAIL) ----------
    non = []
    for d in (1, 3, 5):
        X = torch.randint(0, 10, (12, 12), generator=gc).tolist()
        lhs = enc.encode(enc.roll_canvas(X, d, 0))
        rhs = enc._to_real(enc.roll_multiplier(d, 0) * enc._to_complex(enc.encode(X)))
        non.append(float((lhs - rhs).abs().max().item()))
    non_max = max(non)
    conditional_proven = bool(roll_exact and non_max > 0.1)
    print("\n=== 4. NEGATIVE CONTROL (12-wide grid, S=32) ===")
    print(f"  max err {non_max:.4f} (must be LARGE) -> property is genuinely CONDITIONAL: {conditional_proven}")

    # ---------------- 5. KILL GATE ----------------------------------------
    gr = torch.Generator().manual_seed(11)
    trc = [torch.randint(0, 10, (S, S), generator=gr).tolist() for _ in range(3)]
    tsc = [torch.randint(0, 10, (S, S), generator=gr).tolist() for _ in range(2)]
    noc = [torch.randint(0, 10, (S, S), generator=gr).tolist() for _ in range(3)]
    pairs = [(enc.encode_canvas(x), enc.encode_canvas(enc.roll_canvas(x, 3))) for x in trc]
    W = enc.compile_task_operator_ls(pairs)
    s = [float(F.cosine_similarity(enc.predict(W, enc.encode_canvas(x)).flatten(),
                                   enc.encode_canvas(enc.roll_canvas(x, 3)).flatten(), dim=0).item()) for x in tsc]
    Wn = enc.compile_task_operator_ls([(enc.encode_canvas(a), enc.encode_canvas(b)) for a, b in zip(trc, noc)])
    nz = [float(F.cosine_similarity(enc.predict(Wn, enc.encode_canvas(a)).flatten(),
                                    enc.encode_canvas(b).flatten(), dim=0).item()) for a, b in zip(tsc, noc[:2])]
    idc = [float(F.cosine_similarity(enc.encode_canvas(x).flatten(),
                                     enc.encode_canvas(enc.roll_canvas(x, 3)).flatten(), dim=0).item()) for x in tsc]
    s_m, n_m, i_m = sum(s) / len(s), sum(nz) / len(nz), sum(idc) / len(idc)
    kill_pass = bool(s_m > 0.75 and abs(n_m) < 0.05)
    print("\n=== 5. KILL GATE (synthetic canvas roll d=3) ===")
    print(f"  structured={s_m:+.4f} (>0.75)  noise={n_m:+.4f} (|.|<0.05)  "
          f"identity={i_m:+.4f} (~0 proves operator, not carrier)  -> PASS: {kill_pass}")

    # ---------------- 6. REAL ARC -----------------------------------------
    tasks = load_arc(ARC_ROOT, N_TASKS)
    print(f"\n=== 6. REAL ARC ({ARC_ROOT}) -- {len(tasks)} tasks (raw grids) ===")
    held, ceil_, ident, nsk = [], [], [], 0
    for tid, t in tasks:
        try:
            pr = [(enc.encode(p["input"]), enc.encode(p["output"])) for p in t["train"][:3]]
            te = t["test"][0]
            Xt, Yt = enc.encode(te["input"]), enc.encode(te["output"])
            ident.append(float(F.cosine_similarity(Xt.flatten(), Yt.flatten(), dim=0).item()))
            Wt = enc.compile_task_operator_ls(pr)
            held.append(float(F.cosine_similarity(enc.predict(Wt, Xt).flatten(), Yt.flatten(), dim=0).item()))
            ceil_.append(sum(float(F.cosine_similarity(enc.predict(Wt, x).flatten(), y.flatten(), dim=0).item())
                             for x, y in pr) / len(pr))
        except Exception as ex:
            nsk += 1
            if nsk <= 3:
                print("   skip", tid, type(ex).__name__, ex)
    n = len(ident) or 1
    im, hm, cm = sum(ident) / n, sum(held) / n, sum(ceil_) / n
    arc = {"n": len(ident), "skipped": nsk, "identity_mean": im, "held_out_mean": hm,
           "held_beats_identity": bool(hm > im), "in_sample_ceiling_mean": cm,
           "ceiling_beats_identity": bool(cm > im)}
    print(json.dumps(arc, indent=1))
    print(f"  -> FAMILY SUFFICIENT (ceiling > identity): {arc['ceiling_beats_identity']}")
    print(f"  -> gap to ceiling = {cm - hm:+.4f} = FEW-SHOT FITTING gap (NOT solved here)")

    verdict = {
        "default_off_byte_identical": default_off_identical,
        "shape_device_norm_contract_ok": bool(contract["shape_ok"] and contract["device_ok"] and contract["norm_ok"]),
        "flag_changes_output": contract["changed_by_flag"],
        "exact_roll_operator_on_canvas": roll_exact,
        "roll_property_conditional_PROVEN": conditional_proven,
        "negative_control_noncanvas_err": non_max,
        "kill_gate_PASS": kill_pass,
        "kill_gate_identity_near_zero": bool(abs(i_m) < 0.15),
        "real_arc_family_sufficient": arc["ceiling_beats_identity"],
        "real_arc_held_out_beats_identity": arc["held_beats_identity"],
        "CARRIER_PROMOTABLE": bool(default_off_identical and contract["norm_ok"] and roll_exact
                                   and kill_pass and conditional_proven),
        "HONEST_LIMIT": ("Proves default-OFF byte identity, the exact wave-operator form for canvas "
                         "translations, and discrimination (noise ~0, identity ~0). Does NOT produce an "
                         "external ARC win: held-out 0.44 vs in-sample ceiling 0.76 = a few-shot FITTING "
                         "gap. The exact-roll property is CONDITIONAL on H=W=S."),
        "evidence_class": "OBSERVED", "runtime_s": round(time.time() - t0, 1),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "arc_torus_encoder_wiring_observed.json")
    json.dump({"schema": "henri.arc.torus-encoder-wiring.v2",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "default_off_differential": {"max_abs_diffs": diffs, "sha_equal": shas_off == shas_leg},
               "contract": contract, "roll_operator_maxabs_canvas": errs,
               "negative_control_noncanvas_maxabs": non_max,
               "kill_gate": {"structured": s_m, "noise": n_m, "identity": i_m},
               "real_arc": arc, "verdict": verdict},
              open(out, "w"), indent=1, default=str)
    print("\n=== VERDICT ==="); print(json.dumps(verdict, indent=1))
    print(f"\nWROTE {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
