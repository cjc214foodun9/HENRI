"""
Stage-0b-rev frozen parameter generator (B, W, S) — SINGLE deterministic run.
=============================================================================
Reference 3 (gpt-5.6-sol) binding; pre-registration vla_stage0b_rev_contract.md.

Outputs: vla_stage0b_rev_params.npz
  - W        (192, 4) float32   RFF frequency matrix, N(0,1)
  - b        (192,)  float32    RFF phase bias, U[0, 2pi)
  - k        (16, 384) float32  VSA slot keys, N(0,1)
  - calib_mean (4,) float64     from CALIBRATION episodes only
  - calib_std  (4,) float64     from CALIBRATION episodes only
  - meta: seed, m, created_utc, input_sha (calib obs bytes)
"""
import hashlib, json, pathlib, sys

import numpy as np

SEED = 20260824
M = 192
OUT = pathlib.Path(__file__).resolve().parent / "vla_stage0b_rev_params.npz"
CORPUS_DIR = pathlib.Path(__file__).resolve().parent / "vla_stage0c_corpus"


def load_corpus(corpus_dir):
    """Load the verified corpus; return {episode_seed: [records]} in numeric order."""
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    episodes = {}
    for fname, finfo in manifest["files"].items():
        recs = []
        for line in (corpus_dir / fname).read_text(encoding="utf-8").splitlines():
            recs.append(json.loads(line))
        episodes[int(finfo["episode_seed"])] = recs
    return episodes


def load_calib_obs():
    """Return float32 (171,4) calibration obs_t.

    Established split (Reference 3 + sealed Stage-0c audit ce697efd):
    calibration = first 10 episodes in LEXICOGRAPHIC FILENAME ORDER =
    seeds {101,1010,1111,1212,1313,1414,1515,202,303,404} -> 171 records;
    evaluation = last 5 = {505,606,707,808,909} -> 133 records.
    Numeric-seed-order first-10 (213/91) is REJECTED: it would change the
    calibration partition vs the sealed prior audit, breaking comparability.
    """
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    files = sorted(manifest["files"].items())  # lexicographic filename order
    assert len(files) == 15, f"expected 15 episodes, got {len(files)}"
    rows = []
    for fname, finfo in files[:10]:
        for line in (CORPUS_DIR / fname).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            rows.append(np.asarray(rec["obs_t"], dtype=np.float32))
    X = np.stack(rows)  # (171, 4)
    assert X.shape[0] == 171, f"expected 171 calibration obs_t, got {X.shape[0]}"
    return X


def main():
    rng = np.random.default_rng(SEED)
    X = load_calib_obs()

    W = rng.normal(0.0, 1.0, size=(M, 4)).astype(np.float32)
    b = rng.uniform(0.0, 2.0 * np.pi, size=(M,)).astype(np.float32)
    k = rng.normal(0.0, 1.0, size=(16, 384)).astype(np.float32)

    calib_mean = X.mean(axis=0)
    calib_std = X.std(axis=0)
    # Guard against zero std (a frozen dim with no spread): replace 0 with 1.
    calib_std = np.where(calib_std < 1e-12, 1.0, calib_std)

    input_sha = hashlib.sha256(X.tobytes()).hexdigest()
    meta = {
        "seed": SEED,
        "m": M,
        "n_calib": int(X.shape[0]),
        "created_utc": "2026-08-24T00:00:00Z",  # set by caller if needed
        "input_sha": input_sha,
        "spec": "contract vla_stage0b_rev_contract.md",
    }

    np.savez_compressed(
        OUT,
        W=W, b=b, k=k, calib_mean=calib_mean, calib_std=calib_std,
        meta=json.dumps(meta).encode("utf-8"),
    )
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print("PARAMS_NPZ_SHA", digest[:24])
    print("PARAMS_SHAPES", W.shape, b.shape, k.shape)
    print("INPUT_SHA", input_sha[:24], "N_CALIB", X.shape[0])
    print("CALIB_STD", calib_std.tolist())


if __name__ == "__main__":
    sys.exit(main())
