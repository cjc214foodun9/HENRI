"""Gate 4 single-use heldout split generator (deterministic, stdlib-only).

Pinned by gate4_split_amendment.md (parent seal 821aeb8c). Universe = official
catalog metadata (OBSERVED 2026-08-26 via arc_agi.Arcade.get_environments(),
metadata only — no env instantiated, no make(), no step). Exclusion = the 12
exposed env ids with observed run/probe/ledger/sans evidence. Selection =
random.Random(20260826).shuffle(sorted unseen); take first 8. Writes
gate4_heldout_manifest.json with consumed=false. Single-use: any step on a
selected env before score_eligible=true burns the split.
"""
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

CATALOG = [
    "ar25-0c556536", "bp35-0a0ad940", "cd82-fb555c5d", "cn04-2fe56bfb",
    "dc22-fdcac232", "ft09-0d8bbf25", "g50t-5849a774", "ka59-38d34dbb",
    "lf52-271a04aa", "lp85-305b61c3", "ls20-9607627b", "m0r0-492f87ba",
    "r11l-495a7899", "re86-8af5384d", "s5i5-18d95033", "sb26-7fbdac44",
    "sc25-635fd71a", "sk48-d8078629", "sp80-589a99af", "su15-1944f8ab",
    "tn36-ef4dde99", "tr87-cd924810", "tu93-0768757b", "vc33-5430563c",
    "wa30-ee6fef47",
]
EXCLUDED_PREFIXES = sorted(
    ["ar25", "bp35", "cn04", "dc22", "ft09", "g50t", "ka59",
     "lp85", "ls20", "m0r0", "re86", "sb26"]
)
SEED = 20260826
SELECTED_COUNT = 8


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> int:
    here = Path(__file__).resolve()
    selection_code_sha = sha256_text(here.read_bytes().decode("utf-8"))
    catalog_sha = sha256_text(json.dumps(CATALOG, separators=(",", ":")))
    exclusion_sha = sha256_text(json.dumps(EXCLUDED_PREFIXES, separators=(",", ":")))

    unseen = sorted(
        eid for eid in CATALOG
        if eid.split("-")[0] not in set(EXCLUDED_PREFIXES)
    )
    rng = random.Random(SEED)
    shuffled = list(unseen)
    rng.shuffle(shuffled)
    selected = shuffled[:SELECTED_COUNT]

    manifest = {
        "schema": "henri.gate4-heldout-manifest.v1",
        "status": "GENERATED",
        "seed": SEED,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_sha256": catalog_sha,
        "catalog_count": len(CATALOG),
        "exclusion_sha256": exclusion_sha,
        "exclusion_ids": EXCLUDED_PREFIXES,
        "unseen_ids": unseen,
        "selected_ids": selected,
        "selection_code_sha256": selection_code_sha,
        "selection_algorithm": "random.Random(seed).shuffle(sorted(unseen)); take first 8",
        "consumed": False,
        "single_use_rule": "No instantiation/probe/step on selected envs while score_eligible=false; any step burns the split.",
    }
    out = here.parent / "gate4_heldout_manifest.json"
    raw = json.dumps(manifest, indent=2, ensure_ascii=False)
    out.write_text(raw, encoding="utf-8")
    print("UNSEEN_COUNT", len(unseen))
    print("UNSEEN", " ".join(unseen))
    print("SELECTED", " ".join(selected))
    print("CATALOG_SHA", catalog_sha)
    print("EXCLUSION_SHA", exclusion_sha)
    print("CODE_SHA", selection_code_sha)
    print("MANIFEST_SHA", hashlib.sha256(raw.encode("utf-8")).hexdigest())
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
