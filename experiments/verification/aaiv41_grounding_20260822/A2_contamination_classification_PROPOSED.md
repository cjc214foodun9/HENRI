# Amendment A2 (PROPOSED - pending ratification) - 264-item contamination gate
- Status: PROPOSED. A1 (RATIFIED) pinned corpus aggregate b20b5144... + detector v3.1.
- Full-surface scan (264 items: HE 164 prompt+test+imports, MBPP 100 prompt+test_list+imports)
  fires 5 files / 9 hits; the 30-item pilot surface was CLEAN.
- Classification (OBSERVED, classify_contamination_264.py):
  * itertools.rst - 1 genuine hit: `def is_prime(n):` (stdlib Recipes is_prime example)
    vs HumanEval/31 -> BM25 would likely retrieve a complete correct solution -> real leakage.
  * collections.rst / re.rst - prose shingles ("from the right side", "return tuple containing all")
    firing only because v3.1 counts single prose-common keywords (from/return) as code signals.
  * random.rst / statistics.rst - bare `import random` / `import math` lines (generic idiom).
- A2 actions (if ratified):
  1. Corpus: excise itertools.rst Recipes section; update itertools.rst sha256 in manifest;
     new corpus aggregate (replaces ratified b20b5144...).
  2. Detector v3.2: v3.1 rules + exclude bare import lines (no underscore/compound) and
     shingles whose only code signal is a single prose-common keyword; compound code lines
     (def/class/= />>> / underscore identifiers) still fire.
  3. No change to the 264 surface, arms, prompts, pairing, or kill criterion.
     Historical v3.1 blocked receipts preserved. Expected post-A2 preflight: CLEAN.
- Launch remains BLOCKED until ratification (fail-closed per A1).
