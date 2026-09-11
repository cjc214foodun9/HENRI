#!/bin/bash
# Pin the DSN mismatch: masked file content, env after source, resolver outputs.
echo "== ENV FILE (password masked) =="
sed -E 's#(ZONE_C_PROD_DSN=postgres://[^:]+:)[^@]+@#\1***@#' /workspace/zonec_prod.env
echo "== PORT LITERALS IN FILE =="
grep -o "543[0-9]\|10100" /workspace/zonec_prod.env | sort | uniq -c
echo "== ZONE_C_ENV LINE PRESENT? =="
grep -c "^ZONE_C_ENV=" /workspace/zonec_prod.env
echo "== AFTER SOURCE: ENV (masked) =="
set -a; source /workspace/zonec_prod.env; set +a
env | grep -E "^ZONE_C" | sed -E 's#(PROD_DSN=postgres://[^:]+:)[^@]+@#\1***@#'
echo "== RESOLVER PROD (masked) =="
cd "/workspace/henri_semantic_wt/HENRI V2" && PYTHONPATH=. /venv/main/bin/python - <<'PY'
import os, re
from zone_c_env import resolve_zone_c_dsn
dsn = resolve_zone_c_dsn()
print(re.sub(r'(postgres://[^:]+:)[^@]+@', r'\1***@', dsn))
PY
echo "== RESOLVER DEV (masked, unset prod) =="
cd "/workspace/henri_semantic_wt/HENRI V2" && env -u ZONE_C_ENV -u ZONE_C_PROD_DSN PYTHONPATH=. /venv/main/bin/python - <<'PY'
import re
from zone_c_env import resolve_zone_c_dsn
dsn = resolve_zone_c_dsn()
print(re.sub(r'(postgres://[^:]+:)[^@]+@', r'\1***@', dsn))
PY
