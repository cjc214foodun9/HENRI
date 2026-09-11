#!/bin/bash
# Zone C connectivity diagnostic — NEVER prints the password.
echo "== DSN PARSE (password masked) =="
/venv/main/bin/python - <<'PY'
import re, psycopg
from pathlib import Path
content = Path('/workspace/zonec_prod.env').read_text()
m = re.search(r'ZONE_C_PROD_DSN=(.+)', content)
if not m:
    print('NO_DSN_LINE')
    raise SystemExit
dsn = m.group(1).strip()
try:
    info = psycopg.conninfo.conninfo_to_dict(dsn)
    print({k: ('***' if k == 'password' else v) for k, v in info.items()})
except Exception as e:
    print('PARSE_FAIL', type(e).__name__, str(e)[:200])
    raise SystemExit
try:
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        print('CONNECT_OK', conn.execute('select current_database(), current_user').fetchone())
except Exception as e:
    print('CONNECT_FAIL', type(e).__name__, str(e)[:400])
PY
echo "== ROLES/DBS (peer admin, port 5432) =="
sudo -u postgres psql -p 5432 -tAc "SELECT rolname FROM pg_roles WHERE rolname='henri_runner'" 2>&1
sudo -u postgres psql -p 5432 -tAc "SELECT datname FROM pg_database WHERE datname='henri'" 2>&1
echo "== ORIGINAL PREFLIGHT EXCEPTION =="
grep -E "psycopg|OperationalError|FATAL|refused|denied|authentication|Exception" /tmp/g4_preflight_stdout.log 2>/dev/null | head -10
echo "== LISTENERS =="
ss -tlnp 2>/dev/null | grep -E ":5432|:10100"
