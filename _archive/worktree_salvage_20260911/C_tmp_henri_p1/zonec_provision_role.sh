#!/bin/bash
# Zone C runtime role + sealed DSN (never printed) + grants.
set -e
PSQL="sudo -u postgres /usr/lib/postgresql/16/bin/psql -v ON_ERROR_STOP=1"

if ! $PSQL -tAc "SELECT 1 FROM pg_roles WHERE rolname='henri_runner'" | grep -q 1; then
  PW=$(openssl rand -hex 24)
  $PSQL -c "CREATE ROLE henri_runner LOGIN PASSWORD '$PW';"
  umask 077
  printf 'ZONE_C_PROD_DSN=postgresql://henri_runner:%s@127.0.0.1:5432/henri\n' "$PW" > /workspace/zonec_prod.env
  chmod 600 /workspace/zonec_prod.env
  echo ROLE_CREATED
else
  echo ROLE_EXISTS
fi

$PSQL -d henri -c "GRANT CONNECT ON DATABASE henri TO henri_runner;" >/dev/null
$PSQL -d henri -c "GRANT USAGE ON SCHEMA public TO henri_runner;" >/dev/null
$PSQL -d henri -c "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO henri_runner;" >/dev/null
$PSQL -d henri -c "GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO henri_runner;" >/dev/null
$PSQL -d henri -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO henri_runner;" >/dev/null

HBA=/etc/postgresql/16/main/pg_hba.conf
grep -q "127.0.0.1/32.*scram" "$HBA" || echo "host all all 127.0.0.1/32 scram-sha-256" >> "$HBA"
sudo -u postgres /usr/lib/postgresql/16/bin/pg_ctl -D /workspace/pgdata reload >/dev/null
echo HBA_RELOADED

DSN=$(grep ZONE_C_PROD_DSN /workspace/zonec_prod.env | cut -d= -f2-)
ZONE_C_PROD_DSN="$DSN" /venv/main/bin/python -c "
import os, psycopg
conn = psycopg.connect(os.environ['ZONE_C_PROD_DSN'], connect_timeout=10)
with conn.cursor() as cur:
    cur.execute('SELECT current_user, current_database()')
    print('CONNECT_OK', cur.fetchone())
conn.close()
"
echo PROVISION_ROLE_DONE
