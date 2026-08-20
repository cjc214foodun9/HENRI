# -*- coding: utf-8 -*-
"""Phase 8.38 plan probe: does the production query use the HNSW index?"""
import json
import os

import psycopg

NUM_BLOCKS = 8192
q_list = "[" + ",".join(f"{v:.6f}" for v in [0.01] * 2000) + "]"
SQL = """
SELECT engram_wave_bytes,
       1 - (semantic_index <=> %s::vector) AS similarity,
       EXTRACT(EPOCH FROM (now() - timestamp)) / 3600.0 AS age_hours
FROM phylogenetic_engrams_65536
WHERE timestamp > now() - (%s || ' hours')::interval
ORDER BY semantic_index <=> %s::vector
LIMIT %s
"""
SQL_NO_AGE = """
SELECT engram_wave_bytes,
       1 - (semantic_index <=> %s::vector) AS similarity,
       EXTRACT(EPOCH FROM (now() - timestamp)) / 3600.0 AS age_hours
FROM phylogenetic_engrams_65536
ORDER BY semantic_index <=> %s::vector
LIMIT %s
"""


def plan(conn, sql, params, label):
    cur = conn.cursor()
    cur.execute("EXPLAIN (ANALYZE, BUFFERS, COSTS, FORMAT JSON) " + sql, params)
    (rows,) = cur.fetchone()
    node = rows[0]["Plan"]
    total = node.get("Actual Total Time", -1)
    loops = node.get("Actual Loops", 1)
    print(f"--- {label}: total_ms={total:.3f} loops={loops}")

    def walk(n, depth=0):
        name = n.get("Node Type")
        st = n.get("Startup Cost")
        tt = n.get("Actual Total Time")
        rows_a = n.get("Actual Rows")
        rows_e = n.get("Plan Rows")
        extra = ""
        if name == "Index Scan":
            extra = f" idx={n.get('Index Name')}"
        elif name == "Seq Scan":
            extra = f" filt={n.get('Filter', '')[:80]}"
        elif name in ("Sort", "Incremental Sort"):
            extra = f" sort_key={n.get('Sort Key', '')}"
        print(f"{'  ' * depth}{name} startup={st:.1f} total={tt:.1f}ms "
              f"rows={rows_a}/{rows_e}{extra}")
        for c in n.get("Plans", []):
            walk(c, depth + 1)

    walk(node)


def main():
    dsn = os.environ["ZONE_C_PROD_DSN"]
    with psycopg.connect(dsn, connect_timeout=8) as conn:
        plan(conn, SQL, (q_list, 8760.0, q_list, 15), "production+age")
        plan(conn, SQL_NO_AGE, (q_list, q_list, 15), "no-age")


if __name__ == "__main__":
    main()
