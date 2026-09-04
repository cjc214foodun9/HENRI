"""goal_dist_var helper for the production runner (2026-08-28).

Repair for BLOCKED_INFRASTRUCTURE: production_arc_run.py:2605 crashed with
`TypeError: type NoneType doesn't define __round__ method` when the EFE
table's goal_distance entries were all None (or absent).

Contract (tests/contract/test_arc_goal_dist.py):
- empty table -> None
- fewer than 2 valid observations -> None (never round(None))
- sample variance (statistics.variance, n-1 denominator) rounded to 6 dp
- None entries excluded
- malformed entries -> None (fail closed, never raise)
"""

import statistics


def compute_goal_dist_var(efe_table):
    """Return rounded sample variance of goal_distance over efe_table rows.

    Returns None when the table is empty, has fewer than two valid
    observations, or contains a malformed entry (fail-closed).
    """
    if not efe_table:
        return None
    goal_dists = []
    for row in efe_table:
        try:
            v = row.get("goal_distance")
            if v is None:
                continue
            goal_dists.append(float(v))
        except (TypeError, ValueError):
            return None
    if len(goal_dists) > 1:
        return round(statistics.variance(goal_dists), 6)
    return None
