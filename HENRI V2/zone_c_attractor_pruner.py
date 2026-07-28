"""
Zone C Background Attractor Pruner for Project HENRI V2.

Triggers `consolidate_attractors()` on Zone C TimescaleDB segment store to
cluster redundant engrams in vector space, merge them into canonical low-entropy
attractors, and prune redundant source rows to preserve O(1) retrieval speeds.
"""

import logging
import os
import sys
import time
from zone_c_segment_cache import SegmentCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [AttractorPruner] - %(message)s")


class ZoneCAttractorPruner:
    """Background worker for continuous attractor consolidation."""

    def __init__(self, dsn: str = None, num_blocks: int = 8192, cosine_threshold: float = 0.95):
        self.dsn = dsn or os.environ.get("ZONE_C_PROD_DSN", "postgresql://postgres:postgres@localhost:10100/henri")
        self.num_blocks = num_blocks
        self.cosine_threshold = cosine_threshold
        self.cache = SegmentCache(num_blocks=num_blocks, dsn=self.dsn)

    def run_pruning_pass(self, dry_run: bool = False) -> dict:
        """Executes one consolidation pass over stored Zone C engrams."""
        logging.info(f"Running Attractor Pruning Pass (cosine_threshold={self.cosine_threshold})...")
        res = self.cache.consolidate_attractors(cosine_threshold=self.cosine_threshold, dry_run=dry_run)
        logging.info(f"  Pruning Results: {res}")
        return res


def main():
    dsn = os.environ.get("ZONE_C_PROD_DSN", "postgresql://postgres:postgres@localhost:10100/henri")
    pruner = ZoneCAttractorPruner(dsn=dsn, num_blocks=8192, cosine_threshold=0.95)
    res = pruner.run_pruning_pass(dry_run=False)
    print(f"[Done] Zone C Attractor Consolidation: {res}")


if __name__ == "__main__":
    main()
