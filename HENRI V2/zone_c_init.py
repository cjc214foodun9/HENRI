import asyncio
import logging
import os
from phylogenetic_memory import EngramStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ZoneC_Init")

async def main():
    logger.info("Starting Zone C Phylogenetic Engram Schema Initialization...")
    
    # Get DSN from environment or default to local TimescaleDB instance
    dsn = os.environ.get("TIMESCALEDB_DSN", "postgresql://user:pass@localhost:5432/henri")
    logger.info(f"Connecting to database (credentials hidden): {dsn.split('@')[-1]}")
    
    store = EngramStore(dsn)
    
    try:
        await store.initialize_schema()
        logger.info("Successfully bound continuous wave storage to discrete TimescaleDB hypertable.")
    except Exception as e:
        logger.error(f"Failed to initialize Zone C schema: {e}")
    finally:
        await store.close()

if __name__ == "__main__":
    asyncio.run(main())
