import psycopg
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from henri_contract import complex_to_db, DIMS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")

try:
    print("[*] Connecting to local TimescaleDB Docker container...")
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            conn.autocommit = True
            
            print("[*] Ensuring required extensions are enabled...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector CASCADE;")
            
            print("[*] Creating 4096D tables (if not exist)...")
            
            # 1. Canonical Lexicon
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hrr_canonical_lexicon (
                    concept_hash UUID PRIMARY KEY,
                    semantic_label TEXT NOT NULL,
                    domain_tag TEXT,
                    hrr_wavefront vector(8192) NOT NULL,
                    epiplexity_weight FLOAT DEFAULT 1.0,
                    last_verified TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    raw_text TEXT
                );
            """)
            try:
                cur.execute("ALTER TABLE hrr_canonical_lexicon ADD COLUMN IF NOT EXISTS raw_text TEXT;")
            except Exception as migration_err:
                print(f"[WARNING] Migration failed for hrr_canonical_lexicon: {migration_err}")
            print("[+] Table 'hrr_canonical_lexicon' verified.")
            
            # 2. Thermodynamic Ledger
            cur.execute("""
                CREATE TABLE IF NOT EXISTS thermodynamic_ledger (
                    timestamp TIMESTAMPTZ NOT NULL,
                    inference_id UUID NOT NULL,
                    langevin_heat_injected FLOAT NOT NULL,
                    sagnac_error_delta FLOAT NOT NULL,
                    attractor_locked BOOLEAN NOT NULL,
                    hrr_trajectory vector(8192)
                );
            """)
            print("[+] Table 'thermodynamic_ledger' verified.")
            
            # 3. Gemma Latent History
            cur.execute("""
                CREATE TABLE IF NOT EXISTS gemma_latent_history (
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    step_id BIGSERIAL,
                    token_id INT,
                    generated_text TEXT,
                    epiplexity_value DOUBLE PRECISION,
                    sagnac_delta_value DOUBLE PRECISION,
                    langevin_heat_value DOUBLE PRECISION,
                    rate_of_annealing_value DOUBLE PRECISION DEFAULT 0.0,
                    alpha_value DOUBLE PRECISION DEFAULT 1.0,
                    latent_wave_vector vector(8192),
                    PRIMARY KEY (timestamp, step_id)
                );
            """)
            try:
                # Add newer columns if not present
                cur.execute("ALTER TABLE gemma_latent_history ADD COLUMN IF NOT EXISTS rate_of_annealing_value DOUBLE PRECISION DEFAULT 0.0;")
                cur.execute("ALTER TABLE gemma_latent_history ADD COLUMN IF NOT EXISTS alpha_value DOUBLE PRECISION DEFAULT 1.0;")
            except Exception:
                pass
            print("[+] Table 'gemma_latent_history' verified.")
            
            # 4. LoRA Adapters Registry
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lora_adapters_registry (
                    domain_tag TEXT PRIMARY KEY,
                    adapter_path TEXT NOT NULL,
                    sagnac_error_delta FLOAT NOT NULL,
                    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("[+] Table 'lora_adapters_registry' verified.")
            
            # 5. Stirrup Telemetry Ledger
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stirrup_telemetry_ledger (
                    timestamp TIMESTAMPTZ NOT NULL,
                    inference_id UUID NOT NULL,
                    selected_plan_index INT NOT NULL,
                    thermodynamic_stress_cost DOUBLE PRECISION NOT NULL,
                    sigreg_disentanglement_score DOUBLE PRECISION NOT NULL,
                    transduced_motor_token_id INT NOT NULL,
                    actuated_command TEXT NOT NULL,
                    success BOOLEAN NOT NULL
                );
            """)
            print("[+] Table 'stirrup_telemetry_ledger' verified.")
            
            print("[*] Converting tables to TimescaleDB Hypertables...")
            try:
                cur.execute("SELECT create_hypertable('thermodynamic_ledger', 'timestamp');")
                print("[+] thermodynamic_ledger converted to hypertable.")
            except Exception as e:
                if "already a hypertable" not in str(e).lower():
                    print(f"[WARNING] hypertable creation thermodynamic_ledger: {e}")
                    
            try:
                cur.execute("SELECT create_hypertable('gemma_latent_history', 'timestamp');")
                print("[+] gemma_latent_history converted to hypertable.")
            except Exception as e:
                if "already a hypertable" not in str(e).lower():
                    print(f"[WARNING] hypertable creation gemma_latent_history: {e}")

            try:
                cur.execute("SELECT create_hypertable('stirrup_telemetry_ledger', 'timestamp', if_not_exists => TRUE);")
                print("[+] stirrup_telemetry_ledger converted to hypertable.")
            except Exception as e:
                if "already a hypertable" not in str(e).lower():
                    print(f"[WARNING] hypertable creation stirrup_telemetry_ledger: {e}")
            
            print("[*] Setting up TimescaleDB compression policies (12-hour window)...")
            for table in ['thermodynamic_ledger', 'gemma_latent_history', 'stirrup_telemetry_ledger']:
                try:
                    cur.execute(f"ALTER TABLE {table} SET (timescaledb.compress);")
                    cur.execute(f"SELECT add_compression_policy('{table}', INTERVAL '12 hours', if_not_exists => true);")
                    print(f"[+] Compression policy configured for {table}.")
                except Exception as comp_err:
                    if "already exists" not in str(comp_err).lower() and "already has" not in str(comp_err).lower() and "already enabled" not in str(comp_err).lower():
                        try:
                            cur.execute(f"SELECT add_compression_policy('{table}', INTERVAL '12 hours');")
                            print(f"[+] Compression policy configured for {table} (fallback).")
                        except Exception as comp_err2:
                            if "already exists" not in str(comp_err2).lower() and "already has" not in str(comp_err2).lower():
                                print(f"[WARNING] Failed to enable compression policy for {table}: {comp_err2}")
            
            print("[*] Creating StreamingDiskANN index via vectorscale for 4096D hrr_canonical_lexicon...")
            try:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_hrr_wavefront 
                    ON hrr_canonical_lexicon USING diskann (hrr_wavefront vector_cosine_ops);
                """)
                print("[+] StreamingDiskANN index verified on hrr_canonical_lexicon.")
            except Exception as e:
                # Fallback to standard ivfflat or hnsw index if diskann/vectorscale is unavailable
                print(f"[INFO] DiskANN index creation failed or unsupported: {e}. Trying standard hnsw/ivfflat index...")
                try:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_hrr_wavefront_hnsw
                        ON hrr_canonical_lexicon USING hnsw (hrr_wavefront vector_cosine_ops);
                    """)
                    print("[+] HNSW index verified on hrr_canonical_lexicon.")
                except Exception as e2:
                    print(f"[WARNING] Standard index creation also failed: {e2}")
            
            print("[*] Creating Materialized View for continuous learning updates...")
            try:
                cur.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS continuous_learning_view
                    AS
                    SELECT 
                        time_bucket('1 minute', timestamp) AS bucket,
                        AVG(sagnac_error_delta) as avg_error_energy,
                        MAX(langevin_heat_injected) as peak_thermal_variance
                    FROM thermodynamic_ledger
                    GROUP BY bucket;
                """)
                print("[+] Materialized view verified.")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"[WARNING] Materialized view creation failed: {e}")
            
            # 4. Seed Zone C Axioms if table is empty
            cur.execute("SELECT COUNT(*) FROM hrr_canonical_lexicon;")
            count = cur.fetchone()[0]
            if count == 0:
                print("[*] Seeding hrr_canonical_lexicon with Zone C axioms in 4096D...")
                try:
                    import sys
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    from physics_manifold import (
                        CODATA_CONSTANTS,
                        QUDT_QUANTITY_KINDS,
                        FUNDAMENTAL_EQUATIONS,
                        TOPOLOGICAL_INVARIANTS,
                        SYMMETRY_PRINCIPLES,
                        RELATIVISTIC_INVARIANTS,
                        make_atomic_vector,
                        bind
                    )
                    import uuid
                    
                    seeded = 0
                    
                    # Seed constants and quantity kinds
                    for label in list(CODATA_CONSTANTS.keys()) + list(QUDT_QUANTITY_KINDS.keys()):
                        vec = make_atomic_vector(label) # shape: (4096,) complex
                        vector_str = complex_to_db(vec, DIMS.hrr_dim)
                        concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, label))
                        
                        cur.execute(
                            """
                            INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront)
                            VALUES (%s, %s, %s, %s::vector)
                            ON CONFLICT (concept_hash) DO NOTHING;
                            """,
                            (concept_hash, label[:128], "constant_or_kind", vector_str),
                        )
                        seeded += cur.rowcount
                        
                    # Seed equations (bind terms)
                    for eq_name, eq_data in FUNDAMENTAL_EQUATIONS.items():
                        atoms = [make_atomic_vector(t) for t in eq_data["terms"]]
                        vec = bind(*atoms) if len(atoms) > 1 else atoms[0]
                        vector_str = complex_to_db(vec, DIMS.hrr_dim)
                        concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, eq_name))
                        
                        cur.execute(
                            """
                            INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront)
                            VALUES (%s, %s, %s, %s::vector)
                            ON CONFLICT (concept_hash) DO NOTHING;
                            """,
                            (concept_hash, eq_name[:128], "equation", vector_str),
                        )
                        seeded += cur.rowcount

                    # Seed topological invariants, symmetry principles, and relativistic invariants
                    other_invariants = [
                        (TOPOLOGICAL_INVARIANTS, "topological_invariant"),
                        (SYMMETRY_PRINCIPLES, "symmetry_principle"),
                        (RELATIVISTIC_INVARIANTS, "relativistic_invariant")
                    ]
                    for items_dict, domain_tag in other_invariants:
                        for name, data in items_dict.items():
                            atoms = [make_atomic_vector(t) for t in data["terms"]]
                            vec = bind(*atoms) if len(atoms) > 1 else atoms[0]
                            vector_str = complex_to_db(vec, DIMS.hrr_dim)
                            concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, name))
                            
                            cur.execute(
                                """
                                INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront)
                                VALUES (%s, %s, %s, %s::vector)
                                ON CONFLICT (concept_hash) DO NOTHING;
                                """,
                                (concept_hash, name[:128], domain_tag, vector_str),
                            )
                            seeded += cur.rowcount
                            
                    print(f"[SUCCESS] Seeded {seeded} Zone C axioms into hrr_canonical_lexicon.")
                except Exception as seed_err:
                    print(f"[WARNING] Failed to seed from physics_manifold: {seed_err}. Running random unit-modulus fallback seeding...")
                    try:
                        import uuid
                        concepts = [
                            "Planck_Constant", "Boltzmann_Constant", "Speed_of_Light",
                            "Thermodynamic_Conservation", "Sagnac_Threshold_Limit", "Lipschitz_Bound",
                            "Dirichlet_Invariant", "Neumann_Active_Flow", "Optics_Resonance",
                            "SCADA_Pressure_Control", "SCADA_Thermal_Clamping", "Attractor_Converged"
                        ]
                        seeded = 0
                        for label in concepts:
                            # Generate a random unit-magnitude complex vector of size hrr_dim
                            phases = (np.random.rand(DIMS.hrr_dim) * 2 * np.pi) - np.pi
                            vec = np.exp(1j * phases)
                            vector_str = complex_to_db(vec, DIMS.hrr_dim)
                            concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, label))
                            
                            cur.execute(
                                """
                                INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront)
                                VALUES (%s, %s, %s, %s::vector)
                                ON CONFLICT (concept_hash) DO NOTHING;
                                """,
                                (concept_hash, label[:128], "fallback_axiom", vector_str),
                            )
                            seeded += cur.rowcount
                        print(f"[SUCCESS] Seeded {seeded} fallback Zone C axioms into hrr_canonical_lexicon.")
                    except Exception as fallback_err:
                        print(f"[ERROR] Fallback seeding failed: {fallback_err}")
            else:
                print(f"[INFO] hrr_canonical_lexicon already has {count} rows. Skipping seeding.")
            
            print("[SUCCESS] Local Zone C database initialized with scaled 4096D custom schema.")
            
except Exception as e:
    print("[ERROR] Failed to initialize local 4096D database:", e)
