import numpy as np
import time
import sqlite3

class SSDManifoldEngine:
    def __init__(self, db_path="zone_c_telemetry.db"):
        self.db_path = db_path
        self.titan_decay_rate = 0.05
        self.forget_threshold = 0.1
        self.ace_similarity_threshold = 0.90
        self._init_tables()

    def _init_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS anti_attractors (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            challenge_id TEXT,
                            vector_blob BLOB,
                            weight REAL,
                            last_updated REAL,
                            merge_count INTEGER)''')

    def apply_titan_decay(self):
        """Executes Titan weight decay directly on the SSD."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE anti_attractors SET weight = weight * ?", (1.0 - self.titan_decay_rate,))
            conn.execute("DELETE FROM anti_attractors WHERE weight < ?", (self.forget_threshold,))

    def log_and_deduplicate(self, challenge_id: str, new_vector: np.ndarray, initial_weight: float):
        """ACE Deduplication via SSD streaming."""
        new_vec_norm = new_vector / np.linalg.norm(new_vector)
        merged = False
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, vector_blob, weight, merge_count FROM anti_attractors WHERE challenge_id = ?", (challenge_id,))
            for row_id, blob, weight, count in cursor:
                existing_vec = np.frombuffer(blob, dtype=np.float32)
                existing_norm = existing_vec / np.linalg.norm(existing_vec)
                similarity = np.dot(new_vec_norm, existing_norm)
                
                if similarity > self.ace_similarity_threshold:
                    merged_vec = ((existing_vec * count) + new_vector) / (count + 1)
                    new_weight = min(5.0, weight + initial_weight)
                    conn.execute("UPDATE anti_attractors SET vector_blob = ?, weight = ?, merge_count = ?, last_updated = ? WHERE id = ?",
                                 (merged_vec.astype(np.float32).tobytes(), new_weight, count + 1, time.time(), row_id))
                    merged = True
                    break
            
            if not merged:
                conn.execute("INSERT INTO anti_attractors (challenge_id, vector_blob, weight, last_updated, merge_count) VALUES (?, ?, ?, ?, ?)",
                             (challenge_id, new_vector.astype(np.float32).tobytes(), initial_weight, time.time(), 1))

    def get_dominant_scars(self, challenge_id: str, limit: int = 5):
        """Yields the highest weight scars for topological repulsion."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT vector_blob, weight FROM anti_attractors WHERE challenge_id = ? ORDER BY weight DESC LIMIT ?", (challenge_id, limit))
            return [(np.frombuffer(row[0], dtype=np.float32), row[1]) for row in cursor.fetchall()]
