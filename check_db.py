import psycopg
import sys
import os

def main():
    try:
        db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/henri")
        conn = psycopg.connect(db_url)
        cur = conn.cursor()
        
        # Query total count
        cur.execute("SELECT COUNT(*) FROM hrr_canonical_lexicon")
        total_count = cur.fetchone()[0]
        print(f"Total entries in hrr_canonical_lexicon: {total_count}")
        
        # Query syntax repeller count
        cur.execute("SELECT COUNT(*) FROM hrr_canonical_lexicon WHERE semantic_label LIKE 'repeller_syntax_%'")
        repeller_count = cur.fetchone()[0]
        print(f"Syntax repellers count: {repeller_count}")
        
        # Query 5 syntax repellers and their raw_text
        cur.execute("""
            SELECT semantic_label, raw_text 
            FROM hrr_canonical_lexicon 
            WHERE semantic_label LIKE 'repeller_syntax_%' 
            LIMIT 5
        """)
        rows = cur.fetchall()
        print("\n=== Sample Syntax Repellers ===")
        for label, text in rows:
            print(f"Label: {label}")
            print(f"Text:\n{text}")
            print("-" * 50)
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

