import psycopg
import sys

def main():
    try:
        conn = psycopg.connect('postgresql://postgres:password@127.0.0.1:5432/henri')
        cur = conn.cursor()
        
        # Query 5 syntax repellers and their raw_text
        cur.execute("""
            SELECT semantic_label, raw_text 
            FROM hrr_canonical_lexicon 
            WHERE semantic_label LIKE 'repeller_syntax_%' 
            LIMIT 5
        """)
        rows = cur.fetchall()
        print("=== Sample Syntax Repellers ===")
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
