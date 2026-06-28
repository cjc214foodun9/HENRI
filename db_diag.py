import psycopg

try:
    conn = psycopg.connect("postgresql://postgres:password@127.0.0.1:5432/henri")
    cur = conn.cursor()
    
    # Audit 1: Summary Stats
    cur.execute("""
        SELECT 
            COUNT(*) as total_harvested_axioms,
            COUNT(DISTINCT task_origin) as unique_tasks_mastered,
            MAX(resonance_score) as peak_resonance,
            AVG(resonance_score) as mean_resonance_score
        FROM henri_sub_axiom_harvest;
    """)
    summary = cur.fetchone()
    print("--- HARVEST LEDGER SUMMARY ---")
    print(f"Total Harvested Axioms:   {summary[0]}")
    print(f"Unique Tasks Mastered:    {summary[1]}")
    print(f"Peak Resonance Score:     {summary[2] if summary[2] is not None else 0.0:.4f}")
    print(f"Mean Resonance Score:     {summary[3] if summary[3] is not None else 0.0:.4f}")
    
    # Audit 2: Top 10 High Resonance Tasks
    cur.execute("""
        SELECT 
            task_origin, 
            case_index, 
            resonance_score, 
            timestamp 
        FROM henri_sub_axiom_harvest 
        ORDER BY resonance_score DESC 
        LIMIT 10;
    """)
    rows = cur.fetchall()
    print("\n--- TOP 10 HIGHEST-RESONANCE TASKS ---")
    for r in rows:
        print(f"Task: {r[0]} | Case Index: {r[1]} | Resonance: {r[2]:.4f} | Time: {r[3]}")
        
    conn.close()
except Exception as e:
    print(f"Error querying ledger database: {e}")
