import os
import glob
import time

def monitor_zone_c(interval=5):
    adapters_dir = os.path.join("archive", "domain_adapters")
    print(f"============================================================")
    print(f"  [ZONE C MONITOR] Scanning Topological Database... ")
    print(f"  Directory: {os.path.abspath(adapters_dir)}")
    print(f"============================================================\n")
    
    if not os.path.exists(adapters_dir):
        print(f"Directory not found: {adapters_dir}")
        return

    try:
        while True:
            # Find all intuition waves
            intuitions = glob.glob(os.path.join(adapters_dir, "intuition_*.bin"))
            
            print(f"[{time.strftime('%H:%M:%S')}] Active Geometric Intuitions Distilled: {len(intuitions)}")
            
            if intuitions:
                latest = max(intuitions, key=os.path.getmtime)
                size_kb = os.path.getsize(latest) / 1024
                print(f"   -> Latest: {os.path.basename(latest)} ({size_kb:.2f} KB)")
                
            print(f"------------------------------------------------------------")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n[MONITOR] Disconnected.")

if __name__ == "__main__":
    monitor_zone_c()
