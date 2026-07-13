import os
import sys

try:
    import arc_agi
    from arcengine import GameAction
except ImportError:
    print("arc_agi toolkit is not installed. Please install it using 'pip install arc-agi'")
    sys.exit(1)

def main():
    print("Initializing ARC-AGI-3 Environment (ls20)...")
    try:
        arc = arc_agi.Arcade()
        
        # We can run without render_mode to hit 2K FPS, but terminal mode allows us to see it!
        env = arc.make("ls20", render_mode="terminal")
        
        # Take a few actions just to see it work
        for _ in range(10):
            env.step(GameAction.ACTION1)
            
        print("\nFinal Scorecard:")
        print(arc.get_scorecard())
        
    except Exception as e:
        print(f"Failed to run environment: {e}")

if __name__ == "__main__":
    main()
