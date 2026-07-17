import sys
import arc_agi

try:
    arcade = arc_agi.Arcade()
    scorecards = arcade.get_scorecards()
    if not scorecards:
        print("No scorecards found.")
    else:
        # Just print the last scorecard
        sc = scorecards[-1]
        print(f"Final Scorecard ID: {sc.id}")
        for attr in dir(sc):
            if not attr.startswith('_'):
                val = getattr(sc, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
except Exception as e:
    print(f"Error: {e}")
