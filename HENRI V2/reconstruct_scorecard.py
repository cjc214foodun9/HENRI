import json
import collections
import glob

telemetry_files = glob.glob("telemetry_logs/darwinian_arc_production_*.jsonl")
if not telemetry_files:
    print("No telemetry logs found.")
    exit(1)
latest_log = sorted(telemetry_files)[-1]

game_stats = collections.defaultdict(dict)
overall_success = 0
overall_failed = 0
total_tasks = 0

print(f"Reconstructing Scorecard from {latest_log}...")

with open(latest_log, "r") as f:
    for line in f:
        data = json.loads(line)
        task_id = data.get("task_id")
        if task_id:
            game_id = task_id.split("_STEP")[0]
            if task_id not in game_stats[game_id]:
                game_stats[game_id][task_id] = {"success": False, "epochs": 0}
            
            game_stats[game_id][task_id]["epochs"] = data.get("epoch", 0)
            if data.get("is_isothermal_lock", False) or data.get("sagnac_error_delta", 1.0) < 0.05:
                game_stats[game_id][task_id]["success"] = True

scorecard = {
    "summary": {
        "games_attempted": len(game_stats),
        "total_tasks": 0,
        "tasks_solved": 0,
        "tasks_failed": 0,
        "overall_accuracy": 0.0
    },
    "games": {}
}

for game_id, tasks in game_stats.items():
    game_solved = 0
    game_failed = 0
    for task_id, stats in tasks.items():
        scorecard["summary"]["total_tasks"] += 1
        if stats["success"]:
            game_solved += 1
            scorecard["summary"]["tasks_solved"] += 1
        else:
            game_failed += 1
            scorecard["summary"]["tasks_failed"] += 1
            
    scorecard["games"][game_id] = {
        "tasks_attempted": len(tasks),
        "tasks_solved": game_solved,
        "tasks_failed": game_failed,
        "game_accuracy": game_solved / len(tasks) if len(tasks) > 0 else 0
    }

total = scorecard["summary"]["total_tasks"]
if total > 0:
    scorecard["summary"]["overall_accuracy"] = scorecard["summary"]["tasks_solved"] / total

print(json.dumps(scorecard, indent=2))
