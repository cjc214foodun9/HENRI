import subprocess
import os

filename = "vast_ai_sync_20260719_204835.jsonl"
cmd = [
    "ssh", "-o", "StrictHostKeyChecking=no", "-p", "26117", "root@ssh1.vast.ai",
    f"cat 'HENRI V2/telemetry_logs/{filename}'"
]

out_path = f"HENRI V2/telemetry_logs/{filename}"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

print(f"Downloading to {out_path}...")
with open(out_path, "wb") as f:
    subprocess.run(cmd, stdout=f)
print("Done downloading.")

# Strip MOTD
print("Stripping MOTD...")
with open(out_path, "rb") as f:
    lines = f.readlines()

with open(out_path, "wb") as f:
    for line in lines:
        if line.startswith(b"{"):
            f.write(line)
print("Done stripping.")
