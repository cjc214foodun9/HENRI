"""Gate B completion watcher: polls remote until the scorecard JSON appears."""
import json
import subprocess
import sys
import time

SCORECARD_DIR = "/root/telemetry_logs"
POLL_S = 20
TIMEOUT_S = 7200

ssh_base = [
    "ssh", "-p", "45864", "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=10", "root@107.206.71.138",
]


def remote(cmd):
    r = subprocess.run(ssh_base + [cmd], capture_output=True, text=True, timeout=60)
    return r.stdout.strip(), r.returncode


def main():
    t0 = time.time()
    last = ""
    while time.time() - t0 < TIMEOUT_S:
        out, rc = remote(
            "cd /root/henri-839-wt && ls -t /root/telemetry_logs/humaneval_wave_ast_*.json 2>/dev/null | head -1"
        )
        if rc == 0 and out and "No such" not in out:
            scorecard = out.splitlines()[0]
            tail, _ = remote(
                f"grep -E '\\\"solved\\\"|\\\"status\\\"|\\\"accuracy_attempted\\\"|\\\"ast_idf_only\\\"|\\\"item_count\\\"' {scorecard}"
            )
            proc, _ = remote("ps aux | grep -c '[h]umaneval_wave_ast_runner'")
            print(f"[watcher] scorecard={scorecard} procs={proc}")
            print(tail)
            if tail.strip():
                try:
                    summary = json.dumps({l.split(':')[0].strip().strip('"')
                                          for l in tail.splitlines()})
                except Exception:
                    summary = ""
                print("GATE_B_DONE")
                sys.exit(0)
        out2, rc2 = remote("tail -3 /root/telemetry_logs/gate_b_idf.log 2>/dev/null")
        if out2 and out2 != last:
            print(f"[watcher] log: {out2.splitlines()[-1]}")
            last = out2
        time.sleep(POLL_S)
    print("GATE_B_TIMEOUT")
    sys.exit(2)


if __name__ == "__main__":
    main()
