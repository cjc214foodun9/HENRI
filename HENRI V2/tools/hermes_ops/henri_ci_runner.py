"""Windows-safe Hermes cron entrypoint for the HENRI bash CI script."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


script = Path(__file__).with_name("henri_ci.sh").resolve()
bash = "bash"
# Run from the scripts directory and pass a relative name. This avoids both
# Windows argv path handling and MSYS drive-path translation in cron children.
result = subprocess.run([bash, script.name], cwd=str(script.parent), text=True)
raise SystemExit(result.returncode)
