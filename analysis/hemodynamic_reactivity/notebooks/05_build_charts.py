from __future__ import annotations

import subprocess
from pathlib import Path
import os
import tempfile

from common import CHART_SCRIPT_DIR


def main() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "matplotlib-codex-cache"
    cache_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(cache_dir)
    for script in sorted(CHART_SCRIPT_DIR.glob("[0-9][0-9]_*.py")):
        subprocess.run(["python", str(script)], check=True, env=env)


if __name__ == "__main__":
    main()
