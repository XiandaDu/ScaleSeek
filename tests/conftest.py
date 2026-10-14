import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SCALESEEK_ROOT", "/scratch/" + os.environ.get("USER", "x") + "/scaleseek_v2")
os.environ.setdefault("SCALESEEK_AGENTIR_PYTHON", "/nonexistent/agentir/bin/python")
