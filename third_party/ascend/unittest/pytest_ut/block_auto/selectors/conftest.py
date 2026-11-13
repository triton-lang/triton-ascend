"""Make the standalone selector tools importable by their relocated tests."""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[6] / "tools/block_auto_select"
sys.path.insert(0, str(TOOLS))
