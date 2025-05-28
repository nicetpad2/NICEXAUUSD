import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
