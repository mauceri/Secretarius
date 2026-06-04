import sys
from pathlib import Path

# Rend les modules plats de routing/ importables depuis les tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
