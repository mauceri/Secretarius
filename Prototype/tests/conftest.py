import sys
from pathlib import Path

# Allow imports from Prototype/ (secretarius_local, core, adapters, etc.)
sys.path.insert(0, str(Path(__file__).parent.parent))
