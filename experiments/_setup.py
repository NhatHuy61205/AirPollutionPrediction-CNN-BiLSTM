# experiments/_setup.py
import sys
from pathlib import Path

def setup_project_root(levels_up: int = 1):
    root = Path(__file__).resolve()
    for _ in range(levels_up):
        root = root.parent
    root = root.parent if root.name == "experiments" else root  # an toàn

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    return root
