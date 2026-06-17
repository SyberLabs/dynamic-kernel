"""
Ensure the dynamic_kernel package root (flat-module layout) is on sys.path
so that `from kernel import ...` resolves regardless of where pytest is invoked.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent  # dynamic_kernel/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
