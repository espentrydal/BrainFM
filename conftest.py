import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_ADDED = [
    _ROOT,
    _ROOT / "utils",
    _ROOT / "Generator",
    _ROOT / "ShapeID",
]

for path in _ADDED:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Expose the local interpol package for tests that import `interpol`.
try:
    import interpol  # noqa: F401
except Exception:
    try:
        import utils.interpol as _interpol
        sys.modules["interpol"] = _interpol
    except Exception:
        pass
