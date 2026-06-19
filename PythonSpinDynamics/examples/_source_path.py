"""Source-tree import helper for examples.

This keeps examples runnable before the package is installed, even when the
current working directory is `examples/`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def add_src_to_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def load_matplotlib(*, required: bool = True) -> ModuleType | None:
    """Load pyplot with the package's standard optional-dependency message."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        if not required:
            return None
        raise SystemExit(
            "matplotlib is required for this example. Install it with "
            "`python -m pip install -e .[plot]`."
        ) from exc
    return plt
