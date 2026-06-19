"""Source-tree import helper for examples.

This keeps examples runnable before the package is installed, even when the
current working directory is `examples/`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType


def add_src_to_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def load_matplotlib(
    *, required: bool = True, headless: bool = False
) -> ModuleType | None:
    """Load pyplot with the package's standard optional-dependency message.

    Pass ``headless=True`` before saving figures in scripts that must run in
    shells without a usable GUI backend. Leave it false for interactive
    examples that call ``plt.show()``.
    """

    try:
        import matplotlib

        if headless and "MPLBACKEND" not in os.environ:
            matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        if not required:
            return None
        raise SystemExit(
            "matplotlib is required for this example. Install it with "
            "`python -m pip install -e .[plot]`."
        ) from exc
    return plt
