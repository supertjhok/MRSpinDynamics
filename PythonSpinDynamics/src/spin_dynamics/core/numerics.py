"""Small numerical compatibility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def trapezoid(y: Any, x: Any | None = None, axis: int = -1) -> np.ndarray:
    """Integrate with NumPy's trapezoid rule across NumPy versions.

    NumPy 2.x exposes `np.trapezoid`; older NumPy releases expose `np.trapz`.
    Keep this wrapper so public examples work in common Anaconda environments.
    """

    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return integrator(y, x=x, axis=axis)
