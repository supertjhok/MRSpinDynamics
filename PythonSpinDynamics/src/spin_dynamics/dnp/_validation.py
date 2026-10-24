"""Shared finite-array validation for the CW DNP models."""

import numpy as np


def array(value, name: str, *, positive: bool = False, nonnegative=False):
    out = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(out <= 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(out < 0):
        raise ValueError(f"{name} must be non-negative")
    return out


def fraction(value, name: str):
    out = array(value, name, nonnegative=True)
    if np.any(out > 1):
        raise ValueError(f"{name} must lie in [0, 1]")
    return out
