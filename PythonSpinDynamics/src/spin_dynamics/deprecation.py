"""Consistent public-API deprecation warnings and metadata."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import ParamSpec, TypeVar


class SpinDynamicsDeprecationWarning(FutureWarning):
    """Visible warning for a PythonSpinDynamics API scheduled for removal."""


@dataclass(frozen=True)
class DeprecationInfo:
    """Machine-readable lifecycle information attached to deprecated callables."""

    since: str
    removal: str
    alternative: str

    def message(self, name: str) -> str:
        """Return the standard user-facing warning message for ``name``."""

        return (
            f"{name} is deprecated since {self.since} and will be removed in "
            f"{self.removal}; use {self.alternative} instead."
        )


P = ParamSpec("P")
R = TypeVar("R")


def warn_deprecated(
    name: str,
    *,
    since: str,
    removal: str,
    alternative: str,
    stacklevel: int = 2,
) -> None:
    """Emit the standard visible warning for a deprecated public API."""

    info = _deprecation_info(since, removal, alternative)
    warnings.warn(
        info.message(_required_text(name, "name")),
        SpinDynamicsDeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated(
    *,
    since: str,
    removal: str,
    alternative: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a callable deprecated while preserving its signature and metadata."""

    info = _deprecation_info(since, removal, alternative)

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(
                info.message(function.__qualname__),
                SpinDynamicsDeprecationWarning,
                stacklevel=2,
            )
            return function(*args, **kwargs)

        setattr(wrapped, "__deprecated__", info)
        return wrapped

    return decorate


def _deprecation_info(
    since: str,
    removal: str,
    alternative: str,
) -> DeprecationInfo:
    return DeprecationInfo(
        since=_required_text(since, "since"),
        removal=_required_text(removal, "removal"),
        alternative=_required_text(alternative, "alternative"),
    )


def _required_text(value: str, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text
