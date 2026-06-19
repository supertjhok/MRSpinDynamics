"""Data containers for small scalar-coupled spin-1/2 systems."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpinSite:
    """One spin-1/2 site in a coupled spin system."""

    label: str
    isotope: str = "1H"
    offset_hz: float = 0.0


@dataclass(frozen=True)
class CoupledSpinSystem:
    """Small dense spin-1/2 system with scalar couplings in hertz."""

    sites: tuple[SpinSite, ...]
    couplings_hz: np.ndarray

    def __post_init__(self) -> None:
        if not self.sites:
            raise ValueError("at least one spin site is required")
        matrix = np.asarray(self.couplings_hz, dtype=np.float64)
        nspin = len(self.sites)
        if matrix.shape != (nspin, nspin):
            raise ValueError("couplings_hz must be a square nspin by nspin matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("couplings_hz must be finite")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("couplings_hz must be symmetric")
        matrix = matrix.copy()
        np.fill_diagonal(matrix, 0.0)
        object.__setattr__(self, "couplings_hz", matrix)

    @property
    def nspin(self) -> int:
        """Number of spin-1/2 sites."""

        return len(self.sites)

    @property
    def dimension(self) -> int:
        """Hilbert-space dimension for the spin system."""

        return 2**self.nspin

    @property
    def offsets_hz(self) -> np.ndarray:
        """Per-spin resonance offsets in hertz."""

        return np.array([site.offset_hz for site in self.sites], dtype=np.float64)

    @property
    def labels(self) -> tuple[str, ...]:
        """Spin labels in storage order."""

        return tuple(site.label for site in self.sites)


def coupled_spin_system(
    offsets_hz: Iterable[float],
    couplings_hz: Iterable[Iterable[float]],
    *,
    labels: Sequence[str] | None = None,
    isotopes: Sequence[str] | None = None,
) -> CoupledSpinSystem:
    """Build a validated spin-1/2 system from offsets and couplings."""

    offsets = np.asarray(list(offsets_hz), dtype=np.float64).reshape(-1)
    if offsets.size == 0:
        raise ValueError("offsets_hz must contain at least one spin")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("offsets_hz must be finite")
    nspin = int(offsets.size)
    if labels is None:
        labels = tuple(f"S{idx + 1}" for idx in range(nspin))
    if isotopes is None:
        isotopes = tuple("1H" for _ in range(nspin))
    if len(labels) != nspin or len(isotopes) != nspin:
        raise ValueError("labels and isotopes must match the number of offsets")
    sites = tuple(
        SpinSite(str(label), str(isotope), float(offset))
        for label, isotope, offset in zip(labels, isotopes, offsets)
    )
    return CoupledSpinSystem(sites=sites, couplings_hz=np.asarray(couplings_hz))
