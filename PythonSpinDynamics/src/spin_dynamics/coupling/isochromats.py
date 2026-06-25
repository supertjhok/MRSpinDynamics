"""Isochromat ensembles for dense scalar-coupled spin simulations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import equilibrium_density, evolve_density
from spin_dynamics.coupling.hamiltonians import (
    isotropic_j_hamiltonian,
    rf_hamiltonian,
    secular_j_hamiltonian,
    zeeman_hamiltonian,
)
from spin_dynamics.coupling.operators import total_operator
from spin_dynamics.coupling.systems import CoupledSpinSystem, coupled_spin_system


@dataclass(frozen=True)
class CoupledIsochromatEnsemble:
    """Static field maps for a coupled-spin isochromat ensemble.

    Each isochromat contains the same scalar-coupled spin network. The local
    B0 offset is added to the base per-spin offsets after multiplication by
    ``offset_scales``. Local B1 transmit and receive scales follow the same
    relative-map convention used by the Bloch workflow layer.
    """

    base_system: CoupledSpinSystem
    b0_offsets_hz: np.ndarray
    weights: np.ndarray
    b1_tx_scale: np.ndarray
    b1_rx_scale: np.ndarray
    offset_scales: np.ndarray

    def __post_init__(self) -> None:
        b0 = _as_1d(self.b0_offsets_hz, "b0_offsets_hz")
        n_iso = b0.size
        weights = _as_isochromat_values(self.weights, n_iso, "weights")
        b1_tx = _as_isochromat_values(self.b1_tx_scale, n_iso, "b1_tx_scale")
        b1_rx = _as_isochromat_values(self.b1_rx_scale, n_iso, "b1_rx_scale")
        scales = np.asarray(self.offset_scales, dtype=np.float64).reshape(-1)
        if scales.size != self.base_system.nspin:
            raise ValueError("offset_scales must match the number of spins")
        if not np.all(np.isfinite(scales)):
            raise ValueError("offset_scales must be finite")
        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative")
        if np.any(b1_tx < 0.0) or np.any(b1_rx < 0.0):
            raise ValueError("B1 scales must be non-negative")
        object.__setattr__(self, "b0_offsets_hz", b0)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "b1_tx_scale", b1_tx)
        object.__setattr__(self, "b1_rx_scale", b1_rx)
        object.__setattr__(self, "offset_scales", scales)

    @property
    def nisochromats(self) -> int:
        """Number of isochromats in the ensemble."""

        return int(self.b0_offsets_hz.size)

    def local_offsets_hz(self, index: int, b0_offset_hz: float | None = None) -> np.ndarray:
        """Return per-spin offsets for one isochromat."""

        b0 = self.b0_offsets_hz[index] if b0_offset_hz is None else float(b0_offset_hz)
        return self.base_system.offsets_hz + b0 * self.offset_scales

    def local_system(self, index: int, b0_offset_hz: float | None = None) -> CoupledSpinSystem:
        """Return a copy of the base system with local B0-shifted offsets."""

        return coupled_spin_system(
            self.local_offsets_hz(index, b0_offset_hz),
            self.base_system.couplings_hz,
            labels=self.base_system.labels,
            isotopes=tuple(site.isotope for site in self.base_system.sites),
        )


@dataclass(frozen=True)
class CoupledIsochromatStep:
    """One time-independent step for a coupled isochromat ensemble."""

    duration: float
    nutation_hz: float | Sequence[float] = 0.0
    phase: float = 0.0
    b0_offsets_hz: float | Sequence[float] | np.ndarray | None = None
    b1_tx_scale: float | Sequence[float] | np.ndarray | None = None
    indices: Sequence[int] | None = None

    def __post_init__(self) -> None:
        duration = float(self.duration)
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("duration must be non-negative")
        if not np.isfinite(float(self.phase)):
            raise ValueError("phase must be finite")
        nutation = np.asarray(self.nutation_hz, dtype=np.float64)
        if not np.all(np.isfinite(nutation)):
            raise ValueError("nutation_hz must be finite")


@dataclass(frozen=True)
class CoupledIsochromatSequenceResult:
    """Signal and final states from a coupled isochromat sequence."""

    signal: complex
    local_signals: np.ndarray
    final_densities: np.ndarray


def coupled_isochromat_ensemble(
    base_system: CoupledSpinSystem,
    b0_offsets_hz: Iterable[float] | np.ndarray,
    *,
    weights: float | Iterable[float] | np.ndarray = 1.0,
    b1_tx_scale: float | Iterable[float] | np.ndarray = 1.0,
    b1_rx_scale: float | Iterable[float] | np.ndarray | None = None,
    offset_scales: Iterable[float] | np.ndarray | None = None,
) -> CoupledIsochromatEnsemble:
    """Build a coupled-spin isochromat ensemble.

    ``b0_offsets_hz`` are local field offsets in hertz. ``b1_tx_scale`` and
    ``b1_rx_scale`` are relative transmit and receive sensitivities. When
    ``offset_scales`` is omitted, the B0 offset is added equally to every spin.
    """

    b0 = _as_1d(b0_offsets_hz, "b0_offsets_hz")
    if b1_rx_scale is None:
        b1_rx_scale = b1_tx_scale
    if offset_scales is None:
        offset_scales = np.ones(base_system.nspin, dtype=np.float64)
    return CoupledIsochromatEnsemble(
        base_system=base_system,
        b0_offsets_hz=b0,
        weights=_as_isochromat_values(weights, b0.size, "weights"),
        b1_tx_scale=_as_isochromat_values(b1_tx_scale, b0.size, "b1_tx_scale"),
        b1_rx_scale=_as_isochromat_values(b1_rx_scale, b0.size, "b1_rx_scale"),
        offset_scales=np.asarray(offset_scales, dtype=np.float64),
    )


def free_precession_step(
    duration: float,
    *,
    b0_offsets_hz: float | Iterable[float] | np.ndarray | None = None,
) -> CoupledIsochromatStep:
    """Return a free-precession step with optional time-varying B0 offsets."""

    return CoupledIsochromatStep(
        duration=float(duration),
        b0_offsets_hz=b0_offsets_hz,
    )


def rf_step(
    duration: float,
    nutation_hz: float | Sequence[float],
    *,
    phase: float = 0.0,
    b0_offsets_hz: float | Iterable[float] | np.ndarray | None = None,
    b1_tx_scale: float | Iterable[float] | np.ndarray | None = None,
    indices: Sequence[int] | None = None,
) -> CoupledIsochromatStep:
    """Return an RF or spin-lock step with optional local B0/B1 overrides."""

    return CoupledIsochromatStep(
        duration=float(duration),
        nutation_hz=nutation_hz,
        phase=float(phase),
        b0_offsets_hz=b0_offsets_hz,
        b1_tx_scale=b1_tx_scale,
        indices=indices,
    )


def simulate_coupled_isochromat_sequence(
    ensemble: CoupledIsochromatEnsemble,
    steps: Sequence[CoupledIsochromatStep],
    *,
    initial_axis: str = "x",
    detect_axis: str = "x",
    j_mode: str = "isotropic",
) -> CoupledIsochromatSequenceResult:
    """Propagate a coupled-spin sequence over an isochromat ensemble."""

    if not steps:
        raise ValueError("steps must not be empty")
    j_mode = str(j_mode).lower()
    if j_mode not in {"isotropic", "secular"}:
        raise ValueError("j_mode must be 'isotropic' or 'secular'")
    detect = total_operator(ensemble.base_system.nspin, detect_axis)
    initial = equilibrium_density(ensemble.base_system, initial_axis)
    local_signals = np.empty(ensemble.nisochromats, dtype=np.complex128)
    final_densities = np.empty(
        (ensemble.nisochromats, ensemble.base_system.dimension, ensemble.base_system.dimension),
        dtype=np.complex128,
    )

    prepared_steps = []
    for step in steps:
        b0_offsets = _step_isochromat_values(
            step.b0_offsets_hz,
            ensemble.b0_offsets_hz,
            ensemble.nisochromats,
            "b0_offsets_hz",
        )
        b1_scale = _step_isochromat_values(
            step.b1_tx_scale,
            ensemble.b1_tx_scale,
            ensemble.nisochromats,
            "b1_tx_scale",
        )
        if np.any(b1_scale < 0.0):
            raise ValueError("B1 scales must be non-negative")
        nutation = np.asarray(step.nutation_hz, dtype=np.float64)
        prepared_steps.append((step, b0_offsets, b1_scale, nutation))

    if j_mode == "isotropic":
        j_hamiltonian = isotropic_j_hamiltonian(ensemble.base_system)
    else:
        j_hamiltonian = secular_j_hamiltonian(ensemble.base_system)

    for iso_idx in range(ensemble.nisochromats):
        density = initial.copy()
        for step, b0_offsets, b1_scale, nutation in prepared_steps:
            system = ensemble.local_system(iso_idx, float(b0_offsets[iso_idx]))
            hamiltonian = zeeman_hamiltonian(system) + j_hamiltonian
            if np.any(nutation != 0.0):
                hamiltonian = hamiltonian + rf_hamiltonian(
                    system,
                    nutation * b1_scale[iso_idx],
                    phase=step.phase,
                    indices=step.indices,
                )
            if step.duration:
                density = evolve_density(density, hamiltonian, step.duration)
        final_densities[iso_idx] = density
        local_signals[iso_idx] = np.trace(density @ detect)

    signal = np.sum(ensemble.weights * ensemble.b1_rx_scale * local_signals)
    return CoupledIsochromatSequenceResult(
        signal=complex(signal),
        local_signals=local_signals,
        final_densities=final_densities,
    )


def _as_1d(value: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_isochromat_values(
    value: float | Iterable[float] | np.ndarray,
    n_isochromats: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n_isochromats, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.size != n_isochromats:
        raise ValueError(f"{name} must be scalar or match b0_offsets_hz")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _step_isochromat_values(
    value: float | Sequence[float] | np.ndarray | None,
    fallback: np.ndarray,
    n_isochromats: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return fallback
    return _as_isochromat_values(value, n_isochromats, name)
