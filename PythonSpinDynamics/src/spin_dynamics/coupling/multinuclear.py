"""Heteronuclear J-coupled spin systems for zero/ultra-low-field NMR.

At Earth's field (~50 uT) chemical shifts are negligible and the spectrum is
governed by scalar (``J``) couplings between nuclei whose Larmor frequencies all
fall within a couple of kHz. Because the frequency separations ``|nu_i - nu_j|``
can be comparable to the couplings, spins are often *strongly* coupled and the
full isotropic ``J I_i . I_j`` Hamiltonian must be used in a common (lab) frame,
rather than the weak-coupling secular ``J I_iz I_jz`` truncation.

This module provides a :class:`MultinuclearSpinSystem` (mixed spin quantum
numbers, absolute Larmor frequencies from ``gamma_i B0``), the lab-frame
Hamiltonian builders, a high-temperature equilibrium density, and a per-spin
phenomenological ``R1``/``R2`` relaxation superoperator. The latter reproduces
the "extended T1/T2" relaxation used to model quadrupolar line broadening in
Earth's-field J-coupled spectra (Altenhof et al., J. Magn. Reson. 355 (2023)
107540): a fast-relaxing quadrupolar nucleus (e.g. ``14N``) coupled to observed
spins collapses their J-splittings.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.isotopes import nuclear_isotope
from spin_dynamics.coupling.mixed_operators import (
    dot_product_operator,
    embedded_operator,
    hilbert_dimension,
    product_operator,
)

TAU = 2.0 * np.pi


@dataclass(frozen=True)
class MultinuclearSite:
    """One nucleus in a heteronuclear J-coupled system."""

    label: str
    isotope: str
    spin: float
    gamma_hz_per_t: float


@dataclass(frozen=True)
class MultinuclearSpinSystem:
    """Heteronuclear scalar-coupled spin system at a fixed field ``b0_tesla``."""

    sites: tuple[MultinuclearSite, ...]
    couplings_hz: np.ndarray
    b0_tesla: float

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
        b0 = float(self.b0_tesla)
        if not np.isfinite(b0):
            raise ValueError("b0_tesla must be finite")
        for site in self.sites:
            two_spin = round(2.0 * float(site.spin))
            if two_spin < 1 or not np.isclose(2.0 * float(site.spin), two_spin):
                raise ValueError("each site spin must be a positive half-integer")
        matrix = matrix.copy()
        np.fill_diagonal(matrix, 0.0)
        object.__setattr__(self, "couplings_hz", matrix)
        object.__setattr__(self, "b0_tesla", b0)

    @property
    def nspin(self) -> int:
        """Number of nuclei."""

        return len(self.sites)

    @property
    def spins(self) -> tuple[float, ...]:
        """Spin quantum numbers in storage order."""

        return tuple(float(site.spin) for site in self.sites)

    @property
    def dimension(self) -> int:
        """Tensor-product Hilbert-space dimension."""

        return hilbert_dimension(self.spins)

    @property
    def gammas_hz_per_t(self) -> np.ndarray:
        """Per-site gyromagnetic ratios ``|gamma|/2pi`` in Hz/T."""

        return np.array(
            [site.gamma_hz_per_t for site in self.sites], dtype=np.float64
        )

    @property
    def larmor_hz(self) -> np.ndarray:
        """Per-site Larmor frequencies ``gamma_i B0`` in hertz."""

        return self.gammas_hz_per_t * self.b0_tesla

    @property
    def labels(self) -> tuple[str, ...]:
        """Site labels in storage order."""

        return tuple(site.label for site in self.sites)

    @property
    def isotopes(self) -> tuple[str, ...]:
        """Site isotopes in storage order."""

        return tuple(site.isotope for site in self.sites)

    def indices_for_isotope(self, isotope: str) -> tuple[int, ...]:
        """Return the site indices whose isotope matches ``isotope``."""

        return tuple(
            idx for idx, site in enumerate(self.sites) if site.isotope == str(isotope)
        )


def multinuclear_system(
    isotopes: Sequence[str],
    couplings_hz: Iterable[Iterable[float]],
    b0_tesla: float,
    *,
    labels: Sequence[str] | None = None,
    spins: Sequence[float] | None = None,
    gammas_hz_per_t: Sequence[float] | None = None,
) -> MultinuclearSpinSystem:
    """Build a validated heteronuclear system from isotope names and couplings.

    ``spins`` and ``gammas_hz_per_t`` default to the registry values for each
    isotope and may be overridden per site (required for nuclei not in the
    registry). Labels default to ``"<isotope><index>"``.
    """

    isotope_list = [str(iso) for iso in isotopes]
    nspin = len(isotope_list)
    if nspin == 0:
        raise ValueError("at least one isotope is required")
    if labels is not None and len(labels) != nspin:
        raise ValueError("labels must match the number of isotopes")
    if spins is not None and len(spins) != nspin:
        raise ValueError("spins must match the number of isotopes")
    if gammas_hz_per_t is not None and len(gammas_hz_per_t) != nspin:
        raise ValueError("gammas_hz_per_t must match the number of isotopes")

    sites = []
    for idx, isotope in enumerate(isotope_list):
        spin = None if spins is None else float(spins[idx])
        gamma = None if gammas_hz_per_t is None else float(gammas_hz_per_t[idx])
        if spin is None or gamma is None:
            entry = nuclear_isotope(isotope)
            spin = entry.spin if spin is None else spin
            gamma = entry.gamma_hz_per_t if gamma is None else gamma
        label = f"{isotope}{idx + 1}" if labels is None else str(labels[idx])
        sites.append(
            MultinuclearSite(
                label=label,
                isotope=isotope,
                spin=float(spin),
                gamma_hz_per_t=float(gamma),
            )
        )
    return MultinuclearSpinSystem(
        sites=tuple(sites),
        couplings_hz=np.asarray(list(couplings_hz), dtype=np.float64),
        b0_tesla=float(b0_tesla),
    )


def multinuclear_hamiltonian(
    system: MultinuclearSpinSystem,
    *,
    coupling: str = "isotropic",
    include_zeeman: bool = True,
) -> np.ndarray:
    """Return the static lab-frame Hamiltonian in radians per second.

    The Zeeman term uses absolute Larmor frequencies ``gamma_i B0``. ``coupling``
    selects the scalar term: ``"isotropic"`` (full ``J I_i . I_j``, required in
    the strong-coupling ZULF regime) or ``"secular"`` (weak-coupling
    ``J I_iz I_jz``). Set ``include_zeeman=False`` to obtain the coupling term
    alone (e.g. for the true zero-field limit).
    """

    spins = system.spins
    dimension = system.dimension
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)

    if include_zeeman:
        for idx, nu_hz in enumerate(system.larmor_hz):
            if nu_hz:
                hamiltonian = hamiltonian + TAU * float(nu_hz) * embedded_operator(
                    spins, idx, "z"
                )

    couplings = system.couplings_hz
    for idx in range(system.nspin):
        for jdx in range(idx + 1, system.nspin):
            j_hz = float(couplings[idx, jdx])
            if not j_hz:
                continue
            if coupling == "isotropic":
                pair = dot_product_operator(spins, idx, jdx)
            elif coupling == "secular":
                pair = product_operator(spins, [(idx, "z"), (jdx, "z")])
            else:
                raise ValueError("coupling must be 'isotropic' or 'secular'")
            hamiltonian = hamiltonian + TAU * j_hz * pair
    return hamiltonian


def multinuclear_rf_hamiltonian(
    system: MultinuclearSpinSystem,
    nutation_hz: float,
    *,
    phase: float = 0.0,
    indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Return an RF Hamiltonian (rad/s) driving selected spins at one amplitude."""

    spins = system.spins
    selected = tuple(range(system.nspin) if indices is None else indices)
    amplitude = float(nutation_hz)
    if not np.isfinite(amplitude):
        raise ValueError("nutation_hz must be finite")
    cphase = float(np.cos(phase))
    sphase = float(np.sin(phase))
    dimension = system.dimension
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
    for idx in selected:
        hamiltonian = hamiltonian + TAU * amplitude * (
            cphase * embedded_operator(spins, int(idx), "x")
            + sphase * embedded_operator(spins, int(idx), "y")
        )
    return hamiltonian


def multinuclear_equilibrium_density(
    system: MultinuclearSpinSystem,
    *,
    axis: str = "z",
    gamma_weighted: bool = True,
) -> np.ndarray:
    """Return the high-temperature equilibrium density deviation.

    In the high-field prepolarization / high-temperature limit the initial
    magnetization of each nucleus is proportional to its gyromagnetic ratio, so
    ``rho ~ sum_i gamma_i I_iz``. Set ``gamma_weighted=False`` for an unweighted
    ``sum_i I_iz`` (all nuclei equally polarized).
    """

    spins = system.spins
    dimension = system.dimension
    weights = system.gammas_hz_per_t if gamma_weighted else np.ones(system.nspin)
    reference = float(np.max(np.abs(weights))) or 1.0
    density = np.zeros((dimension, dimension), dtype=np.complex128)
    for idx in range(system.nspin):
        density = density + (float(weights[idx]) / reference) * embedded_operator(
            spins, idx, axis
        )
    return density


def _lindblad_superoperator(jump: np.ndarray) -> np.ndarray:
    """Return the column-stacked Lindblad dissipator for one jump operator."""

    jump = np.asarray(jump, dtype=np.complex128)
    dim = jump.shape[0]
    identity = np.eye(dim, dtype=np.complex128)
    metric = jump.conj().T @ jump
    return (
        np.kron(jump.conj(), jump)
        - 0.5 * np.kron(identity, metric)
        - 0.5 * np.kron(metric.T, identity)
    )


def per_spin_relaxation_superoperator(
    spins: Sequence[float] | Iterable[float],
    r1_per_second: float | Sequence[float] | np.ndarray,
    r2_per_second: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return a per-spin phenomenological ``R1``/``R2`` relaxation superoperator.

    Builds an independent-spin ("extended T1/T2") dissipator in column-stacked
    Liouville space. For each spin ``i`` two Lindblad channels are added:

    * longitudinal ladder channels ``sqrt(R1_i/2) I_i^+`` and
      ``sqrt(R1_i/2) I_i^-`` (equal up/down rates: the infinite-temperature bath),
      which relax populations toward uniform, and
    * a pure-dephasing channel ``sqrt(2 R2_i - R1_i) I_iz``.

    For a spin-1/2 these reproduce ``R1`` and ``R2`` exactly (longitudinal decay
    ``R1``; single-quantum coherence decay ``R2``); the requirement
    ``R2_i >= R1_i/2`` is the usual physical bound. For spin > 1/2 the same
    channels give an isotropic single-rate model (as used for the quadrupolar
    ``14N`` in Earth's-field J-coupled spectra), where ``R1_i = R2_i`` is the
    intended usage.
    """

    spin_values = tuple(float(spin) for spin in spins)
    if not spin_values:
        raise ValueError("at least one spin is required")
    nspin = len(spin_values)
    r1 = np.broadcast_to(np.asarray(r1_per_second, dtype=np.float64), (nspin,))
    r2 = np.broadcast_to(np.asarray(r2_per_second, dtype=np.float64), (nspin,))
    if np.any(r1 < 0.0) or np.any(r2 < 0.0):
        raise ValueError("relaxation rates must be non-negative")
    if not (np.all(np.isfinite(r1)) and np.all(np.isfinite(r2))):
        raise ValueError("relaxation rates must be finite")
    if np.any(r2 < 0.5 * r1 - 1e-12):
        raise ValueError("r2_per_second must be at least r1_per_second / 2")

    size = hilbert_dimension(spin_values) ** 2
    out = np.zeros((size, size), dtype=np.complex128)
    for idx in range(nspin):
        ladder_rate = 0.5 * float(r1[idx])
        if ladder_rate > 0.0:
            for axis in ("+", "-"):
                jump = np.sqrt(ladder_rate) * embedded_operator(
                    spin_values, idx, axis
                )
                out = out + _lindblad_superoperator(jump)
        dephasing_rate = 2.0 * float(r2[idx]) - float(r1[idx])
        if dephasing_rate > 0.0:
            jump = np.sqrt(dephasing_rate) * embedded_operator(spin_values, idx, "z")
            out = out + _lindblad_superoperator(jump)
    return out
