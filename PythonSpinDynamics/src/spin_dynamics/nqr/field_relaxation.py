"""Field-dependent Gibbs equilibrium and thermalizing NQR relaxation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.nqr.crossover import boltzmann_populations
from spin_dynamics.nqr.hamiltonians import TAU, nqr_hamiltonian
from spin_dynamics.nqr.operators import spin_matrices
from spin_dynamics.nqr.systems import QuadrupolarSite
from spin_dynamics.relaxation import liouville_hamiltonian, matrix_exponential
from spin_dynamics.relaxation import quadrupolar_tesseral_operators


_PLANCK_J_S = 6.62607015e-34
_BOLTZMANN_J_PER_K = 1.380649e-23


@dataclass(frozen=True)
class FieldEquilibriumResult:
    """Exact Gibbs state of one quadrupolar site at a specified static field."""

    density_matrix_pas: np.ndarray
    density_matrix_eigenbasis: np.ndarray
    populations: np.ndarray
    levels_hz: np.ndarray
    eigenvectors: np.ndarray
    spin_expectation_pas: np.ndarray
    b0_vector_tesla_pas: np.ndarray
    temperature_kelvin: float
    site: QuadrupolarSite


@dataclass(frozen=True)
class FieldRelaxationResult:
    """Density and spin-expectation trajectories under a fixed static field."""

    times_seconds: np.ndarray
    density_matrices_pas: np.ndarray
    spin_expectation_pas: np.ndarray
    equilibrium: FieldEquilibriumResult
    hamiltonian: np.ndarray
    site: QuadrupolarSite


def _validate_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    out = np.asarray(hamiltonian, dtype=np.complex128)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError("hamiltonian must be square")
    if not np.all(np.isfinite(out)):
        raise ValueError("hamiltonian must be finite")
    if not np.allclose(out, out.conj().T):
        raise ValueError("hamiltonian must be Hermitian")
    return 0.5 * (out + out.conj().T)


def _gibbs_data(
    hamiltonian: np.ndarray,
    temperature_kelvin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hamiltonian = _validate_hamiltonian(hamiltonian)
    levels_rad, vectors = np.linalg.eigh(hamiltonian)
    levels_hz = levels_rad / TAU
    populations = boltzmann_populations(levels_hz, temperature_kelvin)
    density = (vectors * populations[np.newaxis, :]) @ vectors.conj().T
    return levels_hz, vectors, populations, density


def field_dependent_equilibrium(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    *,
    temperature_kelvin: float = 300.0,
) -> FieldEquilibriumResult:
    """Return the normalized Gibbs state of the complete ``H_Q + H_Z``."""

    b0 = np.asarray(b0_vector_tesla_pas, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(b0)):
        raise ValueError("b0_vector_tesla_pas must be finite")
    hamiltonian = nqr_hamiltonian(site, b0)
    levels, vectors, populations, density = _gibbs_data(
        hamiltonian,
        temperature_kelvin,
    )
    ops = spin_matrices(site.spin)
    expectation = np.array(
        [np.trace(density @ operator).real for operator in (ops.ix, ops.iy, ops.iz)]
    )
    return FieldEquilibriumResult(
        density_matrix_pas=density,
        density_matrix_eigenbasis=np.diag(populations.astype(np.complex128)),
        populations=populations,
        levels_hz=levels,
        eigenvectors=vectors,
        spin_expectation_pas=expectation,
        b0_vector_tesla_pas=b0,
        temperature_kelvin=float(temperature_kelvin),
        site=site,
    )


def _lindblad_superoperator(jump: np.ndarray) -> np.ndarray:
    dimension = jump.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    metric = jump.conj().T @ jump
    return (
        np.kron(jump.conj(), jump)
        - 0.5 * np.kron(identity, metric)
        - 0.5 * np.kron(metric.T, identity)
    )


def _energy_manifold_projectors(
    levels_hz: np.ndarray,
    vectors: np.ndarray,
    tolerance_hz: float,
) -> tuple[np.ndarray, ...]:
    groups: list[list[int]] = []
    for index, level in enumerate(levels_hz):
        if not groups or abs(level - levels_hz[groups[-1][0]]) > tolerance_hz:
            groups.append([index])
        else:
            groups[-1].append(index)
    return tuple(
        vectors[:, group] @ vectors[:, group].conj().T
        for group in groups
    )


@dataclass(frozen=True)
class FieldDependentRelaxationModel:
    """Completely-positive relaxation toward the Gibbs state of ``H_Q + H_Z``.

    ``thermalization_time_seconds`` controls a Gibbs-reset channel
    ``(rho_eq Tr(rho) - rho) / tau``. This channel is basis-invariant and has
    the exact finite-temperature Gibbs state as its unique normalized fixed
    point. Optional extra dephasing acts only between distinct energy
    manifolds; coherences inside a degenerate manifold are not assigned an
    arbitrary eigenvector-dependent decay.

    This is a robust phenomenological crossover model. It does not claim the
    field-dependent rates of a microscopic bath spectral density.
    """

    temperature_kelvin: float = 300.0
    thermalization_time_seconds: float = np.inf
    dephasing_time_seconds: float = np.inf
    degeneracy_tolerance_hz: float = 1.0e-6

    def __post_init__(self) -> None:
        temperature = float(self.temperature_kelvin)
        thermalization = float(self.thermalization_time_seconds)
        dephasing = float(self.dephasing_time_seconds)
        tolerance = float(self.degeneracy_tolerance_hz)
        if temperature <= 0.0 or not np.isfinite(temperature):
            raise ValueError("temperature_kelvin must be positive and finite")
        for value, name in (
            (thermalization, "thermalization_time_seconds"),
            (dephasing, "dephasing_time_seconds"),
        ):
            if value <= 0.0 or np.isnan(value):
                raise ValueError(f"{name} must be positive")
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise ValueError("degeneracy_tolerance_hz must be non-negative")
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "thermalization_time_seconds", thermalization)
        object.__setattr__(self, "dephasing_time_seconds", dephasing)
        object.__setattr__(self, "degeneracy_tolerance_hz", tolerance)

    def equilibrium_density(self, hamiltonian: np.ndarray) -> np.ndarray:
        """Return the normalized Gibbs fixed point for ``hamiltonian``."""

        return _gibbs_data(hamiltonian, self.temperature_kelvin)[3]

    def superoperator(self, hamiltonian: np.ndarray) -> np.ndarray:
        """Return the trace-preserving thermalization/dephasing generator."""

        hamiltonian = _validate_hamiltonian(hamiltonian)
        dimension = hamiltonian.shape[0]
        size = dimension * dimension
        levels, vectors, _, equilibrium = _gibbs_data(
            hamiltonian,
            self.temperature_kelvin,
        )
        out = np.zeros((size, size), dtype=np.complex128)
        if np.isfinite(self.thermalization_time_seconds):
            rate = 1.0 / self.thermalization_time_seconds
            trace_vector = np.eye(dimension).reshape(-1, order="F")
            equilibrium_vector = equilibrium.reshape(-1, order="F")
            out += rate * (
                np.outer(equilibrium_vector, trace_vector) - np.eye(size)
            )
        if np.isfinite(self.dephasing_time_seconds):
            rate = 1.0 / self.dephasing_time_seconds
            for projector in _energy_manifold_projectors(
                levels,
                vectors,
                self.degeneracy_tolerance_hz,
            ):
                out += _lindblad_superoperator(np.sqrt(rate) * projector)
        return out


def _positive_frequency_components(
    operator: np.ndarray,
    levels_rad_per_second: np.ndarray,
    tolerance_rad_per_second: float,
    cluster_width_rad_per_second: float | None = None,
) -> tuple[tuple[float, np.ndarray], ...]:
    gaps = levels_rad_per_second[:, None] - levels_rad_per_second[None, :]
    candidates = sorted(
        (float(gaps[row, column]), row, column)
        for row in range(gaps.shape[0])
        for column in range(gaps.shape[1])
        if gaps[row, column] > tolerance_rad_per_second
        and abs(operator[row, column]) > 0.0
    )
    width = (
        tolerance_rad_per_second
        if cluster_width_rad_per_second is None
        else float(cluster_width_rad_per_second)
    )
    clusters: list[list[tuple[float, int, int]]] = []
    for candidate in candidates:
        if not clusters or candidate[0] - clusters[-1][0][0] > width:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)
    components: list[tuple[float, np.ndarray]] = []
    for cluster in clusters:
        raising = np.zeros_like(operator, dtype=np.complex128)
        weights = np.array(
            [abs(operator[row, column]) ** 2 for _, row, column in cluster]
        )
        center = float(
            np.average([gap for gap, _, _ in cluster], weights=weights)
        )
        for _, row, column in cluster:
            raising[row, column] = operator[row, column]
        if np.any(np.abs(raising) > 0.0):
            components.append((center, raising))
    return tuple(components)


@dataclass(frozen=True)
class FieldDependentDaviesRelaxationModel:
    """Thermal secular relaxation with field-dependent transition rates.

    Magnetic rank-1 and EFG rank-2 channels are evaluated in the instantaneous
    eigenbasis of the supplied Hamiltonian. A Lorentzian spectral factor
    ``1 / (1 + (omega*tau_c)**2)`` supplies explicit field dependence, while
    upward and downward rates obey finite-temperature detailed balance. Equal
    Bohr frequencies are grouped into common jump operators.

    This completely-positive secular model is appropriate when distinct Bohr
    frequency groups are resolved. Near crossings where their separation is
    comparable to the relaxation rate, use
    :class:`FieldDependentNonsecularRelaxationModel` to retain unresolved
    coherence-transfer terms, or :class:`FieldDependentRelaxationModel` as the
    robust phenomenological fallback.
    """

    spin: float
    temperature_kelvin: float = 300.0
    magnetic_rate_per_second: float = 0.0
    efg_rate_per_second: float = 0.0
    correlation_time_seconds: float = 0.0
    secular_tolerance_hz: float = 1.0e-6

    def __post_init__(self) -> None:
        spin = float(self.spin)
        dimension = int(round(2.0 * spin + 1.0))
        if spin <= 0.0 or not np.isclose(dimension, 2.0 * spin + 1.0):
            raise ValueError("spin must be a positive integer or half-integer")
        temperature = float(self.temperature_kelvin)
        if temperature <= 0.0 or not np.isfinite(temperature):
            raise ValueError("temperature_kelvin must be positive and finite")
        for value, name in (
            (self.magnetic_rate_per_second, "magnetic_rate_per_second"),
            (self.efg_rate_per_second, "efg_rate_per_second"),
            (self.correlation_time_seconds, "correlation_time_seconds"),
            (self.secular_tolerance_hz, "secular_tolerance_hz"),
        ):
            if float(value) < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be non-negative and finite")
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "_dimension", dimension)
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(
            self, "magnetic_rate_per_second", float(self.magnetic_rate_per_second)
        )
        object.__setattr__(
            self, "efg_rate_per_second", float(self.efg_rate_per_second)
        )
        object.__setattr__(
            self, "correlation_time_seconds", float(self.correlation_time_seconds)
        )
        object.__setattr__(
            self, "secular_tolerance_hz", float(self.secular_tolerance_hz)
        )

    def equilibrium_density(self, hamiltonian: np.ndarray) -> np.ndarray:
        """Return the normalized Gibbs fixed point for ``hamiltonian``."""

        return _gibbs_data(hamiltonian, self.temperature_kelvin)[3]

    def _channel_superoperator(
        self,
        operators_pas: Sequence[np.ndarray],
        rate_per_second: float,
        levels_rad: np.ndarray,
        vectors: np.ndarray,
    ) -> np.ndarray:
        dimension = self._dimension
        out = np.zeros((dimension**2, dimension**2), dtype=np.complex128)
        if rate_per_second == 0.0:
            return out
        tolerance = TAU * self.secular_tolerance_hz
        gaps = levels_rad[:, None] - levels_rad[None, :]
        for operator_pas in operators_pas:
            operator = vectors.conj().T @ operator_pas @ vectors
            zero = np.zeros_like(operator, dtype=np.complex128)
            zero[np.abs(gaps) <= tolerance] = operator[np.abs(gaps) <= tolerance]
            if np.any(np.abs(zero) > 0.0):
                jump = vectors @ zero @ vectors.conj().T
                out += _lindblad_superoperator(
                    np.sqrt(rate_per_second) * jump
                )
            for omega, raising_eigen in _positive_frequency_components(
                operator,
                levels_rad,
                tolerance,
                self._frequency_cluster_width_rad_per_second(),
            ):
                spectral_factor = 1.0 / (
                    1.0 + (omega * self.correlation_time_seconds) ** 2
                )
                downward_rate = rate_per_second * spectral_factor
                boltzmann = np.exp(
                    -_PLANCK_J_S
                    * (omega / TAU)
                    / (_BOLTZMANN_J_PER_K * self.temperature_kelvin)
                )
                raising = vectors @ raising_eigen @ vectors.conj().T
                lowering = raising.conj().T
                out += _lindblad_superoperator(
                    np.sqrt(downward_rate) * lowering
                )
                out += _lindblad_superoperator(
                    np.sqrt(downward_rate * boltzmann) * raising
                )
        return out

    def _frequency_cluster_width_rad_per_second(self) -> float:
        return TAU * self.secular_tolerance_hz

    def superoperator(self, hamiltonian: np.ndarray) -> np.ndarray:
        """Return the thermal, trace-preserving field-dependent generator."""

        hamiltonian = _validate_hamiltonian(hamiltonian)
        if hamiltonian.shape != (self._dimension, self._dimension):
            raise ValueError("hamiltonian dimension does not match spin")
        levels_rad, vectors = np.linalg.eigh(hamiltonian)
        spin = spin_matrices(self.spin)
        magnetic = (spin.ix, spin.iy, spin.iz)
        efg = quadrupolar_tesseral_operators(self.spin)
        return self._channel_superoperator(
            magnetic,
            self.magnetic_rate_per_second,
            levels_rad,
            vectors,
        ) + self._channel_superoperator(
            efg,
            self.efg_rate_per_second,
            levels_rad,
            vectors,
        )

    def gibbs_stationarity_error(self, hamiltonian: np.ndarray) -> float:
        """Return ``||R rho_G||`` for the exact Gibbs state of ``hamiltonian``."""

        equilibrium = self.equilibrium_density(hamiltonian)
        derivative = self.superoperator(hamiltonian) @ equilibrium.reshape(
            -1,
            order="F",
        )
        return float(np.linalg.norm(derivative))


@dataclass(frozen=True)
class FieldDependentNonsecularRelaxationModel(
    FieldDependentDaviesRelaxationModel
):
    """Unified-GKLS relaxation for clusters of unresolved Bohr frequencies.

    Transitions separated by at most ``frequency_cluster_width_hz`` are placed
    in a common jump operator, retaining their coherence-transfer cross terms.
    This is completely positive and becomes the fully secular Davies model
    when the cluster width equals ``secular_tolerance_hz``.

    For a finite-width cluster the KMS factor is evaluated at its
    matrix-element-weighted mean frequency. The exact Gibbs state is therefore
    stationary for exactly degenerate clusters and only approximate otherwise;
    :meth:`gibbs_stationarity_error` quantifies that controlled approximation.
    """

    frequency_cluster_width_hz: float = 1.0e3

    def __post_init__(self) -> None:
        super().__post_init__()
        width = float(self.frequency_cluster_width_hz)
        if width < self.secular_tolerance_hz or not np.isfinite(width):
            raise ValueError(
                "frequency_cluster_width_hz must be finite and at least "
                "secular_tolerance_hz"
            )
        object.__setattr__(self, "frequency_cluster_width_hz", width)

    def _frequency_cluster_width_rad_per_second(self) -> float:
        return TAU * self.frequency_cluster_width_hz


def simulate_field_relaxation(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: Sequence[float] | np.ndarray,
    times_seconds: Sequence[float] | np.ndarray,
    *,
    relaxation: FieldDependentRelaxationModel | FieldDependentDaviesRelaxationModel,
    initial_density: np.ndarray | None = None,
) -> FieldRelaxationResult:
    """Propagate a density matrix while relaxing at one fixed static field."""

    b0 = np.asarray(b0_vector_tesla_pas, dtype=np.float64).reshape(3)
    times = np.asarray(times_seconds, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(b0)):
        raise ValueError("b0_vector_tesla_pas must be finite")
    if times.size == 0 or not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("times_seconds must be non-empty, finite, and non-negative")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("times_seconds must be non-decreasing")
    hamiltonian = nqr_hamiltonian(site, b0)
    equilibrium = field_dependent_equilibrium(
        site,
        b0,
        temperature_kelvin=relaxation.temperature_kelvin,
    )
    if initial_density is None:
        density = equilibrium.density_matrix_pas
    else:
        density = np.asarray(initial_density, dtype=np.complex128)
        if density.shape != (site.dimension, site.dimension):
            raise ValueError("initial_density has the wrong shape")
        if not np.allclose(density, density.conj().T):
            raise ValueError("initial_density must be Hermitian")
    generator = liouville_hamiltonian(hamiltonian) + relaxation.superoperator(
        hamiltonian
    )
    initial_vector = density.reshape(-1, order="F")
    densities = np.empty(
        (times.size, site.dimension, site.dimension),
        dtype=np.complex128,
    )
    for index, time in enumerate(times):
        vector = matrix_exponential(generator, float(time)) @ initial_vector
        densities[index] = vector.reshape((site.dimension, site.dimension), order="F")
    ops = spin_matrices(site.spin)
    expectation = np.empty((times.size, 3), dtype=np.float64)
    for index, state in enumerate(densities):
        expectation[index] = [
            np.trace(state @ operator).real for operator in (ops.ix, ops.iy, ops.iz)
        ]
    return FieldRelaxationResult(
        times_seconds=times,
        density_matrices_pas=densities,
        spin_expectation_pas=expectation,
        equilibrium=equilibrium,
        hamiltonian=hamiltonian,
        site=site,
    )
