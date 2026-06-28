"""Scalar NMR relaxation models based on spectral densities.

The helpers here provide a compact Bloembergen-Purcell-Pound (BPP)-style model
for estimating ``T1`` and ``T2`` from a rotational correlation time. The default
rate coefficients use the common dipolar ratios
``R1 ~ J(w0) + 4 J(2 w0)`` and
``R2 ~ 1.5 J(0) + 2.5 J(w0) + J(2 w0)``; the overall scale absorbs the
spin-pair constants and any convention-specific prefactors.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
TAU = 2.0 * np.pi
MU0_OVER_4PI = 1.0e-7
PLANCK = 6.62607015e-34
ANGSTROM = 1.0e-10
PROTON_GAMMA_HZ_PER_T = 42.57747892e6
BOLTZMANN = 1.380649e-23
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27


@dataclass(frozen=True)
class SingleSpinMatrices:
    """Dense angular-momentum matrices for one spin quantum number."""

    spin: float
    m_values: np.ndarray
    identity: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    iz: np.ndarray
    i_plus: np.ndarray
    i_minus: np.ndarray


def spin_dimension(spin: float) -> int:
    """Return the Hilbert-space dimension for one spin."""

    spin = _validate_spin(spin)
    return int(round(2.0 * spin + 1.0))


@lru_cache(maxsize=None)
def single_spin_matrices(spin: float) -> SingleSpinMatrices:
    """Return dense angular-momentum matrices for one spin."""

    spin = _validate_spin(spin)
    dimension = spin_dimension(spin)
    m_values = np.array([spin - idx for idx in range(dimension)], dtype=np.float64)
    i_plus = np.zeros((dimension, dimension), dtype=np.complex128)
    i_minus = np.zeros_like(i_plus)

    by_m = {round(float(m), 12): idx for idx, m in enumerate(m_values)}
    for col, m_value in enumerate(m_values):
        raised = m_value + 1.0
        lowered = m_value - 1.0
        if raised <= spin:
            row = by_m[round(float(raised), 12)]
            i_plus[row, col] = np.sqrt(spin * (spin + 1.0) - m_value * raised)
        if lowered >= -spin:
            row = by_m[round(float(lowered), 12)]
            i_minus[row, col] = np.sqrt(spin * (spin + 1.0) - m_value * lowered)

    ix = 0.5 * (i_plus + i_minus)
    iy = (i_plus - i_minus) / (2.0j)
    iz = np.diag(m_values).astype(np.complex128)
    return SingleSpinMatrices(
        spin=spin,
        m_values=m_values,
        identity=np.eye(dimension, dtype=np.complex128),
        ix=ix,
        iy=iy,
        iz=iz,
        i_plus=i_plus,
        i_minus=i_minus,
    )


@dataclass(frozen=True)
class BPPRelaxationRates:
    """Temperature-dependent BPP rates, times, and spectral densities."""

    temperature_kelvin: np.ndarray
    correlation_time_seconds: np.ndarray
    j0_seconds: np.ndarray
    jw_seconds: np.ndarray
    j2w_seconds: np.ndarray
    r1_per_second: np.ndarray
    r2_per_second: np.ndarray
    t1_seconds: np.ndarray
    t2_seconds: np.ndarray

    def as_parameters(self) -> dict[str, np.ndarray]:
        """Return ``T1`` and ``T2`` fields suitable for workflow parameters."""

        return {"T1": self.t1_seconds.copy(), "T2": self.t2_seconds.copy()}


@dataclass(frozen=True)
class BPPRelaxationModel:
    """Configurable BPP relaxation model with Arrhenius correlation time."""

    angular_frequency_rad_per_s: float | Iterable[float] | np.ndarray
    tau_ref_seconds: float
    coupling_scale_per_second2: float
    reference_temperature_kelvin: float = 298.15
    activation_energy_j_per_mol: float = 0.0
    r1_coefficients: tuple[float, float, float] = (0.0, 1.0, 4.0)
    r2_coefficients: tuple[float, float, float] = (1.5, 2.5, 1.0)
    baseline_r1_per_second: float = 0.0
    baseline_r2_per_second: float = 0.0

    def __post_init__(self) -> None:
        _require_finite_array(
            np.asarray(self.angular_frequency_rad_per_s, dtype=np.float64),
            "angular_frequency_rad_per_s",
        )
        _require_positive(self.tau_ref_seconds, "tau_ref_seconds")
        _require_positive(self.reference_temperature_kelvin, "reference_temperature_kelvin")
        _require_finite_scalar(
            self.activation_energy_j_per_mol,
            "activation_energy_j_per_mol",
        )
        _require_nonnegative(self.coupling_scale_per_second2, "coupling_scale_per_second2")
        _require_nonnegative(self.baseline_r1_per_second, "baseline_r1_per_second")
        _require_nonnegative(self.baseline_r2_per_second, "baseline_r2_per_second")
        _validate_coefficients(self.r1_coefficients, "r1_coefficients")
        _validate_coefficients(self.r2_coefficients, "r2_coefficients")

    def correlation_time(
        self,
        temperature_kelvin: float | Iterable[float] | np.ndarray,
    ) -> np.ndarray:
        """Return Arrhenius rotational correlation times for temperatures."""

        return arrhenius_correlation_time(
            temperature_kelvin,
            tau_ref_seconds=self.tau_ref_seconds,
            reference_temperature_kelvin=self.reference_temperature_kelvin,
            activation_energy_j_per_mol=self.activation_energy_j_per_mol,
        )

    def rates(
        self,
        temperature_kelvin: float | Iterable[float] | np.ndarray,
    ) -> BPPRelaxationRates:
        """Return BPP ``R1``/``R2`` rates and ``T1``/``T2`` times."""

        return bpp_relaxation_rates(
            angular_frequency_rad_per_s=self.angular_frequency_rad_per_s,
            correlation_time_seconds=self.correlation_time(temperature_kelvin),
            temperature_kelvin=temperature_kelvin,
            coupling_scale_per_second2=self.coupling_scale_per_second2,
            r1_coefficients=self.r1_coefficients,
            r2_coefficients=self.r2_coefficients,
            baseline_r1_per_second=self.baseline_r1_per_second,
            baseline_r2_per_second=self.baseline_r2_per_second,
        )


@dataclass(frozen=True)
class PhenomenologicalRelaxationModel:
    """Phenomenological relaxation model in the Hamiltonian energy basis.

    ``t1_seconds`` damps population differences while preserving trace.
    ``t2_seconds`` damps coherences. Both act on density-matrix deviations, so
    the model is suitable for high-temperature spin-dynamics helpers.
    """

    t1_seconds: float = np.inf
    t2_seconds: float = np.inf

    def __post_init__(self) -> None:
        t1_seconds = float(self.t1_seconds)
        t2_seconds = float(self.t2_seconds)
        if not np.isfinite(t1_seconds) and not np.isinf(t1_seconds):
            raise ValueError("t1_seconds must be positive or infinite")
        if not np.isfinite(t2_seconds) and not np.isinf(t2_seconds):
            raise ValueError("t2_seconds must be positive or infinite")
        if t1_seconds <= 0:
            raise ValueError("t1_seconds must be positive or infinite")
        if t2_seconds <= 0:
            raise ValueError("t2_seconds must be positive or infinite")
        object.__setattr__(self, "t1_seconds", t1_seconds)
        object.__setattr__(self, "t2_seconds", t2_seconds)


NQRRelaxationModel = PhenomenologicalRelaxationModel


class RelaxationSuperoperator(Protocol):
    """Protocol for relaxation models that build Hamiltonian-aware Liouvillians."""

    def superoperator(self, hamiltonian: np.ndarray) -> np.ndarray:
        """Return the relaxation Liouvillian for ``hamiltonian``."""


NQRRelaxationSuperoperator = RelaxationSuperoperator


class MotionalAveragingModel(Protocol):
    """Protocol for motional regimes used by microscopic relaxation models."""

    regime: str
    correlation_time_seconds: float

    def covariance_from_source(self, source: DipolarRelaxationSource) -> np.ndarray:
        """Return a target-spin fluctuation covariance for one dipolar source."""

    def spectral_density(self, angular_frequency_rad_per_s: float) -> float:
        """Return the bath spectral density at angular frequency ``omega``."""


@dataclass(frozen=True)
class DipolarRelaxationSource:
    """One fluctuating dipolar bath spin coupled to a target spin."""

    vector_angstrom: Sequence[float] | np.ndarray
    coupling_hz: float | None = None
    gamma_target_hz_per_t: float = 3.0766e6
    gamma_quadrupolar_hz_per_t: float | None = None
    gamma_bath_hz_per_t: float = PROTON_GAMMA_HZ_PER_T
    bath_spin: float = 0.5
    weight: float = 1.0

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector_angstrom, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError("vector_angstrom must be a finite non-zero 3-vector")
        gamma_target = (
            float(self.gamma_target_hz_per_t)
            if self.gamma_quadrupolar_hz_per_t is None
            else float(self.gamma_quadrupolar_hz_per_t)
        )
        coupling = (
            dipolar_coupling_hz(
                norm,
                gamma_a_hz_per_t=gamma_target,
                gamma_b_hz_per_t=self.gamma_bath_hz_per_t,
            )
            if self.coupling_hz is None
            else float(self.coupling_hz)
        )
        if not np.isfinite(coupling):
            raise ValueError("coupling_hz must be finite")
        bath_spin = float(self.bath_spin)
        if bath_spin < 0.0 or not np.isfinite(bath_spin):
            raise ValueError("bath_spin must be non-negative and finite")
        weight = float(self.weight)
        if weight < 0.0 or not np.isfinite(weight):
            raise ValueError("weight must be non-negative and finite")
        object.__setattr__(self, "vector_angstrom", vector)
        object.__setattr__(self, "coupling_hz", coupling)
        object.__setattr__(self, "gamma_target_hz_per_t", gamma_target)
        object.__setattr__(self, "gamma_quadrupolar_hz_per_t", gamma_target)
        object.__setattr__(self, "bath_spin", bath_spin)
        object.__setattr__(self, "weight", weight)

    @property
    def coupling_tensor_rad_per_s(self) -> np.ndarray:
        """Return the dipolar tensor in angular-frequency units."""

        return dipolar_coupling_tensor(
            self.vector_angstrom,
            coupling_hz=float(self.coupling_hz),
        )

    @property
    def covariance_rad2_per_s2(self) -> np.ndarray:
        """Return the target-spin fluctuation covariance from this source."""

        return _bath_scaled_covariance(self.coupling_tensor_rad_per_s, self)


@dataclass(frozen=True)
class RigidSolidMotionalAveraging:
    """Rigid-lattice dipolar fluctuations for solid-state relaxation."""

    correlation_time_seconds: float
    regime: str = "solid"

    def __post_init__(self) -> None:
        _require_positive(self.correlation_time_seconds, "correlation_time_seconds")

    def covariance_from_source(self, source: DipolarRelaxationSource) -> np.ndarray:
        """Return the fixed-frame dipolar covariance for ``source``."""

        return source.covariance_rad2_per_s2

    def spectral_density(self, angular_frequency_rad_per_s: float) -> float:
        """Return a Lorentzian stochastic-bath spectral density."""

        return float(
            spectral_density_lorentzian(
                angular_frequency_rad_per_s,
                self.correlation_time_seconds,
            )
        )


@dataclass(frozen=True)
class IsotropicLiquidMotionalAveraging:
    """Isotropic rotational averaging for liquid-state dipolar relaxation."""

    correlation_time_seconds: float
    regime: str = "isotropic_liquid"

    def __post_init__(self) -> None:
        _require_positive(self.correlation_time_seconds, "correlation_time_seconds")

    def covariance_from_source(self, source: DipolarRelaxationSource) -> np.ndarray:
        """Return the rotationally averaged dipolar covariance for ``source``."""

        fixed = source.covariance_rad2_per_s2
        return float(np.trace(fixed).real / 3.0) * np.eye(3, dtype=np.float64)

    def spectral_density(self, angular_frequency_rad_per_s: float) -> float:
        """Return the isotropic rotational Lorentzian spectral density."""

        return float(
            spectral_density_lorentzian(
                angular_frequency_rad_per_s,
                self.correlation_time_seconds,
            )
        )


@dataclass(frozen=True)
class RedfieldDipolarRelaxationModel:
    """Secular Redfield relaxation model from fluctuating dipolar couplings."""

    spin: float
    coupling_covariance_rad2_per_s2: np.ndarray
    motion: MotionalAveragingModel
    secular_tolerance_rad_per_s: float = 1.0e-6

    def __post_init__(self) -> None:
        spin = float(self.spin)
        dimension = spin_dimension(spin)
        covariance = np.asarray(
            self.coupling_covariance_rad2_per_s2,
            dtype=np.float64,
        )
        if covariance.shape != (3, 3):
            raise ValueError("coupling_covariance_rad2_per_s2 must be 3x3")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("coupling_covariance_rad2_per_s2 must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError("coupling_covariance_rad2_per_s2 must be symmetric")
        eigvals = np.linalg.eigvalsh(covariance)
        if np.min(eigvals) < -1e-9 * max(1.0, float(np.max(np.abs(eigvals)))):
            raise ValueError(
                "coupling_covariance_rad2_per_s2 must be positive semidefinite"
            )
        motion = self.motion
        if not hasattr(motion, "covariance_from_source") or not hasattr(
            motion,
            "spectral_density",
        ):
            raise TypeError("motion must implement MotionalAveragingModel")
        tolerance = float(self.secular_tolerance_rad_per_s)
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise ValueError("secular_tolerance_rad_per_s must be non-negative")
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "_dimension", dimension)
        object.__setattr__(self, "coupling_covariance_rad2_per_s2", covariance)
        object.__setattr__(self, "motion", motion)
        object.__setattr__(self, "secular_tolerance_rad_per_s", tolerance)

    @classmethod
    def from_dipolar_sources(
        cls,
        spin: float,
        sources: Sequence[DipolarRelaxationSource],
        *,
        motion: MotionalAveragingModel | None = None,
        correlation_time_seconds: float | None = None,
        secular_tolerance_rad_per_s: float = 1.0e-6,
    ) -> RedfieldDipolarRelaxationModel:
        """Build a Redfield model from nearby fluctuating dipolar bath spins."""

        if not sources:
            raise ValueError("sources must not be empty")
        if motion is None:
            if correlation_time_seconds is None:
                raise ValueError(
                    "either motion or correlation_time_seconds must be provided"
                )
            motion = RigidSolidMotionalAveraging(correlation_time_seconds)
        elif correlation_time_seconds is not None:
            raise ValueError(
                "pass correlation_time_seconds through the motion model, not both"
            )
        covariance = np.zeros((3, 3), dtype=np.float64)
        for source in sources:
            covariance = covariance + motion.covariance_from_source(source)
        return cls(
            spin=spin,
            coupling_covariance_rad2_per_s2=covariance,
            motion=motion,
            secular_tolerance_rad_per_s=secular_tolerance_rad_per_s,
        )

    @property
    def correlation_time_seconds(self) -> float:
        """Return the motion model correlation time."""

        return self.motion.correlation_time_seconds

    @property
    def regime(self) -> str:
        """Return the motional averaging regime label."""

        return self.motion.regime

    def spectral_density(self, angular_frequency_rad_per_s: float) -> float:
        """Return the motion model spectral density."""

        return self.motion.spectral_density(angular_frequency_rad_per_s)

    def superoperator(self, hamiltonian: np.ndarray) -> np.ndarray:
        """Return the secular Redfield dissipator for ``hamiltonian``."""

        hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
        if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
            raise ValueError("hamiltonian must be square")
        if hamiltonian.shape[0] != self._dimension:
            raise ValueError("hamiltonian dimension does not match spin")
        if np.allclose(self.coupling_covariance_rad2_per_s2, 0.0):
            size = hamiltonian.shape[0] * hamiltonian.shape[0]
            return np.zeros((size, size), dtype=np.complex128)

        energies, vectors = np.linalg.eigh(0.5 * (hamiltonian + hamiltonian.conj().T))
        spin_ops = single_spin_matrices(self.spin)
        ops_pas = (spin_ops.ix, spin_ops.iy, spin_ops.iz)

        covariance_values, covariance_vectors = np.linalg.eigh(
            self.coupling_covariance_rad2_per_s2
        )
        out = np.zeros(
            (hamiltonian.shape[0] * hamiltonian.shape[0],) * 2,
            dtype=np.complex128,
        )
        for strength, axis in zip(covariance_values, covariance_vectors.T):
            if strength <= 0.0:
                continue
            lab_operator = sum(float(axis[i]) * ops_pas[i] for i in range(3))
            energy_operator = vectors.conj().T @ lab_operator @ vectors
            for omega, jump in _secular_components(
                energy_operator,
                energies,
                self.secular_tolerance_rad_per_s,
            ):
                rate = float(strength) * self.spectral_density(omega)
                if rate <= 0.0:
                    continue
                out = out + _lindblad_superoperator(np.sqrt(rate) * jump)
        return out


@dataclass(frozen=True)
class WallCollisionRelaxationModel:
    """Gas-wall collision relaxation from a stochastic spin map.

    The model assumes the fast-diffusion/ballistic-wall limit where a gas atom
    samples container walls as independent Poisson events. Kinetic theory gives
    the wall encounter rate

    ``k = accommodation_probability * mean_speed * (S/V) / 4``.

    Each encounter applies an isotropic depolarizing spin channel with
    probability ``depolarization_probability``. The continuous-time generator is
    therefore ``k * (Phi - I)``, where ``Phi`` is the one-collision quantum map.
    """

    spin: float
    collision_rate_per_second: float
    depolarization_probability: float

    def __post_init__(self) -> None:
        spin = _validate_spin(self.spin)
        collision_rate = float(self.collision_rate_per_second)
        probability = float(self.depolarization_probability)
        if not np.isfinite(collision_rate) or collision_rate < 0.0:
            raise ValueError("collision_rate_per_second must be non-negative")
        if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError("depolarization_probability must be in [0, 1]")
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "_dimension", spin_dimension(spin))
        object.__setattr__(self, "collision_rate_per_second", collision_rate)
        object.__setattr__(self, "depolarization_probability", probability)

    @classmethod
    def from_geometry(
        cls,
        spin: float,
        *,
        surface_to_volume_per_m: float,
        temperature_kelvin: float,
        mass_amu: float,
        depolarization_probability: float,
        accommodation_probability: float = 1.0,
    ) -> WallCollisionRelaxationModel:
        """Build a wall model from gas kinetic theory and container ``S/V``."""

        collision_rate = wall_collision_rate_per_second(
            surface_to_volume_per_m,
            temperature_kelvin=temperature_kelvin,
            mass_amu=mass_amu,
            accommodation_probability=accommodation_probability,
        )
        return cls(
            spin=spin,
            collision_rate_per_second=float(collision_rate),
            depolarization_probability=depolarization_probability,
        )

    @property
    def relaxation_rate_per_second(self) -> float:
        """Return the decay rate of traceless spin magnetization."""

        return self.collision_rate_per_second * self.depolarization_probability

    @property
    def t1_seconds(self) -> float:
        """Return the isotropic wall-limited ``T1``."""

        rate = self.relaxation_rate_per_second
        return np.inf if rate == 0.0 else 1.0 / rate

    @property
    def t2_seconds(self) -> float:
        """Return the isotropic wall-limited ``T2``."""

        return self.t1_seconds

    def equivalent_surface_relaxivity_m_per_s(
        self,
        *,
        mean_speed_m_per_s: float,
    ) -> float:
        """Return ``rho`` such that ``rho S/V`` matches this microscopic rate."""

        mean_speed = float(mean_speed_m_per_s)
        if not np.isfinite(mean_speed) or mean_speed <= 0.0:
            raise ValueError("mean_speed_m_per_s must be positive")
        return 0.25 * mean_speed * self.depolarization_probability

    def superoperator(self, hamiltonian: np.ndarray) -> np.ndarray:
        """Return the Poisson collision-map generator for ``hamiltonian`` size."""

        hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
        if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
            raise ValueError("hamiltonian must be square")
        if hamiltonian.shape[0] != self._dimension:
            raise ValueError("hamiltonian dimension does not match spin")
        dimension = hamiltonian.shape[0]
        size = dimension * dimension
        identity_super = np.eye(size, dtype=np.complex128)
        identity_vector = np.eye(dimension, dtype=np.complex128).reshape(
            -1,
            order="F",
        )
        reset_super = np.outer(identity_vector / dimension, identity_vector)
        one_collision = (
            (1.0 - self.depolarization_probability) * identity_super
            + self.depolarization_probability * reset_super
        )
        return self.collision_rate_per_second * (one_collision - identity_super)


RelaxationModelLike = PhenomenologicalRelaxationModel | RelaxationSuperoperator
NQRRelaxationLike = RelaxationModelLike


def spectral_density_lorentzian(
    angular_frequency_rad_per_s: float | Iterable[float] | np.ndarray,
    correlation_time_seconds: float | Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return the isotropic rotational spectral density ``2 tau/(1+w^2 tau^2)``."""

    omega, tau = np.broadcast_arrays(
        np.asarray(angular_frequency_rad_per_s, dtype=np.float64),
        np.asarray(correlation_time_seconds, dtype=np.float64),
    )
    _require_finite_array(omega, "angular_frequency_rad_per_s")
    _require_finite_array(tau, "correlation_time_seconds")
    if np.any(tau <= 0.0):
        raise ValueError("correlation_time_seconds must be positive")
    return 2.0 * tau / (1.0 + (omega * tau) ** 2)


def arrhenius_correlation_time(
    temperature_kelvin: float | Iterable[float] | np.ndarray,
    *,
    tau_ref_seconds: float,
    reference_temperature_kelvin: float = 298.15,
    activation_energy_j_per_mol: float = 0.0,
) -> np.ndarray:
    """Return ``tau_c(T)`` using an Arrhenius activation energy."""

    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    _require_finite_array(temperature, "temperature_kelvin")
    if np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be positive")
    _require_positive(tau_ref_seconds, "tau_ref_seconds")
    _require_positive(reference_temperature_kelvin, "reference_temperature_kelvin")
    _require_finite_scalar(activation_energy_j_per_mol, "activation_energy_j_per_mol")
    exponent = (
        float(activation_energy_j_per_mol)
        / GAS_CONSTANT_J_PER_MOL_K
        * (1.0 / temperature - 1.0 / float(reference_temperature_kelvin))
    )
    tau = float(tau_ref_seconds) * np.exp(exponent)
    _require_finite_array(tau, "correlation_time_seconds")
    return tau


def stokes_einstein_debye_correlation_time(
    hydrodynamic_radius_m: float | Iterable[float] | np.ndarray,
    viscosity_pa_s: float | Iterable[float] | np.ndarray,
    temperature_kelvin: float | Iterable[float] | np.ndarray,
    *,
    slip_factor: float = 1.0,
) -> np.ndarray:
    """Return the rank-2 rotational correlation time from Stokes-Einstein-Debye.

    For a sphere of hydrodynamic radius ``a`` reorienting in a medium of shear
    viscosity ``eta`` at temperature ``T``, the Debye rotational diffusion
    constant is ``D_R = k_B T / (8 pi eta a^3)``. The rank-2 (``l = 2``)
    reorientational correlation time relevant to dipolar and quadrupolar NMR
    relaxation is

    ``tau_c = 1 / (6 D_R) = 4 pi eta a^3 / (3 k_B T) * f``,

    where the dimensionless ``slip_factor`` ``f`` rescales the stick-limit result
    for microviscosity or slip boundary conditions (``f = 1`` is the stick limit;
    ``f < 1`` approaches slip and is typical for small molecules). Inputs
    broadcast against one another, so a temperature or viscosity sweep returns an
    array of correlation times.
    """

    radius, viscosity, temperature = np.broadcast_arrays(
        np.asarray(hydrodynamic_radius_m, dtype=np.float64),
        np.asarray(viscosity_pa_s, dtype=np.float64),
        np.asarray(temperature_kelvin, dtype=np.float64),
    )
    _require_finite_array(radius, "hydrodynamic_radius_m")
    _require_finite_array(viscosity, "viscosity_pa_s")
    _require_finite_array(temperature, "temperature_kelvin")
    if np.any(radius <= 0.0):
        raise ValueError("hydrodynamic_radius_m must be positive")
    if np.any(viscosity <= 0.0):
        raise ValueError("viscosity_pa_s must be positive")
    if np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be positive")
    slip = float(slip_factor)
    if not np.isfinite(slip) or slip <= 0.0:
        raise ValueError("slip_factor must be positive")
    tau = (
        4.0 * np.pi * viscosity * radius**3 * slip
        / (3.0 * BOLTZMANN * temperature)
    )
    _require_finite_array(tau, "correlation_time_seconds")
    return tau


@lru_cache(maxsize=None)
def _bpp_t1_min_omega_tau(r1_coefficients: tuple[float, float, float]) -> float:
    """Return ``omega0 * tau_c`` at the BPP ``T1`` minimum for ``R1`` weights.

    The ``T1`` minimum is the ``R1`` maximum. With ``J(w) = 2 tau / (1 + w^2
    tau^2)`` and ``R1 ~ a J(0) + b J(w0) + c J(2 w0)``, the stationary condition
    ``dR1/dtau = 0`` is a polynomial in ``u = (w0 tau)^2``:

    ``a (1+u)^2 (1+4u)^2 + b (1-u)(1+4u)^2 + c (1-4u)(1+u)^2 = 0``.

    For the canonical dipolar weights ``(0, 1, 4)`` the positive real root gives
    the textbook ``w0 tau_c ~ 0.6158``.
    """

    a, b, c = r1_coefficients
    p1 = np.array([1.0, 2.0, 1.0])  # (1 + u)^2
    p4 = np.array([16.0, 8.0, 1.0])  # (1 + 4u)^2
    poly = np.polyadd(a * np.convolve(p1, p4), b * np.convolve([-1.0, 1.0], p4))
    poly = np.polyadd(poly, c * np.convolve([-4.0, 1.0], p1))
    roots = np.roots(poly)
    real = roots[np.abs(roots.imag) <= 1.0e-9 * (1.0 + np.abs(roots.real))].real
    positive = real[real > 0.0]
    if positive.size == 0:
        raise ValueError("r1_coefficients do not yield a T1 minimum")
    return float(np.sqrt(np.min(positive)))


def tau_c_from_t1_minimum(
    angular_frequency_rad_per_s: float,
    *,
    r1_coefficients: tuple[float, float, float] = (0.0, 1.0, 4.0),
) -> float:
    """Return the correlation time at the BPP ``T1`` minimum for a Larmor freq.

    A measured ``T1`` minimum versus temperature pins the absolute correlation
    time at that temperature, independent of the dipolar coupling strength: only
    the depth of the minimum scales with the coupling, not its location. For the
    default dipolar weights this returns ``tau_c = 0.6158 / omega0``.
    """

    omega = float(angular_frequency_rad_per_s)
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("angular_frequency_rad_per_s must be positive")
    coefficients = _validate_coefficients(r1_coefficients, "r1_coefficients")
    return _bpp_t1_min_omega_tau(coefficients) / omega


BPP_T1_MINIMUM_OMEGA_TAU = _bpp_t1_min_omega_tau((0.0, 1.0, 4.0))


def gas_mean_speed_m_per_s(
    temperature_kelvin: float | Iterable[float] | np.ndarray,
    mass_amu: float,
) -> np.ndarray:
    """Return Maxwell-Boltzmann mean molecular speed for a gas species."""

    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    _require_finite_array(temperature, "temperature_kelvin")
    if np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be positive")
    mass_kg = float(mass_amu) * ATOMIC_MASS_UNIT_KG
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("mass_amu must be positive")
    return np.sqrt(8.0 * BOLTZMANN * temperature / (np.pi * mass_kg))


def wall_collision_rate_per_second(
    surface_to_volume_per_m: float | Iterable[float] | np.ndarray,
    *,
    temperature_kelvin: float,
    mass_amu: float,
    accommodation_probability: float = 1.0,
) -> np.ndarray:
    """Return gas-wall encounter rate ``vbar * (S/V) / 4``."""

    surface_to_volume = np.asarray(surface_to_volume_per_m, dtype=np.float64)
    _require_finite_array(surface_to_volume, "surface_to_volume_per_m")
    if np.any(surface_to_volume < 0.0):
        raise ValueError("surface_to_volume_per_m must be non-negative")
    accommodation = float(accommodation_probability)
    if not np.isfinite(accommodation) or accommodation < 0.0 or accommodation > 1.0:
        raise ValueError("accommodation_probability must be in [0, 1]")
    mean_speed = gas_mean_speed_m_per_s(temperature_kelvin, mass_amu)
    return accommodation * mean_speed * surface_to_volume / 4.0


def sphere_surface_to_volume_per_m(
    diameter_m: float | Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return ``S/V`` for a sphere from its diameter."""

    diameter = np.asarray(diameter_m, dtype=np.float64)
    _require_finite_array(diameter, "diameter_m")
    if np.any(diameter <= 0.0):
        raise ValueError("diameter_m must be positive")
    return 6.0 / diameter


def cube_surface_to_volume_per_m(
    edge_m: float | Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return ``S/V`` for a cube from its edge length."""

    edge = np.asarray(edge_m, dtype=np.float64)
    _require_finite_array(edge, "edge_m")
    if np.any(edge <= 0.0):
        raise ValueError("edge_m must be positive")
    return 6.0 / edge


def cylinder_surface_to_volume_per_m(
    diameter_m: float | Iterable[float] | np.ndarray,
    *,
    aspect: float,
) -> np.ndarray:
    """Return ``S/V`` for a closed cylinder from diameter and ``length/diameter``."""

    diameter = np.asarray(diameter_m, dtype=np.float64)
    _require_finite_array(diameter, "diameter_m")
    if np.any(diameter <= 0.0):
        raise ValueError("diameter_m must be positive")
    aspect = float(aspect)
    if not np.isfinite(aspect) or aspect <= 0.0:
        raise ValueError("aspect must be positive")
    radius = 0.5 * diameter
    length = aspect * diameter
    return 2.0 / radius + 2.0 / length


def bpp_relaxation_rates(
    *,
    angular_frequency_rad_per_s: float | Iterable[float] | np.ndarray,
    correlation_time_seconds: float | Iterable[float] | np.ndarray,
    temperature_kelvin: float | Iterable[float] | np.ndarray | None = None,
    coupling_scale_per_second2: float = 1.0,
    r1_coefficients: tuple[float, float, float] = (0.0, 1.0, 4.0),
    r2_coefficients: tuple[float, float, float] = (1.5, 2.5, 1.0),
    baseline_r1_per_second: float = 0.0,
    baseline_r2_per_second: float = 0.0,
) -> BPPRelaxationRates:
    """Return BPP relaxation rates from ``J(0)``, ``J(w0)``, and ``J(2w0)``."""

    _require_nonnegative(coupling_scale_per_second2, "coupling_scale_per_second2")
    _require_nonnegative(baseline_r1_per_second, "baseline_r1_per_second")
    _require_nonnegative(baseline_r2_per_second, "baseline_r2_per_second")
    r1c = _validate_coefficients(r1_coefficients, "r1_coefficients")
    r2c = _validate_coefficients(r2_coefficients, "r2_coefficients")

    omega, tau = np.broadcast_arrays(
        np.asarray(angular_frequency_rad_per_s, dtype=np.float64),
        np.asarray(correlation_time_seconds, dtype=np.float64),
    )
    if temperature_kelvin is None:
        temperature = np.full_like(tau, np.nan, dtype=np.float64)
    else:
        temperature = np.broadcast_to(
            np.asarray(temperature_kelvin, dtype=np.float64),
            tau.shape,
        ).copy()
        _require_finite_array(temperature, "temperature_kelvin")
        if np.any(temperature <= 0.0):
            raise ValueError("temperature_kelvin must be positive")

    j0 = spectral_density_lorentzian(0.0, tau)
    jw = spectral_density_lorentzian(omega, tau)
    j2w = spectral_density_lorentzian(2.0 * omega, tau)
    scale = float(coupling_scale_per_second2)
    r1 = scale * (r1c[0] * j0 + r1c[1] * jw + r1c[2] * j2w)
    r2 = scale * (r2c[0] * j0 + r2c[1] * jw + r2c[2] * j2w)
    r1 = r1 + float(baseline_r1_per_second)
    r2 = r2 + float(baseline_r2_per_second)
    return BPPRelaxationRates(
        temperature_kelvin=temperature,
        correlation_time_seconds=tau.copy(),
        j0_seconds=j0,
        jw_seconds=jw,
        j2w_seconds=j2w,
        r1_per_second=r1,
        r2_per_second=r2,
        t1_seconds=_rate_to_time(r1),
        t2_seconds=_rate_to_time(r2),
    )


def apply_relaxation_to_parameters(
    params: Mapping[str, Any] | Any,
    rates: BPPRelaxationRates,
) -> dict[str, Any]:
    """Return a shallow parameter copy with ``T1`` and ``T2`` replaced."""

    if isinstance(params, Mapping):
        updated = dict(params)
    else:
        updated = {
            name: getattr(params, name)
            for name in dir(params)
            if not name.startswith("_") and not callable(getattr(params, name))
        }
    updated["T1"] = rates.t1_seconds.copy()
    updated["T2"] = rates.t2_seconds.copy()
    return updated


def dipolar_coupling_hz(
    distance_angstrom: float,
    *,
    gamma_a_hz_per_t: float = 3.0766e6,
    gamma_b_hz_per_t: float = PROTON_GAMMA_HZ_PER_T,
) -> float:
    """Return the point-dipole coupling prefactor in Hz."""

    distance_m = float(distance_angstrom) * ANGSTROM
    if distance_m <= 0.0 or not np.isfinite(distance_m):
        raise ValueError("distance_angstrom must be positive and finite")
    return (
        MU0_OVER_4PI
        * PLANCK
        * abs(float(gamma_a_hz_per_t) * float(gamma_b_hz_per_t))
        / distance_m**3
    )


def dipolar_coupling_tensor(
    vector_angstrom: Sequence[float] | np.ndarray,
    *,
    coupling_hz: float,
) -> np.ndarray:
    """Return ``2*pi*d*(I - 3 n n^T)`` for a point dipolar coupling."""

    vector = np.asarray(vector_angstrom, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("vector_angstrom must be a finite non-zero 3-vector")
    coupling = float(coupling_hz)
    if not np.isfinite(coupling):
        raise ValueError("coupling_hz must be finite")
    direction = vector / norm
    tensor = np.eye(3, dtype=np.float64) - 3.0 * np.outer(direction, direction)
    return TAU * coupling * tensor


def matrix_exponential(matrix: np.ndarray, duration: float = 1.0) -> np.ndarray:
    """Return ``exp(matrix * duration)`` for a small dense matrix."""

    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    duration = float(duration)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be non-negative and finite")
    if duration == 0:
        return np.eye(matrix.shape[0], dtype=np.complex128)
    values, vectors = np.linalg.eig(matrix)
    return (vectors * np.exp(values * duration)[np.newaxis, :]) @ np.linalg.inv(vectors)


def liouville_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    """Return the commutator Liouvillian for column-stacked density matrices."""

    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be square")
    dim = hamiltonian.shape[0]
    identity = np.eye(dim, dtype=np.complex128)
    return -1j * (np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity))


def relaxation_superoperator(
    dimension: int,
    model: RelaxationModelLike,
    *,
    hamiltonian: np.ndarray | None = None,
) -> np.ndarray:
    """Return a trace-preserving relaxation superoperator."""

    if not isinstance(model, PhenomenologicalRelaxationModel):
        if hamiltonian is None:
            raise ValueError("hamiltonian is required for microscopic relaxation")
        return model.superoperator(hamiltonian)

    dimension = int(dimension)
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    size = dimension * dimension
    out = np.zeros((size, size), dtype=np.complex128)

    if np.isfinite(model.t1_seconds):
        rate = 1.0 / model.t1_seconds
        for row in range(dimension):
            row_index = row + row * dimension
            for col in range(dimension):
                col_index = col + col * dimension
                out[row_index, col_index] += rate / dimension
            out[row_index, row_index] -= rate

    if np.isfinite(model.t2_seconds):
        rate = 1.0 / model.t2_seconds
        for row in range(dimension):
            for col in range(dimension):
                if row == col:
                    continue
                out[row + col * dimension, row + col * dimension] -= rate

    return out


def liouville_superoperator(
    hamiltonian: np.ndarray,
    model: RelaxationModelLike | None = None,
) -> np.ndarray:
    """Return Hamiltonian plus optional relaxation Liouvillian."""

    out = liouville_hamiltonian(hamiltonian)
    if model is not None:
        out = out + relaxation_superoperator(
            hamiltonian.shape[0],
            model,
            hamiltonian=hamiltonian,
        )
    return out


def propagate_density_liouville(
    density: np.ndarray,
    hamiltonian: np.ndarray,
    duration: float,
    *,
    relaxation: RelaxationModelLike | None = None,
) -> np.ndarray:
    """Propagate a density matrix with Hamiltonian and optional relaxation."""

    density = np.asarray(density, dtype=np.complex128)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density must be square")
    superoperator = matrix_exponential(
        liouville_superoperator(hamiltonian, relaxation),
        duration,
    )
    vector = density.reshape(-1, order="F")
    return (superoperator @ vector).reshape(density.shape, order="F")


def cycle_superoperator(
    steps: tuple[tuple[np.ndarray, float], ...] | list[tuple[np.ndarray, float]],
    *,
    relaxation: RelaxationModelLike | None = None,
) -> np.ndarray:
    """Return the Liouville propagator for one repeated pulse-sequence cycle."""

    if not steps:
        raise ValueError("steps must not be empty")
    first = np.asarray(steps[0][0], dtype=np.complex128)
    size = first.shape[0] * first.shape[0]
    out = np.eye(size, dtype=np.complex128)
    for hamiltonian, duration in steps:
        hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
        step = matrix_exponential(
            liouville_superoperator(hamiltonian, relaxation),
            duration,
        )
        out = step @ out
    return out


def effective_decay_time(
    eigenvalues: np.ndarray,
    cycle_duration_seconds: float,
    *,
    steady_tolerance: float = 1e-10,
) -> float:
    """Estimate the dominant non-steady decay time from cycle eigenvalues."""

    cycle_duration_seconds = float(cycle_duration_seconds)
    if cycle_duration_seconds <= 0 or not np.isfinite(cycle_duration_seconds):
        raise ValueError("cycle_duration_seconds must be positive and finite")
    magnitudes = np.abs(np.asarray(eigenvalues, dtype=np.complex128).reshape(-1))
    candidates = magnitudes[
        (magnitudes > 0.0)
        & np.isfinite(magnitudes)
        & (magnitudes < 1.0 - steady_tolerance)
    ]
    if candidates.size == 0:
        return np.inf
    dominant = float(np.max(candidates))
    return -cycle_duration_seconds / np.log(dominant)


def _bath_scaled_covariance(
    tensor_rad_per_s: np.ndarray,
    source: DipolarRelaxationSource,
) -> np.ndarray:
    tensor = np.asarray(tensor_rad_per_s, dtype=np.float64)
    bath_factor = source.bath_spin * (source.bath_spin + 1.0) / 3.0
    return float(source.weight) * bath_factor * (tensor @ tensor.T)


def _lindblad_superoperator(jump: np.ndarray) -> np.ndarray:
    jump = np.asarray(jump, dtype=np.complex128)
    dim = jump.shape[0]
    identity = np.eye(dim, dtype=np.complex128)
    metric = jump.conj().T @ jump
    return (
        np.kron(jump.conj(), jump)
        - 0.5 * np.kron(identity, metric)
        - 0.5 * np.kron(metric.T, identity)
    )


def _secular_components(
    operator: np.ndarray,
    energies_rad_per_s: np.ndarray,
    tolerance_rad_per_s: float,
) -> list[tuple[float, np.ndarray]]:
    dim = operator.shape[0]
    frequencies = energies_rad_per_s[:, None] - energies_rad_per_s[None, :]
    remaining = np.ones((dim, dim), dtype=bool)
    components: list[tuple[float, np.ndarray]] = []
    for row in range(dim):
        for col in range(dim):
            if not remaining[row, col]:
                continue
            omega = float(frequencies[row, col])
            mask = remaining & (np.abs(frequencies - omega) <= tolerance_rad_per_s)
            jump = np.zeros_like(operator, dtype=np.complex128)
            jump[mask] = operator[mask]
            if np.any(np.abs(jump) > 0.0):
                components.append((omega, jump))
            remaining[mask] = False
    return components


def _validate_spin(spin: float) -> float:
    spin = float(spin)
    if spin <= 0:
        raise ValueError("spin must be positive")
    two_spin = round(2.0 * spin)
    if not np.isclose(2.0 * spin, two_spin):
        raise ValueError("spin must be an integer or half-integer")
    return spin


def _rate_to_time(rate: np.ndarray) -> np.ndarray:
    return np.divide(
        1.0,
        rate,
        out=np.full_like(rate, np.inf, dtype=np.float64),
        where=rate > 0.0,
    )


def _validate_coefficients(
    coefficients: tuple[float, float, float],
    name: str,
) -> tuple[float, float, float]:
    if len(coefficients) != 3:
        raise ValueError(f"{name} must contain three coefficients")
    values = tuple(float(value) for value in coefficients)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} must be non-negative")
    return values


def _require_finite_array(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")


def _require_finite_scalar(value: float, name: str) -> None:
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_positive(value: float, name: str) -> None:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(value: float, name: str) -> None:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative")
