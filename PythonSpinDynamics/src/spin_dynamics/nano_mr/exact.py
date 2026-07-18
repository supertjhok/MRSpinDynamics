"""Dense exact-spin models for resolved defect-sensor nano-MR clusters.

The sensor is tensor factor zero and resolved nuclei follow in the order given
to :class:`ResolvedSpinCluster`.  Hamiltonians are returned in radians per
second.  This module deliberately targets small clusters and enforces an
explicit Hilbert-space ceiling before allocating dense matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.coupling.mixed_operators import (
    dot_product_operator,
    embedded_operator,
    hilbert_dimension,
    product_operator,
)
from spin_dynamics.esr.lineshapes import spectrum_from_lines
from spin_dynamics.nano_mr.baths import PLANCK_CONSTANT_J_S
from spin_dynamics.nano_mr.frames import rotation_from_z, unit_vector
from spin_dynamics.nano_mr.geometry import (
    ISOTOPE_GAMMA_HZ_PER_T,
    MU0_OVER_4PI,
    NuclearSpin,
    dipole_spatial_tensor_inverse_m3,
    point_dipolar_hyperfine_tensor_hz,
)
from spin_dynamics.nano_mr.hamiltonians import (
    TAU,
    drive_operator,
    sensor_hamiltonian,
)
from spin_dynamics.nano_mr.sensors import DefectSpinSensor, as_symmetric_tensor


DEFAULT_MAX_HILBERT_DIMENSION = 64
"""Default dense-cluster ceiling.

For a spin-1 sensor this permits four spin-1/2 nuclei (dimension 48) and
rejects five (dimension 96).  Users may raise the per-cluster limit
deliberately when memory and run time have been assessed.
"""

ScalarCouplingModel = Literal["isotropic", "secular"]
NuclearDipolarModel = Literal["full", "secular"]


def _point(values, *, name: str) -> np.ndarray:
    point = np.asarray(values, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be finite")
    return point


@dataclass(frozen=True, eq=False)
class ResolvedNucleus:
    """One explicitly propagated target nucleus.

    ``chemical_shift_ppm`` is a scalar, principal-value vector, or symmetric
    laboratory-frame tensor.  Positive values increase the bare Larmor
    frequency in this package's frequency-shift convention.
    """

    isotope: str
    position_lab_nm: np.ndarray
    gamma_hz_per_t: float
    spin: float = 0.5
    chemical_shift_ppm: float | np.ndarray | tuple[float, ...] = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        target = NuclearSpin(
            isotope=self.isotope,
            position_lab_nm=self.position_lab_nm,
            gamma_hz_per_t=self.gamma_hz_per_t,
            spin=self.spin,
            label=self.label,
        )
        shift = as_symmetric_tensor(
            self.chemical_shift_ppm,
            name="chemical_shift_ppm",
        )
        object.__setattr__(self, "isotope", target.isotope)
        object.__setattr__(self, "position_lab_nm", target.position_lab_nm)
        object.__setattr__(self, "gamma_hz_per_t", target.gamma_hz_per_t)
        object.__setattr__(self, "spin", target.spin)
        object.__setattr__(self, "chemical_shift_ppm", shift)
        object.__setattr__(self, "label", target.label)

    @classmethod
    def from_isotope(
        cls,
        isotope: str,
        position_lab_nm,
        *,
        spin: float = 0.5,
        chemical_shift_ppm: float | np.ndarray | tuple[float, ...] = 0.0,
        label: str = "",
    ) -> "ResolvedNucleus":
        """Construct a resolved nucleus from a built-in isotope preset."""

        try:
            gamma = ISOTOPE_GAMMA_HZ_PER_T[str(isotope)]
        except KeyError as exc:
            raise ValueError(f"unknown isotope preset: {isotope!r}") from exc
        return cls(
            isotope=str(isotope),
            position_lab_nm=position_lab_nm,
            gamma_hz_per_t=gamma,
            spin=spin,
            chemical_shift_ppm=chemical_shift_ppm,
            label=label,
        )

    def as_target(self) -> NuclearSpin:
        """Return the point-dipole geometry representation of this nucleus."""

        return NuclearSpin(
            isotope=self.isotope,
            position_lab_nm=self.position_lab_nm,
            gamma_hz_per_t=self.gamma_hz_per_t,
            spin=self.spin,
            label=self.label,
        )


@dataclass(frozen=True)
class NuclearScalarCoupling:
    """Scalar coupling between two zero-based nuclear indices."""

    nucleus_a: int
    nucleus_b: int
    j_hz: float
    model: ScalarCouplingModel = "isotropic"

    def __post_init__(self) -> None:
        a = int(self.nucleus_a)
        b = int(self.nucleus_b)
        coupling = float(self.j_hz)
        if a < 0 or b < 0 or a == b:
            raise ValueError("scalar coupling requires two distinct nuclear indices")
        if not np.isfinite(coupling):
            raise ValueError("j_hz must be finite")
        if self.model not in {"isotropic", "secular"}:
            raise ValueError("scalar coupling model must be 'isotropic' or 'secular'")
        object.__setattr__(self, "nucleus_a", a)
        object.__setattr__(self, "nucleus_b", b)
        object.__setattr__(self, "j_hz", coupling)


@dataclass(frozen=True, eq=False)
class ResolvedSpinCluster:
    """A defect sensor and a deliberately small set of resolved nuclei."""

    sensor: DefectSpinSensor
    nuclei: tuple[ResolvedNucleus, ...]
    sensor_position_lab_nm: np.ndarray
    scalar_couplings: tuple[NuclearScalarCoupling, ...] = ()
    max_hilbert_dimension: int = DEFAULT_MAX_HILBERT_DIMENSION
    label: str = "resolved cluster"

    def __post_init__(self) -> None:
        if not isinstance(self.sensor, DefectSpinSensor):
            raise TypeError("sensor must be a DefectSpinSensor")
        nuclei = tuple(self.nuclei)
        if not nuclei:
            raise ValueError("a resolved cluster requires at least one nucleus")
        if not all(isinstance(item, ResolvedNucleus) for item in nuclei):
            raise TypeError("nuclei must contain ResolvedNucleus objects")
        couplings = tuple(self.scalar_couplings)
        if not all(isinstance(item, NuclearScalarCoupling) for item in couplings):
            raise TypeError(
                "scalar_couplings must contain NuclearScalarCoupling objects"
            )
        limit = int(self.max_hilbert_dimension)
        if limit < 2:
            raise ValueError("max_hilbert_dimension must be at least two")
        dimension = hilbert_dimension((self.sensor.spin, *(n.spin for n in nuclei)))
        if dimension > limit:
            raise ValueError(
                f"dense Hilbert dimension {dimension} exceeds configured limit "
                f"{limit}; reduce the resolved cluster or explicitly raise "
                "max_hilbert_dimension"
            )
        for coupling in couplings:
            if max(coupling.nucleus_a, coupling.nucleus_b) >= len(nuclei):
                raise ValueError("scalar coupling index is outside the nuclear list")
        object.__setattr__(self, "nuclei", nuclei)
        object.__setattr__(self, "scalar_couplings", couplings)
        object.__setattr__(
            self,
            "sensor_position_lab_nm",
            _point(self.sensor_position_lab_nm, name="sensor_position_lab_nm"),
        )
        object.__setattr__(self, "max_hilbert_dimension", limit)
        object.__setattr__(self, "label", str(self.label))

    @property
    def spins(self) -> tuple[float, ...]:
        """Tensor-factor spin quantum numbers, sensor first."""

        return (self.sensor.spin, *(nucleus.spin for nucleus in self.nuclei))

    @property
    def dimension(self) -> int:
        """Dense Hilbert-space dimension."""

        return hilbert_dimension(self.spins)

    @property
    def nuclear_dimension(self) -> int:
        """Nuclear-only Hilbert-space dimension."""

        return self.dimension // self.sensor.dimension


def nuclear_zeeman_hamiltonian(
    cluster: ResolvedSpinCluster,
    b0_vector_tesla_lab,
) -> np.ndarray:
    """Return nuclear Zeeman plus chemical-shift terms in radians per second."""

    b0 = _point(b0_vector_tesla_lab, name="b0_vector_tesla_lab")
    out = np.zeros((cluster.dimension, cluster.dimension), dtype=np.complex128)
    identity = np.eye(3)
    for nuclear_index, nucleus in enumerate(cluster.nuclei):
        shifted_field = (identity + 1.0e-6 * nucleus.chemical_shift_ppm).T @ b0
        factor_index = nuclear_index + 1
        for axis, component in zip("xyz", shifted_field):
            out -= (
                TAU
                * nucleus.gamma_hz_per_t
                * component
                * embedded_operator(cluster.spins, factor_index, axis)
            )
    return out


def sensor_nuclear_hyperfine_hamiltonian(
    cluster: ResolvedSpinCluster,
) -> np.ndarray:
    """Return the full point-dipole sensor-target interaction."""

    out = np.zeros((cluster.dimension, cluster.dimension), dtype=np.complex128)
    for nuclear_index, nucleus in enumerate(cluster.nuclei):
        tensor_hz = point_dipolar_hyperfine_tensor_hz(
            cluster.sensor,
            nucleus.as_target(),
            sensor_position_lab_nm=cluster.sensor_position_lab_nm,
        )
        for electron_axis, row in zip("xyz", tensor_hz):
            for nuclear_axis, coefficient in zip("xyz", row):
                out += (
                    TAU
                    * coefficient
                    * product_operator(
                        cluster.spins,
                        [(0, electron_axis), (nuclear_index + 1, nuclear_axis)],
                    )
                )
    return out


def _bilinear_spin_operator(
    cluster: ResolvedSpinCluster,
    factor_a: int,
    direction_a,
    factor_b: int,
    direction_b,
) -> np.ndarray:
    """Return ``(direction_a.I_a) (direction_b.I_b)`` in the cluster basis."""

    first = np.asarray(direction_a, dtype=np.float64).reshape(3)
    second = np.asarray(direction_b, dtype=np.float64).reshape(3)
    operator = np.zeros((cluster.dimension, cluster.dimension), dtype=np.complex128)
    for axis_a, coefficient_a in zip("xyz", first):
        for axis_b, coefficient_b in zip("xyz", second):
            operator += coefficient_a * coefficient_b * product_operator(
                cluster.spins,
                [(factor_a, axis_a), (factor_b, axis_b)],
            )
    return operator


def nuclear_scalar_coupling_hamiltonian(
    cluster: ResolvedSpinCluster,
    *,
    quantization_axis_lab=(0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return configured isotropic or field-secular scalar couplings.

    Secular couplings retain ``I_a,n I_b,n`` along
    ``quantization_axis_lab``. The complete cluster builder supplies the
    static-field direction, preserving rotational covariance for tilted
    fields.
    """

    needs_axis = any(item.model == "secular" for item in cluster.scalar_couplings)
    axis = (
        unit_vector(quantization_axis_lab, name="quantization_axis_lab")
        if needs_axis
        else np.array([0.0, 0.0, 1.0])
    )
    out = np.zeros((cluster.dimension, cluster.dimension), dtype=np.complex128)
    for coupling in cluster.scalar_couplings:
        factor_a = coupling.nucleus_a + 1
        factor_b = coupling.nucleus_b + 1
        if coupling.model == "isotropic":
            operator = dot_product_operator(cluster.spins, factor_a, factor_b)
        else:
            operator = _bilinear_spin_operator(
                cluster,
                factor_a,
                axis,
                factor_b,
                axis,
            )
        out += TAU * coupling.j_hz * operator
    return out


def nuclear_dipolar_hamiltonian(
    cluster: ResolvedSpinCluster,
    *,
    model: NuclearDipolarModel = "full",
    quantization_axis_lab=(0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return pairwise nuclear point-dipole couplings.

    ``model='full'`` retains the laboratory-frame tensor. ``'secular'``
    performs the high-field projection along ``quantization_axis_lab``.
    Longitudinal ``I_a,n I_b,n`` terms are retained for every pair; the
    energy-conserving transverse flip-flop term is retained for homonuclear
    pairs.
    """

    if model not in {"full", "secular"}:
        raise ValueError("nuclear dipolar model must be 'full' or 'secular'")
    if model == "secular":
        axis = unit_vector(
            quantization_axis_lab,
            name="quantization_axis_lab",
        )
        transverse_frame = rotation_from_z(axis)
        transverse_x = transverse_frame[:, 0]
        transverse_y = transverse_frame[:, 1]
    out = np.zeros((cluster.dimension, cluster.dimension), dtype=np.complex128)
    for first, nucleus_a in enumerate(cluster.nuclei):
        for second in range(first + 1, len(cluster.nuclei)):
            nucleus_b = cluster.nuclei[second]
            displacement_nm = nucleus_b.position_lab_nm - nucleus_a.position_lab_nm
            spatial = dipole_spatial_tensor_inverse_m3(displacement_nm)
            coefficient_tensor_hz = (
                MU0_OVER_4PI
                * PLANCK_CONSTANT_J_S
                * nucleus_a.gamma_hz_per_t
                * nucleus_b.gamma_hz_per_t
                * spatial
            )
            factor_a = first + 1
            factor_b = second + 1
            if model == "secular":
                longitudinal_coefficient = float(axis @ coefficient_tensor_hz @ axis)
                operator = longitudinal_coefficient * _bilinear_spin_operator(
                    cluster,
                    factor_a,
                    axis,
                    factor_b,
                    axis,
                )
                if nucleus_a.isotope == nucleus_b.isotope:
                    operator -= 0.5 * longitudinal_coefficient * (
                        _bilinear_spin_operator(
                            cluster,
                            factor_a,
                            transverse_x,
                            factor_b,
                            transverse_x,
                        )
                        + _bilinear_spin_operator(
                            cluster,
                            factor_a,
                            transverse_y,
                            factor_b,
                            transverse_y,
                        )
                    )
                out += TAU * operator
                continue
            for axis_a, row in zip("xyz", coefficient_tensor_hz):
                for axis_b, coefficient in zip("xyz", row):
                    out += (
                        TAU
                        * coefficient
                        * product_operator(
                            cluster.spins,
                            [(factor_a, axis_a), (factor_b, axis_b)],
                        )
                    )
    return out


def resolved_cluster_hamiltonian(
    cluster: ResolvedSpinCluster,
    b0_vector_tesla_lab=(0.0, 0.0, 0.0),
    *,
    include_sensor_nuclear: bool = True,
    include_scalar: bool = True,
    include_nuclear_dipolar: bool = True,
    nuclear_dipolar_model: NuclearDipolarModel = "full",
) -> np.ndarray:
    """Return the complete dense cluster Hamiltonian in radians per second."""

    sensor_part = np.kron(
        sensor_hamiltonian(cluster.sensor, b0_vector_tesla_lab),
        np.eye(cluster.nuclear_dimension),
    )
    out = sensor_part + nuclear_zeeman_hamiltonian(cluster, b0_vector_tesla_lab)
    if include_sensor_nuclear:
        out += sensor_nuclear_hyperfine_hamiltonian(cluster)
    if include_scalar:
        out += nuclear_scalar_coupling_hamiltonian(
            cluster,
            quantization_axis_lab=b0_vector_tesla_lab,
        )
    if include_nuclear_dipolar:
        out += nuclear_dipolar_hamiltonian(
            cluster,
            model=nuclear_dipolar_model,
            quantization_axis_lab=b0_vector_tesla_lab,
        )
    return out


def nuclear_rf_hamiltonian(
    cluster: ResolvedSpinCluster,
    nutation_hz: float,
    *,
    phase_rad: float = 0.0,
    nuclear_indices: tuple[int, ...] | list[int] | None = None,
) -> np.ndarray:
    """Return a resonant rotating-frame nuclear-RF control Hamiltonian.

    ``nutation_hz`` is the angular-momentum rotation rate in cycles per second;
    a pi pulse therefore lasts ``1 / (2 * nutation_hz)``.
    """

    nutation = float(nutation_hz)
    phase = float(phase_rad)
    if nutation < 0.0 or not np.isfinite(nutation):
        raise ValueError("nutation_hz must be finite and non-negative")
    if not np.isfinite(phase):
        raise ValueError("phase_rad must be finite")
    selected = (
        tuple(range(len(cluster.nuclei)))
        if nuclear_indices is None
        else tuple(int(index) for index in nuclear_indices)
    )
    if any(index < 0 or index >= len(cluster.nuclei) for index in selected):
        raise ValueError("nuclear_indices must select existing nuclei")
    operator = np.zeros((cluster.dimension, cluster.dimension), dtype=np.complex128)
    for index in selected:
        operator += np.cos(phase) * embedded_operator(
            cluster.spins, index + 1, "x"
        )
        operator += np.sin(phase) * embedded_operator(
            cluster.spins, index + 1, "y"
        )
    return TAU * nutation * operator


@dataclass(frozen=True)
class ResolvedClusterTransition:
    """One microwave-active eigenstate transition of a resolved cluster."""

    lower: int
    upper: int
    frequency_hz: float
    strength: float


@dataclass(frozen=True)
class ResolvedClusterEigensystem:
    """Dense cluster levels and microwave-active transitions."""

    cluster: ResolvedSpinCluster
    b0_vector_tesla_lab: np.ndarray
    levels_hz: np.ndarray
    eigenvectors: np.ndarray
    transitions: tuple[ResolvedClusterTransition, ...]


def diagonalize_resolved_cluster(
    cluster: ResolvedSpinCluster,
    b0_vector_tesla_lab=(0.0, 0.0, 0.0),
    *,
    drive_direction_lab=(1.0, 0.0, 0.0),
    strength_tolerance: float = 1.0e-14,
) -> ResolvedClusterEigensystem:
    """Diagonalize a resolved cluster and inventory microwave transitions."""

    b0 = _point(b0_vector_tesla_lab, name="b0_vector_tesla_lab")
    hamiltonian = resolved_cluster_hamiltonian(cluster, b0)
    values, vectors = np.linalg.eigh(hamiltonian)
    levels_hz = values / TAU
    electron_drive = np.kron(
        drive_operator(cluster.sensor, drive_direction_lab),
        np.eye(cluster.nuclear_dimension),
    )
    drive_eigen = vectors.conj().T @ electron_drive @ vectors
    transitions = []
    for lower in range(cluster.dimension):
        for upper in range(lower + 1, cluster.dimension):
            frequency = float(levels_hz[upper] - levels_hz[lower])
            strength = float(abs(drive_eigen[lower, upper]) ** 2)
            if frequency > 0.0 and strength > strength_tolerance:
                transitions.append(
                    ResolvedClusterTransition(lower, upper, frequency, strength)
                )
    return ResolvedClusterEigensystem(
        cluster=cluster,
        b0_vector_tesla_lab=b0,
        levels_hz=levels_hz,
        eigenvectors=vectors,
        transitions=tuple(transitions),
    )


@dataclass(frozen=True)
class ResolvedClusterSpectrumResult:
    """Broadened fixed-field CW spectrum of a resolved cluster."""

    frequencies_hz: np.ndarray
    spectrum: np.ndarray
    eigensystem: ResolvedClusterEigensystem
    broadening_hz: float
    lineshape: str


def simulate_resolved_cw_spectrum(
    cluster: ResolvedSpinCluster,
    b0_vector_tesla_lab,
    *,
    frequencies_hz=None,
    broadening_hz: float = 50.0e3,
    points: int = 2048,
    span_hz: float | None = None,
    drive_direction_lab=(1.0, 0.0, 0.0),
    lineshape: str = "gaussian",
) -> ResolvedClusterSpectrumResult:
    """Return a broadened microwave transition-strength spectrum."""

    broadening = float(broadening_hz)
    if broadening <= 0.0 or not np.isfinite(broadening):
        raise ValueError("broadening_hz must be positive and finite")
    eigensystem = diagonalize_resolved_cluster(
        cluster,
        b0_vector_tesla_lab,
        drive_direction_lab=drive_direction_lab,
    )
    centers = np.array(
        [transition.frequency_hz for transition in eigensystem.transitions]
    )
    intensities = np.array(
        [transition.strength for transition in eigensystem.transitions]
    )
    if frequencies_hz is None:
        points = int(points)
        if points < 2:
            raise ValueError("points must be at least two")
        if centers.size == 0:
            raise ValueError("cluster has no microwave-active transitions")
        span = (
            max(float(np.ptp(centers)) + 10.0 * broadening, 10.0 * broadening)
            if span_hz is None
            else float(span_hz)
        )
        if span <= 0.0 or not np.isfinite(span):
            raise ValueError("span_hz must be positive and finite")
        center = 0.5 * (float(centers.min()) + float(centers.max()))
        axis = np.linspace(center - 0.5 * span, center + 0.5 * span, points)
    else:
        axis = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
        if axis.size < 2 or not np.all(np.isfinite(axis)):
            raise ValueError("frequencies_hz must contain at least two finite points")
    spectrum = spectrum_from_lines(
        axis,
        centers,
        intensities,
        width=broadening,
        lineshape=lineshape,
    )
    return ResolvedClusterSpectrumResult(
        frequencies_hz=axis,
        spectrum=spectrum,
        eigensystem=eigensystem,
        broadening_hz=broadening,
        lineshape=lineshape,
    )


__all__ = [
    "DEFAULT_MAX_HILBERT_DIMENSION",
    "NuclearDipolarModel",
    "NuclearScalarCoupling",
    "ResolvedClusterEigensystem",
    "ResolvedClusterSpectrumResult",
    "ResolvedClusterTransition",
    "ResolvedNucleus",
    "ResolvedSpinCluster",
    "ScalarCouplingModel",
    "diagonalize_resolved_cluster",
    "nuclear_dipolar_hamiltonian",
    "nuclear_rf_hamiltonian",
    "nuclear_scalar_coupling_hamiltonian",
    "nuclear_zeeman_hamiltonian",
    "resolved_cluster_hamiltonian",
    "sensor_nuclear_hyperfine_hamiltonian",
    "simulate_resolved_cw_spectrum",
]
