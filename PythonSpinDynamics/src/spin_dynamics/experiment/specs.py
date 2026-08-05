"""Declarative experiment specs for the unified workflow facade.

These frozen dataclasses describe *what* to simulate; they never implement
dynamics. ``Experiment.plan()`` resolves a spec against the workflow registry
and reports compatibility issues; ``Experiment.run()`` delegates to the
validated ``spin_dynamics.workflows`` functions. Optional fields left at
their defaults defer to the wrapped workflow's own defaults, so a default
``Experiment`` reproduces the direct workflow call bit for bit.

Fields typed as ``Mapping | None`` accept the same plain-mapping forms the
underlying workflows accept (e.g. ``noise={"sigma": 0.01}``); rich spec
objects such as :class:`~spin_dynamics.noise.NoiseSpec` also work, and
serialize as long as their contents are JSON-representable.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from spin_dynamics.experiment.serialization import decode, encode, register_serializable
from spin_dynamics.sequences import SequenceIR

PROBE_NAMES = ("ideal", "tuned", "untuned", "matched")


def _as_optional_map(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return arr


def _finite_tuple3(value: Iterable[float], name: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain three finite values")
    return values


@register_serializable
@dataclass(frozen=True, eq=False)
class Phantom:
    """Spatial sample description for imaging: density plus optional maps.

    ``rho`` is the 2-D proton-density map; ``t1_map``/``t2_map`` override the
    scalar ``Sample`` relaxation times per voxel when given.
    """

    rho: np.ndarray
    t1_map: np.ndarray | None = None
    t2_map: np.ndarray | None = None

    def __post_init__(self) -> None:
        rho = _as_optional_map(self.rho, "rho")
        if rho is None:
            raise ValueError("rho is required")
        object.__setattr__(self, "rho", rho)
        for name in ("t1_map", "t2_map"):
            arr = _as_optional_map(getattr(self, name), name)
            if arr is not None and arr.shape != rho.shape:
                raise ValueError(f"{name} must have the same shape as rho")
            object.__setattr__(self, name, arr)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phantom):
            return NotImplemented

        def same(a: np.ndarray | None, b: np.ndarray | None) -> bool:
            if a is None or b is None:
                return a is None and b is None
            return bool(np.array_equal(a, b))

        return (
            same(self.rho, other.rho)
            and same(self.t1_map, other.t1_map)
            and same(self.t2_map, other.t2_map)
        )


@register_serializable
@dataclass(frozen=True, eq=False)
class TransportDomain2D:
    """Density and physical axes for 2-D random-walker transport.

    ``rho`` has shape ``(len(x_axis), len(z_axis))``. Axes are strictly
    increasing and use meters; density is non-negative and need not be
    normalized.
    """

    rho: np.ndarray
    x_axis: np.ndarray
    z_axis: np.ndarray

    def __post_init__(self) -> None:
        rho = np.asarray(self.rho, dtype=np.float64)
        x_axis = np.asarray(self.x_axis, dtype=np.float64).reshape(-1)
        z_axis = np.asarray(self.z_axis, dtype=np.float64).reshape(-1)
        if rho.ndim != 2 or rho.shape != (x_axis.size, z_axis.size):
            raise ValueError("rho shape must equal (len(x_axis), len(z_axis))")
        if x_axis.size < 2 or z_axis.size < 2:
            raise ValueError("transport axes must each contain at least two points")
        if not np.all(np.isfinite(rho)) or np.any(rho < 0.0) or not np.any(rho > 0.0):
            raise ValueError("rho must be finite, non-negative, and contain mass")
        for values, name in ((x_axis, "x_axis"), (z_axis, "z_axis")):
            if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
                raise ValueError(f"{name} must be finite and strictly increasing")
        object.__setattr__(self, "rho", rho.copy())
        object.__setattr__(self, "x_axis", x_axis.copy())
        object.__setattr__(self, "z_axis", z_axis.copy())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransportDomain2D):
            return NotImplemented
        return bool(
            np.array_equal(self.rho, other.rho)
            and np.array_equal(self.x_axis, other.x_axis)
            and np.array_equal(self.z_axis, other.z_axis)
        )


@register_serializable
@dataclass(frozen=True)
class UniformFlow2D:
    """Uniform ``(vx, vz)`` transport velocity in meters per second."""

    velocity_m_per_s: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        velocity = tuple(float(value) for value in self.velocity_m_per_s)
        if len(velocity) != 2 or not np.all(np.isfinite(velocity)):
            raise ValueError("velocity_m_per_s must contain two finite values")
        object.__setattr__(self, "velocity_m_per_s", velocity)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.velocity_m_per_s, dtype=np.float64)


@register_serializable
@dataclass(frozen=True, eq=False)
class DEERDistribution:
    """Distance grid and non-negative weights for a DEER experiment."""

    distances_nm: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        distances = np.asarray(self.distances_nm, dtype=np.float64).reshape(-1)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if distances.size < 2 or distances.size != weights.size:
            raise ValueError("DEER distances and weights must have the same size >= 2")
        if not np.all(np.isfinite(distances)) or np.any(distances <= 0.0):
            raise ValueError("DEER distances must be finite and positive")
        if np.any(np.diff(distances) <= 0.0):
            raise ValueError("DEER distances must be strictly increasing")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("DEER weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("DEER weights must contain positive mass")
        object.__setattr__(self, "distances_nm", distances.copy())
        object.__setattr__(self, "weights", weights.copy())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DEERDistribution):
            return NotImplemented
        return bool(
            np.array_equal(self.distances_nm, other.distances_nm)
            and np.array_equal(self.weights, other.weights)
        )


@register_serializable
@dataclass(frozen=True, eq=False)
class SequenceDomain:
    """Spatial sample and field maps for general SequenceIR execution.

    ``axes`` contains one to three physical coordinate axes in meters and
    ``density`` has their Cartesian-product shape. ``b0_map_rad_s`` is angular
    off-resonance; transmit/receive B1 maps are relative sensitivities.
    ``gradient_channels`` maps domain axes to physical Pulseq x/y/z channels.
    """

    axes: tuple[np.ndarray, ...]
    density: np.ndarray
    b0_map_rad_s: np.ndarray | None = None
    b1_tx_map: np.ndarray | None = None
    b1_rx_map: np.ndarray | None = None
    velocity_m_per_s: tuple[float, ...] | None = None
    gradient_channels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        axes = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in self.axes)
        if not 1 <= len(axes) <= 3:
            raise ValueError("SequenceDomain supports one to three axes")
        for index, axis in enumerate(axes):
            if axis.size < 2 or not np.all(np.isfinite(axis)):
                raise ValueError(f"axis {index} must contain at least two finite values")
            if np.any(np.diff(axis) <= 0.0):
                raise ValueError(f"axis {index} must be strictly increasing")
        shape = tuple(axis.size for axis in axes)
        density = np.asarray(self.density, dtype=np.float64)
        if density.shape != shape or not np.all(np.isfinite(density)):
            raise ValueError("density must be finite and match the domain axes")
        if np.any(density < 0.0) or not np.any(density > 0.0):
            raise ValueError("density must be non-negative and contain mass")

        maps: dict[str, np.ndarray | None] = {}
        for name in ("b0_map_rad_s", "b1_tx_map", "b1_rx_map"):
            value = getattr(self, name)
            array = None if value is None else np.asarray(value, dtype=np.float64)
            if array is not None and (
                array.shape != shape or not np.all(np.isfinite(array))
            ):
                raise ValueError(f"{name} must be finite and match the domain axes")
            if name.startswith("b1") and array is not None and np.any(array < 0.0):
                raise ValueError(f"{name} must be non-negative")
            maps[name] = None if array is None else array.copy()

        velocity = self.velocity_m_per_s
        if velocity is not None:
            velocity = tuple(float(value) for value in velocity)
            if len(velocity) != len(axes) or not np.all(np.isfinite(velocity)):
                raise ValueError("velocity_m_per_s must match the spatial dimension")

        channels = self.gradient_channels
        if channels is None:
            channels = {1: ("x",), 2: ("x", "z"), 3: ("x", "y", "z")}[len(axes)]
        channels = tuple(str(channel).lower() for channel in channels)
        if len(channels) != len(axes) or any(
            channel not in ("x", "y", "z") for channel in channels
        ):
            raise ValueError("gradient_channels must map every axis to x, y, or z")
        if len(set(channels)) != len(channels):
            raise ValueError("gradient_channels must not contain duplicates")

        object.__setattr__(self, "axes", tuple(axis.copy() for axis in axes))
        object.__setattr__(self, "density", density.copy())
        for name, value in maps.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "velocity_m_per_s", velocity)
        object.__setattr__(self, "gradient_channels", channels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceDomain):
            return NotImplemented

        def same(left: np.ndarray | None, right: np.ndarray | None) -> bool:
            if left is None or right is None:
                return left is None and right is None
            return bool(np.array_equal(left, right))

        return bool(
            len(self.axes) == len(other.axes)
            and all(np.array_equal(a, b) for a, b in zip(self.axes, other.axes))
            and np.array_equal(self.density, other.density)
            and same(self.b0_map_rad_s, other.b0_map_rad_s)
            and same(self.b1_tx_map, other.b1_tx_map)
            and same(self.b1_rx_map, other.b1_rx_map)
            and self.velocity_m_per_s == other.velocity_m_per_s
            and self.gradient_channels == other.gradient_channels
        )


@register_serializable
@dataclass(frozen=True)
class NanoMRSensor:
    """Compact diamond-NV or SiC-PL6 sensor preset for facade workflows."""

    preset: str = "diamond_nv_minus"
    depth_nm: float = 5.0
    axis_lab: tuple[float, float, float] = (0.0, 0.0, 1.0)
    label: str = "sensor"

    def __post_init__(self) -> None:
        if self.preset not in ("diamond_nv_minus", "sic_pl6"):
            raise ValueError("preset must be 'diamond_nv_minus' or 'sic_pl6'")
        depth = float(self.depth_nm)
        if depth <= 0.0 or not np.isfinite(depth):
            raise ValueError("depth_nm must be positive and finite")
        axis = _finite_tuple3(self.axis_lab, "axis_lab")
        if np.linalg.norm(axis) <= 0.0:
            raise ValueError("axis_lab must be non-zero")
        object.__setattr__(self, "depth_nm", depth)
        object.__setattr__(self, "axis_lab", axis)
        object.__setattr__(self, "label", str(self.label))


@register_serializable
@dataclass(frozen=True)
class NanoMRBathComponent:
    """One isotope in a uniform planar nano-MR sample layer."""

    isotope: str = "1H"
    number_density_m3: float = 6.7e28
    correlation_time_seconds: float = 100.0e-6
    polarization_mode: str = "statistical"
    temperature_kelvin: float = 300.0
    polarization_fraction: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.isotope not in ("1H", "13C", "19F", "29Si", "31P"):
            raise ValueError("isotope must be a built-in nano-MR isotope preset")
        density = float(self.number_density_m3)
        correlation = float(self.correlation_time_seconds)
        temperature = float(self.temperature_kelvin)
        polarization = float(self.polarization_fraction)
        if density < 0.0 or not np.isfinite(density):
            raise ValueError("number_density_m3 must be finite and non-negative")
        if correlation <= 0.0 or not np.isfinite(correlation):
            raise ValueError("correlation_time_seconds must be positive and finite")
        if self.polarization_mode not in ("statistical", "thermal", "fixed"):
            raise ValueError(
                "polarization_mode must be 'statistical', 'thermal', or 'fixed'"
            )
        if temperature <= 0.0 or not np.isfinite(temperature):
            raise ValueError("temperature_kelvin must be positive and finite")
        if not -1.0 <= polarization <= 1.0 or not np.isfinite(polarization):
            raise ValueError("polarization_fraction must lie in [-1, 1]")
        object.__setattr__(self, "number_density_m3", density)
        object.__setattr__(self, "correlation_time_seconds", correlation)
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "polarization_fraction", polarization)
        object.__setattr__(self, "label", str(self.label))


@register_serializable
@dataclass(frozen=True)
class NanoMRLayer:
    """Uniform planar nuclear layer or half-space above a defect sensor."""

    components: tuple[NanoMRBathComponent, ...] = field(
        default_factory=lambda: (NanoMRBathComponent(),)
    )
    surface_normal_lab: tuple[float, float, float] = (0.0, 0.0, 1.0)
    thickness_nm: float | None = None
    bottom_offset_nm: float = 0.0
    label: str = "uniform nuclear layer"

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components or not all(
            isinstance(item, NanoMRBathComponent) for item in components
        ):
            raise TypeError("components must contain NanoMRBathComponent values")
        normal = _finite_tuple3(self.surface_normal_lab, "surface_normal_lab")
        if np.linalg.norm(normal) <= 0.0:
            raise ValueError("surface_normal_lab must be non-zero")
        offset = float(self.bottom_offset_nm)
        if offset < 0.0 or not np.isfinite(offset):
            raise ValueError("bottom_offset_nm must be finite and non-negative")
        thickness = self.thickness_nm
        if thickness is not None:
            thickness = float(thickness)
            if thickness <= 0.0 or not np.isfinite(thickness):
                raise ValueError("thickness_nm must be positive and finite when set")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "surface_normal_lab", normal)
        object.__setattr__(self, "bottom_offset_nm", offset)
        object.__setattr__(self, "thickness_nm", thickness)
        object.__setattr__(self, "label", str(self.label))


@register_serializable
@dataclass(frozen=True)
class NanoMROpticalReadout:
    """Effective optical initialization and photon-count detector settings."""

    initialization_fidelity: float = 1.0
    initialization_seconds: float = 0.0
    bright_count_rate_hz: float = 100.0e3
    readout_contrast: float = 0.2
    readout_seconds: float = 300.0e-9
    background_count_rate_hz: float = 0.0
    dead_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("initialization_fidelity", "readout_contrast"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0 or not np.isfinite(value):
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "initialization_seconds",
            "bright_count_rate_hz",
            "readout_seconds",
            "background_count_rate_hz",
            "dead_time_seconds",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.readout_seconds <= 0.0:
            raise ValueError("readout_seconds must be positive")


@register_serializable
@dataclass(frozen=True, eq=False)
class SampledB0:
    """A spatially-varying static field sampled on the imaging plane.

    ``b0_tesla`` is the static-field vector on the phantom grid, shape
    ``(n0, n1, 3)`` in tesla -- e.g. sampled from a 3-D magnetostatic solve via
    :func:`spin_dynamics.experiment.wiring.sampled_b0_from_solution`.
    Used as ``Hardware.b0`` it drives imaging with a real (inhomogeneous) magnet
    field instead of the idealized uniform one: the per-voxel B0 direction sets
    the transverse-B1 projection, and :meth:`off_resonance` supplies the imaging
    ``b0_map``.

    **Units.** The physical angular off-resonance is ``gamma |B0| - 2 pi
    carrier_hz`` (rad/s). The CPMG imaging kernel normalizes off-resonance by the
    RF **nutation frequency** ``omega_1 = gamma B1`` -- ``del_w_normalized =
    del_w / omega_1`` -- so set ``nutation_rad_s`` to the imaging pulses' ``omega_1``
    to get the map the kernel expects. Leave it at ``1.0`` to keep the physical
    rad/s value (the motion / multislice engine convention).
    """

    b0_tesla: np.ndarray
    carrier_hz: float
    nutation_rad_s: float = 1.0

    def __post_init__(self) -> None:
        arr = np.asarray(self.b0_tesla, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError("b0_tesla must have shape (n0, n1, 3)")
        object.__setattr__(self, "b0_tesla", arr)
        if not (float(self.carrier_hz) > 0.0):
            raise ValueError("carrier_hz must be positive")
        if not (float(self.nutation_rad_s) > 0.0):
            raise ValueError("nutation_rad_s must be positive")

    def magnitude_tesla(self) -> np.ndarray:
        """Return ``|B0|`` (T) on the grid."""

        return np.linalg.norm(self.b0_tesla, axis=-1)

    def off_resonance(self, gamma: float) -> np.ndarray:
        """Return ``(gamma |B0| - 2 pi carrier) / nutation_rad_s``.

        With ``nutation_rad_s = 1`` this is the physical angular off-resonance
        (rad/s); set it to the RF nutation frequency ``omega_1`` to get the
        CPMG imaging kernel's normalized offset (``del_w / omega_1``).
        """

        physical = gamma * self.magnitude_tesla() - 2.0 * np.pi * float(self.carrier_hz)
        return physical / float(self.nutation_rad_s)

    def direction(self) -> np.ndarray:
        """Return the per-voxel unit B0 direction (for transverse-B1 projection)."""

        mag = self.magnitude_tesla()
        return self.b0_tesla / np.where(mag > 0.0, mag, 1.0)[..., np.newaxis]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SampledB0):
            return NotImplemented
        return (
            float(self.carrier_hz) == float(other.carrier_hz)
            and float(self.nutation_rad_s) == float(other.nutation_rad_s)
            and bool(np.array_equal(self.b0_tesla, other.b0_tesla))
        )


@register_serializable
@dataclass(frozen=True)
class Sample:
    """Sample description.

    ``t1_seconds``/``t2_seconds`` and ``diffusion_coefficient`` left as
    ``None`` defer to the wrapped workflow's defaults; asymptotic
    (steady-state) workflows do not consume relaxation or diffusion at all,
    and ``plan()`` warns when they would be ignored.
    Imaging sequences additionally require a :class:`Phantom`; scalar
    relaxation times then broadcast to uniform maps unless the phantom
    carries its own.
    """

    t1_seconds: float | None = None
    t2_seconds: float | None = None
    diffusion_coefficient: float | None = None
    """Isotropic diffusion coefficient in m^2/s for diffusion workflows."""
    transport_domain: TransportDomain2D | None = None
    """Explicit density and axes for random-walker transport workflows."""
    flow: UniformFlow2D | None = None
    """Optional uniform advective velocity for transport workflows."""
    phantom: Phantom | None = None
    site: Any | None = None
    """Quadrupolar site (``spin_dynamics.nqr.QuadrupolarSite``) for NQR sequences."""
    esr_system: Any | None = None
    """Electron spin system (``spin_dynamics.esr.ESRSpinSystem``) for ESR sequences."""
    deer_distribution: DEERDistribution | None = None
    """Distance distribution used by :class:`ESRDEER`."""
    hyperfine_coupling: Any | None = None
    """Electron-nuclear coupling used by ESEEM, HYSCORE, and ENDOR specs."""
    sequence_domain: SequenceDomain | None = None
    """Explicit spatial sample/field domain for :class:`SequenceIRExecution`."""
    nano_mr_layer: NanoMRLayer | None = None
    """Uniform planar nuclear layer for statistical nano-MR workflows."""
    label: str = ""


@register_serializable
@dataclass(frozen=True)
class Hardware:
    """Transmit/receive hardware description.

    ``probe`` plus the circuit perturbations apply to all workflows. The
    geometry fields (``b0``, ``tx_coil``, ``rx_coil``, ``plane`` — see
    :mod:`spin_dynamics.experiment.hardware`) drive an automatic Biot-Savart
    field solve for imaging sequences: the transverse-B1 maps are computed on
    the phantom grid and passed to the workflow, replacing its synthetic
    default. Nano-MR workflows use the explicit defect sensor and optional
    optical photon-count model below rather than an inductive probe.
    """

    probe: str = "ideal"
    q_value: float | None = None
    mistuning_offset: float | None = None
    radiation_damping: Mapping[str, Any] | Any | None = None
    absolute_phase: Mapping[str, Any] | Any | None = None
    b0: Any | None = None
    tx_coil: Any | None = None
    rx_coil: Any | None = None
    plane: Any | None = None
    nano_mr_sensor: NanoMRSensor | None = None
    """Defect-spin sensor preset for nano-MR workflows."""
    nano_mr_optical_readout: NanoMROpticalReadout | None = None
    """Optional effective photon-count readout for Qdyne."""


@register_serializable
@dataclass(frozen=True, eq=False)
class Acquisition:
    """Offset grid, receiver noise, and rephasing-guard configuration."""

    numpts: int = 101
    maxoffs: float = 10.0
    noise: Mapping[str, Any] | float | Any | None = None
    receiver_noise_std: float = 0.0
    receiver_noise_covariance: np.ndarray | None = None
    receiver_noise_seed: int | None = None
    sense_acceleration: int | None = None
    sense_axis: str = "x"
    sense_offset: int = 0
    sense_regularization: float = 0.0
    auto_refine_grid: bool = False
    rephase_safety_factor: float = 1.25
    rephase_action: str = "warn"

    def __post_init__(self) -> None:
        std = float(self.receiver_noise_std)
        if not np.isfinite(std) or std < 0.0:
            raise ValueError("receiver_noise_std must be finite and non-negative")
        object.__setattr__(self, "receiver_noise_std", std)
        if self.receiver_noise_covariance is not None:
            if std > 0.0:
                raise ValueError(
                    "provide either receiver_noise_std or "
                    "receiver_noise_covariance, not both"
                )
            covariance = np.asarray(
                self.receiver_noise_covariance, dtype=np.complex128
            )
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise ValueError("receiver_noise_covariance must be a square matrix")
            object.__setattr__(self, "receiver_noise_covariance", covariance)
        if self.receiver_noise_seed is not None and (
            not isinstance(self.receiver_noise_seed, int)
            or self.receiver_noise_seed < 0
        ):
            raise ValueError(
                "receiver_noise_seed must be a non-negative integer when set"
            )
        acceleration = self.sense_acceleration
        if acceleration is not None and (
            not isinstance(acceleration, int) or acceleration < 1
        ):
            raise ValueError("sense_acceleration must be a positive integer when set")
        if self.sense_axis not in {"x", "z"}:
            raise ValueError("sense_axis must be 'x' or 'z'")
        if not isinstance(self.sense_offset, int) or self.sense_offset < 0:
            raise ValueError("sense_offset must be a non-negative integer")
        regularization = float(self.sense_regularization)
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("sense_regularization must be finite and non-negative")
        object.__setattr__(self, "sense_regularization", regularization)
        if acceleration is None:
            if (
                self.sense_axis != "x"
                or self.sense_offset != 0
                or regularization != 0.0
            ):
                raise ValueError(
                    "SENSE settings require sense_acceleration to be specified"
                )
        elif self.sense_offset >= acceleration:
            raise ValueError("sense_offset must be smaller than sense_acceleration")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Acquisition):
            return NotImplemented
        return bool(encode(self) == encode(other))


@register_serializable
@dataclass(frozen=True)
class CPMG:
    """Asymptotic (infinite-train) CPMG echo, no relaxation."""


@register_serializable
@dataclass(frozen=True)
class CPMGTrain:
    """Finite CPMG echo train with relaxation."""

    num_echoes: int = 8


@register_serializable
@dataclass(frozen=True)
class CPMGIRTrain:
    """Finite CPMG echo train preceded by an inversion-recovery delay sweep."""

    num_echoes: int = 10
    echo_spacing_seconds: float = 0.5e-3
    tauvect: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.tauvect is not None and not isinstance(self.tauvect, tuple):
            values: Iterable[float] = np.asarray(self.tauvect, dtype=np.float64).reshape(-1)
            object.__setattr__(self, "tauvect", tuple(float(v) for v in values))


@register_serializable
@dataclass(frozen=True)
class CPMGImaging:
    """Phase-encoded CPMG imaging (2-D spin-warp on a phantom).

    ``fov``, ``ny``, and ``maxoffs`` are the imaging workflow's own
    acquisition-geometry parameters (units per the workflow); the physical
    placement of the phantom for coil-field solving is set separately via
    ``Hardware.plane`` in meters.
    """

    num_echoes: int = 2
    echo_spacing_seconds: float = 0.2e-3
    gradient_duration_seconds: float = 0.5e-3
    fov: tuple[float, float] = (20.0, 20.0)
    ny: int = 9
    maxoffs: float = 5.0

    def __post_init__(self) -> None:
        fov = tuple(float(v) for v in self.fov)
        if len(fov) != 2:
            raise ValueError("fov must contain two values")
        object.__setattr__(self, "fov", fov)


@register_serializable
@dataclass(frozen=True)
class PGSE:
    """Deterministic pulsed-gradient spin echo diffusion encoding.

    ``gradient_amplitude`` is in T/m; all timing fields are seconds and
    ``gamma`` is in rad/s/T. The sample's ``diffusion_coefficient`` and
    ``t2_seconds`` control diffusion and transverse-relaxation attenuation.
    """

    num_echoes: int = 1
    gradient_amplitude: float = 0.05
    gradient_duration: float = 2.0e-3
    diffusion_time: float = 20.0e-3
    first_echo_time_seconds: float | None = None
    echo_spacing_seconds: float | None = None
    gamma: float = 2.675e8


@register_serializable
@dataclass(frozen=True)
class PGSEWalkers:
    """Explicit random-walker PGSE with diffusion and optional uniform flow."""

    num_echoes: int = 1
    gradient_amplitude: float = 0.05
    gradient_duration: float = 2.0e-3
    diffusion_time: float = 20.0e-3
    gamma: float = 2.675e8
    gradient_axis: str = "x"
    walkers_per_cell: int = 128
    seed: int | None = None
    jitter: bool = False
    excitation_duration: float = 100.0e-6
    refocusing_duration: float = 200.0e-6
    echo_spacing_seconds: float | None = None
    boundary: str = "reflect"
    substeps_per_interval: int = 8


def _validate_nqr_common(spec: Any) -> None:
    if spec.pulse_duration_seconds <= 0:
        raise ValueError("pulse_duration_seconds must be positive")
    if spec.nutation_hz <= 0:
        raise ValueError("nutation_hz must be positive")
    if spec.orientations not in ("powder", "single"):
        raise ValueError("orientations must be 'powder' or 'single'")
    if spec.t2e_seconds is not None and spec.t2e_seconds <= 0:
        raise ValueError("t2e_seconds must be positive when set")


@register_serializable
@dataclass(frozen=True)
class NQRSLSE:
    """Spin-lock spin-echo NQR detection train.

    ``nutation_hz`` uses the reduced engine's convention: the *effective*
    two-level Rabi rate of the addressed transition at full RF coupling, so
    the on-resonance flip angle is ``2*pi*nutation_hz*pulse_duration_seconds``
    (90 degrees at ``nutation_hz * duration = 0.25``). The conversion to the
    bare ``gamma*B1/(2*pi)`` the full-model engine expects happens in the
    adapter. ``model="auto"`` picks the reduced two-level or full
    density-matrix engine via ``select_nqr_model``; ``transition="auto"``
    addresses the strongest line.
    """

    pulse_duration_seconds: float
    nutation_hz: float
    echo_spacing_seconds: float
    num_echoes: int
    transition: str = "auto"
    model: str = "auto"
    phase: float = 0.0
    rf_frequency_hz: float | None = None
    orientations: str = "powder"
    b0_tesla: float = 0.0
    t2e_seconds: float | None = None

    def __post_init__(self) -> None:
        _validate_nqr_common(self)
        if self.model not in ("auto", "reduced", "full"):
            raise ValueError("model must be 'auto', 'reduced', or 'full'")
        if self.num_echoes <= 0:
            raise ValueError("num_echoes must be positive")
        if self.echo_spacing_seconds < 0:
            raise ValueError("echo_spacing_seconds must be non-negative")


@register_serializable
@dataclass(frozen=True)
class NQRSORC:
    """Strong off-resonance comb NQR train (reduced spin-1 engine only).

    Same ``nutation_hz`` convention as :class:`NQRSLSE`.
    """

    pulse_duration_seconds: float
    nutation_hz: float
    half_spacing_seconds: float
    num_pulses: int
    transition: str = "auto"
    phase: float = 0.0
    rf_frequency_hz: float | None = None
    orientations: str = "powder"
    b0_tesla: float = 0.0
    t2e_seconds: float | None = None

    def __post_init__(self) -> None:
        _validate_nqr_common(self)
        if self.num_pulses <= 0:
            raise ValueError("num_pulses must be positive")
        if self.half_spacing_seconds < 0:
            raise ValueError("half_spacing_seconds must be non-negative")


@register_serializable
@dataclass(frozen=True)
class NQRFID:
    """Single-pulse NQR FID using the full density-matrix engine.

    Unlike the selective reduced-model specs, ``nutation_hz`` is the bare
    ``gamma * B1 / (2*pi)`` rate used by the full-model pulse Hamiltonian.
    """

    nutation_hz: float
    pulse_duration_seconds: float
    acquisition_seconds: float
    num_points: int = 512
    rf_frequency_hz: float | None = None
    phase: float = 0.0
    b0_tesla: float = 0.0

    def __post_init__(self) -> None:
        if self.nutation_hz <= 0 or self.pulse_duration_seconds <= 0:
            raise ValueError("nutation_hz and pulse_duration_seconds must be positive")
        if self.acquisition_seconds <= 0 or self.num_points <= 1:
            raise ValueError("acquisition_seconds must be positive, num_points > 1")


@register_serializable
@dataclass(frozen=True)
class NQRPopulationTransfer:
    """Selective perturbation followed by reduced spin-1 SLSE detection."""

    perturbation_duration_seconds: float
    perturbation_nutation_hz: float
    detection_duration_seconds: float
    detection_nutation_hz: float
    echo_spacing_seconds: float
    num_echoes: int
    perturbation_transition: str = "auto"
    detection_transition: str = "auto"
    perturbation_phase: float = 0.0
    detection_phase: float = 0.0
    perturbation_frequency_hz: float | None = None
    detection_frequency_hz: float | None = None
    orientations: str = "powder"
    b0_tesla: float = 0.0
    t2e_seconds: float | None = None

    def __post_init__(self) -> None:
        durations = (
            self.perturbation_duration_seconds,
            self.detection_duration_seconds,
        )
        rates = (self.perturbation_nutation_hz, self.detection_nutation_hz)
        if any(value <= 0 for value in durations + rates):
            raise ValueError("pulse durations and nutation rates must be positive")
        if self.echo_spacing_seconds < 0 or self.num_echoes <= 0:
            raise ValueError("echo spacing must be non-negative and num_echoes positive")
        if self.orientations not in ("powder", "single"):
            raise ValueError("orientations must be 'powder' or 'single'")
        if self.t2e_seconds is not None and self.t2e_seconds <= 0:
            raise ValueError("t2e_seconds must be positive when set")


@register_serializable
@dataclass(frozen=True)
class ESRFID:
    """Pulsed ESR free-induction decay (rotating frame, single isochromat).

    ``nutation_hz`` is the electron Rabi rate (engine convention); the static
    field comes from ``Hardware.b0`` (a :class:`UniformB0` with
    ``field_tesla`` set). The acquisition grid is
    ``num_points`` samples over ``acquisition_seconds``.
    """

    nutation_hz: float
    pulse_duration_seconds: float
    acquisition_seconds: float
    num_points: int = 512
    rf_frequency_hz: float | None = None
    phase: float = 0.0
    t2_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.nutation_hz <= 0 or self.pulse_duration_seconds <= 0:
            raise ValueError("nutation_hz and pulse_duration_seconds must be positive")
        if self.acquisition_seconds <= 0 or self.num_points <= 1:
            raise ValueError("acquisition_seconds must be positive, num_points > 1")
        if self.t2_seconds is not None and self.t2_seconds <= 0:
            raise ValueError("t2_seconds must be positive when set")


@register_serializable
@dataclass(frozen=True)
class ESRHahnEcho:
    """Two-pulse ESR Hahn echo (single isochromat).

    ``refocus_duration_seconds`` left as ``None`` uses twice the excitation
    duration (a 90-180 pair at the same B1); ``acquisition_seconds`` left as
    ``None`` samples a window of ``2 * tau_seconds`` after the refocusing
    pulse, centering the echo.
    """

    nutation_hz: float
    excitation_duration_seconds: float
    tau_seconds: float
    refocus_duration_seconds: float | None = None
    acquisition_seconds: float | None = None
    num_points: int = 512
    rf_frequency_hz: float | None = None
    excitation_phase: float = 0.0
    refocus_phase: float | None = None
    t2_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.nutation_hz <= 0 or self.excitation_duration_seconds <= 0:
            raise ValueError(
                "nutation_hz and excitation_duration_seconds must be positive"
            )
        if self.tau_seconds <= 0:
            raise ValueError("tau_seconds must be positive")
        if self.refocus_duration_seconds is not None and self.refocus_duration_seconds <= 0:
            raise ValueError("refocus_duration_seconds must be positive when set")
        if self.acquisition_seconds is not None and self.acquisition_seconds <= 0:
            raise ValueError("acquisition_seconds must be positive when set")
        if self.num_points <= 1:
            raise ValueError("num_points must be greater than 1")
        if self.t2_seconds is not None and self.t2_seconds <= 0:
            raise ValueError("t2_seconds must be positive when set")


@register_serializable
@dataclass(frozen=True)
class ESRCWSweep:
    """Continuous-wave ESR field sweep at fixed microwave frequency."""

    microwave_frequency_hz: float
    orientations: str = "single"
    broadening_tesla: float = 1.0e-4
    num_points: int = 1024
    span_tesla: float | None = None
    lineshape: str = "gaussian"
    detection_mode: str = "absorption"

    def __post_init__(self) -> None:
        if self.microwave_frequency_hz <= 0 or self.broadening_tesla <= 0:
            raise ValueError("microwave frequency and broadening must be positive")
        if self.num_points <= 1:
            raise ValueError("num_points must be greater than 1")
        if self.span_tesla is not None and self.span_tesla <= 0:
            raise ValueError("span_tesla must be positive when set")
        if self.orientations not in ("powder", "single"):
            raise ValueError("orientations must be 'powder' or 'single'")
        if self.lineshape not in ("gaussian", "lorentzian"):
            raise ValueError("lineshape must be 'gaussian' or 'lorentzian'")
        if self.detection_mode not in ("absorption", "derivative"):
            raise ValueError("unsupported detection_mode")


@register_serializable
@dataclass(frozen=True)
class ESRDEER:
    """DEER form factor calculated from ``Sample.deer_distribution``."""

    acquisition_seconds: float
    num_points: int = 512
    lambda_depth: float = 1.0
    n_theta: int = 2001
    g_a: float = 2.00231930436256
    g_b: float = 2.00231930436256

    def __post_init__(self) -> None:
        if self.acquisition_seconds <= 0 or self.num_points <= 1:
            raise ValueError("acquisition_seconds must be positive, num_points > 1")
        if not 0.0 <= self.lambda_depth <= 1.0:
            raise ValueError("lambda_depth must be between zero and one")
        if self.n_theta < 2:
            raise ValueError("n_theta must be at least 2")
        if self.g_a <= 0 or self.g_b <= 0:
            raise ValueError("g values must be positive")


def _validate_eseem(spec: Any) -> None:
    if spec.acquisition_seconds <= 0 or spec.num_points <= 1:
        raise ValueError("acquisition_seconds must be positive, num_points > 1")
    if spec.model not in ("auto", "analytic", "quantum"):
        raise ValueError("model must be 'auto', 'analytic', or 'quantum'")
    if spec.zero_fill < 1:
        raise ValueError("zero_fill must be at least 1")


@register_serializable
@dataclass(frozen=True)
class ESRTwoPulseESEEM:
    """Two-pulse ESEEM trace and frequency spectrum."""

    acquisition_seconds: float
    num_points: int = 512
    model: str = "auto"
    electron_offset_hz: float = 0.0
    zero_fill: int = 4

    def __post_init__(self) -> None:
        _validate_eseem(self)


@register_serializable
@dataclass(frozen=True)
class ESRThreePulseESEEM:
    """Three-pulse stimulated-echo ESEEM trace and spectrum."""

    acquisition_seconds: float
    tau_seconds: float
    num_points: int = 512
    model: str = "auto"
    zero_fill: int = 4

    def __post_init__(self) -> None:
        _validate_eseem(self)
        if self.tau_seconds < 0:
            raise ValueError("tau_seconds must be non-negative")


@register_serializable
@dataclass(frozen=True)
class ESRHYSCORE:
    """Two-dimensional HYSCORE time grid and spectrum."""

    evolution1_seconds: float
    evolution2_seconds: float
    tau_seconds: float
    num_points1: int = 128
    num_points2: int = 128
    zero_fill: int = 2

    def __post_init__(self) -> None:
        if self.evolution1_seconds <= 0 or self.evolution2_seconds <= 0:
            raise ValueError("HYSCORE evolution windows must be positive")
        if self.tau_seconds < 0:
            raise ValueError("tau_seconds must be non-negative")
        if self.num_points1 <= 1 or self.num_points2 <= 1:
            raise ValueError("HYSCORE axes must each contain at least two points")
        if self.zero_fill < 1:
            raise ValueError("zero_fill must be at least 1")


def _validate_endor(spec: Any) -> None:
    if spec.num_points <= 1 or spec.linewidth_hz <= 0:
        raise ValueError("num_points must be > 1 and linewidth_hz positive")
    if (
        spec.frequency_min_hz is not None
        and spec.frequency_max_hz is not None
        and spec.frequency_max_hz <= spec.frequency_min_hz
    ):
        raise ValueError("frequency_max_hz must exceed frequency_min_hz")


@register_serializable
@dataclass(frozen=True)
class ESRDaviesENDOR:
    """One-dimensional Davies ENDOR radiofrequency sweep."""

    num_points: int = 1024
    linewidth_hz: float = 1.0e5
    frequency_min_hz: float | None = None
    frequency_max_hz: float | None = None

    def __post_init__(self) -> None:
        _validate_endor(self)


@register_serializable
@dataclass(frozen=True)
class ESRMimsENDOR:
    """One-dimensional Mims ENDOR sweep with blind-spot weighting."""

    tau_seconds: float
    num_points: int = 1024
    linewidth_hz: float = 1.0e5
    frequency_min_hz: float | None = None
    frequency_max_hz: float | None = None

    def __post_init__(self) -> None:
        _validate_endor(self)
        if self.tau_seconds <= 0:
            raise ValueError("tau_seconds must be positive")


@register_serializable
@dataclass(frozen=True)
class SequenceIRExecution:
    """Execute a backend-neutral :class:`SequenceIR` through the facade."""

    ir: SequenceIR
    system_frequency_hz: float | None = None
    walkers_per_cell: int = 1
    seed: int | None = 0
    jitter: bool = False
    boundary: str = "reflect"
    default_substeps: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.ir, SequenceIR):
            raise TypeError("ir must be a spin_dynamics.sequences.SequenceIR")
        if self.system_frequency_hz is not None and (
            not np.isfinite(self.system_frequency_hz)
            or self.system_frequency_hz <= 0.0
        ):
            raise ValueError("system_frequency_hz must be finite and positive when set")
        if not isinstance(self.walkers_per_cell, int) or self.walkers_per_cell <= 0:
            raise ValueError("walkers_per_cell must be a positive integer")
        if self.seed is not None and (
            not isinstance(self.seed, int) or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer when set")
        if self.boundary not in ("reflect", "periodic", "clip"):
            raise ValueError("boundary must be 'reflect', 'periodic', or 'clip'")
        if not isinstance(self.default_substeps, int) or self.default_substeps <= 0:
            raise ValueError("default_substeps must be a positive integer")


@register_serializable
@dataclass(frozen=True)
class NanoMRStatisticalSpectrum:
    """Two-sided statistical field-noise spectrum of a uniform nuclear layer."""

    b0_vector_tesla_lab: tuple[float, float, float] = (0.0, 0.0, 0.05)
    angular_frequency_min_rad_s: float = -2.0 * np.pi * 5.0e6
    angular_frequency_max_rad_s: float = 2.0 * np.pi * 5.0e6
    num_points: int = 2001

    def __post_init__(self) -> None:
        field_vector = _finite_tuple3(
            self.b0_vector_tesla_lab, "b0_vector_tesla_lab"
        )
        if np.linalg.norm(field_vector) <= 0.0:
            raise ValueError("b0_vector_tesla_lab must be non-zero")
        lower = float(self.angular_frequency_min_rad_s)
        upper = float(self.angular_frequency_max_rad_s)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError(
                "angular_frequency_max_rad_s must exceed the finite minimum"
            )
        if not isinstance(self.num_points, int) or self.num_points < 2:
            raise ValueError("num_points must be an integer of at least two")
        object.__setattr__(self, "b0_vector_tesla_lab", field_vector)
        object.__setattr__(self, "angular_frequency_min_rad_s", lower)
        object.__setattr__(self, "angular_frequency_max_rad_s", upper)


@register_serializable
@dataclass(frozen=True)
class NanoMRQdyne:
    """Clocked coherent-tone Qdyne acquisition with explicit error budgets."""

    signal_frequency_hz: float = 2.0e6
    field_amplitude_tesla: float = 20.0e-9
    signal_phase_rad: float = 0.0
    shot_count: int = 1024
    sensing_duration_seconds: float = 20.0e-6
    repetition_interval_seconds: float = 100.0e-6
    reference_frequency_hz: float = 1.999e6
    xy_order: int = 8
    xy_repetitions: int = 1
    pulse_duration_seconds: float = 0.0
    pulse_model: str = "ideal"
    baseline_bright_probability: float = 0.5
    analysis_contrast: float = 1.0
    analysis_phase_rad: float = 0.0
    sensor_coherence_seconds: float | None = None
    sensor_stretch_exponent: float = 1.0
    sample_coherence_seconds: float | None = None
    diffusion_correlation_seconds: float | None = None
    memory_coherence_seconds: float | None = None
    fractional_frequency_offset: float = 0.0
    interval_fractional_frequency_instability: float = 0.0
    trigger_jitter_seconds: float = 0.0
    seed: int | None = 0

    def __post_init__(self) -> None:
        for name in (
            "signal_frequency_hz",
            "sensing_duration_seconds",
            "repetition_interval_seconds",
            "sensor_stretch_exponent",
        ):
            value = float(getattr(self, name))
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        amplitude = float(self.field_amplitude_tesla)
        if amplitude < 0.0 or not np.isfinite(amplitude):
            raise ValueError("field_amplitude_tesla must be finite and non-negative")
        object.__setattr__(self, "field_amplitude_tesla", amplitude)
        if self.repetition_interval_seconds < self.sensing_duration_seconds:
            raise ValueError(
                "repetition_interval_seconds cannot be shorter than sensing_duration_seconds"
            )
        if not isinstance(self.shot_count, int) or self.shot_count < 2:
            raise ValueError("shot_count must be an integer of at least two")
        if self.xy_order not in (4, 8, 16):
            raise ValueError("xy_order must be 4, 8, or 16")
        if not isinstance(self.xy_repetitions, int) or self.xy_repetitions <= 0:
            raise ValueError("xy_repetitions must be a positive integer")
        pulse_duration = float(self.pulse_duration_seconds)
        if pulse_duration < 0.0 or not np.isfinite(pulse_duration):
            raise ValueError("pulse_duration_seconds must be finite and non-negative")
        object.__setattr__(self, "pulse_duration_seconds", pulse_duration)
        if self.pulse_model not in ("ideal", "finite"):
            raise ValueError("pulse_model must be 'ideal' or 'finite'")
        for name in ("baseline_bright_probability", "analysis_contrast"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0 or not np.isfinite(value):
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "sensor_coherence_seconds",
            "sample_coherence_seconds",
            "diffusion_correlation_seconds",
            "memory_coherence_seconds",
        ):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if value <= 0.0 or not np.isfinite(value):
                    raise ValueError(f"{name} must be positive and finite when set")
                object.__setattr__(self, name, value)
        for name in (
            "interval_fractional_frequency_instability",
            "trigger_jitter_seconds",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "signal_phase_rad",
            "reference_frequency_hz",
            "analysis_phase_rad",
            "fractional_frequency_offset",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.seed is not None and (
            not isinstance(self.seed, int) or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer when set")


SEQUENCE_TYPES: tuple[type, ...] = (
    CPMG,
    CPMGTrain,
    CPMGIRTrain,
    CPMGImaging,
    PGSE,
    PGSEWalkers,
    NQRSLSE,
    NQRSORC,
    NQRFID,
    NQRPopulationTransfer,
    ESRFID,
    ESRHahnEcho,
    ESRCWSweep,
    ESRDEER,
    ESRTwoPulseESEEM,
    ESRThreePulseESEEM,
    ESRHYSCORE,
    ESRDaviesENDOR,
    ESRMimsENDOR,
    SequenceIRExecution,
    NanoMRStatisticalSpectrum,
    NanoMRQdyne,
)


@register_serializable
@dataclass(frozen=True)
class Experiment:
    """A complete declarative experiment description."""

    sequence: Any
    sample: Sample = field(default_factory=Sample)
    hardware: Hardware = field(default_factory=Hardware)
    acquisition: Acquisition = field(default_factory=Acquisition)

    def plan(self, *, estimate: bool = True) -> "Any":
        from spin_dynamics.experiment.plan import plan_experiment

        return plan_experiment(self, estimate=estimate)

    def run(self, **execution: Any) -> "Any":
        from spin_dynamics.experiment.runner import run_experiment

        return run_experiment(self, **execution)

    def to_dict(self) -> dict[str, Any]:
        return encode(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Experiment":
        experiment = decode(dict(data))
        if not isinstance(experiment, Experiment):
            raise TypeError("serialized payload does not describe an Experiment")
        return experiment

    @staticmethod
    def from_json(payload: str) -> "Experiment":
        return Experiment.from_dict(json.loads(payload))


def _differs(value: Any, default: Any) -> bool:
    if isinstance(value, np.ndarray) or isinstance(default, np.ndarray):
        return not np.array_equal(np.asarray(value), np.asarray(default))
    return bool(value != default)


def non_default_fields(experiment: Experiment) -> dict[str, Any]:
    """Return dotted spec-field names whose values differ from the defaults.

    The sequence spec is excluded: its type selects the workflow, so all of
    its fields are honored by construction.
    """

    out: dict[str, Any] = {}
    for group_name in ("sample", "hardware", "acquisition"):
        group = getattr(experiment, group_name)
        for spec_field in dataclasses.fields(group):
            value = getattr(group, spec_field.name)
            if spec_field.default is not dataclasses.MISSING:
                default = spec_field.default
            elif spec_field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                default = spec_field.default_factory()  # type: ignore[misc]
            else:
                continue
            if _differs(value, default):
                out[f"{group_name}.{spec_field.name}"] = value
    return out
