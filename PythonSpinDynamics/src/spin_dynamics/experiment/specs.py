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

PROBE_NAMES = ("ideal", "tuned", "untuned", "matched")


def _as_optional_map(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return arr


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
    default. Non-inductive detectors join in a later milestone.
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


@register_serializable
@dataclass(frozen=True)
class Acquisition:
    """Offset grid, receiver noise, and rephasing-guard configuration."""

    numpts: int = 101
    maxoffs: float = 10.0
    noise: Mapping[str, Any] | float | Any | None = None
    auto_refine_grid: bool = False
    rephase_safety_factor: float = 1.25
    rephase_action: str = "warn"


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


SEQUENCE_TYPES: tuple[type, ...] = (
    CPMG,
    CPMGTrain,
    CPMGIRTrain,
    CPMGImaging,
    PGSE,
    PGSEWalkers,
    NQRSLSE,
    NQRSORC,
    ESRFID,
    ESRHahnEcho,
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
