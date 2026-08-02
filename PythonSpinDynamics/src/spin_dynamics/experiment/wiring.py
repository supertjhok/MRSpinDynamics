"""Automatic geometry -> field-map wiring for imaging experiments.

This closes the manual glue gap between the ``fields`` solvers and the
imaging workflows: given ``Hardware`` coil/B0/plane specs and a ``Sample``
phantom, :func:`solve_imaging_field_maps` evaluates the Biot-Savart B1 of
each coil on the phantom grid, projects it perpendicular to the local B0
direction, normalizes it to a rho-weighted mean of one (transmit calibration
at the sample), and assembles the ``ImagingFieldMaps`` container the
workflows consume.

Solves are cached per process keyed on a hash of the phantom arrays and the
geometry specs, so ``plan()`` and ``run()`` (and repeated runs) share one
solve.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from spin_dynamics.experiment.hardware import (
    ImagingPlane,
    RxArray,
    RxCoil,
    TxCoil,
    UniformB0,
)
from spin_dynamics.experiment.serialization import encode
from spin_dynamics.experiment.specs import Experiment, Hardware, Phantom, SampledB0
from spin_dynamics.fields.magnetostatics import GAMMA_PROTON, biot_savart
from spin_dynamics.motion import circular_b1_component, transverse_b1_magnitude
from spin_dynamics.workflows.imaging import make_imaging_field_maps
from spin_dynamics.workflows.imaging_types import ImagingFieldMaps

_PLANE_AXES = {"xz": (0, 2), "xy": (0, 1), "yz": (1, 2)}

_SOLVE_CACHE: dict[str, "_SolvedFields"] = {}


@dataclass(frozen=True)
class ReceiveSensitivityMaps:
    """Absolute and normalized reciprocal sensitivity maps for receive channels.

    Arrays are channel-leading. ``b1_vector_t_per_a`` has shape
    ``(n_channels, *spatial_shape, 3)`` and preserves the unit-current Cartesian
    Biot-Savart field. ``b1_minus_t_per_a`` has shape
    ``(n_channels, *spatial_shape)`` and uses the package convention
    ``B1-=(Bx-1j*By)/2`` for ``B0`` along +z. ``normalized_magnitude`` is the
    legacy-compatible map ``abs(B1-)/rho_weighted_mean(abs(B1-))``.
    """

    b1_vector_t_per_a: np.ndarray
    b1_minus_t_per_a: np.ndarray
    normalization_t_per_a: np.ndarray
    normalized_magnitude: np.ndarray
    channel_labels: tuple[str, ...]

    @property
    def n_channels(self) -> int:
        """Number of receive channels."""

        return int(self.b1_minus_t_per_a.shape[0])


class _SolvedFields:
    """Cached solve: the B1 maps plus per-coil transmit-efficiency diagnostics."""

    __slots__ = (
        "b1_tx_map",
        "b1_rx_map",
        "receive_sensitivities",
        "diagnostics",
    )

    def __init__(
        self,
        b1_tx_map: np.ndarray | None,
        b1_rx_map: np.ndarray | None,
        receive_sensitivities: ReceiveSensitivityMaps | None,
        diagnostics: dict[str, float],
    ) -> None:
        self.b1_tx_map = b1_tx_map
        self.b1_rx_map = b1_rx_map
        self.receive_sensitivities = receive_sensitivities
        self.diagnostics = diagnostics


def uses_hardware_fields(hardware: Hardware) -> bool:
    """True when the hardware spec requests a coil-field solve."""

    return hardware.tx_coil is not None or hardware.rx_coil is not None


def _validate_hardware(hardware: Hardware) -> None:
    if hardware.b0 is not None and not isinstance(hardware.b0, (UniformB0, SampledB0)):
        raise ValueError("hardware.b0 must be a UniformB0 or SampledB0 spec")
    if hardware.tx_coil is not None and not isinstance(hardware.tx_coil, TxCoil):
        raise ValueError("hardware.tx_coil must be a TxCoil spec")
    if hardware.rx_coil is not None and not isinstance(
        hardware.rx_coil, (RxCoil, RxArray)
    ):
        raise ValueError("hardware.rx_coil must be an RxCoil or RxArray spec")
    if hardware.plane is not None and not isinstance(hardware.plane, ImagingPlane):
        raise ValueError("hardware.plane must be an ImagingPlane spec")


def grid_positions_m(shape: tuple[int, int], plane: ImagingPlane) -> np.ndarray:
    """Physical voxel positions (meters) of a phantom grid on the plane."""

    n0, n1 = shape
    ax0, ax1 = _PLANE_AXES[plane.plane]
    c0 = np.linspace(-plane.extent_m[0] / 2, plane.extent_m[0] / 2, n0)
    c1 = np.linspace(-plane.extent_m[1] / 2, plane.extent_m[1] / 2, n1)
    points = np.tile(np.asarray(plane.center_m, dtype=np.float64), (n0, n1, 1))
    points[..., ax0] += c0[:, np.newaxis]
    points[..., ax1] += c1[np.newaxis, :]
    return points


def sampled_b0_from_solution(
    solution,
    plane: ImagingPlane,
    shape: tuple[int, int],
    carrier_hz: float,
    nutation_rad_s: float = 1.0,
) -> SampledB0:
    """Sample a solved 3-D field onto an imaging plane as a :class:`SampledB0`.

    ``solution`` is anything with ``sample(x, y, z) -> (Bx, By, Bz)`` (e.g. a
    :class:`~spin_dynamics.fields.scalar_potential_3d.ScalarPotentialSolution`).
    ``plane`` and ``shape`` place and size the phantom grid in the solver frame,
    so the resulting off-resonance and B0-direction maps come from the real,
    inhomogeneous magnet field. ``nutation_rad_s`` normalizes the off-resonance to
    the CPMG kernel's offset units (the RF ``omega_1``; see :class:`SampledB0`).
    Wire it in as ``Hardware.b0``.
    """

    points = grid_positions_m(shape, plane)
    b_x, b_y, b_z = solution.sample(points[..., 0], points[..., 1], points[..., 2])
    return SampledB0(
        b0_tesla=np.stack([b_x, b_y, b_z], axis=-1),
        carrier_hz=carrier_hz,
        nutation_rad_s=nutation_rad_s,
    )


def _normalized_transverse_b1(
    coil_segments: list,
    current: float,
    points: np.ndarray,
    b0_direction: np.ndarray,
    rho: np.ndarray,
    which: str,
) -> tuple[np.ndarray, float]:
    """Solve, project, and normalize one coil's B1; also report efficiency.

    Returns the rho-weighted-mean-normalized transverse map plus the
    transverse fraction (mean transverse / mean total field magnitude over
    the sample) — the transmit-efficiency diagnostic surfaced by ``plan()``.
    """

    b1_vec = biot_savart(points.reshape(-1, 3), coil_segments, current)
    b1_vec = b1_vec.reshape(points.shape)
    b0_vec = np.broadcast_to(b0_direction, points.shape)
    b1_perp = transverse_b1_magnitude(b0_vec, b1_vec)
    b1_total = np.linalg.norm(b1_vec, axis=-1)

    weights = np.abs(rho)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("phantom rho must not be identically zero")
    reference = float(np.sum(b1_perp * weights)) / total
    mean_total = float(np.sum(b1_total * weights)) / total
    peak = float(np.max(b1_perp))
    if reference <= 0 or (peak > 0 and reference / peak < 1e-6):
        raise ValueError(
            f"the {which} coil produces no transverse B1 over the sample "
            "(is the coil axis parallel to B0?)"
        )
    fraction = reference / mean_total if mean_total > 0 else 0.0
    return b1_perp / reference, fraction


def _receive_sensitivity_maps(
    receiver: RxCoil | RxArray,
    points: np.ndarray,
    b0_direction: np.ndarray,
    rho: np.ndarray,
) -> tuple[ReceiveSensitivityMaps, np.ndarray]:
    """Solve absolute complex B1- and legacy normalized magnitudes per channel."""

    channels = receiver.channels if isinstance(receiver, RxArray) else (receiver,)
    b0_vector = np.broadcast_to(b0_direction, points.shape)
    weights = np.abs(rho)
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise ValueError("phantom rho must not be identically zero")

    vectors: list[np.ndarray] = []
    b1_minus: list[np.ndarray] = []
    normalizations: list[float] = []
    normalized: list[np.ndarray] = []
    transverse_fractions: list[float] = []
    for index, channel in enumerate(channels):
        vector = biot_savart(
            points.reshape(-1, 3), channel.geometry.segments(), 1.0
        ).reshape(points.shape)
        component = circular_b1_component(
            b0_vector,
            vector,
            handedness=-1,
        )
        magnitude = np.abs(component)
        normalization = float(np.sum(magnitude * weights)) / total_weight
        transverse = transverse_b1_magnitude(b0_vector, vector)
        transverse_reference = float(np.sum(transverse * weights)) / total_weight
        total_field = np.linalg.norm(vector, axis=-1)
        mean_total = float(np.sum(total_field * weights)) / total_weight
        peak = float(np.max(transverse))
        if normalization <= 0 or (peak > 0 and transverse_reference / peak < 1e-6):
            raise ValueError(
                f"receive channel {index} produces no transverse B1 over the sample "
                "(is the coil axis parallel to B0?)"
            )
        vectors.append(np.asarray(vector, dtype=np.float64))
        b1_minus.append(component)
        normalizations.append(normalization)
        normalized.append(magnitude / normalization)
        transverse_fractions.append(
            transverse_reference / mean_total if mean_total > 0 else 0.0
        )

    result = ReceiveSensitivityMaps(
        b1_vector_t_per_a=np.stack(vectors, axis=0),
        b1_minus_t_per_a=np.stack(b1_minus, axis=0),
        normalization_t_per_a=np.asarray(normalizations, dtype=np.float64),
        normalized_magnitude=np.stack(normalized, axis=0),
        channel_labels=tuple(f"rx{index}" for index in range(len(channels))),
    )
    return result, np.asarray(transverse_fractions, dtype=np.float64)


def _cache_key(phantom: Phantom, hardware: Hardware) -> str:
    digest = hashlib.sha256()
    for arr in (phantom.rho, phantom.t1_map, phantom.t2_map):
        if arr is None:
            digest.update(b"none")
        else:
            digest.update(str(arr.shape).encode())
            digest.update(np.ascontiguousarray(arr).tobytes())
    geometry = {
        "tx_coil": encode(hardware.tx_coil),
        "rx_coil": encode(hardware.rx_coil),
        "plane": encode(hardware.plane),
    }
    # Hash a SampledB0's array directly (encoding it to JSON would be huge).
    if isinstance(hardware.b0, SampledB0):
        digest.update(np.ascontiguousarray(hardware.b0.b0_tesla).tobytes())
        digest.update(str(float(hardware.b0.carrier_hz)).encode())
    else:
        geometry["b0"] = encode(hardware.b0)
    digest.update(json.dumps(geometry, sort_keys=True).encode())
    return digest.hexdigest()


def solve_imaging_field_maps(
    phantom: Phantom,
    hardware: Hardware,
    *,
    t1_seconds: float | None = None,
    t2_seconds: float | None = None,
) -> ImagingFieldMaps:
    """Assemble (and cache) the imaging field maps for a phantom + hardware.

    Without coils this reduces to the workflow's own defaults (synthetic B1);
    with coils the solved, projected, normalized maps replace them.
    """

    _validate_hardware(hardware)
    if isinstance(hardware.rx_coil, RxArray):
        raise ValueError(
            "CPMG imaging does not yet carry a receiver-channel axis; use "
            "solve_receive_sensitivities() to obtain the channel-resolved B1- "
            "maps during this foundational phase"
        )

    def t_map(map_arr: np.ndarray | None, scalar: float | None) -> np.ndarray | None:
        if map_arr is not None:
            return map_arr
        if scalar is not None:
            return scalar * np.ones_like(phantom.rho)
        return None

    t1_map = t_map(phantom.t1_map, t1_seconds)
    t2_map = t_map(phantom.t2_map, t2_seconds)

    # A SampledB0 supplies a spatially-varying off-resonance map (a real magnet
    # field); a UniformB0 (or no b0) leaves the workflow's zero off-resonance.
    b0_map = None
    if isinstance(hardware.b0, SampledB0):
        b0_map = hardware.b0.off_resonance(GAMMA_PROTON)

    if not uses_hardware_fields(hardware):
        return make_imaging_field_maps(
            phantom.rho, t1_map=t1_map, t2_map=t2_map, b0_map=b0_map
        )

    solved = _solve_coil_fields(phantom, hardware)
    return make_imaging_field_maps(
        phantom.rho,
        t1_map=t1_map,
        t2_map=t2_map,
        b0_map=b0_map,
        b1_tx_map=solved.b1_tx_map,
        b1_rx_map=solved.b1_rx_map,
    )


def _solve_coil_fields(phantom: Phantom, hardware: Hardware) -> _SolvedFields:
    key = _cache_key(phantom, hardware)
    cached = _SOLVE_CACHE.get(key)
    if cached is not None:
        return cached

    plane = hardware.plane if hardware.plane is not None else ImagingPlane()
    b0_spec = hardware.b0 if hardware.b0 is not None else UniformB0()
    if isinstance(b0_spec, SampledB0):
        direction = b0_spec.direction()  # per-voxel (n0, n1, 3)
    else:
        direction = np.asarray(b0_spec.direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
    points = grid_positions_m(phantom.rho.shape, plane)

    diagnostics: dict[str, float] = {}
    b1_tx = b1_rx = None
    receive_sensitivities = None
    tx = hardware.tx_coil
    if tx is not None:
        b1_tx, fraction = _normalized_transverse_b1(
            tx.geometry.segments(),
            tx.current_amps,
            points,
            direction,
            phantom.rho,
            "transmit",
        )
        diagnostics["tx_transverse_fraction"] = fraction
    rx = hardware.rx_coil
    if rx is not None:
        receive_sensitivities, fractions = _receive_sensitivity_maps(
            rx, points, direction, phantom.rho
        )
        diagnostics["rx_transverse_fraction"] = float(np.min(fractions))
        for index, fraction in enumerate(fractions):
            diagnostics[f"rx_{index}_transverse_fraction"] = float(fraction)
        if isinstance(rx, RxCoil):
            b1_rx = receive_sensitivities.normalized_magnitude[0]

    solved = _SolvedFields(
        b1_tx,
        b1_rx,
        receive_sensitivities,
        diagnostics,
    )
    _SOLVE_CACHE[key] = solved
    return solved


def solve_receive_sensitivities(
    phantom: Phantom,
    hardware: Hardware,
) -> ReceiveSensitivityMaps:
    """Return absolute complex reciprocal sensitivity maps for all Rx channels.

    The returned arrays retain physical T/A scaling and channel phase. The
    companion ``normalized_magnitude`` field reproduces the scalar sensitivity
    convention consumed by existing single-channel imaging workflows.
    """

    _validate_hardware(hardware)
    if hardware.rx_coil is None:
        raise ValueError("hardware.rx_coil must be set")
    result = _solve_coil_fields(phantom, hardware).receive_sensitivities
    if result is None:  # Defensive: the condition above guarantees a solve.
        raise RuntimeError("receive sensitivity solve produced no result")
    return result


def solve_diagnostics(phantom: Phantom, hardware: Hardware) -> dict[str, float]:
    """Return the per-coil transmit-efficiency diagnostics (cached solve)."""

    _validate_hardware(hardware)
    if not uses_hardware_fields(hardware):
        return {}
    return dict(_solve_coil_fields(phantom, hardware).diagnostics)


def solve_for_experiment(experiment: Experiment) -> ImagingFieldMaps:
    """Solve field maps for an experiment's sample + hardware specs."""

    if experiment.sample.phantom is None:
        raise ValueError("imaging sequences require sample.phantom")
    return solve_imaging_field_maps(
        experiment.sample.phantom,
        experiment.hardware,
        t1_seconds=experiment.sample.t1_seconds,
        t2_seconds=experiment.sample.t2_seconds,
    )
