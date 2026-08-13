"""Quantitative validation helpers for harmonic full-wave field solutions.

The checks in this module deliberately separate solver completion from physical
trust.  They operate on solver-neutral :class:`HarmonicEMSolution` objects,
with one openEMS-specific power check for its raw electric/current-density
dumps and port metadata.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from spin_dynamics.fields.harmonic import (
    MU_0,
    HarmonicConvergence,
    HarmonicEMSolution,
)
from spin_dynamics.fields.interpolate import dlinear_sample


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _json_metadata(value: Mapping[str, object], name: str) -> dict[str, object]:
    result = dict(value)
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-serializable") from exc
    return result


@dataclass(frozen=True)
class FullWaveValidationCheck:
    """One auditable validation metric and its pass/fail threshold."""

    name: str
    passed: bool
    metric: float | None = None
    tolerance: float | None = None
    required: bool = True
    summary: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("validation check name must be nonempty")
        for value, label in ((self.metric, "metric"), (self.tolerance, "tolerance")):
            if value is not None:
                _finite_nonnegative(value, label)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "summary", str(self.summary))
        object.__setattr__(
            self,
            "metadata",
            _json_metadata(self.metadata, "validation check metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "metric": self.metric,
            "tolerance": self.tolerance,
            "required": self.required,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FullWaveValidationReport:
    """Collection of validation checks for one model or convergence series."""

    checks: tuple[FullWaveValidationCheck, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        checks = tuple(self.checks)
        if not checks:
            raise ValueError("validation report must contain at least one check")
        if not all(isinstance(item, FullWaveValidationCheck) for item in checks):
            raise TypeError("checks must contain FullWaveValidationCheck objects")
        if len({item.name for item in checks}) != len(checks):
            raise ValueError("validation check names must be unique")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(
            self,
            "metadata",
            _json_metadata(self.metadata, "validation report metadata"),
        )

    @property
    def passed(self) -> bool:
        """Whether every required check passed."""

        return all(item.passed for item in self.checks if item.required)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "python-spin-dynamics.fullwave-validation/v1",
            "passed": self.passed,
            "checks": [item.to_dict() for item in self.checks],
            "metadata": dict(self.metadata),
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the report as deterministic JSON and return its path."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


def _positions3(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.all(np.isfinite(result)):
        raise ValueError("positions must have shape (num_points, 3) and be finite")
    return result


def sample_harmonic_vector(
    solution: HarmonicEMSolution,
    positions_m: Sequence[Sequence[float]] | np.ndarray,
    *,
    field_name: str = "B",
) -> np.ndarray:
    """Trilinearly sample canonical complex E or B without boundary clamping."""

    if len(solution.domain.axes) != 3:
        raise ValueError("full-wave validation requires a three-dimensional domain")
    positions = _positions3(positions_m)
    for index, axis in enumerate(solution.domain.axes):
        if np.any(positions[:, index] < axis[0]) or np.any(
            positions[:, index] > axis[-1]
        ):
            raise ValueError("validation positions must lie inside the field domain")
    if field_name.upper() == "B":
        values = solution.magnetic_flux_density()
    elif field_name.upper() == "E":
        values = solution.electric_field()
    else:
        raise ValueError("field_name must be 'E' or 'B'")
    return np.column_stack(
        [
            dlinear_sample(values[..., component], solution.domain.axes, positions)
            for component in range(3)
        ]
    )


def numerical_termination_check(
    solution: HarmonicEMSolution,
) -> FullWaveValidationCheck:
    """Check that time stepping reached the requested energy-decay criterion."""

    convergence = solution.convergence
    metadata = {} if convergence is None else dict(convergence.metadata)
    terminated = bool(metadata.get("time_domain_terminated", False))
    residual = None if convergence is None else convergence.relative_residual
    tolerance_raw = metadata.get("end_criteria")
    tolerance = None if tolerance_raw is None else float(tolerance_raw)
    return FullWaveValidationCheck(
        "time_domain_termination",
        terminated,
        metric=residual,
        tolerance=tolerance,
        summary=(
            "FDTD energy reached the requested decay criterion"
            if terminated
            else "FDTD termination could not be verified"
        ),
        metadata={
            "iterations": None if convergence is None else convergence.iterations,
            "final_energy_db": metadata.get("final_energy_db"),
        },
    )


def mesh_convergence_check(
    coarse: HarmonicEMSolution,
    fine: HarmonicEMSolution,
    positions_m: Sequence[Sequence[float]] | np.ndarray,
    *,
    relative_tolerance: float = 0.05,
    name: str = "mesh_convergence",
) -> FullWaveValidationCheck:
    """Compare normalized complex E and B on common physical probe points."""

    tolerance = _finite_nonnegative(relative_tolerance, "relative_tolerance")
    if not np.isclose(coarse.frequency_hz, fine.frequency_hz, rtol=1.0e-12):
        raise ValueError("mesh solutions must use the same frequency")
    if coarse.normalization.kind != fine.normalization.kind:
        raise ValueError("mesh solutions must use the same normalization")
    positions = _positions3(positions_m)

    def relative_rms(field_name: str) -> float:
        left = sample_harmonic_vector(coarse, positions, field_name=field_name)
        right = sample_harmonic_vector(fine, positions, field_name=field_name)
        denominator = float(np.linalg.norm(right))
        if denominator <= np.finfo(np.float64).tiny:
            return 0.0 if np.linalg.norm(left - right) == 0.0 else float("inf")
        return float(np.linalg.norm(left - right) / denominator)

    electric_error = relative_rms("E")
    magnetic_error = relative_rms("B")
    metric = max(electric_error, magnetic_error)
    return FullWaveValidationCheck(
        name,
        bool(np.isfinite(metric) and metric <= tolerance),
        metric=metric if np.isfinite(metric) else None,
        tolerance=tolerance,
        summary="relative RMS change of complex E and B on common probes",
        metadata={
            "electric_relative_rms": electric_error,
            "magnetic_relative_rms": magnetic_error,
            "num_probe_points": int(positions.shape[0]),
            "coarse_model_hash": coarse.provenance.model_hash,
            "fine_model_hash": fine.provenance.model_hash,
        },
    )


def low_frequency_loop_check(
    solution: HarmonicEMSolution,
    *,
    center_m: Sequence[float] = (0.0, 0.0, 0.0),
    normal: Sequence[float] = (1.0, 0.0, 0.0),
    radius_m: float,
    turns: int = 1,
    relative_tolerance: float = 0.05,
) -> FullWaveValidationCheck:
    """Compare the loop-center B/I magnitude with the Biot--Savart limit."""

    if solution.normalization.kind != "per_ampere":
        raise ValueError("loop benchmark requires per_ampere normalization")
    radius = float(radius_m)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    if int(turns) < 1:
        raise ValueError("turns must be positive")
    direction = np.asarray(normal, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("normal must be a finite 3-vector")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("normal must be nonzero")
    direction /= norm
    sampled = sample_harmonic_vector(
        solution,
        np.asarray(center_m, dtype=np.float64).reshape(1, 3),
        field_name="B",
    )[0]
    actual = float(abs(np.dot(sampled, direction)))
    expected = MU_0 * int(turns) / (2.0 * radius)
    relative_error = abs(actual - expected) / expected
    tolerance = _finite_nonnegative(relative_tolerance, "relative_tolerance")
    return FullWaveValidationCheck(
        "low_frequency_biot_savart",
        relative_error <= tolerance,
        metric=relative_error,
        tolerance=tolerance,
        summary="loop-center |B/I| relative error versus Biot--Savart",
        metadata={
            "actual_t_per_a": actual,
            "expected_t_per_a": expected,
            "frequency_hz": solution.frequency_hz,
            "electrical_size_ka": 2.0
            * np.pi
            * solution.frequency_hz
            * radius
            / 299_792_458.0,
        },
    )


def reciprocity_check(
    drive_port_a: HarmonicEMSolution,
    drive_port_b: HarmonicEMSolution,
    *,
    port_a: int,
    port_b: int,
    relative_tolerance: float = 0.02,
    orientation_invariant: bool = True,
) -> FullWaveValidationCheck:
    """Compare reciprocal transfer impedances from two port-excitation runs."""

    def port(solution: HarmonicEMSolution, index: int):
        try:
            return next(item for item in solution.ports if item.index == index)
        except StopIteration as exc:
            raise ValueError(f"solution has no port {index}") from exc

    a_drive = port(drive_port_a, port_a)
    b_receive = port(drive_port_a, port_b)
    b_drive = port(drive_port_b, port_b)
    a_receive = port(drive_port_b, port_a)
    if a_drive.current_a is None or b_drive.current_a is None:
        raise ValueError("driven ports must contain current phasors")
    if b_receive.voltage_v is None or a_receive.voltage_v is None:
        raise ValueError("receiving ports must contain voltage phasors")
    if abs(a_drive.current_a) == 0.0 or abs(b_drive.current_a) == 0.0:
        raise ValueError("driven port currents must be nonzero")
    z_ba = b_receive.voltage_v / a_drive.current_a
    z_ab = a_receive.voltage_v / b_drive.current_a
    difference = abs(z_ba - z_ab)
    if orientation_invariant:
        difference = min(difference, abs(z_ba + z_ab))
    scale = max(abs(z_ba), abs(z_ab), np.finfo(np.float64).tiny)
    relative_error = float(difference / scale)
    tolerance = _finite_nonnegative(relative_tolerance, "relative_tolerance")
    return FullWaveValidationCheck(
        "port_reciprocity",
        relative_error <= tolerance,
        metric=relative_error,
        tolerance=tolerance,
        summary="relative mismatch between reciprocal transfer impedances",
        metadata={
            "z_ba_ohm": [float(z_ba.real), float(z_ba.imag)],
            "z_ab_ohm": [float(z_ab.real), float(z_ab.imag)],
            "orientation_invariant": orientation_invariant,
        },
    )


def integrate_conductive_loss(
    electric_field: np.ndarray,
    current_density: np.ndarray,
    axes_m: Sequence[np.ndarray],
) -> float:
    """Integrate ``0.5 Re(E dot conj(J))`` over a rectilinear volume."""

    electric = np.asarray(electric_field, dtype=np.complex128)
    current = np.asarray(current_density, dtype=np.complex128)
    axes = tuple(np.asarray(axis, dtype=np.float64) for axis in axes_m)
    expected = tuple(axis.size for axis in axes) + (3,)
    if electric.shape != expected or current.shape != expected:
        raise ValueError(f"E and J must both have shape {expected}")
    density = 0.5 * np.real(np.sum(electric * np.conjugate(current), axis=-1))
    integral: np.ndarray | np.float64 = density
    for axis in reversed(axes):
        if hasattr(np, "trapezoid"):
            integral = np.trapezoid(integral, x=axis, axis=-1)
        else:  # NumPy < 2.0
            integral = np.trapz(integral, x=axis, axis=-1)  # type: ignore[attr-defined]
    result = float(integral)
    if not np.isfinite(result):
        raise ValueError("integrated conductive loss is not finite")
    return result


def openems_conductive_loss_check(
    directory: str | Path,
    *,
    relative_tolerance: float = 0.1,
) -> FullWaveValidationCheck:
    """Compare openEMS volume E/J loss with accepted feed-port power.

    The unaccounted nonnegative remainder may contain radiation, PML absorption,
    and conductor loss.  A volume loss larger than accepted power beyond the
    tolerance is therefore a hard consistency failure; a positive remainder is
    reported rather than silently called an exact balance.
    """

    from spin_dynamics.fields.openems import (
        load_openems_field_dump,
        load_openems_project,
    )

    root = Path(directory)
    project = load_openems_project(root / "openems_project.json")
    electric = load_openems_field_dump(
        root / "E_fd.h5", frequency_hz=project.settings.frequency_hz
    )
    current = load_openems_field_dump(
        root / "J_fd.h5", frequency_hz=project.settings.frequency_hz
    )
    if electric.domain.shape != current.domain.shape or any(
        not np.array_equal(left, right)
        for left, right in zip(electric.domain.axes, current.domain.axes)
    ):
        raise ValueError("openEMS E and J dumps use different spatial meshes")
    loss = integrate_conductive_loss(
        electric.values,
        current.values,
        electric.domain.axes,
    )
    raw_ports = json.loads((root / "openems_port.json").read_text(encoding="utf-8"))
    excited = next(item for item in project.ports if item.excite)
    port_data = next(item for item in raw_ports if int(item["number"]) == excited.number)
    accepted = float(port_data["accepted_power_w"])
    if not np.isfinite(accepted) or accepted <= 0.0:
        raise ValueError("accepted port power must be positive and finite")
    excess = max(0.0, (loss - accepted) / accepted)
    tolerance = _finite_nonnegative(relative_tolerance, "relative_tolerance")
    return FullWaveValidationCheck(
        "accepted_power_bounds_conductive_loss",
        loss >= 0.0 and excess <= tolerance,
        metric=excess,
        tolerance=tolerance,
        summary="conductive volume loss must not exceed accepted port power",
        metadata={
            "conductive_loss_raw": loss,
            "accepted_power_raw": accepted,
            "conductive_fraction": loss / accepted,
            "unaccounted_fraction": (accepted - loss) / accepted,
        },
    )


def apply_validation_report(
    solution: HarmonicEMSolution,
    report: FullWaveValidationReport,
) -> HarmonicEMSolution:
    """Return a copy whose convergence record includes a validation report."""

    previous = solution.convergence
    metadata = {} if previous is None else dict(previous.metadata)
    metadata["fullwave_validation"] = report.to_dict()
    power = next(
        (
            item
            for item in report.checks
            if item.name == "accepted_power_bounds_conductive_loss"
        ),
        None,
    )
    convergence = HarmonicConvergence(
        converged=report.passed,
        relative_residual=(None if previous is None else previous.relative_residual),
        iterations=None if previous is None else previous.iterations,
        mesh_cells=None if previous is None else previous.mesh_cells,
        minimum_cell_m=None if previous is None else previous.minimum_cell_m,
        maximum_cell_m=None if previous is None else previous.maximum_cell_m,
        energy_balance_relative_error=None if power is None else power.metric,
        metadata=metadata,
    )
    return replace(solution, convergence=convergence)


__all__ = [
    "FullWaveValidationCheck",
    "FullWaveValidationReport",
    "apply_validation_report",
    "integrate_conductive_loss",
    "low_frequency_loop_check",
    "mesh_convergence_check",
    "numerical_termination_check",
    "openems_conductive_loss_check",
    "reciprocity_check",
    "sample_harmonic_vector",
]
