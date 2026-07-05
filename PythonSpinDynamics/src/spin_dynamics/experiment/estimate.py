"""Advisory runtime/memory estimation for planned experiments.

Workflow cost is modeled as ``seconds = a + b * work_units`` where
``work_units`` counts kernel operations (isochromats x pulse segments x
phase-cycle branches x inversion delays) supplied per workflow by the
registry's :class:`CostModel` builders. The affine constants ``(a, b)`` are
calibrated once per process by timing two small ideal-CPMG-train dry runs on
this machine with the active kernel backend, so estimates track the actual
host and backend rather than a benchmark reference host.

Estimates are advisory (target accuracy roughly a factor of two): probe
circuit setup and backend JIT warm-up are not modeled, and very large grids
may amortize vectorization better than the calibration sizes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from spin_dynamics.core.kernels import get_arb10_backend


@dataclass(frozen=True)
class CostModel:
    """Abstract cost of one planned workflow execution."""

    work_units: float
    memory_bytes: int | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeEstimate:
    """Advisory runtime/memory prediction for a planned experiment."""

    seconds: float
    work_units: float
    backend: str
    memory_bytes: int | None = None
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        parts = [
            f"~{format_seconds(self.seconds)} on '{self.backend}' backend (advisory)"
        ]
        if self.memory_bytes is not None:
            parts.append(f"memory ~{format_bytes(self.memory_bytes)}")
        return "; ".join(parts)


@dataclass(frozen=True)
class _Calibration:
    overhead_seconds: float
    seconds_per_unit: float
    backend: str
    manual: bool = False


_CALIBRATION: _Calibration | None = None

# Two ideal-train dry runs bracket the fixed Python assembly overhead and the
# per-isochromat kernel cost. Sized to finish in well under 0.1 s combined.
_CAL_SMALL = (256, 4)
_CAL_LARGE = (4001, 16)


def _train_units(numpts: int, num_echoes: int) -> float:
    # Two phase-cycle branches; excitation encodes as 2 segments, each echo
    # as 3 (free precession, refocusing pulse, free precession + acquire).
    return 2.0 * float(numpts) * (2.0 + 3.0 * float(num_echoes))


def _time_ideal_train(numpts: int, num_echoes: int) -> float:
    from spin_dynamics.workflows import run_ideal_cpmg_train

    start = time.perf_counter()
    run_ideal_cpmg_train(
        numpts=numpts,
        maxoffs=10.0,
        num_echoes=num_echoes,
        rephase_action="ignore",
    )
    return time.perf_counter() - start


def calibrate(force: bool = False) -> _Calibration:
    """Measure (or return the cached) affine cost constants for this host."""

    global _CALIBRATION
    backend = get_arb10_backend()
    if (
        _CALIBRATION is not None
        and not force
        and (_CALIBRATION.manual or _CALIBRATION.backend == backend)
    ):
        return _CALIBRATION

    units_small = _train_units(*_CAL_SMALL)
    units_large = _train_units(*_CAL_LARGE)
    _time_ideal_train(*_CAL_SMALL)  # warm caches/JIT before timing
    t_small = _time_ideal_train(*_CAL_SMALL)
    t_large = _time_ideal_train(*_CAL_LARGE)

    per_unit = max(t_large - t_small, 1e-12) / (units_large - units_small)
    overhead = max(t_small - per_unit * units_small, 0.0)
    _CALIBRATION = _Calibration(
        overhead_seconds=overhead, seconds_per_unit=per_unit, backend=backend
    )
    return _CALIBRATION


def set_calibration(
    overhead_seconds: float,
    seconds_per_unit: float,
    backend: str = "manual",
) -> None:
    """Pin the cost constants (useful for tests and known hosts)."""

    global _CALIBRATION
    if seconds_per_unit <= 0:
        raise ValueError("seconds_per_unit must be positive")
    if overhead_seconds < 0:
        raise ValueError("overhead_seconds must be non-negative")
    _CALIBRATION = _Calibration(
        overhead_seconds=float(overhead_seconds),
        seconds_per_unit=float(seconds_per_unit),
        backend=str(backend),
        manual=True,
    )


def estimate_runtime(cost: CostModel) -> RuntimeEstimate:
    """Convert an abstract cost model to a host-calibrated estimate."""

    cal = calibrate()
    seconds = cal.overhead_seconds + cal.seconds_per_unit * cost.work_units
    return RuntimeEstimate(
        seconds=seconds,
        work_units=cost.work_units,
        backend=cal.backend,
        memory_bytes=cost.memory_bytes,
        notes=cost.notes,
    )


def format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return "<1 ms"
    if seconds < 1.0:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 120.0:
        return f"{seconds:.1f} s"
    return f"{seconds / 60.0:.1f} min"


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "kB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.0f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"
