"""Inverse Laplace transform utilities for relaxation and diffusion analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np


KernelName = Literal["t1", "t1_recovery", "t1_ir", "t2", "diffusion"]


@dataclass(frozen=True)
class Regularization:
    """Tikhonov regularization settings for inverse Laplace solves.

    `strength` is the non-negative penalty weight. `order=0` damps amplitudes,
    `order=1` damps slopes on the distribution grid, and `order=2` damps
    curvature. The default curvature penalty is a conservative choice for
    smooth relaxation spectra.
    """

    strength: float = 1e-2
    order: int = 2


@dataclass(frozen=True)
class ILTResult1D:
    """Result returned by one-dimensional inverse Laplace transforms."""

    distribution: np.ndarray
    axis: np.ndarray
    sample_axis: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    kernel: np.ndarray
    regularization: Regularization
    nonnegative: bool
    residual_norm: float
    solution_norm: float


@dataclass(frozen=True)
class ILTResult2D:
    """Result returned by separable two-dimensional inverse Laplace transforms."""

    distribution: np.ndarray
    axis1: np.ndarray
    axis2: np.ndarray
    sample_axis1: np.ndarray
    sample_axis2: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    kernel1: np.ndarray
    kernel2: np.ndarray
    regularization: tuple[Regularization, Regularization]
    nonnegative: bool
    residual_norm: float
    solution_norm: float


def t2_kernel(echo_times: np.ndarray, t2_values: np.ndarray) -> np.ndarray:
    """Return the CPMG decay kernel ``exp(-te / T2)``."""

    times = _positive_vector(echo_times, "echo_times", allow_zero=True)
    t2 = _positive_vector(t2_values, "t2_values")
    return np.exp(-times[:, np.newaxis] / t2[np.newaxis, :])


def t1_kernel(
    recovery_times: np.ndarray,
    t1_values: np.ndarray,
    *,
    mode: Literal["saturation", "inversion"] = "saturation",
) -> np.ndarray:
    """Return a T1 recovery or inversion-recovery kernel.

    Saturation recovery uses ``1 - exp(-tau / T1)`` and is non-negative.
    Inversion recovery uses ``1 - 2 exp(-tau / T1)``, matching the ideal
    inversion-preparation contrast used by the imaging workflow.

    The inversion-recovery kernel is *signed* (negative for ``tau < T1 ln 2``),
    so it must be fit against signed, phase-corrected real data. Magnitude data
    folds that negative lobe upward and is incompatible with this kernel; pass
    it through ``saturation`` mode or phase-correct first. ``invert_laplace_1d``
    and ``invert_laplace_2d`` emit a warning if they detect magnitude-like input.
    """

    times = _positive_vector(recovery_times, "recovery_times", allow_zero=True)
    t1 = _positive_vector(t1_values, "t1_values")
    decay = np.exp(-times[:, np.newaxis] / t1[np.newaxis, :])
    if mode == "saturation":
        return 1.0 - decay
    if mode == "inversion":
        return 1.0 - 2.0 * decay
    raise ValueError("mode must be 'saturation' or 'inversion'")


def diffusion_kernel(b_values: np.ndarray, diffusion_values: np.ndarray) -> np.ndarray:
    """Return the diffusion attenuation kernel ``exp(-b D)``."""

    b_axis = _positive_vector(b_values, "b_values", allow_zero=True)
    diffusion = _positive_vector(diffusion_values, "diffusion_values", allow_zero=True)
    return np.exp(-b_axis[:, np.newaxis] * diffusion[np.newaxis, :])


def laplace_kernel(
    sample_axis: np.ndarray,
    distribution_axis: np.ndarray,
    *,
    kind: KernelName = "t2",
) -> np.ndarray:
    """Build a named one-dimensional Laplace kernel."""

    if kind == "t2":
        return t2_kernel(sample_axis, distribution_axis)
    if kind in {"t1", "t1_recovery"}:
        return t1_kernel(sample_axis, distribution_axis, mode="saturation")
    if kind == "t1_ir":
        return t1_kernel(sample_axis, distribution_axis, mode="inversion")
    if kind == "diffusion":
        return diffusion_kernel(sample_axis, distribution_axis)
    raise ValueError(
        "kind must be 't1', 't1_recovery', 't1_ir', 't2', or 'diffusion'"
    )


def _warn_if_magnitude_like_signal(
    kernel_matrix: np.ndarray, signal: np.ndarray, nonnegative: bool
) -> None:
    """Warn when a signed kernel is paired with an all-non-negative signal.

    Inversion-recovery kernels are negative at short recovery times, so valid
    signed data must contain negative samples there. An entirely non-negative
    signal is the signature of magnitude data, which folds the negative lobe
    upward and biases the fit. This is a heuristic warning, not an error: it
    can also fire if the shortest recovery time already exceeds the inversion
    null, in which case it is safe to ignore.
    """

    if not nonnegative:
        return
    real = np.real(np.asarray(signal, dtype=np.float64))
    if real.size == 0:
        return
    scale = float(np.max(np.abs(real))) or 1.0
    tol = 1e-12 * scale
    if np.nanmin(kernel_matrix) < -tol and np.all(real >= -tol):
        warnings.warn(
            "a signed (inversion-recovery) kernel was paired with an entirely "
            "non-negative signal, which looks like magnitude data. The signed "
            "kernel cannot fit magnitudes correctly; provide signed, "
            "phase-corrected real data or use saturation-recovery mode.",
            RuntimeWarning,
            stacklevel=3,
        )


def invert_laplace_1d(
    signal: np.ndarray,
    sample_axis: np.ndarray,
    distribution_axis: np.ndarray,
    *,
    kernel: KernelName | np.ndarray = "t2",
    regularization: float | Regularization = Regularization(),
    regularization_order: int | None = None,
    nonnegative: bool = True,
) -> ILTResult1D:
    """Estimate a non-negative 1D distribution from Laplace-domain data.

    `signal` must be real, or complex with negligible imaginary residual after
    phase correction. `kernel` may be a named kernel or a precomputed matrix
    with shape ``(len(sample_axis), len(distribution_axis))``.
    """

    samples = _vector(sample_axis, "sample_axis")
    axis = _positive_vector(distribution_axis, "distribution_axis", allow_zero=True)
    y = _vector(signal, "signal")
    if y.size != samples.size:
        raise ValueError("signal and sample_axis must have the same length")

    reg = _regularization(regularization, regularization_order)
    kernel_matrix = _kernel_matrix(kernel, samples, axis)
    if kernel_matrix.shape != (samples.size, axis.size):
        raise ValueError(
            "kernel matrix must have shape "
            f"({samples.size}, {axis.size}); got {kernel_matrix.shape}"
        )
    _warn_if_magnitude_like_signal(kernel_matrix, y, nonnegative)
    return _invert_laplace_1d_precomputed(
        y,
        samples,
        axis,
        kernel_matrix,
        regularization=reg,
        nonnegative=nonnegative,
    )


def invert_laplace_2d(
    data: np.ndarray,
    sample_axis1: np.ndarray,
    sample_axis2: np.ndarray,
    distribution_axis1: np.ndarray,
    distribution_axis2: np.ndarray,
    *,
    kernel1: KernelName | np.ndarray,
    kernel2: KernelName | np.ndarray,
    regularization: (
        float
        | tuple[float, float]
        | Regularization
        | tuple[Regularization, Regularization]
    ) = Regularization(),
    regularization_order: int | tuple[int, int] | None = None,
    nonnegative: bool = True,
) -> ILTResult2D:
    """Estimate a 2D distribution from separable Laplace-domain data.

    The forward model is ``data = K1 @ distribution @ K2.T``. This covers
    T1-T2 maps, where axis 1 is recovery/inversion time and axis 2 is echo
    time, and D-T2 maps, where axis 1 is b-value and axis 2 is echo time.
    """

    x1 = _vector(sample_axis1, "sample_axis1")
    x2 = _vector(sample_axis2, "sample_axis2")
    axis1 = _positive_vector(distribution_axis1, "distribution_axis1", allow_zero=True)
    axis2 = _positive_vector(distribution_axis2, "distribution_axis2", allow_zero=True)
    matrix = np.asarray(data)
    if matrix.shape != (x1.size, x2.size):
        raise ValueError(
            "data must have shape "
            f"({x1.size}, {x2.size}); got {matrix.shape}"
        )

    reg1, reg2 = _regularization_pair(regularization, regularization_order)
    k1 = _kernel_matrix(kernel1, x1, axis1)
    k2 = _kernel_matrix(kernel2, x2, axis2)
    if k1.shape != (x1.size, axis1.size):
        raise ValueError(f"kernel1 has unexpected shape {k1.shape}")
    if k2.shape != (x2.size, axis2.size):
        raise ValueError(f"kernel2 has unexpected shape {k2.shape}")
    _warn_if_magnitude_like_signal(k1, matrix, nonnegative)
    _warn_if_magnitude_like_signal(k2, matrix, nonnegative)

    design = np.kron(k2, k1)
    return _invert_laplace_2d_precomputed(
        matrix,
        x1,
        x2,
        axis1,
        axis2,
        k1,
        k2,
        design,
        regularization=(reg1, reg2),
        nonnegative=nonnegative,
    )


def _invert_laplace_1d_precomputed(
    signal: np.ndarray,
    samples: np.ndarray,
    axis: np.ndarray,
    kernel_matrix: np.ndarray,
    *,
    regularization: Regularization,
    nonnegative: bool,
) -> ILTResult1D:
    """Solve a 1D ILT with validated axes and a precomputed kernel."""

    if signal.size != samples.size:
        raise ValueError("signal and sample_axis must have the same length")

    distribution = _solve_tikhonov(
        kernel_matrix,
        signal,
        (regularization,),
        (axis.size,),
        nonnegative=nonnegative,
    )
    prediction = kernel_matrix @ distribution
    residual = prediction - signal
    return ILTResult1D(
        distribution=distribution,
        axis=axis,
        sample_axis=samples,
        prediction=prediction,
        residual=residual,
        kernel=kernel_matrix,
        regularization=regularization,
        nonnegative=nonnegative,
        residual_norm=float(np.linalg.norm(residual.ravel())),
        solution_norm=float(np.linalg.norm(distribution.ravel())),
    )


def _invert_laplace_2d_precomputed(
    data: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    design: np.ndarray,
    *,
    regularization: tuple[Regularization, Regularization],
    nonnegative: bool,
) -> ILTResult2D:
    """Solve a 2D ILT with validated axes and precomputed separable kernels."""

    if data.shape != (x1.size, x2.size):
        raise ValueError(
            "data must have shape "
            f"({x1.size}, {x2.size}); got {data.shape}"
        )
    reg1, reg2 = regularization
    distribution_flat = _solve_tikhonov(
        design,
        data.reshape(-1, order="F"),
        (reg1, reg2),
        (axis1.size, axis2.size),
        nonnegative=nonnegative,
    )
    distribution = distribution_flat.reshape((axis1.size, axis2.size), order="F")
    prediction = k1 @ distribution @ k2.T
    residual = prediction - data
    return ILTResult2D(
        distribution=distribution,
        axis1=axis1,
        axis2=axis2,
        sample_axis1=x1,
        sample_axis2=x2,
        prediction=prediction,
        residual=residual,
        kernel1=k1,
        kernel2=k2,
        regularization=(reg1, reg2),
        nonnegative=nonnegative,
        residual_norm=float(np.linalg.norm(residual.ravel())),
        solution_norm=float(np.linalg.norm(distribution.ravel())),
    )


def invert_t2(
    signal: np.ndarray,
    echo_times: np.ndarray,
    t2_axis: np.ndarray,
    **kwargs,
) -> ILTResult1D:
    """Convenience wrapper for a 1D T2 inverse Laplace transform."""

    return invert_laplace_1d(signal, echo_times, t2_axis, kernel="t2", **kwargs)


def invert_t1(
    signal: np.ndarray,
    recovery_times: np.ndarray,
    t1_axis: np.ndarray,
    *,
    mode: Literal["saturation", "inversion"] = "saturation",
    **kwargs,
) -> ILTResult1D:
    """Convenience wrapper for a 1D T1 recovery or inversion-recovery ILT.

    With ``mode="inversion"`` the kernel is signed (``1 - 2 exp(-tau / T1)``)
    and requires signed, phase-corrected real ``signal`` data; magnitude data
    is incompatible and triggers a warning. Use ``mode="saturation"`` for
    magnitude or saturation-recovery data.
    """

    kernel = "t1_ir" if mode == "inversion" else "t1"
    return invert_laplace_1d(signal, recovery_times, t1_axis, kernel=kernel, **kwargs)


def invert_t1_t2(
    data: np.ndarray,
    recovery_times: np.ndarray,
    echo_times: np.ndarray,
    t1_axis: np.ndarray,
    t2_axis: np.ndarray,
    *,
    t1_mode: Literal["saturation", "inversion"] = "saturation",
    **kwargs,
) -> ILTResult2D:
    """Convenience wrapper for a separable T1-T2 inverse Laplace transform."""

    kernel1: KernelName = "t1_ir" if t1_mode == "inversion" else "t1"
    return invert_laplace_2d(
        data,
        recovery_times,
        echo_times,
        t1_axis,
        t2_axis,
        kernel1=kernel1,
        kernel2="t2",
        **kwargs,
    )


def invert_d_t2(
    data: np.ndarray,
    b_values: np.ndarray,
    echo_times: np.ndarray,
    diffusion_axis: np.ndarray,
    t2_axis: np.ndarray,
    **kwargs,
) -> ILTResult2D:
    """Convenience wrapper for a separable D-T2 inverse Laplace transform."""

    return invert_laplace_2d(
        data,
        b_values,
        echo_times,
        diffusion_axis,
        t2_axis,
        kernel1="diffusion",
        kernel2="t2",
        **kwargs,
    )


def invert_t2_t2(
    data: np.ndarray,
    encode_times: np.ndarray,
    detect_times: np.ndarray,
    t2_axis_encode: np.ndarray,
    t2_axis_detect: np.ndarray | None = None,
    **kwargs,
) -> ILTResult2D:
    """Convenience wrapper for a T2-T2 (relaxation exchange) inverse transform.

    The forward model is ``data = K1 @ distribution @ K2.T`` with both kernels
    of T2 decay form, where axis 1 is the encode echo time and axis 2 is the
    detect echo time. The recovered ``distribution`` is the relaxation exchange
    map: diagonal peaks come from spins whose T2 is unchanged across the mixing
    interval, while off-diagonal cross peaks reveal spins that moved to a site
    with a different T2 (chemical or compartmental exchange). Pair it with
    ``spin_dynamics.exchange.simulate_relaxation_exchange_2d`` for the forward
    side. ``t2_axis_detect`` defaults to ``t2_axis_encode``.
    """

    if t2_axis_detect is None:
        t2_axis_detect = t2_axis_encode
    return invert_laplace_2d(
        data,
        encode_times,
        detect_times,
        t2_axis_encode,
        t2_axis_detect,
        kernel1="t2",
        kernel2="t2",
        **kwargs,
    )


def _solve_tikhonov(
    design: np.ndarray,
    observations: np.ndarray,
    regularization: tuple[Regularization, ...],
    distribution_shape: tuple[int, ...],
    *,
    nonnegative: bool,
) -> np.ndarray:
    design2 = np.asarray(design, dtype=np.float64)
    y = np.asarray(observations)
    if np.iscomplexobj(y):
        if not np.allclose(np.imag(y), 0.0, atol=1e-12, rtol=1e-12):
            raise ValueError(
                "inverse Laplace data must be real. Pass a magnitude or "
                "phase-corrected real signal for complex acquisitions."
            )
        y = np.real(y)
    system = design2
    rhs = np.asarray(y, dtype=np.float64)

    penalties = _penalty_rows(regularization, distribution_shape)
    if penalties.size:
        system = np.vstack((system, penalties))
        rhs = np.concatenate((rhs, np.zeros(penalties.shape[0], dtype=np.float64)))

    column_norms = np.linalg.norm(system, axis=0)
    column_norms[column_norms == 0.0] = 1.0
    scaled_system = system / column_norms[np.newaxis, :]

    if nonnegative:
        try:
            from scipy.optimize import nnls
        except ImportError as exc:  # pragma: no cover - exercised where SciPy absent
            raise ImportError(
                "nonnegative inverse Laplace solves require SciPy. "
                "Install python-spin-dynamics[opt] or pass nonnegative=False."
            ) from exc
        scaled_solution, _ = nnls(scaled_system, rhs)
    else:
        scaled_solution, *_ = np.linalg.lstsq(scaled_system, rhs, rcond=None)
    return scaled_solution / column_norms


def _penalty_rows(
    regularization: tuple[Regularization, ...],
    distribution_shape: tuple[int, ...],
) -> np.ndarray:
    total_size = int(np.prod(distribution_shape))
    rows: list[np.ndarray] = []
    if len(distribution_shape) == 1:
        reg = regularization[0]
        if reg.strength > 0.0:
            rows.append(
                np.sqrt(reg.strength)
                * _difference_matrix(distribution_shape[0], reg.order)
            )
    elif len(distribution_shape) == 2:
        n1, n2 = distribution_shape
        reg1, reg2 = regularization
        if reg1.strength > 0.0:
            l1 = _difference_matrix(n1, reg1.order)
            rows.append(np.sqrt(reg1.strength) * np.kron(np.eye(n2), l1))
        if reg2.strength > 0.0:
            l2 = _difference_matrix(n2, reg2.order)
            rows.append(np.sqrt(reg2.strength) * np.kron(l2, np.eye(n1)))
    else:  # pragma: no cover - internal guard
        raise ValueError("distribution_shape must be one- or two-dimensional")

    if not rows:
        return np.empty((0, total_size), dtype=np.float64)
    return np.vstack(rows)


def _difference_matrix(size: int, order: int) -> np.ndarray:
    if order not in {0, 1, 2}:
        raise ValueError("regularization order must be 0, 1, or 2")
    if order == 0 or size <= order:
        return np.eye(size, dtype=np.float64)
    if order == 1:
        matrix = np.zeros((size - 1, size), dtype=np.float64)
        for idx in range(size - 1):
            matrix[idx, idx] = -1.0
            matrix[idx, idx + 1] = 1.0
        return matrix
    matrix = np.zeros((size - 2, size), dtype=np.float64)
    for idx in range(size - 2):
        matrix[idx, idx] = 1.0
        matrix[idx, idx + 1] = -2.0
        matrix[idx, idx + 2] = 1.0
    return matrix


def _kernel_matrix(
    kernel: KernelName | np.ndarray,
    sample_axis: np.ndarray,
    distribution_axis: np.ndarray,
) -> np.ndarray:
    if isinstance(kernel, str):
        return laplace_kernel(sample_axis, distribution_axis, kind=kernel)
    return np.asarray(kernel, dtype=np.float64)


def _regularization(
    value: float | Regularization,
    order_override: int | None,
) -> Regularization:
    if isinstance(value, Regularization):
        order = value.order if order_override is None else int(order_override)
        strength = value.strength
    else:
        order = 2 if order_override is None else int(order_override)
        strength = float(value)
    if strength < 0.0:
        raise ValueError("regularization strength must be non-negative")
    if order not in {0, 1, 2}:
        raise ValueError("regularization order must be 0, 1, or 2")
    return Regularization(strength=float(strength), order=order)


def _regularization_pair(
    value: (
        float
        | tuple[float, float]
        | Regularization
        | tuple[Regularization, Regularization]
    ),
    order_override: int | tuple[int, int] | None,
) -> tuple[Regularization, Regularization]:
    if isinstance(order_override, tuple):
        order1, order2 = order_override
    else:
        order1 = order2 = order_override

    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("2D regularization tuple must have length 2")
        return (
            _regularization(value[0], order1),
            _regularization(value[1], order2),
        )
    reg = _regularization(value, None)
    return (
        _regularization(reg, order1),
        _regularization(reg, order2),
    )


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_vector(
    values: np.ndarray,
    name: str,
    *,
    allow_zero: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _vector(array, name)
    if allow_zero:
        ok = np.all(array >= 0.0)
    else:
        ok = np.all(array > 0.0)
    if not ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain only {qualifier} values")
    return array
