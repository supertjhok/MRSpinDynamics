"""Reference-channel RFI cancellers with masked fitting and adaptation.

The cancellers in this module are deliberately numerical: they consume a
primary channel ``y``, stacked reference channels ``X`` with shape ``(K, N)``,
and boolean masks that say where coefficients may be estimated or adapted. That
keeps them independent of any particular NQR/ESR/MRI simulation object.

Stationary interferers can often be removed by a fixed complex scalar or a short
multi-reference FIR transfer function fitted on baseline samples. Time-varying
interferers, including randomly modulated AM carriers or drifting coupling
paths, need adaptive updates; the LMS/NLMS and RLS helpers predict continuously
but update only where ``update_mask`` is true.

Offline use has a wider design space when the primary channel remains linear
(no LNA/ADC saturation). Windowed ridge fits a separate FIR transfer function in
each time block and regularizes neighboring blocks toward one another, giving a
batch estimator for slow random modulation without requiring real-time updates.
Joint signal/reference fits add an explicit signal subspace, such as an SLSE
echo-train basis, so offline cleanup can separate structured NQR response from
reference-correlated RFI instead of treating all primary-channel structure as
interference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CancellationResult:
    """Cleaned primary data and the RFI estimate produced by a canceller."""

    cleaned: np.ndarray
    predicted: np.ndarray
    coefficients: np.ndarray
    fit_mask: np.ndarray | None = None
    update_mask: np.ndarray | None = None
    coefficient_history: np.ndarray | None = None
    signal_estimate: np.ndarray | None = None
    signal_coefficients: np.ndarray | None = None


@dataclass(frozen=True)
class LinearCancellerModel:
    """Fixed multi-reference FIR canceller fitted by gated ridge least squares."""

    coefficients: np.ndarray
    taps: int
    ridge: float
    fit_mask: np.ndarray

    def predict(self, references: np.ndarray) -> np.ndarray:
        """Return the predicted RFI contribution for ``references``."""

        x = _reference_matrix(references)
        coeff = np.asarray(self.coefficients, dtype=np.complex128)
        if coeff.ndim == 1:
            coeff = coeff.reshape(x.shape[0], self.taps)
        if coeff.shape != (x.shape[0], self.taps):
            raise ValueError("coefficient shape does not match references and taps")
        design = _tapped_design(x, self.taps)
        return design @ coeff.reshape(-1)

    def apply(self, primary: np.ndarray, references: np.ndarray) -> CancellationResult:
        """Apply the fixed model to a primary/reference record."""

        y = _primary_vector(primary)
        predicted = self.predict(references)
        if predicted.size != y.size:
            raise ValueError("primary and reference lengths must match")
        return CancellationResult(
            cleaned=y - predicted,
            predicted=predicted,
            coefficients=self.coefficients,
            fit_mask=self.fit_mask,
        )


def _primary_vector(primary: np.ndarray) -> np.ndarray:
    y = np.asarray(primary, dtype=np.complex128).reshape(-1)
    if y.size == 0:
        raise ValueError("primary must contain at least one sample")
    if not np.all(np.isfinite(y)):
        raise ValueError("primary samples must be finite")
    return y


def _reference_matrix(references: np.ndarray) -> np.ndarray:
    x = np.asarray(references, dtype=np.complex128)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("references must have shape (K, N)")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("references must contain at least one channel and sample")
    if not np.all(np.isfinite(x)):
        raise ValueError("reference samples must be finite")
    return x


def _mask(mask: np.ndarray | None, num_samples: int, name: str) -> np.ndarray:
    if mask is None:
        return np.ones(num_samples, dtype=bool)
    out = np.asarray(mask, dtype=bool).reshape(-1)
    if out.size != num_samples:
        raise ValueError(f"{name} length must match the sample count")
    return out


def _validate_xy(primary: np.ndarray, references: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = _primary_vector(primary)
    x = _reference_matrix(references)
    if x.shape[1] != y.size:
        raise ValueError("primary and reference lengths must match")
    return y, x


def _positive_int(value: int, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _nonnegative_float(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return out


def _tapped_design(references: np.ndarray, taps: int) -> np.ndarray:
    x = _reference_matrix(references)
    taps = _positive_int(taps, "taps")
    channels, samples = x.shape
    design = np.zeros((samples, channels * taps), dtype=np.complex128)
    for tap in range(taps):
        if tap == 0:
            design[:, tap::taps] = x.T
        else:
            design[tap:, tap::taps] = x[:, :-tap].T
    return design


def _basis_design(signal_basis: np.ndarray, num_samples: int) -> np.ndarray:
    basis = np.asarray(signal_basis, dtype=np.complex128)
    if basis.ndim == 1:
        basis = basis.reshape(1, -1)
    if basis.ndim != 2:
        raise ValueError("signal_basis must have shape (M, N)")
    if basis.shape[1] != num_samples:
        raise ValueError("signal_basis sample dimension must match primary")
    if basis.shape[0] == 0:
        raise ValueError("signal_basis must contain at least one basis vector")
    if not np.all(np.isfinite(basis)):
        raise ValueError("signal_basis samples must be finite")
    return basis.T


def fit_gated_ridge_fir(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray,
    *,
    taps: int = 1,
    ridge: float = 0.0,
) -> LinearCancellerModel:
    """Fit a fixed FIR reference canceller on baseline samples only.

    ``references`` has shape ``(K, N)``. The fitted model predicts
    ``sum_k sum_l h[k, l] * X[k, n-l]`` and returns coefficients with shape
    ``(K, taps)``. ``fit_mask`` should normally be the baseline set ``B``.
    """

    y, x = _validate_xy(primary, references)
    taps = _positive_int(taps, "taps")
    ridge = _nonnegative_float(ridge, "ridge")
    fit = _mask(fit_mask, y.size, "fit_mask")
    valid = fit.copy()
    valid[: taps - 1] = False
    if not np.any(valid):
        raise ValueError("fit_mask selects no samples with enough FIR history")

    design = _tapped_design(x, taps)
    a = design[valid]
    target = y[valid]
    normal = a.conj().T @ a
    if ridge > 0.0:
        normal = normal + ridge * np.eye(normal.shape[0], dtype=np.complex128)
    rhs = a.conj().T @ target
    coeff = np.linalg.solve(normal, rhs).reshape(x.shape[0], taps)
    return LinearCancellerModel(coefficients=coeff, taps=taps, ridge=ridge, fit_mask=fit)


def gated_ridge_fir_canceller(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray,
    *,
    taps: int = 1,
    ridge: float = 0.0,
) -> CancellationResult:
    """Fit and apply a gated ridge-LS FIR canceller."""

    model = fit_gated_ridge_fir(primary, references, fit_mask, taps=taps, ridge=ridge)
    return model.apply(primary, references)


def fit_scalar_canceller(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray,
    *,
    ridge: float = 0.0,
) -> LinearCancellerModel:
    """Fit a zero-lag multi-reference scalar canceller on ``fit_mask`` samples."""

    return fit_gated_ridge_fir(primary, references, fit_mask, taps=1, ridge=ridge)


def scalar_canceller(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray,
    *,
    ridge: float = 0.0,
) -> CancellationResult:
    """Fit and apply a zero-lag multi-reference scalar canceller."""

    return gated_ridge_fir_canceller(primary, references, fit_mask, taps=1, ridge=ridge)


def windowed_ridge_fir_canceller(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray,
    *,
    taps: int = 1,
    window_samples: int = 1024,
    ridge: float = 0.0,
    smoothness: float = 0.0,
) -> CancellationResult:
    """Apply an offline windowed ridge-FIR canceller with smooth coefficients.

    A separate FIR transfer function is fitted in each contiguous time window.
    The batch objective is the sum of masked least-squares residuals, ridge
    penalties, and ``smoothness * ||h_w - h_{w-1}||^2`` between adjacent windows.
    This is useful for randomly modulated AM-like RFI whose coupling changes
    too much for one stationary fit but can be estimated offline.
    """

    y, x = _validate_xy(primary, references)
    taps = _positive_int(taps, "taps")
    window_samples = _positive_int(window_samples, "window_samples")
    ridge = _nonnegative_float(ridge, "ridge")
    smoothness = _nonnegative_float(smoothness, "smoothness")
    fit = _mask(fit_mask, y.size, "fit_mask")
    valid = fit.copy()
    valid[: taps - 1] = False
    if not np.any(valid):
        raise ValueError("fit_mask selects no samples with enough FIR history")

    design = _tapped_design(x, taps)
    num_windows = int(np.ceil(y.size / window_samples))
    coeff_size = x.shape[0] * taps
    total_size = num_windows * coeff_size
    normal = np.zeros((total_size, total_size), dtype=np.complex128)
    rhs = np.zeros(total_size, dtype=np.complex128)

    for window in range(num_windows):
        start = window * window_samples
        stop = min(y.size, start + window_samples)
        selected = valid[start:stop]
        sl = slice(window * coeff_size, (window + 1) * coeff_size)
        if np.any(selected):
            a = design[start:stop][selected]
            target = y[start:stop][selected]
            normal[sl, sl] += a.conj().T @ a
            rhs[sl] += a.conj().T @ target
        if ridge > 0.0:
            normal[sl, sl] += ridge * np.eye(coeff_size, dtype=np.complex128)

    if smoothness > 0.0 and num_windows > 1:
        eye = smoothness * np.eye(coeff_size, dtype=np.complex128)
        for window in range(num_windows - 1):
            left = slice(window * coeff_size, (window + 1) * coeff_size)
            right = slice((window + 1) * coeff_size, (window + 2) * coeff_size)
            normal[left, left] += eye
            normal[right, right] += eye
            normal[left, right] -= eye
            normal[right, left] -= eye

    coefficients = np.linalg.solve(normal, rhs).reshape(num_windows, x.shape[0], taps)
    window_index = np.minimum(np.arange(y.size) // window_samples, num_windows - 1)
    history = coefficients[window_index]
    predicted = np.einsum("nkt,nkt->n", design.reshape(y.size, x.shape[0], taps), history)
    return CancellationResult(
        cleaned=y - predicted,
        predicted=predicted,
        coefficients=coefficients,
        fit_mask=fit,
        coefficient_history=history,
    )


def joint_signal_reference_canceller(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray,
    signal_basis: np.ndarray,
    *,
    taps: int = 1,
    reference_ridge: float = 0.0,
    signal_ridge: float = 0.0,
) -> CancellationResult:
    """Fit reference-derived RFI and structured signal terms jointly.

    ``signal_basis`` has shape ``(M, N)`` and should contain the expected
    signal subspace sampled on the same clock as ``primary``. For pulsed NQR,
    these can be SLSE echo templates with the expected phase progression and
    decay envelope. The joint model

    ``primary ~= reference_design @ h + signal_basis.T @ a``

    is fitted on ``fit_mask`` with separate ridge penalties for the reference
    and signal blocks. The returned ``predicted`` field is only the RFI
    estimate; ``cleaned`` is ``primary - predicted`` so the NQR signal remains
    in the output.
    """

    y, x = _validate_xy(primary, references)
    taps = _positive_int(taps, "taps")
    reference_ridge = _nonnegative_float(reference_ridge, "reference_ridge")
    signal_ridge = _nonnegative_float(signal_ridge, "signal_ridge")
    fit = _mask(fit_mask, y.size, "fit_mask")
    valid = fit.copy()
    valid[: taps - 1] = False
    if not np.any(valid):
        raise ValueError("fit_mask selects no samples with enough FIR history")

    reference_design = _tapped_design(x, taps)
    signal_design = _basis_design(signal_basis, y.size)
    design = np.hstack([reference_design, signal_design])
    a = design[valid]
    target = y[valid]
    normal = a.conj().T @ a
    ref_size = reference_design.shape[1]
    sig_size = signal_design.shape[1]
    if reference_ridge > 0.0:
        normal[:ref_size, :ref_size] += reference_ridge * np.eye(
            ref_size, dtype=np.complex128
        )
    if signal_ridge > 0.0:
        sl = slice(ref_size, ref_size + sig_size)
        normal[sl, sl] += signal_ridge * np.eye(sig_size, dtype=np.complex128)
    rhs = a.conj().T @ target
    coeff = np.linalg.solve(normal, rhs)
    reference_coeff = coeff[:ref_size].reshape(x.shape[0], taps)
    signal_coeff = coeff[ref_size:]
    predicted = reference_design @ coeff[:ref_size]
    signal_estimate = signal_design @ signal_coeff
    return CancellationResult(
        cleaned=y - predicted,
        predicted=predicted,
        coefficients=reference_coeff,
        fit_mask=fit,
        signal_estimate=signal_estimate,
        signal_coefficients=signal_coeff,
    )


def windowed_joint_signal_reference_canceller(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray,
    signal_basis: np.ndarray,
    *,
    taps: int = 1,
    window_samples: int = 1024,
    reference_ridge: float = 0.0,
    smoothness: float = 0.0,
    signal_ridge: float = 0.0,
) -> CancellationResult:
    """Fit windowed reference RFI and structured signal terms jointly.

    This is the echo-aware counterpart of ``windowed_ridge_fir_canceller``.
    Reference coefficients are local to each time window and optionally
    smoothed between neighboring windows, while the ``signal_basis`` coefficients
    are global. In pulsed-NQR workflows this lets the fit use all receive
    samples, including echoes, without forcing the reference model to explain an
    SLSE-shaped signal.
    """

    y, x = _validate_xy(primary, references)
    taps = _positive_int(taps, "taps")
    window_samples = _positive_int(window_samples, "window_samples")
    reference_ridge = _nonnegative_float(reference_ridge, "reference_ridge")
    smoothness = _nonnegative_float(smoothness, "smoothness")
    signal_ridge = _nonnegative_float(signal_ridge, "signal_ridge")
    fit = _mask(fit_mask, y.size, "fit_mask")
    valid = fit.copy()
    valid[: taps - 1] = False
    if not np.any(valid):
        raise ValueError("fit_mask selects no samples with enough FIR history")

    reference_design = _tapped_design(x, taps)
    signal_design = _basis_design(signal_basis, y.size)
    num_windows = int(np.ceil(y.size / window_samples))
    ref_size = x.shape[0] * taps
    sig_size = signal_design.shape[1]
    total_ref_size = num_windows * ref_size
    total_size = total_ref_size + sig_size
    normal = np.zeros((total_size, total_size), dtype=np.complex128)
    rhs = np.zeros(total_size, dtype=np.complex128)
    sig_slice = slice(total_ref_size, total_size)

    for window in range(num_windows):
        start = window * window_samples
        stop = min(y.size, start + window_samples)
        selected = valid[start:stop]
        ref_slice = slice(window * ref_size, (window + 1) * ref_size)
        if np.any(selected):
            ref_block = reference_design[start:stop][selected]
            sig_block = signal_design[start:stop][selected]
            target = y[start:stop][selected]
            normal[ref_slice, ref_slice] += ref_block.conj().T @ ref_block
            normal[ref_slice, sig_slice] += ref_block.conj().T @ sig_block
            normal[sig_slice, ref_slice] += sig_block.conj().T @ ref_block
            normal[sig_slice, sig_slice] += sig_block.conj().T @ sig_block
            rhs[ref_slice] += ref_block.conj().T @ target
            rhs[sig_slice] += sig_block.conj().T @ target
        if reference_ridge > 0.0:
            normal[ref_slice, ref_slice] += reference_ridge * np.eye(
                ref_size, dtype=np.complex128
            )

    if smoothness > 0.0 and num_windows > 1:
        eye = smoothness * np.eye(ref_size, dtype=np.complex128)
        for window in range(num_windows - 1):
            left = slice(window * ref_size, (window + 1) * ref_size)
            right = slice((window + 1) * ref_size, (window + 2) * ref_size)
            normal[left, left] += eye
            normal[right, right] += eye
            normal[left, right] -= eye
            normal[right, left] -= eye

    if signal_ridge > 0.0:
        normal[sig_slice, sig_slice] += signal_ridge * np.eye(
            sig_size, dtype=np.complex128
        )

    coeff = np.linalg.solve(normal, rhs)
    reference_coeff = coeff[:total_ref_size].reshape(num_windows, x.shape[0], taps)
    signal_coeff = coeff[total_ref_size:]
    window_index = np.minimum(np.arange(y.size) // window_samples, num_windows - 1)
    history = reference_coeff[window_index]
    predicted = np.einsum(
        "nkt,nkt->n",
        reference_design.reshape(y.size, x.shape[0], taps),
        history,
    )
    signal_estimate = signal_design @ signal_coeff
    return CancellationResult(
        cleaned=y - predicted,
        predicted=predicted,
        coefficients=reference_coeff,
        fit_mask=fit,
        coefficient_history=history,
        signal_estimate=signal_estimate,
        signal_coefficients=signal_coeff,
    )


def adaptive_lms_canceller(
    primary: np.ndarray,
    references: np.ndarray,
    update_mask: np.ndarray | None = None,
    *,
    taps: int = 1,
    step: float = 0.1,
    normalized: bool = True,
    epsilon: float = 1.0e-12,
    leak: float = 0.0,
    initial_coefficients: np.ndarray | None = None,
) -> CancellationResult:
    """Apply mask-aware LMS/NLMS cancellation for time-varying transfer paths.

    Prediction is produced at every sample. Coefficients are updated only where
    ``update_mask`` is true, so adaptation can run in baseline windows and freeze
    across expected NQR signal gaps. ``normalized=True`` gives the usual NLMS
    update, which is the recommended default for amplitude-modulated carriers.
    """

    y, x = _validate_xy(primary, references)
    taps = _positive_int(taps, "taps")
    step = float(step)
    if not np.isfinite(step) or step < 0.0:
        raise ValueError("step must be non-negative and finite")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    leak = _nonnegative_float(leak, "leak")
    if leak >= 1.0:
        raise ValueError("leak must be less than 1")
    update = _mask(update_mask, y.size, "update_mask")
    design = _tapped_design(x, taps)
    coeff = _initial_coefficients(initial_coefficients, x.shape[0], taps)
    predicted = np.zeros_like(y)
    history = np.zeros((y.size, coeff.size), dtype=np.complex128)

    for idx, phi in enumerate(design):
        predicted[idx] = phi @ coeff
        error = y[idx] - predicted[idx]
        if update[idx]:
            scale = epsilon + float(np.vdot(phi, phi).real) if normalized else 1.0
            coeff = (1.0 - leak) * coeff + (step / scale) * phi.conj() * error
        history[idx] = coeff

    return CancellationResult(
        cleaned=y - predicted,
        predicted=predicted,
        coefficients=coeff.reshape(x.shape[0], taps),
        update_mask=update,
        coefficient_history=history.reshape(y.size, x.shape[0], taps),
    )


def adaptive_rls_canceller(
    primary: np.ndarray,
    references: np.ndarray,
    update_mask: np.ndarray | None = None,
    *,
    taps: int = 1,
    forgetting: float = 0.995,
    initial_covariance: float = 1.0e3,
    initial_coefficients: np.ndarray | None = None,
) -> CancellationResult:
    """Apply a mask-aware recursive least-squares reference canceller.

    RLS adapts faster than LMS when reference channels are correlated or when a
    randomly modulated carrier changes coupling on a timescale comparable to the
    receive gaps. Updates freeze wherever ``update_mask`` is false.
    """

    y, x = _validate_xy(primary, references)
    taps = _positive_int(taps, "taps")
    forgetting = float(forgetting)
    if not np.isfinite(forgetting) or forgetting <= 0.0 or forgetting > 1.0:
        raise ValueError("forgetting must be in (0, 1]")
    initial_covariance = float(initial_covariance)
    if not np.isfinite(initial_covariance) or initial_covariance <= 0.0:
        raise ValueError("initial_covariance must be positive and finite")
    update = _mask(update_mask, y.size, "update_mask")
    design = _tapped_design(x, taps)
    coeff = _initial_coefficients(initial_coefficients, x.shape[0], taps)
    covariance = initial_covariance * np.eye(coeff.size, dtype=np.complex128)
    predicted = np.zeros_like(y)
    history = np.zeros((y.size, coeff.size), dtype=np.complex128)

    for idx, phi in enumerate(design):
        predicted[idx] = phi @ coeff
        error = y[idx] - predicted[idx]
        if update[idx]:
            p_phi = covariance @ phi.conj()
            denom = forgetting + phi @ p_phi
            gain = p_phi / denom
            coeff = coeff + gain * error
            covariance = (covariance - np.outer(gain, phi @ covariance)) / forgetting
        history[idx] = coeff

    return CancellationResult(
        cleaned=y - predicted,
        predicted=predicted,
        coefficients=coeff.reshape(x.shape[0], taps),
        update_mask=update,
        coefficient_history=history.reshape(y.size, x.shape[0], taps),
    )


def _initial_coefficients(
    coefficients: np.ndarray | None,
    channels: int,
    taps: int,
) -> np.ndarray:
    if coefficients is None:
        return np.zeros(channels * taps, dtype=np.complex128)
    coeff = np.asarray(coefficients, dtype=np.complex128)
    if coeff.shape == (channels, taps):
        return coeff.reshape(-1).copy()
    if coeff.shape == (channels * taps,):
        return coeff.copy()
    raise ValueError("initial_coefficients must have shape (K, taps) or (K*taps,)")
