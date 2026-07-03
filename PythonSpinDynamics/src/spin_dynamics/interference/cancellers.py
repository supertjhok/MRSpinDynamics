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

Ordinary (L2) least squares assumes the fit residual is Gaussian. Impulsive
pickup -- Poisson-timed switching transients whose bursts are large and rare --
violates that: a handful of spiky samples dominate the squared-error normal
equations and drag the fitted transfer function away from the coherent coupling.
The robust FIR canceller replaces the L2 loss with a Huber loss solved by
iteratively reweighted least squares, so samples in the impulsive tail are
down-weighted and the fit tracks the stationary RFI path instead of the spikes.
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
    fit_weights: np.ndarray | None = None
    iterations: int | None = None
    transfer_function: np.ndarray | None = None
    coherence: np.ndarray | None = None
    frequencies: np.ndarray | None = None
    sparse_component: np.ndarray | None = None


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


def _huber_weights(
    residual: np.ndarray,
    delta: float,
    scale: float | None,
) -> tuple[np.ndarray, float]:
    """Return Huber IRLS weights and the robust residual scale.

    ``scale`` is a robust dispersion of the residual magnitudes; when ``None`` it
    is estimated as ``1.4826 * median(|residual|)`` (a MAD-style proxy for the
    RMS of the coherent-fit residual, robust to the impulsive tail). Samples with
    ``|r| > delta * scale`` receive weight ``delta * scale / |r|`` so their
    contribution to the normal equations grows linearly rather than
    quadratically; the rest keep unit weight.
    """

    magnitude = np.abs(residual)
    if scale is None:
        median = float(np.median(magnitude))
        s = 1.4826 * median
    else:
        s = float(scale)
    if not np.isfinite(s) or s <= 0.0:
        s = 1.0
    threshold = delta * s
    weights = np.ones(magnitude.shape, dtype=np.float64)
    tail = magnitude > threshold
    weights[tail] = threshold / magnitude[tail]
    return weights, s


def _fit_robust_fir(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray,
    *,
    taps: int,
    ridge: float,
    huber_delta: float,
    max_iter: int,
    tol: float,
    scale: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Huber IRLS core: return (coeff (K,taps), full-length weights, fit, iters)."""

    y, x = _validate_xy(primary, references)
    taps = _positive_int(taps, "taps")
    ridge = _nonnegative_float(ridge, "ridge")
    huber_delta = _nonnegative_float(huber_delta, "huber_delta")
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive")
    max_iter = _positive_int(max_iter, "max_iter")
    tol = _nonnegative_float(tol, "tol")
    if scale is not None:
        scale = _nonnegative_float(scale, "scale")
        if scale <= 0.0:
            raise ValueError("scale must be positive when supplied")
    fit = _mask(fit_mask, y.size, "fit_mask")
    valid = fit.copy()
    valid[: taps - 1] = False
    if not np.any(valid):
        raise ValueError("fit_mask selects no samples with enough FIR history")

    design = _tapped_design(x, taps)
    a = design[valid]
    target = y[valid]
    ridge_eye = ridge * np.eye(a.shape[1], dtype=np.complex128) if ridge > 0.0 else None

    def _solve(weights: np.ndarray) -> np.ndarray:
        root = np.sqrt(weights)[:, np.newaxis]
        aw = a * root
        normal = aw.conj().T @ aw
        if ridge_eye is not None:
            normal = normal + ridge_eye
        rhs = aw.conj().T @ (target * np.sqrt(weights))
        return np.linalg.solve(normal, rhs)

    weights = np.ones(a.shape[0], dtype=np.float64)
    coeff = _solve(weights)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        residual = target - a @ coeff
        weights, _ = _huber_weights(residual, huber_delta, scale)
        new_coeff = _solve(weights)
        change = np.linalg.norm(new_coeff - coeff)
        coeff = new_coeff
        if change <= tol * (np.linalg.norm(coeff) + 1.0e-30):
            break

    full_weights = np.ones(y.size, dtype=np.float64)
    full_weights[valid] = weights
    return coeff.reshape(x.shape[0], taps), full_weights, fit, iterations


def fit_robust_fir(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray,
    *,
    taps: int = 1,
    ridge: float = 0.0,
    huber_delta: float = 1.345,
    max_iter: int = 25,
    tol: float = 1.0e-6,
    scale: float | None = None,
) -> LinearCancellerModel:
    """Fit an outlier-robust FIR canceller by Huber IRLS on baseline samples.

    This is the impulsive-pickup counterpart of :func:`fit_gated_ridge_fir`. The
    fitted transfer function minimises a Huber loss instead of squared error, so
    rare, large switching transients in the fit residual are down-weighted rather
    than allowed to dominate the normal equations. ``huber_delta`` is the
    transition point in units of the robust residual scale (MAD-based by default;
    override with ``scale``); smaller values reject more aggressively.

    Returns the same :class:`LinearCancellerModel` as the L2 fit, so it can be
    applied to fresh records with :meth:`LinearCancellerModel.apply`.
    """

    coeff, _, fit, _ = _fit_robust_fir(
        primary,
        references,
        fit_mask,
        taps=taps,
        ridge=ridge,
        huber_delta=huber_delta,
        max_iter=max_iter,
        tol=tol,
        scale=scale,
    )
    return LinearCancellerModel(
        coefficients=coeff, taps=coeff.shape[1], ridge=float(ridge), fit_mask=fit
    )


def robust_fir_canceller(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray,
    *,
    taps: int = 1,
    ridge: float = 0.0,
    huber_delta: float = 1.345,
    max_iter: int = 25,
    tol: float = 1.0e-6,
    scale: float | None = None,
) -> CancellationResult:
    """Fit and apply a Huber-IRLS robust FIR canceller.

    The returned :class:`CancellationResult` carries ``fit_weights`` (the final
    per-sample IRLS weights, ``1.0`` outside the fit set) and ``iterations``, so
    callers can see which samples were treated as impulsive outliers.
    """

    coeff, full_weights, fit, iterations = _fit_robust_fir(
        primary,
        references,
        fit_mask,
        taps=taps,
        ridge=ridge,
        huber_delta=huber_delta,
        max_iter=max_iter,
        tol=tol,
        scale=scale,
    )
    model = LinearCancellerModel(
        coefficients=coeff, taps=coeff.shape[1], ridge=float(ridge), fit_mask=fit
    )
    result = model.apply(primary, references)
    return CancellationResult(
        cleaned=result.cleaned,
        predicted=result.predicted,
        coefficients=result.coefficients,
        fit_mask=fit,
        fit_weights=full_weights,
        iterations=iterations,
    )


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


def _soft_threshold(values: np.ndarray, penalty: float) -> np.ndarray:
    """Complex soft-threshold (vector shrinkage) toward zero by ``penalty``."""

    magnitude = np.abs(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(magnitude > penalty, 1.0 - penalty / magnitude, 0.0)
    return values * scale


def sparse_reference_canceller(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray,
    *,
    sparse_penalty: float,
    taps: int = 1,
    ridge: float = 0.0,
    max_iter: int = 50,
    tol: float = 1.0e-6,
) -> CancellationResult:
    """Split the primary into reference-explained RFI plus a sparse impulse term.

    The robust FIR (:func:`robust_fir_canceller`) keeps impulsive outliers from
    biasing the fitted transfer, but the impulses themselves remain in the
    cleaned output. This canceller instead *models* them: it minimises

        ||y - D h - s||^2 (on the fit set) + ridge ||h||^2 + sparse_penalty ||s||_1

    by alternating a ridge least-squares update of the FIR coefficients ``h``
    (fitted on ``fit_mask`` with ``s`` removed, so the coherent transfer stays
    unbiased) with a soft-threshold update of the sparse impulse estimate ``s``
    over the whole record. ``sparse_penalty`` sets the impulse threshold: choose
    it above the NQR echo amplitude and below the switching-transient amplitude,
    so the bursts are captured while the signal passes through.

    ``cleaned`` removes both the coherent RFI ``D h`` and the impulses ``s``;
    ``sparse_component`` returns the estimated impulse train, and ``predicted``
    is their sum.
    """

    y, x = _validate_xy(primary, references)
    taps = _positive_int(taps, "taps")
    ridge = _nonnegative_float(ridge, "ridge")
    sparse_penalty = _nonnegative_float(sparse_penalty, "sparse_penalty")
    if sparse_penalty <= 0.0:
        raise ValueError("sparse_penalty must be positive")
    max_iter = _positive_int(max_iter, "max_iter")
    tol = _nonnegative_float(tol, "tol")
    fit = _mask(fit_mask, y.size, "fit_mask")
    valid = fit.copy()
    valid[: taps - 1] = False
    if not np.any(valid):
        raise ValueError("fit_mask selects no samples with enough FIR history")

    design = _tapped_design(x, taps)
    a = design[valid]
    normal = a.conj().T @ a
    if ridge > 0.0:
        normal = normal + ridge * np.eye(normal.shape[0], dtype=np.complex128)

    sparse = np.zeros(y.size, dtype=np.complex128)
    coeff = np.zeros(x.shape[0] * taps, dtype=np.complex128)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        rhs = a.conj().T @ (y - sparse)[valid]
        new_coeff = np.linalg.solve(normal, rhs)
        residual = y - design @ new_coeff
        new_sparse = _soft_threshold(residual, sparse_penalty)
        change = np.linalg.norm(new_coeff - coeff) + np.linalg.norm(new_sparse - sparse)
        coeff = new_coeff
        sparse = new_sparse
        if change <= tol * (np.linalg.norm(coeff) + np.linalg.norm(sparse) + 1.0e-30):
            break

    coherent = design @ coeff
    predicted = coherent + sparse
    return CancellationResult(
        cleaned=y - predicted,
        predicted=predicted,
        coefficients=coeff.reshape(x.shape[0], taps),
        fit_mask=fit,
        sparse_component=sparse,
        iterations=iterations,
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


def _segment_starts(num_samples: int, length: int, hop: int) -> list[int]:
    if length > num_samples:
        raise ValueError("segment_length must not exceed the record length")
    starts = list(range(0, num_samples - length + 1, hop))
    last = num_samples - length
    if not starts:
        starts = [0]
    elif starts[-1] != last:
        starts.append(last)
    return starts


def frequency_domain_canceller(
    primary: np.ndarray,
    references: np.ndarray,
    fit_mask: np.ndarray | None = None,
    *,
    segment_length: int = 256,
    hop: int | None = None,
    ridge: float = 0.0,
    sample_rate_hz: float = 1.0,
) -> CancellationResult:
    """Cancel RFI with a per-frequency multi-reference Wiener transfer function.

    Where the FIR cancellers estimate a short time-domain transfer, this fits a
    complex transfer ``W_k(f)`` in every DFT bin from averaged cross-spectra and
    applies it by weighted overlap-add (WOLA). It suits interference whose
    reference-to-primary coupling is strongly frequency dependent (resonant
    probes, long impulse responses), where a compact FIR would need many taps.

    Cross-spectra are accumulated over Hann-windowed segments that lie entirely
    inside ``fit_mask`` (normally the baseline set ``B``); the NQR signal is
    uncorrelated with the references, so leaving it in only adds estimator
    variance, but gating removes it. The per-bin solution is
    ``W(f) = (S_xx(f) + ridge I)^-1 s_xy(f)`` and the predicted RFI spectrum is
    ``W(f)^H X(f)``. The returned result carries ``transfer_function`` (shape
    ``(K, segment_length)``), the multiple-``coherence`` spectrum in ``[0, 1]``
    (the fraction of primary power the references explain at each frequency), and
    the ``frequencies`` grid.

    Application is exact for a pass-through (references equal to the primary) but
    only approximate for a non-trivial ``W`` when the coupling impulse response
    approaches ``segment_length`` (per-segment circular convolution), so choose
    ``segment_length`` well above the expected impulse-response length. The first
    and last ``segment_length`` samples are tapered by the overlap-add windows;
    score results and downstream estimates on the interior.
    """

    y, x = _validate_xy(primary, references)
    length = _positive_int(segment_length, "segment_length")
    ridge = _nonnegative_float(ridge, "ridge")
    sample_rate_hz = float(sample_rate_hz)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")
    step = length // 2 if hop is None else _positive_int(hop, "hop")
    channels = x.shape[0]
    window = np.hanning(length + 1)[:-1] if length > 1 else np.ones(1)
    fit = _mask(fit_mask, y.size, "fit_mask")
    starts = _segment_starts(y.size, length, step)

    s_xx = np.zeros((length, channels, channels), dtype=np.complex128)
    s_xy = np.zeros((length, channels), dtype=np.complex128)
    s_yy = np.zeros(length, dtype=np.float64)
    fitted_segments = 0
    for start in starts:
        stop = start + length
        if not np.all(fit[start:stop]):
            continue
        xf = np.fft.fft(x[:, start:stop] * window, axis=1)  # (K, L)
        yf = np.fft.fft(y[start:stop] * window)  # (L,)
        s_xx += np.einsum("kf,jf->fkj", xf, np.conj(xf))
        s_xy += (xf * np.conj(yf)[np.newaxis, :]).T
        s_yy += np.abs(yf) ** 2
        fitted_segments += 1
    if fitted_segments == 0:
        raise ValueError("fit_mask contains no full segment for spectral estimation")

    eye = np.eye(channels, dtype=np.complex128)
    transfer = np.zeros((length, channels), dtype=np.complex128)
    coherence = np.zeros(length, dtype=np.float64)
    for bin_index in range(length):
        loaded = s_xx[bin_index] + ridge * eye
        weights = np.linalg.solve(loaded, s_xy[bin_index])
        transfer[bin_index] = weights
        if s_yy[bin_index] > 0.0:
            explained = float(np.real(np.vdot(s_xy[bin_index], weights)))
            coherence[bin_index] = min(max(explained / s_yy[bin_index], 0.0), 1.0)

    numerator = np.zeros(y.size, dtype=np.complex128)
    denominator = np.zeros(y.size, dtype=np.float64)
    window_energy = window * window
    for start in starts:
        stop = start + length
        xf = np.fft.fft(x[:, start:stop] * window, axis=1)  # (K, L)
        predicted_spectrum = np.einsum("fk,kf->f", np.conj(transfer), xf)
        segment = np.fft.ifft(predicted_spectrum)
        numerator[start:stop] += window * segment
        denominator[start:stop] += window_energy
    predicted = np.zeros(y.size, dtype=np.complex128)
    valid = denominator > 0.0
    predicted[valid] = numerator[valid] / denominator[valid]

    frequencies = np.fft.fftfreq(length, d=1.0 / sample_rate_hz)
    return CancellationResult(
        cleaned=y - predicted,
        predicted=predicted,
        coefficients=transfer.T,
        fit_mask=fit,
        transfer_function=transfer.T,
        coherence=coherence,
        frequencies=frequencies,
    )
