"""Parametric, reference-free RFI trackers.

The reference cancellers in :mod:`~spin_dynamics.interference.cancellers` need a
correlated reference channel. When the interferer is narrowband -- an amplitude-
modulated broadcast carrier, the canonical in-band NQR nuisance -- its structure
can be exploited directly, with no reference at all: model it as one or more
carriers at known (or separately estimated) frequencies whose complex amplitudes
drift slowly, and track those amplitudes with a Kalman filter.

For a real primary each carrier ``f_j`` contributes ``I_j cos(w_j n) -
Q_j sin(w_j n)``; the state is the stacked quadratures ``[I_j, Q_j]`` following a
random walk, and the observation is the (scalar) primary sample. The predicted
interferer is the *a priori* estimate -- it uses only past samples -- so the
current sample's NQR content is never subtracted into the carrier estimate.
Measurement updates can be frozen wherever ``update_mask`` is false (the expected
signal gaps), so the tracker coasts through NQR echoes instead of absorbing them.

``process_std`` sets how fast the amplitudes may drift (larger tracks faster but
admits more noise); ``measurement_std`` is the observation-noise scale. Both are
the usual Kalman trade-off knobs.
"""

from __future__ import annotations

import numpy as np

from spin_dynamics.interference.cancellers import CancellationResult


def _real_vector(primary: np.ndarray) -> np.ndarray:
    arr = np.asarray(primary)
    if np.iscomplexobj(arr) and np.any(np.abs(arr.imag) > 0.0):
        raise ValueError(
            "kalman_harmonic_canceller models real carriers; pass a real primary "
            "channel (e.g. one I/Q component)"
        )
    y = np.real(arr).astype(np.float64).reshape(-1)
    if y.size == 0:
        raise ValueError("primary must contain at least one sample")
    if not np.all(np.isfinite(y)):
        raise ValueError("primary samples must be finite")
    return y


def _positive_float(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def kalman_harmonic_canceller(
    primary: np.ndarray,
    frequencies_hz: np.ndarray | list[float] | tuple[float, ...],
    sample_rate_hz: float,
    *,
    update_mask: np.ndarray | None = None,
    process_std: float = 1.0e-3,
    measurement_std: float = 1.0,
    initial_amplitude_std: float = 1.0,
) -> CancellationResult:
    """Track and subtract drifting narrowband carriers with a Kalman filter.

    ``frequencies_hz`` lists the carrier frequencies to track (reference-free).
    The returned :class:`~spin_dynamics.interference.cancellers.CancellationResult`
    carries ``cleaned`` (primary minus the tracked interferer), ``predicted`` (the
    a-priori interferer estimate), ``coefficients`` (the final complex amplitude
    ``I_j - i Q_j`` per carrier), and ``coefficient_history`` (the complex
    amplitude trajectory, shape ``(N, J)``). Measurement updates are applied only
    where ``update_mask`` is true (default everywhere); elsewhere the filter
    predicts without correcting, so it coasts through expected signal gaps.
    """

    y = _real_vector(primary)
    n = y.size
    sample_rate_hz = _positive_float(sample_rate_hz, "sample_rate_hz")
    process_var = _positive_float(process_std, "process_std") ** 2
    measurement_var = _positive_float(measurement_std, "measurement_std") ** 2
    initial_var = _positive_float(initial_amplitude_std, "initial_amplitude_std") ** 2

    freqs = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    if freqs.size == 0:
        raise ValueError("frequencies_hz must contain at least one carrier")
    if not np.all(np.isfinite(freqs)):
        raise ValueError("frequencies_hz must be finite")
    num_carriers = freqs.size
    dim = 2 * num_carriers

    if update_mask is None:
        update = np.ones(n, dtype=bool)
    else:
        update = np.asarray(update_mask, dtype=bool).reshape(-1)
        if update.size != n:
            raise ValueError("update_mask length must match the primary")

    omega = 2.0 * np.pi * freqs / sample_rate_hz
    times = np.arange(n, dtype=np.float64)
    cosines = np.cos(omega[:, np.newaxis] * times[np.newaxis, :])  # (J, N)
    sines = np.sin(omega[:, np.newaxis] * times[np.newaxis, :])  # (J, N)

    state = np.zeros(dim, dtype=np.float64)
    covariance = initial_var * np.eye(dim, dtype=np.float64)
    process = process_var * np.eye(dim, dtype=np.float64)
    predicted = np.zeros(n, dtype=np.float64)
    history = np.zeros((n, num_carriers), dtype=np.complex128)

    row = np.zeros(dim, dtype=np.float64)
    for k in range(n):
        row[0::2] = cosines[:, k]
        row[1::2] = -sines[:, k]
        # Predict (random-walk transition is the identity).
        covariance = covariance + process
        prior = float(row @ state)
        predicted[k] = prior
        if update[k]:
            gain_num = covariance @ row
            innovation_var = float(row @ gain_num) + measurement_var
            gain = gain_num / innovation_var
            state = state + gain * (y[k] - prior)
            covariance = covariance - np.outer(gain, gain_num)
        history[k] = state[0::2] - 1j * state[1::2]

    amplitudes = state[0::2] - 1j * state[1::2]
    return CancellationResult(
        cleaned=y - predicted,
        predicted=predicted,
        coefficients=amplitudes,
        update_mask=update,
        coefficient_history=history,
        frequencies=freqs,
    )
