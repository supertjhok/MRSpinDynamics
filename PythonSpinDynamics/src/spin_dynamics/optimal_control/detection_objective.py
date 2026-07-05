"""Detector-aware GRAPE: optimize a pulse for a non-inductive flux readout.

The PGSE objective (:mod:`~spin_dynamics.optimal_control.diffusion`) scores the
detected echo through a tuned-probe :class:`ReceiverResponse` over a *scalar*
noise floor -- the inductive-coil picture. A SQUID or OPM is a field sensor with
a **frequency-dependent field-noise floor** ``N^2(f)`` (flat for a SQUID, a
Lorentzian band-pass for an RF-OPM) and no ``dPhi/dt`` weighting. This module
scores the same acquired echo with the field-referred matched filter

    ``SNR = sqrt(integral |S(f)|^2 / N^2(f) df)``   (:func:`detected_field_snr_jax`),

so the optimizer shapes the pulse to place signal where the *detector* is quiet.
The detector's ``N^2(f)`` is a fixed array over the acquisition frequency grid --
it does not depend on the controls -- so it is precomputed once with NumPy
(:func:`detector_noise_grid`, sidestepping any non-JAX detector internals) and
passed as a static weight into the differentiable objective. With a flat
``N^2 = sigma^2`` grid this reproduces
:func:`~spin_dynamics.optimal_control.diffusion.detected_echo_snr` with
``noise=sigma`` (Parseval), so it is a strict generalization.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE
from spin_dynamics.optimal_control.control_response import build_control_delivery
from spin_dynamics.optimal_control.diffusion import detected_echo_signal

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    from spin_dynamics.optimal_control._jax_propagation import propagate_batched_controls


def _require_jax() -> None:
    if not JAX_AVAILABLE:
        raise ImportError(
            "detector-aware optimal control requires the optional 'jax' extra. "
            "Install it with `python -m pip install -e .[jax]` (or `.[perf]`)."
        )


def detector_noise_grid(detector, n_acq, dt_acq, *, center_hz=None) -> np.ndarray:
    """Field-noise PSD ``N^2(f)`` of a detector on the acquisition FFT grid.

    Samples ``detector.field_noise_psd`` on ``fftfreq(n_acq, dt_acq)`` shifted by
    ``center_hz`` -- the detector's response centre in the rotating frame, so a
    band-limited detector (an RF-OPM tuned to the line) lands its pass-band at
    baseband ``0``. ``center_hz`` defaults to ``detector.center_hz`` if present
    (the OPM carrier) else ``0``. Returns a length-``n_acq`` positive array.
    """

    if center_hz is None:
        center_hz = float(getattr(detector, "center_hz", 0.0))
    freqs = np.fft.fftfreq(int(n_acq), float(dt_acq)) + float(center_hz)
    n2 = np.asarray(detector.field_noise_psd(freqs), dtype=np.float64).reshape(-1)
    if n2.size != int(n_acq):
        raise ValueError("detector.field_noise_psd must return one value per frequency")
    if np.any(~np.isfinite(n2)) or np.any(n2 <= 0):
        raise ValueError("detector field-noise PSD must be finite and positive")
    return n2


def detected_field_snr_jax(signal, dt_acq, noise_psd_grid, *, xp=None):
    """Field-referred matched-filter SNR of a baseband echo against ``N^2(f)``.

    ``noise_psd_grid`` is ``N^2`` on the ``fftfreq(len(signal), dt_acq)`` grid
    (see :func:`detector_noise_grid`). Uses ``Re(S conj S)`` (NaN-gradient-safe)
    and is normalized so a flat ``N^2 = sigma^2`` reproduces
    ``detected_echo_snr(signal, dt_acq, noise=sigma)`` exactly (Parseval).
    """

    xp = xp if xp is not None else (jnp if JAX_AVAILABLE else np)
    s = xp.asarray(signal)
    n = s.shape[0]
    spectrum = xp.fft.fft(s)
    power = xp.real(spectrum * xp.conj(spectrum))  # |S(f)|^2, gradient-safe at 0
    snr2 = xp.sum(power / noise_psd_grid) * (dt_acq / n)
    return xp.sqrt(snr2)


def make_detected_snr_objective(
    *,
    drift_batch: Sequence[np.ndarray],
    hx_batch: Sequence[np.ndarray],
    hy_batch: Sequence[np.ndarray],
    psi0: np.ndarray,
    detection_operator: np.ndarray,
    weights: np.ndarray,
    offsets_hz: np.ndarray,
    dt: float,
    n_segments: int,
    amplitude_template: np.ndarray,
    acquisition_points: int,
    acquisition_dt: float,
    detector=None,
    noise_psd_grid: np.ndarray | None = None,
    noise_center_hz: float | None = None,
    optimize_amplitude: bool = False,
    per_rf_power: bool = False,
    rf_response=None,
    propagator: str = "expm",
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Build a ``value_and_grad`` maximizing detector-referred detected SNR.

    Optimizes the RF pulse (phase only, or amplitude+phase if
    ``optimize_amplitude``) over a ``(B0, B1)`` ensemble so the acquired echo has
    maximum SNR through ``detector`` (its ``N^2(f)`` sets the frequency weighting;
    pass ``noise_psd_grid`` directly to override, or neither for a flat floor).
    ``x`` is ``phase[n]`` (default) or ``concat([amplitude[n], phase[n]])`` when
    ``optimize_amplitude``; amplitude bounds belong on the command via
    :mod:`~spin_dynamics.optimal_control.parameterization`. No gradient channel --
    this is the excitation/refocusing analogue of the PGSE objective.

    With ``per_rf_power`` the score is detected SNR **per unit RF amplitude**,
    ``SNR / sqrt(integral |B1(t)|^2 dt)`` -- the efficiency figure of merit. This
    is what makes the readout's noise shape the pulse: raw SNR always rewards more
    excitation (out-of-band signal never hurts), whereas per-power SNR penalizes
    spending amplitude where the detector is noisy, so a band-limited detector
    concentrates the excitation in-band and a flat one spreads it.
    """

    _require_jax()
    n_segments = int(n_segments)
    n_acq = int(acquisition_points)

    drift = jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in drift_batch])
    hx = jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in hx_batch])
    hy = jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in hy_batch])
    psi0_j = jnp.asarray(psi0, dtype=jnp.complex128)
    psi0_batch = jnp.broadcast_to(psi0_j, (drift.shape[0],) + psi0_j.shape)
    detect = jnp.asarray(detection_operator, dtype=jnp.complex128)
    weights_j = jnp.asarray(weights, dtype=jnp.float64)
    offsets_j = jnp.asarray(offsets_hz, dtype=jnp.float64)
    amp_template = jnp.asarray(amplitude_template, dtype=jnp.float64)

    if noise_psd_grid is not None:
        grid = np.asarray(noise_psd_grid, dtype=np.float64).reshape(-1)
        if grid.size != n_acq or np.any(grid <= 0) or np.any(~np.isfinite(grid)):
            raise ValueError("noise_psd_grid must be length acquisition_points and positive")
    elif detector is not None:
        grid = detector_noise_grid(detector, n_acq, acquisition_dt, center_hz=noise_center_hz)
    else:
        grid = np.ones(n_acq, dtype=np.float64)
    noise_grid_j = jnp.asarray(grid, dtype=jnp.float64)

    deliver, _n_fine, _dt_fine = build_control_delivery(n_segments, dt, rf_response=rf_response)
    zeros_grad = jnp.zeros(n_segments, dtype=jnp.float64)

    def _coherence(psi):
        return jnp.vdot(psi, detect @ psi)

    def score(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        if optimize_amplitude:
            amplitude_cmd = x[:n_segments]
            phase = x[n_segments : 2 * n_segments]
        else:
            amplitude_cmd = amp_template
            phase = x[:n_segments]
        amplitude, phase_f, _grad_f, dt_eff = deliver(amplitude_cmd, phase, zeros_grad)

        psi_finals = propagate_batched_controls(
            drift, hx, hy, amplitude, phase_f, dt_eff, psi0_batch, method=propagator
        )
        coherences = jax.vmap(_coherence)(psi_finals)
        _tau, signal = detected_echo_signal(
            coherences, weights_j, offsets_j, n_acq=n_acq, dt_acq=float(acquisition_dt)
        )
        snr = detected_field_snr_jax(signal, float(acquisition_dt), noise_grid_j, xp=jnp)
        if per_rf_power:
            # SNR per unit RF amplitude: divide by the delivered pulse's B1 norm.
            # Floor keeps the gradient finite at zero amplitude (a no-op otherwise).
            rf_energy = jnp.sum(amplitude * amplitude) * dt_eff
            snr = snr / jnp.sqrt(rf_energy + 1e-30)
        return snr

    _vg = jax.jit(jax.value_and_grad(score))

    def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = _vg(jnp.asarray(x, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=np.float64)

    return value_and_grad
