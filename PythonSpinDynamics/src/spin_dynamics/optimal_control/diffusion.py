"""PGSE diffusion optimal control: q-vector moments and detected-echo SNR.

The diffusion capstone (design note ``docs/optimal_control_pgse_diffusion.md``)
co-optimizes the RF pulses and the gradient waveform of a pulsed-gradient spin
echo so that, on realistic hardware (a tuned RF probe on transmit and receive, a
gradient driver with ``L/R`` slew + eddy currents, and an inhomogeneous
``B0``/``B1`` field), the sequence delivers the intended diffusion encoding
(**correct q-vector**) and maximizes the **detected echo SNR**.

Three pieces, all differentiable (JAX), built on the M4 hardware-response layer
and the M2 gradient channel:

* :func:`gradient_moments` -- the q-vector ``q(t) = integral g dt`` of the
  *delivered* gradient, its encode-lobe value and its residual at the echo
  (which must return to zero for a static-spin echo), matching the convention of
  ``workflows.pgse.gradient_moment_b_value``.
* :func:`detected_echo_signal` / :func:`detected_echo_snr` -- the coherently
  summed transverse magnetization over the ``(position, B0, B1)`` ensemble,
  acquired over a short window as an isochromat FID, filtered through a
  :class:`~optimal_control.control_response.ReceiverResponse` (the Rx tuned
  probe) and matched-filtered to a detected amplitude over a fixed noise floor.
* :func:`make_pgse_objective` -- the weighted ``value_and_grad`` closure the
  example optimizes: maximize detected SNR minus q-accuracy and refocusing
  penalties.

Diffusion is modelled as a static position ensemble (the gradient dephases and
rephases the position grid); explicit restricted diffusion remains the domain of
the random-walker backends in ``workflows.pgse``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import pi

import numpy as np

from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE
from spin_dynamics.optimal_control.control_response import build_control_delivery

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    from spin_dynamics.optimal_control._jax_propagation import (
        propagate_batched_grad_controls,
    )


def _require_jax() -> None:
    if not JAX_AVAILABLE:
        raise ImportError(
            "PGSE optimal control requires the optional 'jax' extra. "
            "Install it with `python -m pip install -e .[jax]` (or `.[perf]`)."
        )


def gradient_moments(gradient, dt, *, encode_end, coherence_sign=None):
    """q-vector moments of a (delivered) piecewise-constant gradient waveform.

    ``q(t) = integral_0^t g(tau) s(tau) dtau`` accumulated over the fine grid,
    where ``s(t) = +-1`` is the coherence-order sign (``coherence_sign``, same
    length as ``gradient``): a 180 refocusing pulse flips it, so two same-polarity
    physical lobes straddling a 180 refocus (net ``q`` back to zero), matching the
    effective-gradient convention of ``workflows.pgse``. Without
    ``coherence_sign`` the raw gradient is integrated (``s == +1``).

    Returns ``(q_of_t, q_encode, q_residual, b_value)``: ``q_encode = q`` at the
    end of the encode lobe (fine-grid index ``encode_end``), ``q_residual = q`` at
    the end of the waveform (must be ~0 for a static-spin echo -- the eddy tail is
    included, so eddy residual moments are penalized too), and
    ``b_value = integral q(t)^2 dt`` (matches
    ``workflows.pgse.gradient_moment_b_value``'s convention).
    """

    effective = gradient if coherence_sign is None else gradient * coherence_sign
    q = jnp.cumsum(effective) * dt
    q_encode = q[int(encode_end) - 1]
    q_residual = q[-1]
    b_value = jnp.sum(q * q) * dt
    return q, q_encode, q_residual, b_value


def detected_echo_signal(coherences, weights, offsets_hz, *, n_acq, dt_acq):
    """Acquire the coherently-summed echo over a readout window.

    ``coherences[k] = <I+>`` for ensemble member ``k`` at the echo center;
    ``weights[k]`` its (density) weight and ``offsets_hz[k]`` its B0 offset. The
    acquired signal re-dephases each isochromat over a window centered on the
    echo, ``s(tau) = sum_k w_k * c_k * exp(i 2 pi offset_k tau)``, peaking where
    the offsets rephase. Returns ``(tau, s)`` with ``tau`` centered at 0.
    """

    tau = (jnp.arange(n_acq) - (n_acq - 1) / 2.0) * dt_acq
    ramp = jnp.exp(2j * pi * offsets_hz[:, None] * tau[None, :])
    s = jnp.sum((weights * coherences)[:, None] * ramp, axis=0)
    return tau, s


def detected_echo_snr(signal, dt_acq, *, rx_response=None, noise=1.0):
    """Matched-filter detected amplitude of an acquired echo over a noise floor.

    Filters the echo through the Rx probe (``rx_response``, an
    :class:`~optimal_control.control_response.ReceiverResponse`) if given, then
    returns the matched-filter detected amplitude ``sqrt(integral |s|^2 dtau) /
    noise`` -- for the coherent echo the matched-filter peak is its energy, so
    with a fixed noise this is the quantity to maximize. Reduces to the raw
    coherent-echo amplitude when no receiver is supplied.
    """

    s = signal if rx_response is None else rx_response.apply(signal, dt_acq, xp=jnp)
    # |s|^2 as Re(s*conj(s)), NOT jnp.abs(s)**2: the latter has a NaN gradient
    # wherever the echo crosses zero (0 * d|s|/dx), which poisons the optimizer.
    energy = jnp.sum(jnp.real(s * jnp.conj(s))) * dt_acq
    return jnp.sqrt(energy) / noise


def make_pgse_objective(
    *,
    drift_batch: Sequence[np.ndarray],
    hx_batch: Sequence[np.ndarray],
    hy_batch: Sequence[np.ndarray],
    hgrad_batch: Sequence[np.ndarray],
    psi0: np.ndarray,
    detection_operator: np.ndarray,
    weights: np.ndarray,
    offsets_hz: np.ndarray,
    dt: float,
    n_segments: int,
    amplitude_template: np.ndarray,
    gradient_mask: np.ndarray,
    coherence_sign: np.ndarray | None = None,
    encode_end_segment: int,
    q_target: float,
    acquisition_points: int,
    acquisition_dt: float,
    noise: float = 1.0,
    q_weight: float = 1.0,
    refocus_weight: float = 1.0,
    rf_response=None,
    gradient_response=None,
    rx_response=None,
    propagator: str = "expm",
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Build the weighted PGSE ``value_and_grad`` closure.

    Optimizes ``x = concat([phase (n_segments), gradient (n_segments)])`` (RF
    phase-only -- the primary mode for switching-amplifier hardware; broadband
    phase-modulated refocusing, e.g. RP2 / symmetric-phase-alternating pulses,
    is what recovers the echo across a grossly inhomogeneous field -- plus the
    gradient waveform), scoring
    ``detected_SNR - q_weight*(q_encode - q_target)^2 - refocus_weight*q_residual^2``
    over the ``(position, B0, B1)`` ensemble. Each member ``k`` is defined by
    ``drift_batch[k]`` (its B0 offset), ``hx_batch[k]``/``hy_batch[k]`` (its B1
    scale) and ``hgrad_batch[k] = r_k * h_grad`` (its position). The RF envelope
    follows a fixed ``amplitude_template`` (the 90/180 on/off pattern) with the
    optimized phase; the commanded gradient is masked by ``gradient_mask`` (the
    lobe windows) before the driver. ``rf_response``/``gradient_response`` apply
    the Tx probe and gradient driver; ``rx_response`` the Rx probe on the
    acquired echo. ``coherence_sign`` (``+-1`` per segment, flipping at each 180
    pulse) sets the effective-gradient sign for the moment integral so
    same-polarity lobes straddling a 180 refocus. ``propagator`` defaults to
    ``"expm"`` (degeneracy-safe: on-resonance spins have ``H_seg == 0`` during
    RF/gradient gaps, whose ``eigh`` gradient is NaN). Requires the ``jax`` extra
    and a uniform scalar ``dt``.
    """

    _require_jax()
    n_segments = int(n_segments)

    drift = jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in drift_batch])
    hx = jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in hx_batch])
    hy = jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in hy_batch])
    hgrad = jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in hgrad_batch])
    psi0_j = jnp.asarray(psi0, dtype=jnp.complex128)
    psi0_batch = jnp.broadcast_to(psi0_j, (drift.shape[0],) + psi0_j.shape)
    detect = jnp.asarray(detection_operator, dtype=jnp.complex128)
    weights_j = jnp.asarray(weights, dtype=jnp.float64)
    offsets_j = jnp.asarray(offsets_hz, dtype=jnp.float64)
    amp_template = jnp.asarray(amplitude_template, dtype=jnp.float64)
    grad_mask = jnp.asarray(gradient_mask, dtype=jnp.float64)
    q_target = float(q_target)
    sign_cmd = (
        np.ones(n_segments) if coherence_sign is None
        else np.asarray(coherence_sign, dtype=np.float64).reshape(-1)
    )

    deliver, n_fine, _dt_fine = build_control_delivery(
        n_segments, dt, rf_response=rf_response, gradient_response=gradient_response
    )
    oversample = 1 if _dt_fine is None else int(round(dt / _dt_fine))
    encode_end_fine = int(encode_end_segment) * oversample
    # Coherence sign on the fine grid: hold each command segment's sign across its
    # upsampled samples; the ring-down/eddy tail keeps the final (post-180) sign.
    n_tail = n_fine - n_segments * oversample
    sign_fine = jnp.asarray(
        np.concatenate([np.repeat(sign_cmd, oversample), np.full(n_tail, sign_cmd[-1])]),
        dtype=jnp.float64,
    )

    def _coherence(psi):
        return jnp.vdot(psi, detect @ psi)

    def score(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        phase = x[:n_segments]
        gradient_cmd = x[n_segments : 2 * n_segments] * grad_mask
        amplitude, phase_f, gradient_f, dt_eff = deliver(amp_template, phase, gradient_cmd)

        psi_finals = propagate_batched_grad_controls(
            drift, hx, hy, hgrad, amplitude, phase_f, gradient_f, dt_eff, psi0_batch,
            method=propagator,
        )
        coherences = jax.vmap(_coherence)(psi_finals)
        _tau, signal = detected_echo_signal(
            coherences, weights_j, offsets_j,
            n_acq=int(acquisition_points), dt_acq=float(acquisition_dt),
        )
        snr = detected_echo_snr(
            signal, float(acquisition_dt), rx_response=rx_response, noise=float(noise)
        )

        _q, q_encode, q_residual, _b = gradient_moments(
            gradient_f, dt_eff, encode_end=encode_end_fine, coherence_sign=sign_fine
        )
        penalty = (
            q_weight * (q_encode - q_target) ** 2 + refocus_weight * q_residual ** 2
        )
        return snr - penalty

    _vg = jax.jit(jax.value_and_grad(score))

    def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = _vg(jnp.asarray(x, dtype=jnp.float64))
        return float(value), np.asarray(grad)

    return value_and_grad


__all__ = [
    "gradient_moments",
    "detected_echo_signal",
    "detected_echo_snr",
    "make_pgse_objective",
]
