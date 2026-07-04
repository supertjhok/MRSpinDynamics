"""Hardware response (probe + gradient-driver) as differentiable LTI filters.

GRAPE (:mod:`optimal_control.objectives`) optimizes an *ideal* piecewise-
constant control and assumes the spins see it verbatim. Real hardware does not
deliver it: a tuned/matched RF probe has finite bandwidth (loaded ``Q``) and a
ring-down tail, and a gradient driver has a finite ``L/R`` slew plus eddy-
current droop. Each is, to excellent approximation, a **linear time-invariant
(LTI) filter** between the *commanded* waveform and the field the spins see.

This module models those filters as one differentiable stage that the objective
inserts between the command and the Hamiltonian (see the design note
``docs/optimal_control_hardware_response.md``). Because the whole chain is JAX,
autodiff supplies the filter's adjoint for free -- no hand-derived circuit
gradient. Box constraints stay on the *command* (what the amplifier produces),
so the optimizer pre-emphasizes only up to real hardware limits.

Two families, matching the physics and the chosen numerics:

* **RF probe** -> :class:`EnvelopeResponse` subclasses act on the complex control
  *envelope* in the rotating frame via an **FFT multiply** by the baseband-
  equivalent transfer function ``H_env(f)``. Tuned and matched probes have
  genuinely different dynamics (single resonance vs. a matching network) and are
  separate classes, per :class:`TunedProbeResponse` /
  :class:`MatchedProbeResponse` / :class:`UntunedProbeResponse`.
* **Gradient driver** -> :class:`GradientDriverResponse` acts on the (real)
  gradient waveform via a causal **state-space** cascade: one ``L/R`` slew pole
  and a shelf per eddy-current term, so the post-pulse eddy tail is explicit.

Both upsample the coarse GRAPE grid by ``oversample`` before filtering
(distortion + interpolation, Motzoi/Gambetta/Wilhelm) and append a ring-down /
eddy **tail** (:class:`TailSpec`) so the tail's rotation of the spins is scored.

:class:`ReceiverResponse` is the output-side mirror: an LTI filter (resampled
onto the caller's frequency grid) for objectives defined on the *detected*
signal rather than the magnetization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import pi

import numpy as np

try:  # pragma: no cover - exercised by environment, not logic
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tail policy (ring-down / eddy window)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailSpec:
    """How long to propagate past the commanded window for the ring-down/eddy tail.

    Two policies, selected by which field is set (the switch the design note
    resolves): ``window_seconds`` gives an explicit acquisition/dead-time window;
    otherwise the tail is ``factor`` times the response's dominant time constant
    (default -- long enough that the ring-down/eddy transient has decayed, which
    for the FFT-based RF filter also suppresses circular-convolution wrap-around).
    """

    factor: float = 5.0
    window_seconds: float | None = None

    def seconds(self, dominant_tau: float) -> float:
        """Resolve the tail length in seconds for a given dominant time constant."""

        if self.window_seconds is not None:
            if self.window_seconds < 0:
                raise ValueError("window_seconds must be non-negative")
            return float(self.window_seconds)
        if self.factor < 0:
            raise ValueError("factor must be non-negative")
        return float(self.factor) * float(dominant_tau)


# ---------------------------------------------------------------------------
# RF probe: envelope FFT-multiply responses
# ---------------------------------------------------------------------------


class EnvelopeResponse:
    """Base class for an RF probe modelled as an LTI filter on the control envelope.

    Subclasses define :meth:`transfer`, the baseband-equivalent (rotating-frame)
    transfer function ``H_env(f)`` acting on the complex envelope
    ``u = amplitude * exp(i*phase)``. :meth:`apply` upsamples nothing itself --
    the objective handles oversampling and the zero tail -- it just multiplies in
    the frequency domain (linear convolution, provided the caller's zero tail
    covers the ring-down so the circular FFT does not wrap).
    """

    #: Piecewise-constant upsampling factor applied before filtering.
    oversample: int = 8
    #: Tail policy for the ring-down window.
    tail: TailSpec = TailSpec()
    #: Dominant time constant (seconds) -- sets the tail length and cost.
    dominant_tau: float = 0.0

    def transfer(self, freqs, *, xp=np):
        """Return ``H_env(freqs)`` (complex), evaluated with backend ``xp``."""

        raise NotImplementedError

    def apply(self, envelope, dt, *, xp=np):
        """Filter a complex control envelope sampled uniformly at spacing ``dt``.

        ``envelope`` is the (already upsampled, zero-tailed) complex envelope on
        the fine grid; ``dt`` its sample spacing. Returns the delivered complex
        envelope, same length. Uses ``xp.fft`` so the JAX path is differentiable.
        """

        n = envelope.shape[0]
        freqs = xp.fft.fftfreq(n, d=dt)
        h = self.transfer(freqs, xp=xp)
        return xp.fft.ifft(xp.fft.fft(envelope) * h)


@dataclass(frozen=True)
class TunedProbeResponse(EnvelopeResponse):
    """Single-resonance (tuned) probe: a one-pole envelope lowpass with ring-down.

    A series/parallel RLC probe is a 2nd-order bandpass at ``f0``; its baseband
    equivalent on the envelope is a **single complex pole** with ring-down time
    constant ``tau = 2*Q/omega0 = Q/(pi*f0) = 2*L/R``:

    ``H_env(f) = 1 / (1 + i*2*pi*tau*(f - detuning_hz))``

    On resonance (``detuning_hz = 0``) the DC gain is 1 and the response rolls
    off with cutoff ``1/(2*pi*tau)``. A non-zero ``detuning_hz`` (probe resonance
    minus RF carrier) moves the pole off the carrier: the on-carrier component is
    then attenuated and phase-shifted, exactly the mistuned-probe behaviour.
    """

    tau: float
    detuning_hz: float = 0.0
    oversample: int = 8
    tail: TailSpec = TailSpec()

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if int(self.oversample) < 1:
            raise ValueError("oversample must be >= 1")
        object.__setattr__(self, "oversample", int(self.oversample))
        object.__setattr__(self, "dominant_tau", float(self.tau))

    @classmethod
    def from_quality_factor(
        cls, *, f0_hz: float, quality_factor: float, carrier_hz: float | None = None, **kw
    ) -> "TunedProbeResponse":
        """Build from resonance ``f0``, loaded ``Q`` and (optional) RF carrier.

        ``tau = Q/(pi*f0)``; ``detuning_hz = f0 - carrier`` (0 when the carrier
        sits on resonance, the default).
        """

        if f0_hz <= 0 or quality_factor <= 0:
            raise ValueError("f0_hz and quality_factor must be positive")
        tau = quality_factor / (pi * f0_hz)
        detuning = 0.0 if carrier_hz is None else (f0_hz - carrier_hz)
        return cls(tau=tau, detuning_hz=detuning, **kw)

    @classmethod
    def from_circuit(
        cls, *, inductance_h: float, resistance_ohm: float, capacitance_f: float,
        carrier_hz: float | None = None, **kw
    ) -> "TunedProbeResponse":
        """Build from series-RLC component values (``tau = 2L/R``, ``f0 = 1/2pi*sqrt(LC)``)."""

        if inductance_h <= 0 or resistance_ohm <= 0 or capacitance_f <= 0:
            raise ValueError("L, R, C must be positive")
        tau = 2.0 * inductance_h / resistance_ohm
        f0 = 1.0 / (2.0 * pi * np.sqrt(inductance_h * capacitance_f))
        detuning = 0.0 if carrier_hz is None else (f0 - carrier_hz)
        return cls(tau=tau, detuning_hz=detuning, **kw)

    def transfer(self, freqs, *, xp=np):
        return 1.0 / (1.0 + 1j * (2.0 * pi * self.tau) * (freqs - self.detuning_hz))


@dataclass(frozen=True)
class UntunedProbeResponse(EnvelopeResponse):
    """Untuned (low-``Q``, broadband) probe: a near-flat envelope with mild phase.

    A series ``RL`` coil, ``H(s) = 1/(R + sL)``. Around the carrier the envelope
    response is ``H_env(f) = 1 / (1 + i*2*pi*f * L/z0)`` with
    ``z0 = R + i*omega_carrier*L`` (normalized to unity at the carrier). Because
    ``omega_carrier*L >> R`` at NMR frequencies the effective time constant is
    tiny, so magnitude is nearly flat -- the model correctly predicts minimal
    distortion, and is here mainly for A/B comparison against a tuned probe.
    """

    inductance_h: float
    resistance_ohm: float
    carrier_hz: float
    oversample: int = 4
    tail: TailSpec = TailSpec()

    def __post_init__(self) -> None:
        if self.inductance_h <= 0 or self.resistance_ohm <= 0 or self.carrier_hz <= 0:
            raise ValueError("inductance_h, resistance_ohm, carrier_hz must be positive")
        object.__setattr__(self, "oversample", int(self.oversample))
        z0 = self.resistance_ohm + 1j * 2.0 * pi * self.carrier_hz * self.inductance_h
        tau_c = self.inductance_h / z0  # complex effective time constant
        object.__setattr__(self, "_tau_complex", tau_c)
        object.__setattr__(self, "dominant_tau", float(abs(tau_c)))

    def transfer(self, freqs, *, xp=np):
        return 1.0 / (1.0 + 1j * (2.0 * pi) * freqs * self._tau_complex)


@dataclass(frozen=True)
class MatchedProbeResponse(EnvelopeResponse):
    """Impedance-matched probe: a two-pole envelope, distinct from a tuned probe.

    A matched probe adds a matching network (the ``matched`` circuit model is a
    3rd-order system) on top of the tuned coil. Near band it is **not** the tuned
    probe's single pole: the network contributes an extra pole that changes the
    wideband roll-off and, importantly, the group delay / phase. We model the
    dominant envelope response as the loaded resonance pole in series with the
    extra network pole:

    ``H_env(f) = norm / [(1 + i*2*pi*tau_main*(f - detuning_hz))
                         * (1 + i*2*pi*tau_extra*(f - extra_detuning_hz))]``

    normalized to unity at the main resonance. Set ``tau_extra`` from the
    matching network's second pole; the general two-pole form also accepts the
    poles directly. For a single-resonance probe use :class:`TunedProbeResponse`.
    """

    tau_main: float
    tau_extra: float
    detuning_hz: float = 0.0
    extra_detuning_hz: float = 0.0
    oversample: int = 8
    tail: TailSpec = TailSpec()

    def __post_init__(self) -> None:
        if self.tau_main <= 0 or self.tau_extra <= 0:
            raise ValueError("tau_main and tau_extra must be positive")
        object.__setattr__(self, "oversample", int(self.oversample))
        object.__setattr__(self, "dominant_tau", float(max(self.tau_main, self.tau_extra)))
        # Normalize so the main resonance passes unity (the extra pole otherwise
        # attenuates it): gain = value of the extra-pole factor at f = detuning.
        gain = 1.0 + 1j * (2.0 * pi * self.tau_extra) * (self.detuning_hz - self.extra_detuning_hz)
        object.__setattr__(self, "_gain", gain)

    @classmethod
    def from_loaded_q(
        cls, *, f0_hz: float, loaded_q: float, carrier_hz: float | None = None,
        extra_pole_hz: float | None = None, **kw
    ) -> "MatchedProbeResponse":
        """Build from resonance ``f0``, loaded ``Q`` and an optional extra-pole corner.

        ``tau_main = loaded_q/(pi*f0)`` (like a tuned probe at the *loaded* Q);
        ``extra_pole_hz`` sets the matching network's second corner
        (``tau_extra = 1/(2*pi*extra_pole_hz)``), defaulting to ~5x the main
        bandwidth away (a mild, physically typical extra roll-off) when omitted.
        """

        if f0_hz <= 0 or loaded_q <= 0:
            raise ValueError("f0_hz and loaded_q must be positive")
        tau_main = loaded_q / (pi * f0_hz)
        main_bw = 1.0 / (2.0 * pi * tau_main)
        corner = 5.0 * main_bw if extra_pole_hz is None else float(extra_pole_hz)
        tau_extra = 1.0 / (2.0 * pi * corner)
        detuning = 0.0 if carrier_hz is None else (f0_hz - carrier_hz)
        return cls(tau_main=tau_main, tau_extra=tau_extra, detuning_hz=detuning, **kw)

    def transfer(self, freqs, *, xp=np):
        main = 1.0 + 1j * (2.0 * pi * self.tau_main) * (freqs - self.detuning_hz)
        extra = 1.0 + 1j * (2.0 * pi * self.tau_extra) * (freqs - self.extra_detuning_hz)
        return self._gain / (main * extra)


# ---------------------------------------------------------------------------
# Gradient driver: state-space (RL + eddy) response
# ---------------------------------------------------------------------------


def _first_order_lowpass_numpy(x, a):
    """Causal one-pole lowpass ``y[n] = a*y[n-1] + (1-a)*x[n]`` (ZOH, DC gain 1)."""

    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    prev = 0.0
    one_minus_a = 1.0 - a
    for i in range(x.shape[0]):
        prev = a * prev + one_minus_a * x[i]
        y[i] = prev
    return y


@dataclass(frozen=True)
class GradientDriverResponse:
    """Gradient driver: an ``L/R`` slew pole in series with eddy-current shelves.

    The delivered gradient is the command through one slew pole
    ``1/(1 + s*tau_rl)`` followed by, per eddy term ``k``, a first-order shelf
    ``(1 - alpha_k) + alpha_k/(1 + s*tau_k)``. Each shelf has DC gain 1, so the
    steady-state gradient equals the command; the transient droop
    ``G(t) = G_cmd*[1 - sum_k alpha_k*e^{-t/tau_k}]`` is the classic MRI eddy
    model. Implemented as a causal state-space cascade (exact zero-order-hold
    per sample), so the post-command eddy tail is represented directly and the
    filter is differentiable through :func:`jax.lax.scan`.
    """

    tau_rl: float
    eddy_terms: tuple[tuple[float, float], ...] = ()
    oversample: int = 8
    tail: TailSpec = TailSpec()

    def __post_init__(self) -> None:
        if self.tau_rl <= 0:
            raise ValueError("tau_rl must be positive")
        for alpha, tau in self.eddy_terms:
            if tau <= 0:
                raise ValueError("every eddy tau must be positive")
            if not (0.0 <= alpha < 1.0):
                raise ValueError("every eddy alpha must be in [0, 1)")
        object.__setattr__(self, "eddy_terms", tuple((float(a), float(t)) for a, t in self.eddy_terms))
        object.__setattr__(self, "oversample", int(self.oversample))
        taus = [self.tau_rl, *(t for _a, t in self.eddy_terms)]
        object.__setattr__(self, "dominant_tau", float(max(taus)))

    def _apply_numpy(self, waveform, dt):
        a_rl = float(np.exp(-dt / self.tau_rl))
        y = _first_order_lowpass_numpy(waveform, a_rl)
        for alpha, tau in self.eddy_terms:
            a = float(np.exp(-dt / tau))
            y = (1.0 - alpha) * y + alpha * _first_order_lowpass_numpy(y, a)
        return y

    def _apply_jax(self, waveform, dt):
        def lowpass(x, a):
            one_minus_a = 1.0 - a

            def body(prev, x_n):
                nxt = a * prev + one_minus_a * x_n
                return nxt, nxt

            _, y = lax.scan(body, jnp.asarray(0.0, dtype=x.dtype), x)
            return y

        a_rl = jnp.exp(-dt / self.tau_rl)
        y = lowpass(waveform, a_rl)
        for alpha, tau in self.eddy_terms:
            a = jnp.exp(-dt / tau)
            y = (1.0 - alpha) * y + alpha * lowpass(y, a)
        return y

    def apply(self, waveform, dt, *, xp=np):
        """Filter a real gradient waveform sampled uniformly at spacing ``dt``."""

        if xp is np:
            return self._apply_numpy(waveform, dt)
        return self._apply_jax(waveform, dt)


def eddy_terms_from_step_response(
    times: Sequence[float],
    response: Sequence[float],
    *,
    n_terms: int = 1,
    steady_value: float | None = None,
) -> tuple[tuple[float, float], ...]:
    """Fit eddy ``(alpha_k, tau_k)`` terms from a measured/simulated step response.

    Given a gradient step response ``G(t)`` (settling towards a steady value),
    fits ``G(t) = G_inf*(1 - sum_k alpha_k*e^{-t/tau_k})`` and returns the
    ``(alpha_k, tau_k)`` pairs for use as :class:`GradientDriverResponse`'s
    ``eddy_terms``. ``G_inf`` is taken from ``steady_value`` or the last sample.
    Whether the fastest fitted pole is physically the ``L/R`` slew (``tau_rl``)
    or an eddy term is the caller's assignment; the fit does not distinguish
    them. Requires SciPy (host-side; not differentiated).
    """

    from scipy.optimize import curve_fit

    t = np.asarray(times, dtype=np.float64).reshape(-1)
    g = np.asarray(response, dtype=np.float64).reshape(-1)
    if t.shape != g.shape or t.size < 2 * n_terms + 1:
        raise ValueError("times and response must match and be long enough for n_terms")
    g_inf = float(g[-1]) if steady_value is None else float(steady_value)
    if g_inf == 0.0:
        raise ValueError("steady value is zero; cannot normalize the step response")
    deficit = 1.0 - g / g_inf  # sum_k alpha_k exp(-t/tau_k), -> 0 as t grows

    span = float(t[-1] - t[0]) or 1.0

    def model(tt, *params):
        out = np.zeros_like(tt)
        for k in range(n_terms):
            alpha = params[2 * k]
            tau = params[2 * k + 1]
            out = out + alpha * np.exp(-tt / tau)
        return out

    p0: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for k in range(n_terms):
        p0 += [max(deficit[0], 1e-3) / n_terms, span / (2 ** (k + 1))]
        lo += [0.0, span * 1e-4]
        hi += [1.0, span * 10.0]
    popt, _ = curve_fit(model, t, deficit, p0=p0, bounds=(lo, hi), maxfev=20000)
    terms = tuple((float(popt[2 * k]), float(popt[2 * k + 1])) for k in range(n_terms))
    return tuple(sorted(terms, key=lambda at: at[1]))


# ---------------------------------------------------------------------------
# Receiver: output-side LTI filter on the detected signal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReceiverResponse:
    """Output-side LTI filter for objectives on the *detected* signal.

    When the objective is the acquired echo/FID (e.g. CPMG SNR-per-time) rather
    than the magnetization or commanded trajectory, the receiver chain band-
    limits what is detected. This wraps its transfer function ``H_rx(f)`` and,
    unlike the transmit probe, is genuinely per-frequency: it is applied to the
    detected signal/spectrum, resampling ``H_rx`` onto the caller's frequency
    grid (:meth:`from_samples`) so the objective and the propagation ensemble
    share one grid.

    Provide either an analytic single-pole receiver (``tau``, ``detuning_hz``)
    or sampled ``(freqs_hz, transfer)`` from a probe's Rx transfer function via
    :meth:`from_samples` (linear interpolation of real/imag parts onto the
    target grid).
    """

    tau: float | None = None
    detuning_hz: float = 0.0
    _sample_freqs: np.ndarray | None = field(default=None, repr=False)
    _sample_tf: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        analytic = self.tau is not None
        sampled = self._sample_freqs is not None
        if analytic == sampled:
            raise ValueError("provide exactly one of an analytic tau or sampled (freqs, tf)")
        if analytic and self.tau <= 0:
            raise ValueError("tau must be positive")
        if sampled:
            f = np.asarray(self._sample_freqs, dtype=np.float64).reshape(-1)
            tf = np.asarray(self._sample_tf, dtype=np.complex128).reshape(-1)
            if f.shape != tf.shape or f.size < 2:
                raise ValueError("sample freqs and tf must match and have >= 2 points")
            order = np.argsort(f)
            object.__setattr__(self, "_sample_freqs", f[order])
            object.__setattr__(self, "_sample_tf", tf[order])

    @classmethod
    def from_samples(cls, freqs_hz, transfer) -> "ReceiverResponse":
        """Build from a sampled transfer function (e.g. ``tuned_probe_rx_tf``)."""

        return cls(_sample_freqs=np.asarray(freqs_hz), _sample_tf=np.asarray(transfer))

    def transfer(self, freqs, *, xp=np):
        """Return ``H_rx(freqs)`` resampled onto ``freqs`` (the caller's grid)."""

        if self.tau is not None:
            return 1.0 / (1.0 + 1j * (2.0 * pi * self.tau) * (freqs - self.detuning_hz))
        f = np.asarray(freqs, dtype=np.float64)
        re = np.interp(f, self._sample_freqs, self._sample_tf.real)
        im = np.interp(f, self._sample_freqs, self._sample_tf.imag)
        h = re + 1j * im
        return xp.asarray(h) if xp is not np else h

    def apply(self, signal, dt, *, xp=np):
        """Filter a detected time-domain signal sampled uniformly at spacing ``dt``."""

        n = signal.shape[0]
        freqs = xp.fft.fftfreq(n, d=dt)
        return xp.fft.ifft(xp.fft.fft(signal) * self.transfer(freqs, xp=xp))


def _uniform_dt_scalar(dt) -> float:
    """Return the scalar segment duration, requiring a uniform grid.

    The hardware-response filters (FFT envelope multiply, state-space eddy
    cascade) live on a uniformly sampled time axis, so a per-segment ``dt``
    array is only accepted when all entries are equal.
    """

    arr = np.asarray(dt, dtype=np.float64).reshape(-1)
    if arr.size == 1 or np.allclose(arr, arr[0]):
        return float(arr[0])
    raise ValueError(
        "hardware response filters (rf_response/gradient_response) require a uniform dt"
    )


def build_control_delivery(n_segments, dt, *, rf_response=None, gradient_response=None):
    """Build the commanded-to-delivered control transform for GRAPE objectives.

    Returns ``(deliver, n_fine, dt_fine)`` where ``deliver(amplitude, phase,
    gradient)`` maps commanded per-segment controls to delivered fine-grid
    controls plus the effective segment duration:
    ``(amplitude_f, phase_f, gradient_f, dt_eff)``. When neither response is
    given, ``deliver`` is the identity (returns the inputs and ``dt`` unchanged),
    ``n_fine == n_segments`` and ``dt_fine is None``.

    Otherwise controls are upsampled by the responses' (max) ``oversample``,
    a ring-down/eddy tail is appended, the RF envelope is filtered by an FFT
    multiply and the gradient by the state-space driver, all on a uniform fine
    grid (requires a scalar/uniform ``dt``). Shared by
    :func:`optimal_control.objectives.make_grape_objective` and the PGSE
    objective in :mod:`optimal_control.diffusion`; requires the ``jax`` extra.
    """

    if not JAX_AVAILABLE:  # pragma: no cover - import guard
        raise ImportError("build_control_delivery requires the optional 'jax' extra")

    responses_active = rf_response is not None or gradient_response is not None
    if not responses_active:
        dt_j = jnp.asarray(dt, dtype=jnp.float64)

        def deliver_identity(amplitude, phase, gradient):
            return amplitude, phase, gradient, dt_j

        return deliver_identity, int(n_segments), None

    dt_scalar = _uniform_dt_scalar(dt)
    active = [r for r in (rf_response, gradient_response) if r is not None]
    oversample = max(int(r.oversample) for r in active)
    dt_fine = dt_scalar / oversample
    tail_seconds = max(r.tail.seconds(r.dominant_tau) for r in active)
    n_tail = int(np.ceil(tail_seconds / dt_fine)) if tail_seconds > 0 else 0
    n_fine = int(n_segments) * oversample + n_tail

    rf_transfer_fine = None
    if rf_response is not None:
        freqs_fine = np.fft.fftfreq(n_fine, d=dt_fine)
        rf_transfer_fine = jnp.asarray(
            rf_response.transfer(freqs_fine, xp=np), dtype=jnp.complex128
        )

    def deliver(amplitude, phase, gradient):
        if rf_response is not None:
            envelope = amplitude * jnp.exp(1j * phase)
            env_fine = jnp.concatenate(
                [jnp.repeat(envelope, oversample), jnp.zeros(n_tail, dtype=jnp.complex128)]
            )
            delivered = jnp.fft.ifft(jnp.fft.fft(env_fine) * rf_transfer_fine)
            # Gradient-safe polar form: |u| and angle(u) both have NaN gradients
            # where the delivered envelope hits zero (e.g. a gapped RF template
            # rung down through the probe). The floor + double-where keep the
            # gradient finite there; for constant-amplitude pulses |u| > 0 always,
            # so this is a no-op to ~1e-15.
            re = jnp.real(delivered)
            im = jnp.imag(delivered)
            mag2 = re * re + im * im
            amplitude_f = jnp.sqrt(mag2 + 1e-30)
            safe = mag2 > 1e-24
            re_safe = jnp.where(safe, re, 1.0)
            im_safe = jnp.where(safe, im, 0.0)
            phase_f = jnp.where(safe, jnp.arctan2(im_safe, re_safe), 0.0)
        else:
            amplitude_f = jnp.concatenate(
                [jnp.repeat(amplitude, oversample), jnp.zeros(n_tail)]
            )
            phase_f = jnp.concatenate([jnp.repeat(phase, oversample), jnp.zeros(n_tail)])
        if gradient is None:
            gradient_f = None
        else:
            grad_fine = jnp.concatenate([jnp.repeat(gradient, oversample), jnp.zeros(n_tail)])
            if gradient_response is not None:
                gradient_f = gradient_response.apply(grad_fine, dt_fine, xp=jnp)
            else:
                gradient_f = grad_fine
        return amplitude_f, phase_f, gradient_f, dt_fine

    return deliver, n_fine, dt_fine


__all__ = [
    "TailSpec",
    "EnvelopeResponse",
    "TunedProbeResponse",
    "UntunedProbeResponse",
    "MatchedProbeResponse",
    "GradientDriverResponse",
    "eddy_terms_from_step_response",
    "ReceiverResponse",
    "build_control_delivery",
]
