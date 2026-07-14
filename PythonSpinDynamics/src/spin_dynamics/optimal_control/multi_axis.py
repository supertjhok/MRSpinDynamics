"""Tri-axial (multi-coil) parametric optimization of the NQR SLSE signal.

Where :mod:`optimal_control.objectives` optimizes a free per-segment RF waveform
on one coil, this module optimizes a *parametric* multi-coil pulse: up to three
orthogonal coils, each contributing one **rectangular** sub-pulse whose
amplitude, phase, start (delay), and length are the free parameters -- for both
the excitation and the refocusing pulse of a spin-lock spin-echo (SLSE) train.
Sequential vs simultaneous excitation is not imposed; it emerges from the fitted
delays and lengths (disjoint sub-pulses = sequential, overlapping = simultaneous).

The objective is the refocused powder SLSE signal (echo-train energy), scored with
the full ``(2I+1)`` density-matrix model and matched detection. Each rectangular
sub-pulse is rendered on a fixed fine time grid through a smooth (sigmoid) window,
so the delay/length parameters are differentiable and the whole objective is a
single ``jax.value_and_grad`` closure driving the existing bounded optimizer. The
degeneracy-safe ``expm`` segment propagator is used throughout (quadrupolar
spectra are Kramers-degenerate, which makes ``eigh`` gradients NaN).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from spin_dynamics.nqr.full_dynamics import (
    detection_operator,
    quadrature_detection_operator,
)
from spin_dynamics.nqr.hamiltonians import diagonalize_site
from spin_dynamics.nqr.simulation import equilibrium_density
from spin_dynamics.optimal_control._jax_propagation import (
    JAX_AVAILABLE,
    propagate_unitary_scan_multi_numpy,
)
from spin_dynamics.optimal_control.hamiltonians import nqr_site_control_model

# 4 parameters per coil per pulse: amplitude fraction, phase, start, length.
PARAMS_PER_COIL = 4


@dataclass(frozen=True)
class MultiAxisSLSEConfig:
    """Configuration for a tri-axial parametric SLSE optimization.

    ``n_coils`` orthogonal coils (1 = conventional linear, 2 = circular-style
    quadrature, 3 = tri-axial) each contribute one rectangular sub-pulse to the
    excitation and to the refocusing pulse. ``nutation_scale`` sets the amplitude
    unit (an amplitude fraction of 1 means ``nutation_scale`` hertz of bare
    nutation on that coil), keeping the optimized parameters O(1). ``window_ex``/
    ``window_ref`` are the time budgets the sub-pulses live in; ``n_fine`` is the
    fixed rendering grid per pulse. ``rx_scheme`` selects the receiver: ``"single"``
    (coil 0) or ``"quadrature"`` (coils 0,1 in matched quadrature).
    """

    site: object
    frames: Sequence
    n_coils: int = 3
    rf_frequency_hz: float | None = None
    nutation_scale_hz: float = 20e3
    window_ex_seconds: float = 30e-6
    window_ref_seconds: float = 30e-6
    n_fine: int = 24
    echo_spacing_seconds: float = 200e-6
    num_echoes: int = 6
    rx_scheme: str = "quadrature"
    ramp_segments: float = 1.0
    helicity: int = 1
    _built: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.n_coils < 1 or self.n_coils > 3:
            raise ValueError("n_coils must be 1, 2, or 3")
        if len(self.frames) == 0:
            raise ValueError("frames must contain at least one powder orientation")
        if self.rx_scheme not in ("single", "quadrature"):
            raise ValueError("rx_scheme must be 'single' or 'quadrature'")
        if self.rx_scheme == "quadrature" and self.n_coils < 2:
            raise ValueError("quadrature detection needs at least 2 coils")
        if self.nutation_scale_hz <= 0.0:
            raise ValueError("nutation_scale_hz must be positive")
        if self.window_ex_seconds <= 0.0 or self.window_ref_seconds <= 0.0:
            raise ValueError("pulse windows must be positive")
        if self.n_fine < 1:
            raise ValueError("n_fine must be positive")
        if self.num_echoes < 1:
            raise ValueError("num_echoes must be positive")
        if self.echo_spacing_seconds < self.window_ref_seconds:
            raise ValueError("echo_spacing must be at least window_ref")
        if self.ramp_segments < 0.0:
            raise ValueError("ramp_segments must be non-negative")
        if self.helicity not in (-1, 1):
            raise ValueError("helicity must be +1 or -1")


def _carrier(config: MultiAxisSLSEConfig, eigensystem) -> float:
    if config.rf_frequency_hz is not None:
        return float(config.rf_frequency_hz)
    return max(t.frequency_hz for t in eigensystem.transitions)


def _build_model(config: MultiAxisSLSEConfig) -> dict:
    """Precompute the per-orientation operators, detector, rho0, and free phase.

    Cached on the config so repeated objective/evaluation calls share it. Returns
    a dict of NumPy arrays: ``h_drift`` (dim, dim), ``hx_batch``/``hy_batch``
    (n_orient, n_coils, dim, dim), ``det_batch`` (n_orient, dim, dim), ``rho0``
    (dim, dim), ``free_omega`` (dim,), and scalar ``carrier``.
    """

    if config._built:
        return config._built

    eigensystem = diagonalize_site(config.site, None)
    carrier = _carrier(config, eigensystem)
    dim = eigensystem.site.dimension
    n_coils = config.n_coils

    hx_batch = np.empty((len(config.frames), n_coils, dim, dim), dtype=np.complex128)
    hy_batch = np.empty_like(hx_batch)
    det_batch = np.empty((len(config.frames), dim, dim), dtype=np.complex128)
    h_drift = None
    for f_idx, frame in enumerate(config.frames):
        for c in range(n_coils):
            model = nqr_site_control_model(
                config.site, rf_frequency_hz=carrier,
                b1_direction_pas=frame.axes[:, c],
            )
            hx_batch[f_idx, c] = model.h_x
            hy_batch[f_idx, c] = model.h_y
            if h_drift is None:
                h_drift = model.h_drift
        if config.rx_scheme == "single":
            det_batch[f_idx] = detection_operator(eigensystem, carrier, frame.axes[:, 0])
        else:
            det_batch[f_idx] = quadrature_detection_operator(
                eigensystem, carrier, frame.axes[:, 0], frame.axes[:, 1],
                helicity=config.helicity,
            )

    built = {
        "eigensystem": eigensystem,
        "carrier": carrier,
        "dim": dim,
        "h_drift": h_drift,
        "hx_batch": hx_batch,
        "hy_batch": hy_batch,
        "det_batch": det_batch,
        "rho0": equilibrium_density(eigensystem.levels_hz),
        "free_omega": np.real(np.diag(h_drift)),
        "weights": np.array([f.weight for f in config.frames], dtype=np.float64),
    }
    config._built.update(built)
    return built


def control_bounds(
    config: MultiAxisSLSEConfig, *, amp_max: float = 4.0
) -> list[tuple[float, float]]:
    """Per-parameter ``(lower, upper)`` bounds for one control vector.

    Layout: excitation coils first, then refocusing coils; each coil is
    ``[amp_fraction, phase, start_fraction, length_fraction]``. Amplitude is a
    dimensionless multiple of ``nutation_scale`` (kept O(1) for conditioning),
    phase is unbounded within +-2pi, start/length are fractions of the pulse
    window in ``[0, 1]``.
    """

    per_coil = [(0.0, amp_max), (-2.0 * np.pi, 2.0 * np.pi), (0.0, 1.0), (0.0, 1.0)]
    return per_coil * (2 * config.n_coils)


def _split_params(x, n_coils):
    """Return (excite, refocus) each shaped (n_coils, 4)."""

    block = n_coils * PARAMS_PER_COIL
    if x.size != 2 * block:
        raise ValueError(f"expected {2 * block} control parameters, got {x.size}")
    excite = x[:block].reshape(n_coils, PARAMS_PER_COIL)
    refocus = x[block:].reshape(n_coils, PARAMS_PER_COIL)
    return excite, refocus


if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    from spin_dynamics.optimal_control._jax_propagation import (
        propagate_unitary_batched_multi,
    )

    def _render_rect(pulse_params, window_T, n_fine, nutation_scale, ramp_segments):
        """Render per-coil rectangular sub-pulses onto a fixed fine grid.

        ``pulse_params`` is ``(n_coils, 4)`` = amp_fraction, phase, start_fraction,
        length_fraction. Returns ``amplitude``/``phase`` of shape
        ``(n_fine, n_coils)``: a smooth (sigmoid) window makes start/length
        differentiable while approaching a true rectangle as ``ramp_segments`` -> 0.
        """

        dt = window_T / n_fine
        centers = (jnp.arange(n_fine) + 0.5) * dt  # (n_fine,)
        amp_frac = pulse_params[:, 0]
        phase = pulse_params[:, 1]
        start = pulse_params[:, 2] * window_T
        length = pulse_params[:, 3] * window_T
        end = start + length
        width = jnp.maximum(ramp_segments * dt, 1e-12)
        # (n_fine, n_coils) smooth top-hat
        rise = jax.nn.sigmoid((centers[:, None] - start[None, :]) / width)
        fall = jax.nn.sigmoid((end[None, :] - centers[:, None]) / width)
        window = rise * fall
        amplitude = (amp_frac * nutation_scale)[None, :] * window
        phase_grid = jnp.broadcast_to(phase[None, :], (n_fine, amp_frac.shape[0]))
        return amplitude, phase_grid, dt

    def _make_energy_fn(config: MultiAxisSLSEConfig):
        built = _build_model(config)
        h_drift = jnp.asarray(built["h_drift"])
        hx_batch = jnp.asarray(built["hx_batch"])
        hy_batch = jnp.asarray(built["hy_batch"])
        det_batch = jnp.asarray(built["det_batch"])
        rho0 = jnp.asarray(built["rho0"])
        free_omega = jnp.asarray(built["free_omega"])
        weights = jnp.asarray(built["weights"])
        n_coils = config.n_coils
        n_fine = config.n_fine
        num_echoes = config.num_echoes
        nut = config.nutation_scale_hz
        ramp = config.ramp_segments
        win_ex = config.window_ex_seconds
        win_ref = config.window_ref_seconds
        free_half = 0.5 * (config.echo_spacing_seconds - win_ref)
        free_phase = jnp.exp(-1j * free_omega * free_half)

        def energy(x):
            excite, refocus = _split_params(x, n_coils)
            amp_ex, ph_ex, dt_ex = _render_rect(excite, win_ex, n_fine, nut, ramp)
            amp_re, ph_re, dt_re = _render_rect(refocus, win_ref, n_fine, nut, ramp)

            u_ex = propagate_unitary_batched_multi(
                h_drift, hx_batch, hy_batch, amp_ex, ph_ex, dt_ex
            )
            u_re = propagate_unitary_batched_multi(
                h_drift, hx_batch, hy_batch, amp_re, ph_re, dt_re
            )

            def train(u_e, u_r, det):
                rho = u_e @ rho0 @ u_e.conj().T

                def step(rho, _):
                    rho = (free_phase[:, None] * rho) * jnp.conj(free_phase)[None, :]
                    rho = u_r @ rho @ u_r.conj().T
                    rho = (free_phase[:, None] * rho) * jnp.conj(free_phase)[None, :]
                    return rho, jnp.trace(rho @ det)

                _, echoes = jax.lax.scan(step, rho, None, length=num_echoes)
                return echoes

            echoes = jax.vmap(train)(u_ex, u_re, det_batch)  # (n_orient, num_echoes)
            powder = weights @ echoes  # (num_echoes,) complex, coherent powder sum
            mag = jnp.sqrt(jnp.real(powder * jnp.conj(powder)) + 1e-30)
            return jnp.sum(mag)

        return energy


def make_multi_axis_slse_objective(config: MultiAxisSLSEConfig):
    """Return ``value_and_grad(x) -> (energy, grad)`` for the SLSE train energy.

    ``x`` is the flat control vector (see :func:`control_bounds`). The returned
    callable feeds directly into
    ``optimization._bounded.scipy_maximize_with_grad`` (maximization). Requires the
    optional ``jax`` extra.
    """

    if not JAX_AVAILABLE:
        raise ImportError(
            "Multi-axis GRAPE requires the optional 'jax' extra. Install it with "
            "`python -m pip install -e .[jax]` (or `.[perf]`)."
        )
    energy = _make_energy_fn(config)
    _vg = jax.jit(jax.value_and_grad(energy))

    def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = _vg(jnp.asarray(x, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=np.float64)

    return value_and_grad


def slse_train_amplitudes(config: MultiAxisSLSEConfig, x: np.ndarray) -> np.ndarray:
    """Return the complex powder echo-train (num_echoes,) for a control vector.

    Pure NumPy (no jax needed): renders the rectangular sub-pulses, composes the
    per-coil unitaries, and runs the density-matrix SLSE train. Used for reporting
    and plotting an optimized or baseline pulse.
    """

    built = _build_model(config)
    h_drift = built["h_drift"]
    hx_batch = built["hx_batch"]
    hy_batch = built["hy_batch"]
    det_batch = built["det_batch"]
    rho0 = built["rho0"]
    free_omega = built["free_omega"]
    weights = built["weights"]
    n_coils = config.n_coils
    n_fine = config.n_fine
    excite, refocus = _split_params(np.asarray(x, dtype=np.float64), n_coils)

    def render(pulse, window_T):
        dt = window_T / n_fine
        centers = (np.arange(n_fine) + 0.5) * dt
        amp_frac, phase, start_f, len_f = pulse.T
        start = start_f * window_T
        end = start + len_f * window_T
        width = max(config.ramp_segments * dt, 1e-12)
        rise_arg = (centers[:, None] - start[None, :]) / width
        fall_arg = (end[None, :] - centers[:, None]) / width
        rise = np.exp(-np.logaddexp(0.0, -rise_arg))
        fall = np.exp(-np.logaddexp(0.0, -fall_arg))
        amplitude = (amp_frac * config.nutation_scale_hz)[None, :] * rise * fall
        phase_grid = np.broadcast_to(phase[None, :], (n_fine, n_coils))
        return amplitude, phase_grid, dt

    amp_ex, ph_ex, dt_ex = render(excite, config.window_ex_seconds)
    amp_re, ph_re, dt_re = render(refocus, config.window_ref_seconds)
    free_half = 0.5 * (config.echo_spacing_seconds - config.window_ref_seconds)
    free_phase = np.exp(-1j * free_omega * free_half)

    powder = np.zeros(config.num_echoes, dtype=np.complex128)
    for f_idx in range(len(config.frames)):
        u_e = propagate_unitary_scan_multi_numpy(
            h_drift, hx_batch[f_idx], hy_batch[f_idx], amp_ex, ph_ex, dt_ex)
        u_r = propagate_unitary_scan_multi_numpy(
            h_drift, hx_batch[f_idx], hy_batch[f_idx], amp_re, ph_re, dt_re)
        det = det_batch[f_idx]
        rho = u_e @ rho0 @ u_e.conj().T
        for e in range(config.num_echoes):
            rho = (free_phase[:, None] * rho) * free_phase.conj()[None, :]
            rho = u_r @ rho @ u_r.conj().T
            rho = (free_phase[:, None] * rho) * free_phase.conj()[None, :]
            powder[e] += weights[f_idx] * np.trace(rho @ det)
    return powder


@dataclass(frozen=True)
class MultiAxisSLSEResult:
    """Result of a multistart tri-axial SLSE optimization."""

    config: MultiAxisSLSEConfig
    best_x: np.ndarray
    best_energy: float
    start_energies: np.ndarray
    seed_energies: np.ndarray
    iterations: int

    def excite_params(self) -> np.ndarray:
        """Return the excitation sub-pulse parameters, shape (n_coils, 4)."""
        return _split_params(self.best_x, self.config.n_coils)[0]

    def refocus_params(self) -> np.ndarray:
        """Return the refocusing sub-pulse parameters, shape (n_coils, 4)."""
        return _split_params(self.best_x, self.config.n_coils)[1]

    def train(self) -> np.ndarray:
        """Return the optimized complex powder echo train."""
        return slse_train_amplitudes(self.config, self.best_x)


def simultaneous_seed(config: MultiAxisSLSEConfig, *, amp_fraction: float = 1.5,
                      excite_len: float = 0.5, refocus_len: float = 1.0) -> np.ndarray:
    """A circular-style warm start: all coils simultaneous, quadrature phases.

    Coils start together (delay 0) and span the window; the excitation coils are
    stepped in phase by pi/2 (a rotating field), the refocusing coils by pi/2
    relative to the excitation, matching the circular SLSE of
    ``plot_nqr_circular_polarization_nutation``. A good basin for the optimizer to
    refine and a fair "structured" reference among the random restarts.
    """

    n = config.n_coils
    excite = np.zeros((n, PARAMS_PER_COIL))
    refocus = np.zeros((n, PARAMS_PER_COIL))
    for c in range(n):
        excite[c] = [amp_fraction, c * np.pi / 2.0, 0.0, excite_len]
        refocus[c] = [amp_fraction, np.pi / 2.0 + c * np.pi / 2.0, 0.0, refocus_len]
    return np.concatenate([excite.reshape(-1), refocus.reshape(-1)])


def _random_seed(rng, config, amp_max):
    n = config.n_coils
    seed = np.zeros((2 * n, PARAMS_PER_COIL))
    seed[:, 0] = rng.uniform(0.3, amp_max, size=2 * n)          # amplitude fraction
    seed[:, 1] = rng.uniform(-np.pi, np.pi, size=2 * n)          # phase
    seed[:, 2] = rng.uniform(0.0, 0.4, size=2 * n)              # start fraction
    seed[:, 3] = rng.uniform(0.3, 1.0, size=2 * n)              # length fraction
    return seed.reshape(-1)


def optimize_multi_axis_slse(
    config: MultiAxisSLSEConfig, *, n_starts: int = 8, seed: int = 0,
    amp_max: float = 4.0, include_simultaneous_seed: bool = True,
    options: dict | None = None,
) -> MultiAxisSLSEResult:
    """Multistart-optimize the tri-axial rectangular SLSE pulse parameters.

    Maximizes the refocused powder echo-train energy over each coil's amplitude,
    phase, delay, and length (for excitation and refocusing) via
    ``scipy_maximize_with_grad`` with the ``jax.value_and_grad`` objective. GRAPE
    landscapes are multimodal, so several random restarts are run (plus, by
    default, a circular-style structured start). Returns the best control vector.
    """

    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")
    if amp_max <= 0.0:
        raise ValueError("amp_max must be positive")

    from spin_dynamics.optimization._bounded import scipy_maximize_with_grad

    value_and_grad = make_multi_axis_slse_objective(config)
    bounds = control_bounds(config, amp_max=amp_max)
    rng = np.random.default_rng(seed)

    seeds = []
    if include_simultaneous_seed:
        seeds.append(np.clip(simultaneous_seed(config),
                             [b[0] for b in bounds], [b[1] for b in bounds]))
    while len(seeds) < max(n_starts, 1):
        seeds.append(_random_seed(rng, config, amp_max))

    best_x = None
    best_energy = -np.inf
    start_energies = []
    seed_energies = []
    total_iters = 0
    for x0 in seeds:
        seed_energies.append(value_and_grad(x0)[0])
        run = scipy_maximize_with_grad(
            value_and_grad, x0, bounds=bounds, options=options,
        )
        start_energies.append(run.best_score)
        total_iters += int(getattr(run, "iterations", 0) or 0)
        if run.best_score > best_energy:
            best_energy = run.best_score
            best_x = np.asarray(run.best_x, dtype=np.float64)

    return MultiAxisSLSEResult(
        config=config,
        best_x=best_x,
        best_energy=float(best_energy),
        start_energies=np.asarray(start_energies),
        seed_energies=np.asarray(seed_energies),
        iterations=total_iters,
    )
