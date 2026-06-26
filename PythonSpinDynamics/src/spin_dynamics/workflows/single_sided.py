"""Depth-resolved single-sided (NMR-MOUSE) measurement simulation.

Single-sided NMR profiles a sample by the depth-dependent static field of an open
magnet: an excitation frequency selects the iso-B0 sensitive slice, and the strong
static gradient encodes diffusion. The defining feature is that the spins *move
through a spatially structured field* -- as a molecule diffuses it samples a
changing off-resonance (set by the real gradient) and a changing B1 -- which is
irreducibly spatial and cannot be captured by a fixed off-resonance distribution.

This module therefore drives the moving-isochromat engine directly with the
magnet's own ``B0``/``B1`` maps (from :mod:`spin_dynamics.fields.magnetostatics`):
walkers are seeded around the resonant depth, the finite excitation pulse selects
the slice from the real field, the CPMG train and its diffusion attenuation emerge
from the walkers moving in the real gradient, and sweeping the carrier frequency
profiles the sample in depth. The engine's constant-gradient diffusion physics is
validated against the exact Carr-Purcell law in ``tests/test_single_sided.py``, so
deviations seen here are the real-field physics, not numerics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.magnetostatics import (
    GAMMA_PROTON,
    BarMagnet,
    bar_array_b0,
    sample_magnet_field,
)
from spin_dynamics.motion import initialize_ensemble_from_density, make_motion_field_maps_2d
from spin_dynamics.sequences.motion import run_motion_cpmg_sequence


@dataclass(frozen=True)
class SampleLayer:
    """A depth layer of the sample (depth measured along the through-plane axis)."""

    y_min: float  # m
    y_max: float  # m
    rho: float = 1.0
    t2: float = 0.1  # s
    diffusion: float = 0.0  # m^2/s


@dataclass(frozen=True)
class LayeredSample:
    """A stack of :class:`SampleLayer` ordered by depth."""

    layers: Sequence[SampleLayer]

    def properties(self, depth: np.ndarray) -> dict[str, np.ndarray]:
        """Return per-point ``rho``/``t2``/``diffusion`` at the given depths."""

        depth = np.asarray(depth, dtype=np.float64)
        rho = np.zeros_like(depth)
        t2 = np.full_like(depth, np.inf)
        diff = np.zeros_like(depth)
        for layer in self.layers:
            mask = (depth >= layer.y_min) & (depth < layer.y_max)
            rho[mask] = layer.rho
            t2[mask] = layer.t2
            diff[mask] = layer.diffusion
        return {"rho": rho, "t2": t2, "diffusion": diff}


@dataclass(frozen=True)
class MouseCPMGResult:
    """One simulated CPMG measurement at a fixed excitation frequency."""

    frequency_hz: float
    depth: float  # resonant depth (m)
    local_gradient: float  # |grad |B0|| at the slice (T/m)
    echo_times: np.ndarray  # echo-center times (s)
    echo_amplitudes: np.ndarray  # |signal| at each echo
    excited_signal: float  # first-echo amplitude (the depth-profile point)


@dataclass(frozen=True)
class MouseDepthProfileResult:
    """A depth profile assembled from per-frequency CPMG measurements."""

    frequencies_hz: np.ndarray
    depths: np.ndarray
    local_gradient: np.ndarray
    signal: np.ndarray  # excited signal vs depth
    t2_eff: np.ndarray  # apparent T2 vs depth (s)
    echo_amplitudes: np.ndarray  # (n_freq, num_echoes)
    echo_times: np.ndarray


def _on_axis_profile(bars, yoke_y, y_lo, y_hi, gamma):
    y = np.linspace(y_lo, y_hi, 800)
    bmag = np.hypot(*bar_array_b0(np.zeros_like(y), y, bars, yoke_y=yoke_y))
    return y, gamma * bmag / (2.0 * np.pi)  # depth, Larmor Hz (decreasing with y)


def resonant_depth(
    bars: Sequence[BarMagnet],
    frequency_hz: float,
    *,
    yoke_y: float | None = None,
    depth_range: tuple[float, float] = (0.021, 0.060),
    gamma: float = GAMMA_PROTON,
) -> float:
    """Return the on-axis depth where the proton Larmor frequency equals ``frequency_hz``."""

    y, f = _on_axis_profile(bars, yoke_y, depth_range[0], depth_range[1], gamma)
    return float(np.interp(frequency_hz, f[::-1], y[::-1]))


def simulate_mouse_cpmg(
    bars: Sequence[BarMagnet],
    sample: LayeredSample,
    frequency_hz: float,
    *,
    yoke_y: float | None = None,
    echo_time: float = 2.0e-4,
    num_echoes: int = 64,
    excitation_duration: float = 10.0e-6,
    depth_halfwidth: float = 0.5e-3,
    lateral_halfwidth: float = 3.0e-3,
    n_depth: int = 121,
    n_lateral: int = 3,
    walkers_per_cell: int = 12,
    substeps_per_interval: int = 4,
    coil_segments: Sequence | None = None,
    diffusion_scale: float = 1.0,
    gamma: float = GAMMA_PROTON,
    seed: int = 0,
) -> MouseCPMGResult:
    """Simulate one CPMG measurement at ``frequency_hz`` in the real magnet field.

    Walkers are seeded in a depth window around the resonant slice and diffuse
    through the magnet's actual ``B0`` (off-resonance ``gamma|B0| - 2 pi f``) and,
    if a coil is supplied, its actual transverse ``B1``. The finite excitation
    pulse selects the slice from the field; the echo train and its diffusion
    attenuation emerge from the motion.
    """

    w0 = 2.0 * np.pi * float(frequency_hz)
    y0 = resonant_depth(bars, frequency_hz, yoke_y=yoke_y, gamma=gamma)
    x_axis = np.linspace(-lateral_halfwidth, lateral_halfwidth, n_lateral)
    y_axis = np.linspace(y0 - depth_halfwidth, y0 + depth_halfwidth, n_depth)
    fm = sample_magnet_field(x_axis, y_axis, bars, yoke_y=yoke_y,
                             coil_segments=coil_segments, coil_current=1.0, gamma=gamma)
    offres = gamma * fm.b0_magnitude - w0
    if coil_segments is not None and fm.b1_transverse is not None:
        b1 = fm.b1_transverse / (fm.b1_transverse.max() or 1.0)
    else:
        b1 = np.ones_like(offres)

    # Per-cell sample properties from the depth of each grid point.
    _, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
    props = sample.properties(yy)
    fields = make_motion_field_maps_2d(x_axis, y_axis, b0_map=offres,
                                       b1_tx_map=b1, b1_rx_map=b1)
    ensemble = initialize_ensemble_from_density(
        props["rho"], x_axis, y_axis, walkers_per_cell=walkers_per_cell,
        diffusion_coefficient=float(diffusion_scale) * props["diffusion"].reshape(-1),
        seed=seed, jitter=True,
    )
    t2_walkers = np.repeat(props["t2"].reshape(-1), walkers_per_cell)
    result = run_motion_cpmg_sequence(
        ensemble, fields, num_echoes=num_echoes, echo_spacing=echo_time,
        excitation_duration=excitation_duration, refocusing_duration=excitation_duration,
        t2=t2_walkers, boundary="reflect", substeps_per_interval=substeps_per_interval,
        rng=np.random.default_rng(seed),
    )
    amps = np.abs(result.signal)
    times = echo_time * np.arange(1, num_echoes + 1)
    local_g = float(np.median(fm.b0_gradient[:, n_depth // 2]))
    return MouseCPMGResult(
        frequency_hz=float(frequency_hz),
        depth=float(y0),
        local_gradient=local_g,
        echo_times=times,
        echo_amplitudes=amps,
        excited_signal=float(amps[0]) if amps.size else 0.0,
    )


@dataclass(frozen=True)
class MouseDiffusionResult:
    """Diffusion measured at one depth from the diffusion-on/off echo ratio."""

    frequency_hz: float
    depth: float
    local_gradient: float  # T/m
    echo_time: float
    diffusion: float  # fitted D (m^2/s)
    diffusion_rate: float  # fitted attenuation rate k, with k = (1/12) g^2 G^2 D tE^2


def measure_diffusion_at_depth(
    bars: Sequence[BarMagnet],
    sample: LayeredSample,
    frequency_hz: float,
    *,
    echo_time: float = 1.2e-4,
    num_echoes: int = 40,
    n_seeds: int = 4,
    min_ratio: float = 0.1,
    gamma: float = GAMMA_PROTON,
    **cpmg_kwargs,
) -> MouseDiffusionResult:
    """Measure D at the slice by the diffusion-on / diffusion-off echo ratio.

    Runs the CPMG twice with identical initial walker positions -- once with the
    sample's diffusion, once with diffusion switched off -- so the (messy)
    inhomogeneous-field relaxation/pathway envelope cancels in the ratio, leaving
    the pure diffusion attenuation in the real gradient. The attenuation rate
    ``k = (1/12) gamma^2 G^2 D tE^2`` then gives D using the slice's local
    gradient. Use a short ``echo_time`` so the echoes survive long enough to fit.
    """

    rates: list[float] = []
    local_g = depth = None
    for seed in range(int(n_seeds)):
        on = simulate_mouse_cpmg(bars, sample, frequency_hz, echo_time=echo_time,
                                 num_echoes=num_echoes, diffusion_scale=1.0,
                                 gamma=gamma, seed=seed, **cpmg_kwargs)
        off = simulate_mouse_cpmg(bars, sample, frequency_hz, echo_time=echo_time,
                                  num_echoes=num_echoes, diffusion_scale=0.0,
                                  gamma=gamma, seed=seed, **cpmg_kwargs)
        local_g, depth = on.local_gradient, on.depth
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = on.echo_amplitudes / off.echo_amplitudes
        t = on.echo_times
        mask = np.isfinite(ratio) & (ratio > float(min_ratio)) & (off.echo_amplitudes > 0)
        if mask.sum() >= 3:
            rates.append(float(-np.polyfit(t[mask], np.log(ratio[mask]), 1)[0]))
    rate = float(np.mean(rates)) if rates else float("nan")
    d_fit = rate * 12.0 / (gamma**2 * local_g**2 * echo_time**2)
    return MouseDiffusionResult(
        frequency_hz=float(frequency_hz), depth=float(depth),
        local_gradient=float(local_g), echo_time=float(echo_time),
        diffusion=float(d_fit), diffusion_rate=rate,
    )


def _fit_t2(times: np.ndarray, amps: np.ndarray, min_frac: float = 0.1) -> float:
    mask = amps > float(min_frac) * (amps.max() or 1.0)
    if mask.sum() < 2:
        return float("nan")
    slope = np.polyfit(times[mask], np.log(amps[mask]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else float("nan")


def mouse_depth_profile(
    bars: Sequence[BarMagnet],
    sample: LayeredSample,
    frequencies_hz: Sequence[float],
    *,
    yoke_y: float | None = None,
    **cpmg_kwargs,
) -> MouseDepthProfileResult:
    """Profile a sample in depth by sweeping the excitation frequency.

    Each frequency runs :func:`simulate_mouse_cpmg`; the excited signal traces the
    depth profile of spin density and the echo decay gives the apparent T2 at that
    depth. Extra keyword arguments are forwarded to :func:`simulate_mouse_cpmg`.
    """

    freqs = np.asarray(list(frequencies_hz), dtype=np.float64)
    depths, grads, signals, t2s, trains = [], [], [], [], []
    times = None
    for f in freqs:
        res = simulate_mouse_cpmg(bars, sample, float(f), yoke_y=yoke_y, **cpmg_kwargs)
        depths.append(res.depth)
        grads.append(res.local_gradient)
        signals.append(res.excited_signal)
        t2s.append(_fit_t2(res.echo_times, res.echo_amplitudes))
        trains.append(res.echo_amplitudes)
        times = res.echo_times
    return MouseDepthProfileResult(
        frequencies_hz=freqs,
        depths=np.asarray(depths),
        local_gradient=np.asarray(grads),
        signal=np.asarray(signals),
        t2_eff=np.asarray(t2s),
        echo_amplitudes=np.asarray(trains),
        echo_times=times if times is not None else np.empty(0),
    )
