"""Bloch-McConnell site exchange for inhomogeneous-field relaxation analysis.

This module adds the chemical/site-exchange physics that the rest of the
package deliberately left out: a bath of magnetically distinct sites that
swap magnetization at finite kinetic rates while each relaxes with its own
``T1``/``T2`` and precesses at its own offset. The model is the classical
Bloch-McConnell system, written here as a small dense kinetic generator so it
composes with the existing Liouville-style helpers and the inverse-Laplace
analysis layer.

Two scientific endpoints motivate the design:

* **Lineshape exchange.** ``simulate_exchange_fid`` and ``exchange_spectrum``
  integrate the transverse Bloch-McConnell equations, reproducing the
  slow-exchange (resolved lines) to fast-exchange (coalesced, population-
  averaged line) crossover that diagnoses kinetics directly from a spectrum.
* **Relaxation exchange spectroscopy (REXSY / T2-T2).**
  ``simulate_relaxation_exchange_2d`` produces the encode-mix-detect data set
  whose 2D inverse Laplace transform (``spin_dynamics.analysis.invert_t2_t2``)
  shows diagonal peaks for spins that stay put and off-diagonal cross peaks for
  spins that change site during the mixing interval. This is the relaxation
  analogue of the diffusion-exchange (DEXSY) example already in the package.

Conventions
-----------
* ``exchange_rates_hz[i, j]`` (``i != j``) is the first-order rate constant for
  leaving site ``i`` toward site ``j`` (``k_{i->j}``) in inverse seconds. The
  diagonal is ignored. The kinetic generator built from it conserves total
  magnetization under exchange alone (each column sums to zero).
* ``offset_hz`` is the site resonance offset in the rotating frame. Transverse
  magnetization follows ``dM+/dt = +i 2 pi offset M+`` so a positive offset
  appears at a positive spectral frequency.
* Site ``population`` values are normalized to sum to one. Detailed balance
  (``p_i k_{i->j} = p_j k_{j->i}``) is checked when populations are supplied;
  use ``balance="off"`` to skip the check for intentionally non-equilibrium
  kinetics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    "ExchangeSite",
    "ExchangeSystem",
    "RelaxationExchange2DResult",
    "two_site_exchange",
    "exchange_generator",
    "mixing_propagator",
    "transverse_generator",
    "transverse_propagator",
    "simulate_exchange_fid",
    "exchange_spectrum",
    "simulate_relaxation_exchange_2d",
]


@dataclass(frozen=True)
class ExchangeSite:
    """One magnetically distinct site in a chemical-exchange bath.

    ``population`` is the equilibrium fraction (normalized across the system).
    ``offset_hz`` is the rotating-frame resonance offset. ``t2_seconds`` and
    ``t1_seconds`` are the intrinsic transverse and longitudinal relaxation
    times for spins residing on this site; both default to non-relaxing.
    """

    label: str
    population: float = 1.0
    offset_hz: float = 0.0
    t2_seconds: float = np.inf
    t1_seconds: float = np.inf

    def __post_init__(self) -> None:
        population = float(self.population)
        offset_hz = float(self.offset_hz)
        t1_seconds = float(self.t1_seconds)
        t2_seconds = float(self.t2_seconds)
        if not np.isfinite(population) or population < 0.0:
            raise ValueError("population must be finite and non-negative")
        if not np.isfinite(offset_hz):
            raise ValueError("offset_hz must be finite")
        if t1_seconds <= 0.0:
            raise ValueError("t1_seconds must be positive or infinite")
        if t2_seconds <= 0.0:
            raise ValueError("t2_seconds must be positive or infinite")
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "offset_hz", offset_hz)
        object.__setattr__(self, "t1_seconds", t1_seconds)
        object.__setattr__(self, "t2_seconds", t2_seconds)


@dataclass(frozen=True)
class ExchangeSystem:
    """A set of exchanging sites with a first-order rate matrix.

    ``exchange_rates_hz`` has shape ``(n, n)`` where entry ``(i, j)`` with
    ``i != j`` is the rate constant ``k_{i->j}`` in inverse seconds. The
    diagonal is ignored and replaced by the negative total outflow when the
    kinetic generator is formed.
    """

    sites: tuple[ExchangeSite, ...]
    exchange_rates_hz: np.ndarray
    balance: str = "warn"
    balance_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        sites = tuple(self.sites)
        if len(sites) < 1:
            raise ValueError("an exchange system needs at least one site")
        rates = np.asarray(self.exchange_rates_hz, dtype=np.float64)
        n = len(sites)
        if rates.shape != (n, n):
            raise ValueError(
                f"exchange_rates_hz must have shape ({n}, {n}); got {rates.shape}"
            )
        if not np.all(np.isfinite(rates)):
            raise ValueError("exchange_rates_hz must contain only finite values")
        off_diagonal = rates - np.diag(np.diag(rates))
        if np.any(off_diagonal < 0.0):
            raise ValueError("off-diagonal exchange rates must be non-negative")
        if self.balance not in {"warn", "raise", "off"}:
            raise ValueError("balance must be 'warn', 'raise', or 'off'")

        populations = np.array([site.population for site in sites], dtype=np.float64)
        total = float(populations.sum())
        if total <= 0.0:
            raise ValueError("site populations must sum to a positive value")
        populations = populations / total

        rates = off_diagonal.copy()
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "exchange_rates_hz", rates)
        object.__setattr__(self, "_populations", populations)

        if self.balance != "off" and n > 1:
            self._check_detailed_balance(populations, rates)

    def _check_detailed_balance(
        self, populations: np.ndarray, rates: np.ndarray
    ) -> None:
        flux = populations[:, np.newaxis] * rates
        imbalance = np.abs(flux - flux.T)
        scale = float(np.max(flux)) or 1.0
        worst = float(np.max(imbalance)) / scale
        if worst <= float(self.balance_tolerance):
            return
        message = (
            "exchange rates violate detailed balance "
            f"(relative imbalance {worst:.3e} > {self.balance_tolerance:.3e}). "
            "The supplied populations are not the kinetic steady state; pass "
            "balance='off' if this is intentional."
        )
        if self.balance == "raise":
            raise ValueError(message)
        import warnings

        warnings.warn(message, RuntimeWarning, stacklevel=3)

    @property
    def num_sites(self) -> int:
        return len(self.sites)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(site.label for site in self.sites)

    @property
    def populations(self) -> np.ndarray:
        return np.asarray(self._populations, dtype=np.float64).copy()

    @property
    def offsets_hz(self) -> np.ndarray:
        return np.array([site.offset_hz for site in self.sites], dtype=np.float64)

    @property
    def r2_rates(self) -> np.ndarray:
        return np.array(
            [0.0 if np.isinf(s.t2_seconds) else 1.0 / s.t2_seconds for s in self.sites],
            dtype=np.float64,
        )

    @property
    def r1_rates(self) -> np.ndarray:
        return np.array(
            [0.0 if np.isinf(s.t1_seconds) else 1.0 / s.t1_seconds for s in self.sites],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class RelaxationExchange2DResult:
    """Encode-mix-detect data set for relaxation exchange spectroscopy.

    ``data[i, j]`` is the detected signal for encode time ``encode_times[i]``
    and detect time ``detect_times[j]``. ``mixing_propagator`` is the
    longitudinal exchange map ``G`` applied during the mixing interval, with
    ``G[b, a]`` the magnetization transferred from site ``a`` to site ``b``.
    """

    data: np.ndarray
    encode_times: np.ndarray
    detect_times: np.ndarray
    mixing_time: float
    mixing_propagator: np.ndarray
    populations: np.ndarray
    labels: tuple[str, ...]


def two_site_exchange(
    *,
    offset_a_hz: float,
    offset_b_hz: float,
    k_ab_hz: float,
    k_ba_hz: float | None = None,
    population_a: float | None = None,
    t2_a_seconds: float = np.inf,
    t2_b_seconds: float = np.inf,
    t1_a_seconds: float = np.inf,
    t1_b_seconds: float = np.inf,
    labels: tuple[str, str] = ("A", "B"),
    balance: str = "warn",
) -> ExchangeSystem:
    """Build a two-site exchange system.

    Supply either the explicit backward rate ``k_ba_hz`` or the fractional
    population ``population_a`` of site A; in the latter case ``k_ba_hz`` is
    derived from detailed balance ``p_a k_ab = p_b k_ba``. Exactly one of the
    two must be given.
    """

    k_ab = float(k_ab_hz)
    if k_ab < 0.0:
        raise ValueError("k_ab_hz must be non-negative")
    if (k_ba_hz is None) == (population_a is None):
        raise ValueError("supply exactly one of k_ba_hz or population_a")

    if population_a is not None:
        p_a = float(population_a)
        if not 0.0 < p_a < 1.0:
            raise ValueError("population_a must lie strictly between 0 and 1")
        p_b = 1.0 - p_a
        k_ba = k_ab * p_a / p_b
    else:
        k_ba = float(k_ba_hz)
        if k_ba < 0.0:
            raise ValueError("k_ba_hz must be non-negative")
        if k_ab + k_ba > 0.0:
            p_a = k_ba / (k_ab + k_ba)
        else:
            p_a = 0.5
        p_b = 1.0 - p_a

    sites = (
        ExchangeSite(labels[0], p_a, offset_a_hz, t2_a_seconds, t1_a_seconds),
        ExchangeSite(labels[1], p_b, offset_b_hz, t2_b_seconds, t1_b_seconds),
    )
    rates = np.array([[0.0, k_ab], [k_ba, 0.0]], dtype=np.float64)
    return ExchangeSystem(sites, rates, balance=balance)


def exchange_generator(system: ExchangeSystem) -> np.ndarray:
    """Return the kinetic generator ``X`` for magnetization exchange.

    ``dm/dt = X @ m`` with ``X[i, j] = k_{j->i}`` for ``i != j`` and
    ``X[i, i] = -sum_j k_{i->j}``. Columns sum to zero, so exchange alone
    conserves total magnetization.
    """

    rates = system.exchange_rates_hz
    generator = rates.T.copy()
    outflow = rates.sum(axis=1)
    np.fill_diagonal(generator, -outflow)
    return generator


def transverse_generator(system: ExchangeSystem) -> np.ndarray:
    """Return the complex transverse Bloch-McConnell generator.

    ``dM+/dt = A @ M+`` with ``A = X + diag(i 2 pi offset - R2)`` acting on the
    complex transverse magnetization ``M+ = Mx + i My`` of each site.
    """

    generator = exchange_generator(system).astype(np.complex128)
    diagonal = 1j * 2.0 * np.pi * system.offsets_hz - system.r2_rates
    generator[np.diag_indices_from(generator)] += diagonal
    return generator


def _matrix_exponential(matrix: np.ndarray) -> np.ndarray:
    """Dependency-free matrix exponential via scaling and squaring.

    Robust for the small, well-conditioned kinetic generators used here,
    including defective (non-diagonalizable) cases that eigen-decomposition
    would mishandle.
    """

    a = np.asarray(matrix, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    dim = a.shape[0]
    norm = float(np.max(np.sum(np.abs(a), axis=0))) if a.size else 0.0
    if norm == 0.0:
        return np.eye(dim, dtype=np.complex128)
    squarings = max(0, int(np.ceil(np.log2(norm))))
    scaled = a / (2.0**squarings)
    result = np.eye(dim, dtype=np.complex128)
    term = np.eye(dim, dtype=np.complex128)
    for k in range(1, 19):
        term = term @ scaled / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


def transverse_propagator(system: ExchangeSystem, duration: float) -> np.ndarray:
    """Return ``exp(A * duration)`` for the transverse generator."""

    duration = float(duration)
    if duration < 0.0 or not np.isfinite(duration):
        raise ValueError("duration must be non-negative and finite")
    return _matrix_exponential(transverse_generator(system) * duration)


def mixing_propagator(
    system: ExchangeSystem,
    mixing_time: float,
    *,
    include_t1: bool = True,
) -> np.ndarray:
    """Return the longitudinal exchange map ``G`` for the mixing interval.

    With ``include_t1=False`` the propagator is ``exp(X * t_mix)`` and is
    column-stochastic (each column sums to one), so stored magnetization is
    conserved and only redistributed by exchange. With ``include_t1=True`` each
    site additionally relaxes toward zero with its own ``T1`` during mixing;
    equilibrium recovery is intentionally excluded because the stored
    magnetization in an encode-mix-detect experiment is a deviation that has
    already been tipped away from equilibrium.
    """

    mixing_time = float(mixing_time)
    if mixing_time < 0.0 or not np.isfinite(mixing_time):
        raise ValueError("mixing_time must be non-negative and finite")
    generator = exchange_generator(system).astype(np.float64)
    if include_t1:
        generator = generator - np.diag(system.r1_rates)
    return np.real(_matrix_exponential(generator.astype(np.complex128) * mixing_time))


def simulate_exchange_fid(
    system: ExchangeSystem,
    times_seconds: np.ndarray,
    *,
    initial_magnetization: np.ndarray | None = None,
) -> np.ndarray:
    """Return the complex transverse free-induction decay with exchange.

    The default initial state places each site's population fraction in
    transverse magnetization, i.e. an ideal hard 90 degree excitation. The
    returned signal is the total transverse magnetization summed over sites at
    each requested time.
    """

    times = np.asarray(times_seconds, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times_seconds must not be empty")
    if np.any(times < 0.0) or not np.all(np.isfinite(times)):
        raise ValueError("times_seconds must be non-negative and finite")
    n = system.num_sites
    if initial_magnetization is None:
        state0 = system.populations.astype(np.complex128)
    else:
        state0 = np.asarray(initial_magnetization, dtype=np.complex128).reshape(-1)
        if state0.size != n:
            raise ValueError("initial_magnetization must have one entry per site")
    generator = transverse_generator(system)
    signal = np.empty(times.size, dtype=np.complex128)
    for index, t in enumerate(times):
        state = _matrix_exponential(generator * t) @ state0
        signal[index] = state.sum()
    return signal


def exchange_spectrum(
    system: ExchangeSystem,
    *,
    num_points: int = 4096,
    dwell_seconds: float | None = None,
    span_hz: float | None = None,
    line_broadening_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frequencies_hz, spectrum)`` from an exchange-broadened FID.

    The FID is sampled on a uniform time grid and Fourier transformed. ``span_hz``
    sets the spectral width (and hence the dwell time) when ``dwell_seconds`` is
    not given; it defaults to several times the offset spread plus the exchange
    rate so both resolved and coalesced regimes are captured. ``line_broadening_hz``
    applies an extra exponential apodization for display.
    """

    num_points = int(num_points)
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    offsets = system.offsets_hz
    spread = float(np.ptp(offsets)) if offsets.size > 1 else abs(float(offsets[0]))
    rate_scale = float(np.max(system.exchange_rates_hz)) if system.num_sites > 1 else 0.0
    if dwell_seconds is None:
        if span_hz is None:
            span_hz = 8.0 * spread + 8.0 * rate_scale + 100.0
        span_hz = float(span_hz)
        if span_hz <= 0.0:
            raise ValueError("span_hz must be positive")
        dwell_seconds = 1.0 / span_hz
    dwell_seconds = float(dwell_seconds)
    if dwell_seconds <= 0.0:
        raise ValueError("dwell_seconds must be positive")

    times = np.arange(num_points, dtype=np.float64) * dwell_seconds
    fid = simulate_exchange_fid(system, times)
    if line_broadening_hz > 0.0:
        fid = fid * np.exp(-np.pi * float(line_broadening_hz) * times)
    spectrum = np.fft.fftshift(np.fft.fft(fid))
    frequencies = np.fft.fftshift(np.fft.fftfreq(num_points, d=dwell_seconds))
    return frequencies, spectrum


def simulate_relaxation_exchange_2d(
    system: ExchangeSystem,
    encode_times: np.ndarray,
    detect_times: np.ndarray,
    mixing_time: float,
    *,
    include_t1: bool = True,
) -> RelaxationExchange2DResult:
    """Simulate an encode-mix-detect (T2-T2) relaxation exchange data set.

    The model assumes the encode and detect periods are refocused trains (so
    offsets do not dephase the stored amplitude) during which each site decays
    with its own ``T2``, and that site exchange happens during the longitudinal
    mixing interval. The resulting signal is

    ``S(t1, t2) = sum_b exp(-t2 / T2_b) sum_a G[b, a] p_a exp(-t1 / T2_a)``

    whose 2D inverse Laplace transform is a ``T2``-``T2`` exchange map: diagonal
    peaks for spins that stay on one site and off-diagonal cross peaks for spins
    that change site during mixing. Invert ``data`` with
    ``spin_dynamics.analysis.invert_t2_t2``.
    """

    encode = np.asarray(encode_times, dtype=np.float64).reshape(-1)
    detect = np.asarray(detect_times, dtype=np.float64).reshape(-1)
    if encode.size == 0 or detect.size == 0:
        raise ValueError("encode_times and detect_times must not be empty")
    if np.any(encode < 0.0) or np.any(detect < 0.0):
        raise ValueError("encode_times and detect_times must be non-negative")
    if not (np.all(np.isfinite(encode)) and np.all(np.isfinite(detect))):
        raise ValueError("encode_times and detect_times must be finite")

    r2 = system.r2_rates
    populations = system.populations
    propagator = mixing_propagator(system, mixing_time, include_t1=include_t1)

    encode_kernel = np.exp(-encode[:, np.newaxis] * r2[np.newaxis, :])
    encode_kernel = encode_kernel * populations[np.newaxis, :]
    detect_kernel = np.exp(-detect[:, np.newaxis] * r2[np.newaxis, :])
    data = encode_kernel @ propagator.T @ detect_kernel.T
    return RelaxationExchange2DResult(
        data=data,
        encode_times=encode,
        detect_times=detect,
        mixing_time=float(mixing_time),
        mixing_propagator=propagator,
        populations=populations,
        labels=system.labels,
    )
