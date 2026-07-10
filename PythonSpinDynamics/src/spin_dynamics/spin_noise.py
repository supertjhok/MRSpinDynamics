"""Physical spin-noise models: sample impedance and stochastic source.

Implements the two models planned in ``docs/spin_noise.md``:

- **Option B** -- :class:`SampleCoupling` describes the Hoult & Ginsberg sample
  impedance ``Z_n(dw) = R_n0 * (1 - j*dw*T2) / (1 + (dw*T2)^2)`` that sits in
  series with the receive coil. The ``*_probe_output_noise_density`` functions
  in :mod:`spin_dynamics.noise` accept it via their ``sample=`` keyword and
  add the sample's Nyquist noise ``4*k*T_s*R_n`` at the *sample* temperature
  while the coil resistance stays at the *coil* temperature.
- **Option C** -- :func:`simulate_spin_noise` integrates the rotating-frame
  Bloch equations with a complex Ornstein-Uhlenbeck Langevin force on the
  transverse magnetization plus a coil-temperature Johnson force, coupled to
  the probe through radiation-damping back-action. The coil-noise realization
  is shared between the spin drive and the receiver EMF, which is what
  produces the equilibrium dip / hot-spin bump phenomenology in the time
  domain.

Both models share one coupling constant with the deterministic
radiation-damping module through the identity
``R_n0 = R_coil * T2 / (2 * Trd)``.

Feedback phase: like the deterministic :mod:`spin_dynamics.radiation_damping`
integrators, this module uses the phase-independent feedback form
``feedback target = -mxy / Trd`` (Bloom 1957), under which the damping and the
equilibrium noise statistics are independent of the transverse phase. See the
"Feedback Phase Convention" section of ``docs/radiation_damping.md`` for why
the textbook's conjugate form is not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.radiation_damping import (
    MU0,
    RadiationDampingProbe,
)
from spin_dynamics.sample import Sample

__all__ = [
    "SampleCoupling",
    "SpinNoiseSource",
    "SpinNoiseResult",
    "sample_resistance_on_resonance",
    "sample_coupling_from_sample",
    "sample_coupling_from_radiation_damping",
    "spin_noise_source_from_sample",
    "simulate_spin_noise",
    "spin_noise_output_psd",
    "estimate_spin_noise_spectrum",
]


# ---------------------------------------------------------------------------
# Option B: sample impedance in the receiver circuit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleCoupling:
    """Sample impedance parameters for the receiver noise model.

    ``r_n0`` is the on-resonance NMR sample resistance in ohms (Hoult &
    Ginsberg Eq. 22), ``t2`` the transverse relaxation time in seconds that
    sets the Lorentzian width, and ``temperature`` the *sample* temperature in
    kelvin used for its Nyquist source. ``offset`` optionally shifts the spin
    resonance (rad/s) relative to the probe carrier ``w0``.
    """

    r_n0: float
    t2: float
    temperature: float
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.r_n0 < 0 or not np.isfinite(self.r_n0):
            raise ValueError("r_n0 must be finite and non-negative")
        if self.t2 <= 0 or not np.isfinite(self.t2):
            raise ValueError("t2 must be finite and positive")
        if self.temperature < 0 or not np.isfinite(self.temperature):
            raise ValueError("temperature must be finite and non-negative")
        if not np.isfinite(self.offset):
            raise ValueError("offset must be finite")

    def z_n(self, delta_omega: np.ndarray) -> np.ndarray:
        """Series sample impedance at offset ``delta_omega`` rad/s from ``w0``.

        ``Z_n = R_n0 / (1 + j*(dw - offset)*T2)``: the real part is the
        absorptive Lorentzian ``R_n``, the imaginary part the dispersive
        reactance ``X_n`` responsible for noise-line frequency pulling.
        """

        dw = np.asarray(delta_omega, dtype=np.float64) - self.offset
        return self.r_n0 / (1.0 + 1j * dw * self.t2)

    def r_n(self, delta_omega: np.ndarray) -> np.ndarray:
        """Absorptive sample resistance ``Re(Z_n)`` in ohms."""

        return np.real(self.z_n(delta_omega))


def sample_resistance_on_resonance(
    *,
    fill_factor: float,
    gamma: float,
    magnetization_density: float,
    t2: float,
    omega0: float,
    inductance: float,
) -> float:
    """On-resonance sample resistance ``R_n0 = mu0*eta*gamma*M0*T2*w0*L / 4``.

    Equivalent to Hoult & Ginsberg Eq. 22 with the solenoid relation
    ``B1_hat^2 = mu0 * L / (2 * V_c)`` and the magnetic-energy fill factor
    ``eta = B1_hat^2 * V_s / (mu0 * L)``, and to ``R_coil * T2 / (2 * Trd)``
    with the package radiation-damping convention.
    """

    if not (0 < fill_factor <= 1):
        raise ValueError("fill_factor must be in the interval (0, 1]")
    for label, value in (
        ("gamma", gamma),
        ("magnetization_density", magnetization_density),
        ("t2", t2),
        ("omega0", omega0),
        ("inductance", inductance),
    ):
        if value <= 0 or not np.isfinite(value):
            raise ValueError(f"{label} must be finite and positive")
    return float(
        MU0 * fill_factor * gamma * magnetization_density * t2 * omega0 * inductance / 4.0
    )


def sample_coupling_from_sample(
    sample: Sample,
    *,
    field_tesla: float,
    fill_factor: float,
    inductance: float,
    omega0: float | None = None,
    offset: float = 0.0,
) -> SampleCoupling:
    """Build the receiver-circuit coupling for ``sample`` in a given coil.

    ``omega0`` defaults to ``gamma * B0``. ``sample.t2`` must be finite since
    it sets the noise linewidth.
    """

    if not np.isfinite(sample.t2):
        raise ValueError("sample.t2 must be finite to define the spin-noise linewidth")
    w0 = sample.gamma * float(field_tesla) if omega0 is None else float(omega0)
    r_n0 = sample_resistance_on_resonance(
        fill_factor=fill_factor,
        gamma=sample.gamma,
        magnetization_density=sample.magnetization_density(field_tesla),
        t2=sample.t2,
        omega0=w0,
        inductance=inductance,
    )
    return SampleCoupling(
        r_n0=r_n0,
        t2=sample.t2,
        temperature=sample.temperature,
        offset=float(offset),
    )


def sample_coupling_from_radiation_damping(
    *,
    trd: float,
    coil_resistance: float,
    t2: float,
    temperature: float,
    offset: float = 0.0,
) -> SampleCoupling:
    """Build a coupling from the radiation-damping identity ``R_n0 = R*T2/(2*Trd)``."""

    if trd <= 0 or not np.isfinite(trd):
        raise ValueError("trd must be finite and positive")
    if coil_resistance <= 0 or not np.isfinite(coil_resistance):
        raise ValueError("coil_resistance must be finite and positive")
    return SampleCoupling(
        r_n0=float(coil_resistance) * float(t2) / (2.0 * float(trd)),
        t2=float(t2),
        temperature=float(temperature),
        offset=float(offset),
    )


# ---------------------------------------------------------------------------
# Option C: stochastic source-level model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpinNoiseSource:
    """Stochastic spin-noise drive in normalized magnetization units.

    ``m_rms`` is the stationary RMS transverse fluctuation relative to the
    equilibrium magnetization (``Sample.normalized_transverse_fluctuation``).
    ``t2`` is the spin-bath correlation time in seconds.
    ``sample_temperature`` and ``coil_temperature`` (kelvin) set the two bath
    temperatures; the fluctuation-dissipation calibration is

    ``sigma_s^2 = (2/T2)  * m_rms^2``                      (spin bath)
    ``sigma_c^2 = (2/Trd) * m_rms^2 * (Tc/Ts)``            (coil channel)

    so at ``Tc == Ts`` the two channels are in detailed balance.
    ``spin_offset`` (rad/s) detunes the spins from the probe carrier.
    """

    m_rms: float
    t2: float
    sample_temperature: float
    coil_temperature: float | None = None
    spin_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.m_rms < 0 or not np.isfinite(self.m_rms):
            raise ValueError("m_rms must be finite and non-negative")
        if self.t2 <= 0:
            raise ValueError("t2 must be positive")
        if self.sample_temperature <= 0 or not np.isfinite(self.sample_temperature):
            raise ValueError("sample_temperature must be finite and positive")
        if self.coil_temperature is not None and (
            self.coil_temperature < 0 or not np.isfinite(self.coil_temperature)
        ):
            raise ValueError("coil_temperature must be finite and non-negative")
        if not np.isfinite(self.spin_offset):
            raise ValueError("spin_offset must be finite")

    @property
    def effective_coil_temperature(self) -> float:
        return (
            self.sample_temperature
            if self.coil_temperature is None
            else self.coil_temperature
        )

    def spin_bath_force_density(self) -> float:
        """Langevin force PSD ``sigma_s^2`` of the T2 bath (1/s units)."""

        if not np.isfinite(self.t2):
            return 0.0
        return 2.0 * self.m_rms**2 / self.t2

    def coil_force_density(self, trd: float) -> float:
        """Langevin force PSD ``sigma_c^2`` of the coil channel (1/s units)."""

        return (
            2.0
            * self.m_rms**2
            * self.effective_coil_temperature
            / (self.sample_temperature * float(trd))
        )


def spin_noise_source_from_sample(
    sample: Sample,
    *,
    field_tesla: float,
    coil_temperature: float | None = None,
    spin_offset: float = 0.0,
) -> SpinNoiseSource:
    """Build the normalized stochastic source for ``sample`` at ``field_tesla``."""

    if not np.isfinite(sample.t2):
        raise ValueError("sample.t2 must be finite to define the spin-noise bandwidth")
    return SpinNoiseSource(
        m_rms=sample.normalized_transverse_fluctuation(field_tesla),
        t2=sample.t2,
        sample_temperature=sample.temperature,
        coil_temperature=coil_temperature,
        spin_offset=float(spin_offset),
    )


@dataclass(frozen=True)
class SpinNoiseResult:
    """Stochastic magnetization trajectory and receiver-node signal.

    ``mxy``/``mz`` are the normalized magnetization components, ``coil_noise``
    the coil Johnson EMF-node fluctuation in the same normalized units, and
    ``emf = mxy + coil_noise`` the receiver-node signal whose power spectrum
    shows the spin-noise feature on the coil-noise baseline. ``feedback`` is
    the probe back-action state (per second).
    """

    time: np.ndarray
    mxy: np.ndarray
    mz: np.ndarray
    feedback: np.ndarray
    coil_noise: np.ndarray
    probe: RadiationDampingProbe
    source: SpinNoiseSource
    model: str

    @property
    def emf(self) -> np.ndarray:
        return self.mxy + self.coil_noise


def _complex_normals(
    rng: np.random.Generator, size: int, variance: float
) -> np.ndarray:
    """Complex Gaussians with ``E[|x|^2] = variance`` and ``E[x^2] = 0``."""

    if variance <= 0.0:
        return np.zeros(size, dtype=np.complex128)
    sigma = np.sqrt(variance / 2.0)
    return rng.normal(scale=sigma, size=size) + 1j * rng.normal(scale=sigma, size=size)


def simulate_spin_noise(
    time: np.ndarray,
    probe: RadiationDampingProbe,
    source: SpinNoiseSource,
    *,
    initial_mxy: complex = 0.0 + 0.0j,
    initial_mz: float = 1.0,
    t1: float = np.inf,
    equilibrium_mz: float = 1.0,
    model: str = "instant",
    max_step: float | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> SpinNoiseResult:
    """Integrate the stochastic Bloch equations with probe back-action.

    Deterministic part (phase-independent radiation damping):

    ``dmxy/dt = (j*spin_offset - 1/T2) mxy + mz * feedback``
    ``dmz/dt  = -Re(conj(mxy) * feedback) + (meq - mz)/T1``

    with feedback target ``-(mxy + e(t)) / Trd`` (instant model: applied
    directly; circuit model: through the resonator ringdown ``2Q/w0`` with
    optional probe detuning). ``e(t)`` is the coil Johnson EMF-node noise,
    white with density ``2*Trd*m_rms^2*(Tc/Ts)`` in normalized units; the same
    realization is returned as ``coil_noise`` so the receiver signal
    ``emf = mxy + e`` carries the physical spin/coil noise correlation. The
    spin bath adds the Langevin force ``sigma_s`` on ``mxy``.

    Integration is Euler-Maruyama with substeps resolving ``T2``, ``Trd``,
    ``T1``, and the resonator time constant. With ``m_rms = 0`` and zero
    ``coil_temperature`` the deterministic radiation-damping dynamics are
    recovered (in the phase-independent convention -- see module docstring).
    """

    time = np.asarray(time, dtype=np.float64).reshape(-1)
    if time.size < 2:
        raise ValueError("time must contain at least two samples")
    steps_dt = np.diff(time)
    if not np.all(steps_dt > 0):
        raise ValueError("time must be strictly increasing")
    if t1 <= 0 and np.isfinite(t1):
        raise ValueError("t1 must be positive")
    if model not in {"instant", "circuit"}:
        raise ValueError("model must be 'instant' or 'circuit'")
    if rng is not None and seed is not None:
        raise ValueError("provide either rng or seed, not both")
    generator = rng if rng is not None else np.random.default_rng(seed)

    trd = probe.trd
    t2 = float(source.t2)
    sigma_s2 = source.spin_bath_force_density()
    sigma_c2 = source.coil_force_density(trd)
    # EMF-node coil noise density: e = sigma_c * Trd * u with unit-density u,
    # so S_e = sigma_c^2 * Trd^2 (see docs/spin_noise.md).
    coil_noise_density = sigma_c2 * trd**2

    step_limit_candidates = [trd / 50.0, t2 / 50.0 if np.isfinite(t2) else np.inf]
    if model == "circuit":
        step_limit_candidates.append(probe.resonator_time_constant / 20.0)
    if np.isfinite(t1):
        step_limit_candidates.append(float(t1) / 50.0)
    if max_step is not None:
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        step_limit_candidates.append(float(max_step))
    step_limit = min(c for c in step_limit_candidates if np.isfinite(c))

    mxy = complex(initial_mxy)
    mz = float(initial_mz)
    feedback = 0.0 + 0.0j

    out_mxy = np.zeros(time.size, dtype=np.complex128)
    out_mz = np.zeros(time.size, dtype=np.float64)
    out_feedback = np.zeros(time.size, dtype=np.complex128)
    out_coil = np.zeros(time.size, dtype=np.complex128)
    out_mxy[0] = mxy
    out_mz[0] = mz

    offset = float(source.spin_offset)
    meq = float(equilibrium_mz)

    for idx in range(time.size - 1):
        h_total = float(steps_dt[idx])
        nsub = max(1, int(np.ceil(h_total / step_limit)))
        h = h_total / nsub
        # Per-substep noise: spin-bath increments (variance sigma_s^2 * h) and
        # the piecewise-constant coil EMF-node noise e with E[|e|^2] = S_e / h.
        spin_kicks = _complex_normals(generator, nsub, sigma_s2 * h)
        coil_samples = _complex_normals(
            generator, nsub, coil_noise_density / h if coil_noise_density > 0 else 0.0
        )
        for sub in range(nsub):
            e = coil_samples[sub]
            target = -(mxy + e) / trd
            if model == "instant":
                feedback = target
                dfeedback = 0.0 + 0.0j
            else:
                tau = probe.resonator_time_constant
                dfeedback = (target - feedback) / tau - 1j * probe.detuning * feedback

            dmxy = 1j * offset * mxy + mz * feedback
            if np.isfinite(t2):
                dmxy -= mxy / t2
            dmz = -(mxy.conjugate() * feedback).real
            if np.isfinite(t1):
                dmz += (meq - mz) / t1

            mxy = mxy + h * dmxy + spin_kicks[sub]
            mz = mz + h * dmz
            feedback = feedback + h * dfeedback
        out_mxy[idx + 1] = mxy
        out_mz[idx + 1] = mz
        out_feedback[idx + 1] = feedback
        # Average e over the interval so the stored samples keep the white
        # density S_e at the output rate regardless of the substep count.
        out_coil[idx + 1] = np.mean(coil_samples) if coil_noise_density > 0 else 0.0

    return SpinNoiseResult(
        time=time,
        mxy=out_mxy,
        mz=out_mz,
        feedback=out_feedback,
        coil_noise=out_coil,
        probe=probe,
        source=source,
        model=model,
    )


def spin_noise_output_psd(
    frequencies_hz: np.ndarray,
    *,
    source: SpinNoiseSource,
    trd: float,
) -> np.ndarray:
    """Analytic linear-regime PSD of the receiver-node signal ``emf``.

    Small-fluctuation limit (``mz ~ 1``, instant probe model): with
    ``Gamma = 1/T2 + 1/Trd`` and ``D = j*(w - spin_offset) + Gamma``,

    ``S(w) = sigma_s^2 / |D|^2 + S_e * |1 - (1/Trd)/D|^2``

    two-sided, in normalized-magnetization units squared per Hz (input
    frequencies in Hz). At ``Tc == Ts`` this shows the equilibrium dip
    ``S(0)/S_e = Trd/(T2+Trd)``; hot spins / cold coil turn it into a bump.
    """

    w = 2.0 * np.pi * np.asarray(frequencies_hz, dtype=np.float64)
    gamma_rate = 1.0 / trd + (1.0 / source.t2 if np.isfinite(source.t2) else 0.0)
    d = 1j * (w - source.spin_offset) + gamma_rate
    sigma_s2 = source.spin_bath_force_density()
    s_e = source.coil_force_density(trd) * trd**2
    return sigma_s2 / np.abs(d) ** 2 + s_e * np.abs(1.0 - (1.0 / trd) / d) ** 2


def estimate_spin_noise_spectrum(
    signal: np.ndarray,
    dt: float,
    *,
    block_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bartlett-averaged two-sided PSD of a complex time series.

    Splits ``signal`` into contiguous blocks of ``block_samples`` (Field &
    Bain recommend block lengths of order ``T2``), FFTs each, and averages the
    power. Returns ``(frequencies_hz, psd)`` with the PSD normalized so that
    ``sum(psd) * df = mean |signal|^2`` (units squared per Hz, fftshifted).
    """

    sig = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    if block_samples < 2:
        raise ValueError("block_samples must be at least 2")
    nblocks = sig.size // int(block_samples)
    if nblocks < 1:
        raise ValueError("signal is shorter than one block")
    trimmed = sig[: nblocks * int(block_samples)].reshape(nblocks, int(block_samples))
    spectra = np.fft.fftshift(np.fft.fft(trimmed, axis=1), axes=1)
    psd = np.mean(np.abs(spectra) ** 2, axis=0) * dt / int(block_samples)
    freqs = np.fft.fftshift(np.fft.fftfreq(int(block_samples), d=dt))
    return freqs, psd
