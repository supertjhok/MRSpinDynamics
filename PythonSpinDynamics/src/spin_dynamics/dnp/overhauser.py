"""Liquid-state, high-temperature CW Overhauser rates and nuclear polarization.

Positive-gamma spin-1/2 nuclei and S=1/2 electrons, weak coupling and a single
well-mixed nuclear reservoir. Electron saturation is prescribed. This is not
a coupled electron population solver or a zero-field pumping model.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import array, fraction
from .spectral import TranslationalDiffusionModel

PROTON_GAMMA_HZ_PER_T = 42.57747892e6
ELECTRON_GAMMA_HZ_PER_T = 28.0249514242e9
PLANCK_J_S = 6.62607015e-34
BOLTZMANN_J_PER_K = 1.380649e-23
AVOGADRO_PER_MOL = 6.02214076e23
MU0_OVER_4PI = 1.00000000055e-7

ASSUMPTIONS = (
    "Positive-gamma spin-1/2 nucleus; S=1/2 electron; Zeeman/weak-coupling limit",
    "High temperature: h*gamma_e*B/(2*k*T) <= 0.1; linear thermal polarization",
    "Single homogeneous reservoir with prescribed effective electron saturation",
)


@dataclass(frozen=True)
class OverhauserRates:
    """Nuclear background, paramagnetic self and cross rates in s^-1.

    All arrays broadcast to a common condition shape. Cross rates are signed;
    positive means dipolar-like inversion for positive-gamma nuclei. Require
    |R_sigma| <= R_rho, as for nonnegative ZQ/DQ/SQ transition rates.
    """

    background_r1_per_second: np.ndarray
    self_relaxation_per_second: np.ndarray
    cross_relaxation_per_second: np.ndarray
    spectral_model: str = "measured_rates"

    def __post_init__(self):
        names = (
            "background_r1_per_second",
            "self_relaxation_per_second",
            "cross_relaxation_per_second",
        )
        values = np.broadcast_arrays(
            *(
                array(getattr(self, name), name, nonnegative=index < 2)
                for index, name in enumerate(names)
            )
        )
        if np.any(np.abs(values[2]) > values[1]):
            raise ValueError("|cross_relaxation_per_second| must not exceed self rate")
        array(values[0] + values[1], "total_r1_per_second", nonnegative=True)
        for name, value in zip(names, values):
            object.__setattr__(self, name, value.copy())

    @property
    def total_r1_per_second(self):
        """Return nuclear R1, including the radical contribution."""
        return self.background_r1_per_second + self.self_relaxation_per_second

    @property
    def t1_seconds(self):
        """Return nuclear T1; infinite if there is no relaxation."""
        rates = self.total_r1_per_second
        return np.divide(1.0, rates, out=np.full_like(rates, np.inf), where=rates > 0)

    @property
    def coupling_factor(self):
        """Return xi=R_sigma/R_rho, defined as zero for a zero paramagnetic rate."""
        rho = self.self_relaxation_per_second
        return np.divide(
            self.cross_relaxation_per_second, rho, out=np.zeros_like(rho), where=rho > 0
        )

    @property
    def leakage_factor(self):
        """Return f=R_rho/(R_10+R_rho), zero for a zero total rate."""
        total = self.total_r1_per_second
        return np.divide(
            self.self_relaxation_per_second,
            total,
            out=np.zeros_like(total),
            where=total > 0,
        )

    @classmethod
    def from_relaxivities(
        cls,
        concentration_molar,
        self_relaxivity_per_molar_second,
        cross_relaxivity_per_molar_second,
        *,
        background_r1_per_second=0.0,
    ):
        """Scale empirical relaxivities (s^-1 M^-1) by radical concentration (M)."""
        concentration = array(
            concentration_molar, "concentration_molar", nonnegative=True
        )
        rho = array(
            self_relaxivity_per_molar_second, "self_relaxivity", nonnegative=True
        )
        sigma = array(cross_relaxivity_per_molar_second, "cross_relaxivity")
        if np.any(np.abs(sigma) > rho):
            raise ValueError(
                "absolute cross relaxivity must not exceed self relaxivity"
            )
        return cls(background_r1_per_second, concentration * rho, concentration * sigma)

    @classmethod
    def from_t1(cls, t1_seconds, t10_seconds, *, coupling_factor):
        """Use matched-condition measured T1/T10 (including power-dependent arrays).

        T10 is radical-free. This method does not infer temperature or saturation
        from power; supply those separately to the CW calculation.
        """
        t1 = array(t1_seconds, "t1_seconds", positive=True)
        t10 = array(t10_seconds, "t10_seconds", positive=True)
        xi = array(coupling_factor, "coupling_factor")
        if np.any(np.abs(xi) > 1) or np.any(t1 > t10):
            raise ValueError("require |coupling_factor| <= 1 and T1 <= T10")
        rho = 1 / t1 - 1 / t10
        return cls(1 / t10, rho, xi * rho, "measured_T1_T10")


def translational_relaxation_rates(
    field_tesla,
    model: TranslationalDiffusionModel,
    concentration_molar,
    *,
    background_r1_per_second=0.0,
    scalar_self_rate_per_second=0.0,
    gamma_n_hz_per_t=PROTON_GAMMA_HZ_PER_T,
    gamma_e_hz_per_t=ELECTRON_GAMMA_HZ_PER_T,
) -> OverhauserRates:
    """Compute absolute FFHS dipolar rates and add an optional scalar rate.

    For S=1/2, A=(1/10)*(mu0/(4*pi)*hbar*gammaI*gammaS)^2
    *4*pi*n/(3*d^3), with angular gammas and n=1000*N_A*C. Then
    R_rho=A*(6*Jminus+3*JI+Jplus), R_sigma=A*(6*Jminus-Jplus).
    Positive frequency magnitudes are used: H_Z=+omega_e*S_z-omega_n*I_z,
    so the dipolar double-quantum spin transition is at omega_e-omega_n.
    J is the one-sided normalized-correlation spectrum in seconds.
    Scalar input q adds +q to self and -q to cross relaxation.
    """
    if not isinstance(model, TranslationalDiffusionModel):
        raise TypeError("model must be a TranslationalDiffusionModel")
    field = array(field_tesla, "field_tesla", positive=True)
    concentration = array(concentration_molar, "concentration_molar", nonnegative=True)
    scalar = array(
        scalar_self_rate_per_second, "scalar_self_rate_per_second", nonnegative=True
    )
    gn, ge = _gammas(gamma_n_hz_per_t, gamma_e_hz_per_t)
    wn, we = 2 * np.pi * gn * field, 2 * np.pi * ge * field
    ji = model.spectral_density(wn)
    jm, jp = model.spectral_density(we - wn), model.spectral_density(we + wn)
    hbar = PLANCK_J_S / (2 * np.pi)
    dipole = MU0_OVER_4PI * hbar * (2 * np.pi * gn) * (2 * np.pi * ge)
    variance = (
        4
        * np.pi
        * (1000 * AVOGADRO_PER_MOL * concentration)
        / (3 * model.closest_approach_m**3)
    )
    amplitude = dipole**2 * variance / 10
    return OverhauserRates(
        background_r1_per_second,
        amplitude * (6 * jm + 3 * ji + jp) + scalar,
        amplitude * (6 * jm - jp) - scalar,
        "FFHS_translational_plus_scalar" if np.any(scalar) else "FFHS_translational",
    )


def _gammas(nuclear, electron):
    gn = array(nuclear, "gamma_n_hz_per_t", positive=True)
    ge = array(electron, "gamma_e_hz_per_t", positive=True)
    if gn.ndim or ge.ndim or ge <= gn:
        raise ValueError("gamma magnitudes must be scalar with gamma_e > gamma_n > 0")
    return float(gn), float(ge)


@dataclass(frozen=True)
class OverhauserResult:
    """CW observables on a broadcast condition grid; E includes the baseline 1."""

    enhancement: np.ndarray
    thermal_polarization: np.ndarray
    steady_state_polarization: np.ndarray
    coupling_factor: np.ndarray
    leakage_factor: np.ndarray
    saturation_factor: np.ndarray
    field_tesla: np.ndarray
    temperature_kelvin: np.ndarray
    rates: OverhauserRates
    assumptions: tuple[str, ...] = ASSUMPTIONS

    def to_nano_mr_model(self, *, off_t1_seconds=None):
        """Adapt one CW condition to the existing nano-MR DNPModel.

        Assumes constant thermal polarization across pumping and recovery.
        Grids and zero total rates cannot map to the legacy scalar finite-T1
        helper. Existing nano-MR behavior is otherwise unchanged.
        """
        from spin_dynamics.nano_mr.high_resolution import DNPModel

        t1 = self.rates.t1_seconds
        if self.enhancement.ndim or not np.isfinite(t1):
            raise ValueError("adapter requires one scalar condition with finite T1")
        off = (
            t1
            if off_t1_seconds is None
            else array(off_t1_seconds, "off_t1_seconds", positive=True)
        )
        if np.ndim(off):
            raise ValueError("off_t1_seconds must be scalar")
        return DNPModel(float(self.enhancement), float(t1), float(off))


def cw_overhauser(
    rates: OverhauserRates,
    *,
    field_tesla,
    saturation_factor,
    temperature_kelvin=298.15,
    gamma_n_hz_per_t=PROTON_GAMMA_HZ_PER_T,
    gamma_e_hz_per_t=ELECTRON_GAMMA_HZ_PER_T,
) -> OverhauserResult:
    """Calculate E=1-xi*f*s*|gamma_e|/gamma_n and absolute polarization.

    Inputs broadcast. Thermal polarization is h*gamma_n*B/(2*k*T), consistent
    with the linear high-temperature transfer equation. Reject electron thermal
    arguments above 0.1 and unphysical polarization rather than clipping.
    Rates must be evaluated at the supplied field and temperature conditions.
    """
    if not isinstance(rates, OverhauserRates):
        raise TypeError("rates must be OverhauserRates")
    gn, ge = _gammas(gamma_n_hz_per_t, gamma_e_hz_per_t)
    field, temp, saturation, background, rho, sigma = np.broadcast_arrays(
        array(field_tesla, "field_tesla", positive=True),
        array(temperature_kelvin, "temperature_kelvin", positive=True),
        fraction(saturation_factor, "saturation_factor"),
        rates.background_r1_per_second,
        rates.self_relaxation_per_second,
        rates.cross_relaxation_per_second,
    )
    electron_argument = PLANCK_J_S * ge * field / (2 * BOLTZMANN_J_PER_K * temp)
    if np.any(electron_argument > 0.1):
        raise ValueError("outside high-temperature regime: h*gamma_e*B/(2*k*T) > 0.1")
    expanded = OverhauserRates(background, rho, sigma, rates.spectral_model)
    enhancement = (
        1 - expanded.coupling_factor * expanded.leakage_factor * saturation * ge / gn
    )
    thermal = electron_argument * gn / ge
    steady = enhancement * thermal
    if not np.all(np.isfinite(steady)) or np.any(np.abs(steady) > 1):
        raise ValueError(
            "model predicts invalid polarization; use a thermal population model"
        )
    return OverhauserResult(
        enhancement,
        thermal,
        steady,
        expanded.coupling_factor,
        expanded.leakage_factor,
        saturation.copy(),
        field.copy(),
        temp.copy(),
        expanded,
    )


def overhauser_from_factors(
    *,
    coupling_factor,
    leakage_factor,
    t1_seconds,
    field_tesla,
    saturation_factor,
    temperature_kelvin=298.15,
    gamma_n_hz_per_t=PROTON_GAMMA_HZ_PER_T,
    gamma_e_hz_per_t=ELECTRON_GAMMA_HZ_PER_T,
) -> OverhauserResult:
    """Calculate CW DNP from measured xi/f/s and a finite nuclear T1.

    T1 sets the buildup and recovery rate. The factor path is separate from
    the microscopic model so that competing parameterizations cannot conflict.
    """
    xi = array(coupling_factor, "coupling_factor")
    if np.any(np.abs(xi) > 1):
        raise ValueError("coupling_factor must lie in [-1, 1]")
    leakage = fraction(leakage_factor, "leakage_factor")
    r1 = 1 / array(t1_seconds, "t1_seconds", positive=True)
    rho = leakage * r1
    return cw_overhauser(
        OverhauserRates((1 - leakage) * r1, rho, xi * rho, "measured_factors"),
        field_tesla=field_tesla,
        saturation_factor=saturation_factor,
        temperature_kelvin=temperature_kelvin,
        gamma_n_hz_per_t=gamma_n_hz_per_t,
        gamma_e_hz_per_t=gamma_e_hz_per_t,
    )


@dataclass(frozen=True)
class OverhauserBuildup:
    """Time-leading polarization with shape (n_times, *condition_shape)."""

    times_seconds: np.ndarray
    polarization: np.ndarray
    target_polarization: np.ndarray
    pumping: bool


def polarization_buildup(
    result: OverhauserResult,
    times_seconds,
    *,
    initial_polarization=None,
    pumping=True,
    off_t1_seconds=None,
) -> OverhauserBuildup:
    """Evolve toward the CW steady state or the microwave-off thermal state.

    Default initial state is thermal when pumping, CW steady state on recovery.
    Recovery defaults to the same sample T1 (including radicals); supply a
    finite ``off_t1_seconds`` if heating changes it. Build a separate off-state
    result if its thermal polarization changes. A zero total rate holds the
    initial state. This is a prescribed-electron, single-exponential model.
    """
    times = array(times_seconds, "times_seconds", nonnegative=True)
    if times.ndim != 1 or not times.size:
        raise ValueError("times_seconds must be a non-empty 1-D axis")
    if pumping and off_t1_seconds is not None:
        raise ValueError("off_t1_seconds applies only to recovery")
    target = (
        result.steady_state_polarization if pumping else result.thermal_polarization
    )
    default_initial = (
        result.thermal_polarization if pumping else result.steady_state_polarization
    )
    initial = array(
        default_initial if initial_polarization is None else initial_polarization,
        "initial_polarization",
    )
    if np.any(np.abs(initial) > 1):
        raise ValueError("initial_polarization must lie in [-1, 1]")
    rate = result.rates.total_r1_per_second
    if off_t1_seconds is not None:
        rate = 1 / array(off_t1_seconds, "off_t1_seconds", positive=True)
    initial, target, rate = np.broadcast_arrays(initial, target, rate)
    t = times.reshape((times.size,) + (1,) * target.ndim)
    polarization = target + (initial - target) * np.exp(-t * rate)
    return OverhauserBuildup(times.copy(), polarization, target.copy(), bool(pumping))
