"""Independent translational, rotational dipolar and scalar-contact CW channels.

All rates describe one positive-gamma spin-1/2 nuclear reservoir and S=1/2
electrons. Add independent rates, not coupling factors. Correlated stochastic
interactions require cross-correlation terms and cannot use this composition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import array, fraction
from .overhauser import (
    ELECTRON_GAMMA_HZ_PER_T,
    MU0_OVER_4PI,
    PLANCK_J_S,
    PROTON_GAMMA_HZ_PER_T,
    OverhauserRates,
    OverhauserResult,
    _gammas,
    cw_overhauser,
    translational_relaxation_rates,
)
from .spectral import TranslationalDiffusionModel, rotational_spectral_density


def _scalar(value, name, *, positive=False):
    value = array(value, name, positive=positive, nonnegative=not positive)
    if value.ndim:
        raise ValueError(f"{name} must be scalar")
    return float(value)


@dataclass(frozen=True)
class RotationalDipolarModel:
    """Isotropic rotational diffusion of fixed-length electron–nuclear vectors.

    ``correlation_time_seconds`` is the rank-2 integral correlation time, equal
    to 1/(6*D_rot) for isotropic angular diffusion. ``pair_count`` is the mean
    number of independent pairs per observed nucleus (including occupancy),
    not radical molarity. Bound-pair and FFHS populations must not be counted
    twice. Each independent vector has the same supplied distance and time.
    """

    distance_m: float
    correlation_time_seconds: float
    pair_count: float = 1.0

    def __post_init__(self):
        for name in ("distance_m", "correlation_time_seconds", "pair_count"):
            object.__setattr__(
                self,
                name,
                _scalar(getattr(self, name), name, positive=name != "pair_count"),
            )

    def spectral_density(self, angular_frequency_rad_per_s):
        """Return the one-sided normalized rank-2 Lorentzian in seconds."""
        return rotational_spectral_density(
            angular_frequency_rad_per_s, self.correlation_time_seconds
        )

    def relaxation_rates(
        self,
        field_tesla,
        *,
        gamma_n_hz_per_t=PROTON_GAMMA_HZ_PER_T,
        gamma_e_hz_per_t=ELECTRON_GAMMA_HZ_PER_T,
    ) -> OverhauserRates:
        """Calculate absolute dipolar self/cross rates from distance and motion.

        b=(mu0/4pi)*hbar*gamma_n*gamma_e/r^3 with angular gammas.
        A=pair_count*b^2/10 for S=1/2 and the one-sided spectrum.
        DQ uses |omega_e|-omega_n, ZQ uses |omega_e|+omega_n.
        """
        field = array(field_tesla, "field_tesla", positive=True)
        gn, ge = _gammas(gamma_n_hz_per_t, gamma_e_hz_per_t)
        wn, we = 2 * np.pi * gn * field, 2 * np.pi * ge * field
        jm = self.spectral_density(we - wn)
        ji = self.spectral_density(wn)
        jp = self.spectral_density(we + wn)
        coupling = (
            MU0_OVER_4PI
            * (PLANCK_J_S / (2 * np.pi))
            * (2 * np.pi * gn)
            * (2 * np.pi * ge)
            / self.distance_m**3
        )
        amplitude = self.pair_count * coupling**2 / 10
        return OverhauserRates(
            np.zeros_like(field),
            amplitude * (6 * jm + 3 * ji + jp),
            amplitude * (6 * jm - jp),
            "rotational_dipolar",
        )


@dataclass(frozen=True)
class ScalarContactModel:
    """Exponentially correlated fluctuations of isotropic hyperfine contact.

    The fluctuating Hamiltonian is 2*pi*delta_A(t)*S.I (rad/s), where delta_A
    has zero mean and RMS ``rms_hyperfine_hz`` in Hz during contact. The total
    autocovariance is contact_fraction*A_rms^2*exp(-|t|/tau). The contact fraction
    is a supplied population/occupancy weight; it is not inferred from molarity.
    This simple stochastic model does not derive collision statistics or binding.
    A constant mean hyperfine coupling alone produces no scalar relaxation.
    """

    rms_hyperfine_hz: float
    correlation_time_seconds: float
    contact_fraction: float = 1.0

    def __post_init__(self):
        object.__setattr__(
            self, "rms_hyperfine_hz", _scalar(self.rms_hyperfine_hz, "rms_hyperfine_hz")
        )
        object.__setattr__(
            self,
            "correlation_time_seconds",
            _scalar(
                self.correlation_time_seconds, "correlation_time_seconds", positive=True
            ),
        )
        contact = fraction(self.contact_fraction, "contact_fraction")
        if contact.ndim:
            raise ValueError("contact_fraction must be scalar")
        object.__setattr__(self, "contact_fraction", float(contact))

    def spectral_density(self, angular_frequency_rad_per_s):
        """Return the one-sided normalized scalar autocorrelation spectrum."""
        return rotational_spectral_density(
            angular_frequency_rad_per_s, self.correlation_time_seconds
        )

    def relaxation_rates(
        self,
        field_tesla,
        *,
        gamma_n_hz_per_t=PROTON_GAMMA_HZ_PER_T,
        gamma_e_hz_per_t=ELECTRON_GAMMA_HZ_PER_T,
    ) -> OverhauserRates:
        """Return R_rho=q, R_sigma=-q with q=p*(2*pi*A_rms)^2*J(we+wn)/2.

        For H_Z=+we*Sz-wn*Iz, the scalar flip-flop transition samples the sum
        of positive frequency magnitudes. The factor 1/2 follows from its
        spin-1/2 matrix element and the one-sided Fourier convention.
        """
        field = array(field_tesla, "field_tesla", positive=True)
        gn, ge = _gammas(gamma_n_hz_per_t, gamma_e_hz_per_t)
        q = (
            self.contact_fraction
            * (2 * np.pi * self.rms_hyperfine_hz) ** 2
            * (self.spectral_density(2 * np.pi * (ge + gn) * field))
            / 2
        )
        return OverhauserRates(
            np.zeros_like(field), q, -q, "scalar_contact_exponential"
        )


@dataclass(frozen=True)
class MotionBasedCWModel:
    """Compose independent FFHS, bound-vector rotation and contact fluctuations.

    Only the FFHS channel uses ``concentration_molar``. Rotational pair counts
    and scalar contact fractions are separately supplied populations. Their
    concentration/temperature dependence is not inferred by this model.
    Electron saturation remains prescribed; this is not electron dynamics.
    """

    translational_model: TranslationalDiffusionModel | None = None
    concentration_molar: float = 0.0
    rotational_models: tuple[RotationalDipolarModel, ...] = ()
    scalar_models: tuple[ScalarContactModel, ...] = ()

    def __post_init__(self):
        concentration = _scalar(self.concentration_molar, "concentration_molar")
        object.__setattr__(self, "concentration_molar", concentration)
        if self.translational_model is not None and not isinstance(
            self.translational_model, TranslationalDiffusionModel
        ):
            raise TypeError("translational_model must be a TranslationalDiffusionModel")
        if concentration and self.translational_model is None:
            raise ValueError("concentration_molar requires a translational_model")
        for name, cls in (
            ("rotational_models", RotationalDipolarModel),
            ("scalar_models", ScalarContactModel),
        ):
            models = tuple(getattr(self, name))
            if not all(isinstance(model, cls) for model in models):
                raise TypeError(f"{name} must contain {cls.__name__} objects")
            object.__setattr__(self, name, models)

    def rate_components(
        self,
        field_tesla,
        *,
        background_r1_per_second=0.0,
        gamma_n_hz_per_t=PROTON_GAMMA_HZ_PER_T,
        gamma_e_hz_per_t=ELECTRON_GAMMA_HZ_PER_T,
    ) -> dict[str, OverhauserRates]:
        """Return each independent channel and the background contribution.

        Keys are background, translational (if present), rotational[i], scalar[i].
        Components broadcast with the supplied field/background arrays; background
        is included exactly once. No correlation terms between channels are used.
        """
        field = array(field_tesla, "field_tesla", positive=True)
        gn, ge = _gammas(gamma_n_hz_per_t, gamma_e_hz_per_t)
        gammas = dict(gamma_n_hz_per_t=gn, gamma_e_hz_per_t=ge)
        zeros = np.zeros_like(field)
        parts = {
            "background": OverhauserRates(
                background_r1_per_second, zeros, zeros, "background"
            )
        }
        if self.translational_model is not None:
            parts["translational"] = translational_relaxation_rates(
                field, self.translational_model, self.concentration_molar, **gammas
            )
        for prefix, models in (
            ("rotational", self.rotational_models),
            ("scalar", self.scalar_models),
        ):
            for index, model in enumerate(models):
                parts[f"{prefix}[{index}]"] = model.relaxation_rates(field, **gammas)
        return parts

    def relaxation_rates(
        self,
        field_tesla,
        *,
        background_r1_per_second=0.0,
        gamma_n_hz_per_t=PROTON_GAMMA_HZ_PER_T,
        gamma_e_hz_per_t=ELECTRON_GAMMA_HZ_PER_T,
    ) -> OverhauserRates:
        """Sum independent relaxation rates before calculating any CW factors."""
        parts = self.rate_components(
            field_tesla,
            background_r1_per_second=background_r1_per_second,
            gamma_n_hz_per_t=gamma_n_hz_per_t,
            gamma_e_hz_per_t=gamma_e_hz_per_t,
        ).values()
        return OverhauserRates(
            sum(part.background_r1_per_second for part in parts),
            sum(part.self_relaxation_per_second for part in parts),
            sum(part.cross_relaxation_per_second for part in parts),
            "independent_motion_channels",
        )

    def cw(
        self,
        field_tesla,
        *,
        saturation_factor,
        temperature_kelvin=298.15,
        background_r1_per_second=0.0,
        gamma_n_hz_per_t=PROTON_GAMMA_HZ_PER_T,
        gamma_e_hz_per_t=ELECTRON_GAMMA_HZ_PER_T,
    ) -> OverhauserResult:
        """Calculate CW observables from molecular motion and supplied saturation."""
        rates = self.relaxation_rates(
            field_tesla,
            background_r1_per_second=background_r1_per_second,
            gamma_n_hz_per_t=gamma_n_hz_per_t,
            gamma_e_hz_per_t=gamma_e_hz_per_t,
        )
        return cw_overhauser(
            rates,
            field_tesla=field_tesla,
            saturation_factor=saturation_factor,
            temperature_kelvin=temperature_kelvin,
            gamma_n_hz_per_t=gamma_n_hz_per_t,
            gamma_e_hz_per_t=gamma_e_hz_per_t,
        )
