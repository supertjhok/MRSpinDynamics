"""Resolved-line, high-temperature nitroxide CW saturation populations.

Electron exchange follows the population-swap collision model of Bates and
Drozdoski, as described in DOI 10.1039/C5CP00935A (supporting information).
This is a solution ensemble saturation model, separate from a closed electron–
proton pair. Nitrogen Zeeman and hyperfine Boltzmann corrections are neglected:
the thermal state is an electron Boltzmann distribution times uniform nitrogen.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import array
from .overhauser import (
    BOLTZMANN_J_PER_K,
    ELECTRON_GAMMA_HZ_PER_T,
    PLANCK_J_S,
)


def _scalar(value, name, *, positive=False):
    result = array(value, name, positive=positive, nonnegative=not positive)
    if result.ndim:
        raise ValueError(f"{name} must be scalar")
    return float(result)


@dataclass(frozen=True)
class NitroxideSaturationResult:
    """CW populations (..., nitrogen m, electron +/-), and total saturation.

    Electron polarization is p_minus - p_plus; line polarizations are
    conditional on nitrogen m. Total saturation uses all lines, not only the
    irradiated line. Nitrogen populations sum over the final electron axis.
    """

    nitrogen_m: np.ndarray
    populations: np.ndarray
    line_polarizations: np.ndarray
    line_saturation: np.ndarray
    saturation_factor: np.ndarray
    thermal_electron_polarization: np.ndarray
    exchange_rate_per_second: np.ndarray
    t2e_seconds: np.ndarray
    microwave_transition_rates_per_second: np.ndarray

    @property
    def nitrogen_populations(self):
        """Return the probability in each nitrogen hyperfine manifold."""
        return self.populations.sum(axis=-1)

    @property
    def electron_polarization(self):
        """Return ensemble p_minus - p_plus, summed over nitrogen states."""
        return np.sum(self.populations[..., 1] - self.populations[..., 0], axis=-1)


@dataclass(frozen=True)
class NitroxideSaturationModel:
    """CW electron/nitrogen populations for resolved 14N or 15N nitroxides.

    Supply measured intrinsic T1e, T2e and hyperfine splitting in Hz. The
    optional T2 broadening coefficient adds coefficient*C to 1/T2e, independently
    of exchange; it is not automatically inferred from the collision rate.
    C is mol/L, exchange coefficient is M^-1 s^-1, and k_ex=coefficient*C is
    the decay rate of electron–nitrogen population correlations in a random
    electron-swap collision ensemble. Other exchange conventions can differ.

    Nitrogen relaxation is an adjacent-m symmetric transition rate w_N, s^-1
    per direction (not an asserted nitrogen T1). The electron bath has common
    polarization tanh(h gamma_e B / 2 k T) on all lines. Microwave rates are
    W_m=Omega1^2*T2/[2*(1+Delta_m^2*T2^2)], with Delta_m=Delta+2*pi*A*m.
    Coherences have been eliminated for CW steady state. This does not model
    exchange coherence transfer, line coalescence, coherent transients,
    anisotropic spectra, heating, or concentration-dependent molecular motion.
    """

    hyperfine_hz: float = 45e6
    electron_t1_seconds: float = 0.5e-6
    electron_t2_seconds: float = 0.1e-6
    nitrogen_transition_per_second: float = 0.0
    exchange_coefficient_per_molar_second: float = 0.0
    transverse_broadening_per_molar_second: float = 0.0
    nitrogen_spin: float = 1.0

    def __post_init__(self):
        for name in ("hyperfine_hz", "electron_t1_seconds", "electron_t2_seconds"):
            object.__setattr__(
                self, name, _scalar(getattr(self, name), name, positive=True)
            )
        for name in (
            "nitrogen_transition_per_second",
            "exchange_coefficient_per_molar_second",
            "transverse_broadening_per_molar_second",
        ):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if self.nitrogen_spin not in (0.5, 1.0):
            raise ValueError("nitrogen_spin must be 1 (14N) or 0.5 (15N)")
        if self.electron_t2_seconds > 2 * self.electron_t1_seconds:
            raise ValueError("electron_t2_seconds must be <= 2*T1e")

    @property
    def nitrogen_m(self):
        """Return ascending nitrogen projections labeling the hyperfine lines."""
        return np.arange(-self.nitrogen_spin, self.nitrogen_spin + 1)

    def _parameters(
        self,
        field_tesla,
        concentration_molar,
        nutation_rad_per_s,
        detuning_rad_per_s,
        temperature_kelvin,
    ):
        field, concentration, omega, delta, temperature = np.broadcast_arrays(
            array(field_tesla, "field_tesla", positive=True),
            array(concentration_molar, "concentration_molar", nonnegative=True),
            array(nutation_rad_per_s, "nutation_rad_per_s", nonnegative=True),
            array(detuning_rad_per_s, "detuning_rad_per_s"),
            array(temperature_kelvin, "temperature_kelvin", positive=True),
        )
        argument = (
            PLANCK_J_S
            * ELECTRON_GAMMA_HZ_PER_T
            * field
            / (2 * BOLTZMANN_J_PER_K * temperature)
        )
        if np.any(argument > 0.1):
            raise ValueError(
                "resolved-line model requires high temperature (electron argument <= 0.1)"
            )
        if np.any(
            ELECTRON_GAMMA_HZ_PER_T * field
            < 10 * self.hyperfine_hz * self.nitrogen_spin
        ):
            raise ValueError("field is too low for the secular resolved-line model")
        with np.errstate(over="ignore", invalid="ignore"):
            exchange = self.exchange_coefficient_per_molar_second * concentration
            t2 = 1 / (
                1 / self.electron_t2_seconds
                + self.transverse_broadening_per_molar_second * concentration
            )
            offsets = delta[..., None] + 2 * np.pi * self.hyperfine_hz * self.nitrogen_m
            microwave = (
                0.5
                * (omega[..., None] * np.sqrt(t2[..., None])) ** 2
                / (1 + (offsets * t2[..., None]) ** 2)
            )
        if not (
            np.isfinite(exchange).all()
            and np.isfinite(microwave).all()
            and np.all(t2 > 0)
        ):
            raise ValueError("parameters exceed finite numerical rate range")
        return np.tanh(argument), exchange, t2, microwave

    def steady_state(
        self,
        *,
        field_tesla,
        concentration_molar,
        nutation_rad_per_s,
        detuning_rad_per_s=0.0,
        temperature_kelvin=298.15,
    ):
        """Solve the CW population balance with initially uniform nitrogen.

        Inputs broadcast; the final axes of populations are (2I+1, 2). At zero
        nitrogen relaxation, uniform nitrogen is a specified invariant sector,
        not a claim of a unique stationary state over all nitrogen marginals.
        The exact collision term k*(p_N outer p_e - p) reduces to a linear
        line-polarization system in this sector. No population clipping is used.
        """
        thermal, exchange, t2, microwave = self._parameters(
            field_tesla,
            concentration_molar,
            nutation_rad_per_s,
            detuning_rad_per_s,
            temperature_kelvin,
        )
        n = len(self.nitrogen_m)
        nitrogen_laplacian = np.zeros((n, n))
        for index in range(n - 1):
            nitrogen_laplacian[index : index + 2, index : index + 2] += np.array(
                [[1, -1], [-1, 1]]
            )
        matrix = (
            self.nitrogen_transition_per_second * nitrogen_laplacian
            + exchange[..., None, None] * (np.eye(n) - np.ones((n, n)) / n)
            + (1 / self.electron_t1_seconds + 2 * microwave)[..., :, None] * np.eye(n)
        )
        if not np.isfinite(matrix).all() or np.any(np.linalg.cond(matrix) > 1e12):
            raise ValueError(
                "population balance is numerically unresolved at these rate ratios"
            )
        # Solve saturation directly: M*s=2W. This gives exactly zero with no
        # drive and retains relative precision for weak microwave pumping.
        saturation = np.linalg.solve(matrix, (2 * microwave)[..., None])[..., 0]
        polarization = (1 - saturation) * thermal[..., None]
        populations = np.stack((1 - polarization, 1 + polarization), axis=-1) / (2 * n)
        return NitroxideSaturationResult(
            self.nitrogen_m,
            populations,
            polarization,
            saturation,
            np.mean(saturation, axis=-1),
            thermal,
            exchange,
            t2,
            microwave,
        )

    def population_derivative(
        self,
        populations,
        *,
        field_tesla,
        concentration_molar,
        nutation_rad_per_s,
        detuning_rad_per_s=0.0,
        temperature_kelvin=298.15,
    ):
        """Evaluate full population kinetics, including nonlinear swap collisions.

        Accept a single normalized nonnegative (2I+1, 2) population array and
        scalar experimental parameters. Exchange alone preserves both nitrogen
        marginals and total electron magnetization for arbitrary populations.
        """
        p = array(populations, "populations", nonnegative=True)
        if p.shape != (len(self.nitrogen_m), 2) or not np.isclose(
            p.sum(), 1, atol=1e-12, rtol=0
        ):
            raise ValueError("populations must be normalized with shape (2I+1, 2)")
        thermal, exchange, _, microwave = self._parameters(
            field_tesla,
            concentration_molar,
            nutation_rad_per_s,
            detuning_rad_per_s,
            temperature_kelvin,
        )
        if thermal.ndim:
            raise ValueError(
                "population_derivative requires scalar experimental parameters"
            )
        flux = ((1 + thermal) * p[:, 0] - (1 - thermal) * p[:, 1]) / (
            2 * self.electron_t1_seconds
        )
        flux += microwave * (p[:, 0] - p[:, 1])
        derivative = np.stack((-flux, flux), axis=-1)
        for index in range(len(p) - 1):
            flow = self.nitrogen_transition_per_second * (p[index + 1] - p[index])
            derivative[index] += flow
            derivative[index + 1] -= flow
        # Keep the pair-collision loss proportional to total probability, too.
        # It equals p on the normalized simplex and prevents numerical mass
        # errors from being amplified during intermediate integration stages.
        derivative += exchange * (
            p.sum(axis=1)[:, None] * p.sum(axis=0)[None, :] - p * p.sum()
        )
        return derivative
