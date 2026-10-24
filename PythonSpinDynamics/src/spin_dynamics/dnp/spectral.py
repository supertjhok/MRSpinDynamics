"""One-sided normalized-correlation spectra for liquid-state Overhauser DNP.

Hwang and Freed, JCP 63, 4017 (1975), DOI 10.1063/1.431841,
Appendix A. Here J = Re integral_0^infinity C(t)/C(0) exp(-i*w*t) dt
has units seconds. It excludes spin density, radial variance and spin factors.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import array


def ffhs_spectral_density(angular_frequency_rad_per_s, diffusion_time_seconds):
    """Return the FFHS spectrum in seconds, with J(0) = 4*tau_D/9.

    ``tau_D = d**2 / D_relative``. Frequencies may be signed; the spectrum
    is even. The shape is Re[(1+z/4)/(1+z+4*z**2/9+z**3/9)],
    z=sqrt(i*abs(w)*tau_D). A real rational form avoids complex cancellation
    at high frequency. This is half the two-sided Fourier convention used
    by the shared rotational relaxation helper.
    """
    omega, tau = np.broadcast_arrays(
        array(angular_frequency_rad_per_s, "angular_frequency_rad_per_s"),
        array(diffusion_time_seconds, "diffusion_time_seconds", positive=True),
    )
    with np.errstate(over="ignore"):
        a = np.sqrt(np.abs(omega)) * np.sqrt(tau) / np.sqrt(2.0)
    # Horner polynomials in a=sqrt(|omega|*tau/2), scaled for a>1.
    # Numerator: 1 + 5a/4 + a**2/2.
    # Denominator: 1+2a+2a**2+4a**3/3+64a**4/81+32a**5/81+8a**6/81.
    small = a <= 1
    v = np.where(small, a, np.divide(1.0, a, where=a != 0, out=np.zeros_like(a)))
    numerator_low = 1 + v * (1.25 + 0.5 * v)
    denominator_low = 1 + v * (
        2 + v * (2 + v * (4 / 3 + v * (64 / 81 + v * (32 / 81 + v * 8 / 81))))
    )
    denominator_high = 8 / 81 + v * (
        32 / 81 + v * (64 / 81 + v * (4 / 3 + v * (2 + v * (2 + v))))
    )
    # Multiply by tau before taking the fourth inverse power: the shape alone
    # can underflow even when the dimensional spectrum is representable.
    high = (((tau * v) * v) * v) * v * (0.5 + v * (1.25 + v)) / denominator_high
    low = tau * numerator_low / denominator_low
    return (4 / 9) * np.where(small, low, high)


def rotational_spectral_density(angular_frequency_rad_per_s, correlation_time_seconds):
    """Return the one-sided exponential-correlation spectrum tau/(1+(w*tau)^2)."""
    omega, tau = np.broadcast_arrays(
        array(angular_frequency_rad_per_s, "angular_frequency_rad_per_s"),
        array(correlation_time_seconds, "correlation_time_seconds", positive=True),
    )
    with np.errstate(over="ignore"):
        return tau / (1 + (omega * tau) ** 2)


@dataclass(frozen=True)
class TranslationalDiffusionModel:
    """Uniform FFHS solution of S=1/2 radicals outside a reflecting sphere.

    Distances are metres and relative diffusion is m^2/s (sum of radical and
    solvent diffusivities). No pair correlations, binding, or spin exchange
    corrections are included. Parameters are scalar; frequency axes broadcast.
    """

    closest_approach_m: float
    relative_diffusion_m2_per_s: float

    def __post_init__(self):
        for name in ("closest_approach_m", "relative_diffusion_m2_per_s"):
            value = array(getattr(self, name), name, positive=True)
            if value.ndim:
                raise ValueError(f"{name} must be scalar")
            object.__setattr__(self, name, float(value))
        array(self.diffusion_time_seconds, "diffusion_time_seconds", positive=True)

    @property
    def diffusion_time_seconds(self) -> float:
        """Return tau_D=d^2/D_relative, not the integral correlation time."""
        return self.closest_approach_m**2 / self.relative_diffusion_m2_per_s

    def spectral_density(self, angular_frequency_rad_per_s):
        """Return the normalized one-sided FFHS spectral density in seconds."""
        return ffhs_spectral_density(
            angular_frequency_rad_per_s, self.diffusion_time_seconds
        )
