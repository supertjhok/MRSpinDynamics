"""Effective CW saturation models; nitroxide hyperfine mixing is an input."""

import numpy as np

from ._validation import array, fraction


def power_saturation(power_watts, half_saturation_power_watts, *, maximum=1.0):
    """Return effective s_max*p/(p+p_half) with explicitly calibrated powers."""
    power, half, maximum = np.broadcast_arrays(
        array(power_watts, "power_watts", nonnegative=True),
        array(
            half_saturation_power_watts, "half_saturation_power_watts", positive=True
        ),
        fraction(maximum, "maximum"),
    )
    scale = np.maximum(power, half)
    return maximum * (power / scale) / (power / scale + half / scale)


def bloch_saturation(
    nutation_rad_per_s,
    t1e_seconds,
    t2e_seconds,
    *,
    detuning_rad_per_s=0.0,
    maximum=1.0,
):
    """Return homogeneous-line CW Bloch saturation, optionally scaled by s_max.

    Nutation is the rotating-frame angular frequency. For a linearly polarized
    laboratory B1 use the co-rotating half-amplitude. This does not calculate
    nitrogen relaxation or exchange-driven mixing between nitroxide lines.
    """
    omega1, t1, t2, delta, maximum = np.broadcast_arrays(
        array(nutation_rad_per_s, "nutation_rad_per_s", nonnegative=True),
        array(t1e_seconds, "t1e_seconds", positive=True),
        array(t2e_seconds, "t2e_seconds", positive=True),
        array(detuning_rad_per_s, "detuning_rad_per_s"),
        fraction(maximum, "maximum"),
    )
    # Logarithms prevent overflow in intermediate frequency/time products.
    with np.errstate(divide="ignore"):
        log_drive = 2 * np.log(omega1) + np.log(t1) + np.log(t2)
        log_detuning = 2 * (np.log(np.abs(delta)) + np.log(t2))
    denominator = np.logaddexp(log_drive, np.logaddexp(0, log_detuning))
    return maximum * np.exp(log_drive - denominator)
