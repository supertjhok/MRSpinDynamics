"""Explicit field/power grids for the liquid-state CW Overhauser model."""

from dataclasses import dataclass

import numpy as np

from spin_dynamics.dnp import (
    OverhauserResult,
    TranslationalDiffusionModel,
    cw_overhauser,
    power_saturation,
    translational_relaxation_rates,
)
from spin_dynamics.dnp._validation import array


@dataclass(frozen=True)
class OverhauserPowerSweep:
    """CW observables with power rows and field columns."""

    field_tesla: np.ndarray
    power_watts: np.ndarray
    cw: OverhauserResult


def run_overhauser_power_sweep(
    field_tesla,
    power_watts,
    *,
    model: TranslationalDiffusionModel,
    concentration_molar,
    half_saturation_power_watts,
    maximum_saturation=1.0,
    background_r1_per_second=0.0,
    scalar_self_rate_per_second=0.0,
    temperature_kelvin=298.15,
) -> OverhauserPowerSweep:
    """Run a proton CW power/field sweep with a prescribed FFHS motion model.

    Conditions broadcast to (n_power, n_field); supply heating-dependent
    temperature/background-rate arrays in that shape when known. Diffusivity
    remains the supplied model's value; temperature does not infer motion.
    For measured T1(p)/T10(p), use OverhauserRates.from_t1 and cw_overhauser
    directly rather than mixing measured and microscopic self-relaxation rates.
    """
    fields = np.atleast_1d(array(field_tesla, "field_tesla", positive=True))
    powers = np.atleast_1d(array(power_watts, "power_watts", nonnegative=True))
    if fields.ndim != 1 or powers.ndim != 1 or not fields.size or not powers.size:
        raise ValueError("field_tesla and power_watts must be non-empty 1-D axes")
    saturation = power_saturation(
        powers[:, None], half_saturation_power_watts, maximum=maximum_saturation
    )
    rates = translational_relaxation_rates(
        fields[None, :],
        model,
        concentration_molar,
        background_r1_per_second=background_r1_per_second,
        scalar_self_rate_per_second=scalar_self_rate_per_second,
    )
    result = cw_overhauser(
        rates,
        field_tesla=fields[None, :],
        saturation_factor=saturation,
        temperature_kelvin=temperature_kelvin,
    )
    if result.enhancement.shape != (powers.size, fields.size):
        raise ValueError("condition arrays must broadcast to (n_power, n_field)")
    return OverhauserPowerSweep(fields.copy(), powers.copy(), result)
