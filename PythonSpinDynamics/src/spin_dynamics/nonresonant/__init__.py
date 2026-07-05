"""Nonresonant (field-reversal / CSAR) spin dynamics.

Simulate nuclear magnetization manipulated by sudden field switching and adiabatic
rotations rather than resonant RF pulses (Brill et al., *Nonresonant Multiple Spin
Echoes*, Science 297, 369, 2002), including the dephasing by a non-reversible
background field (Earth's field) and its suppression by combined sudden-switching and
adiabatic rotation (CSAR) sequences.
"""

from spin_dynamics.nonresonant.field_reversal import (
    EARTH_FIELD_TESLA,
    PROTON_GAMMA_HZ_PER_T,
    FieldReversalResult,
    FieldSegment,
    IsochromatEnsemble,
    NonresonantFieldModel,
    evolve_segment,
    rodrigues_rotate,
    sample_isochromats,
    sequence_waveform,
    simulate_field_reversal_echoes,
)
from spin_dynamics.nonresonant.sequences import (
    basic_reversal_sequence,
    csar_double_reversal_sequence,
    csar_sequence,
    csar_supercycle_sequence,
    effective_rotation,
)

__all__ = [
    "EARTH_FIELD_TESLA",
    "PROTON_GAMMA_HZ_PER_T",
    "FieldReversalResult",
    "FieldSegment",
    "IsochromatEnsemble",
    "NonresonantFieldModel",
    "basic_reversal_sequence",
    "csar_double_reversal_sequence",
    "csar_sequence",
    "csar_supercycle_sequence",
    "effective_rotation",
    "evolve_segment",
    "rodrigues_rotate",
    "sample_isochromats",
    "sequence_waveform",
    "simulate_field_reversal_echoes",
]
