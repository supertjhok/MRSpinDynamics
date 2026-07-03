"""RFI modelling and reference-coil cancellation for magnetic resonance.

This package models radio-frequency interference (RFI) as a vector magnetic
field, senses it with reference and primary pickup coils, and (in later phases)
cancels it as a masked/gated estimation problem. It is deliberately
sequence-agnostic so low-field MRI and ESR can reuse it; the NQR-specific
mask adapter lives in :mod:`spin_dynamics.nqr`.

Phase 0 provides the sample-clock and receive-valid masks; Phase 1 provides the
synthetic RFI sources (uniform far-field and near-field dipole) and the
reference-coil pickup model. See ``docs/roadmap.md`` section 8.
"""

from spin_dynamics.interference.coils import (
    ReferenceCoil,
    coil_voltage,
    coupling_matrix,
    reference_matrix,
)
from spin_dynamics.interference.cancellers import (
    CancellationResult,
    LinearCancellerModel,
    adaptive_lms_canceller,
    adaptive_rls_canceller,
    fit_gated_ridge_fir,
    fit_scalar_canceller,
    gated_ridge_fir_canceller,
    joint_signal_reference_canceller,
    scalar_canceller,
    windowed_joint_signal_reference_canceller,
    windowed_ridge_fir_canceller,
)
from spin_dynamics.interference.diagnostics import (
    DesignMatrixDiagnostics,
    MatchedFilterImprovementResult,
    RFISuppressionResult,
    ResidualLine,
    ResidualSpectrumResult,
    SaturationDiagnostics,
    SignalBiasResult,
    matched_filter_snr_improvement,
    reference_design_diagnostics,
    residual_spectral_lines,
    rfi_suppression_db,
    saturation_diagnostics,
    signal_bias,
)
from spin_dynamics.interference.masks import (
    AcquisitionMask,
    Interval,
    SampleLabel,
    blank_mask,
    mask_from_intervals,
)
from spin_dynamics.interference.sources import (
    MagneticDipoleSource,
    RFISource,
    RFIWaveform,
    UniformPlaneWaveSource,
    am_carrier_waveform,
    chirp_waveform,
    colored_noise_waveform,
    impulsive_waveform,
    tone_waveform,
    total_field,
)

__all__ = [
    "AcquisitionMask",
    "CancellationResult",
    "DesignMatrixDiagnostics",
    "Interval",
    "LinearCancellerModel",
    "MatchedFilterImprovementResult",
    "MagneticDipoleSource",
    "RFISource",
    "RFISuppressionResult",
    "RFIWaveform",
    "ReferenceCoil",
    "ResidualLine",
    "ResidualSpectrumResult",
    "SampleLabel",
    "SaturationDiagnostics",
    "SignalBiasResult",
    "UniformPlaneWaveSource",
    "adaptive_lms_canceller",
    "adaptive_rls_canceller",
    "am_carrier_waveform",
    "blank_mask",
    "chirp_waveform",
    "coil_voltage",
    "colored_noise_waveform",
    "coupling_matrix",
    "fit_gated_ridge_fir",
    "fit_scalar_canceller",
    "gated_ridge_fir_canceller",
    "impulsive_waveform",
    "joint_signal_reference_canceller",
    "matched_filter_snr_improvement",
    "mask_from_intervals",
    "reference_design_diagnostics",
    "reference_matrix",
    "residual_spectral_lines",
    "rfi_suppression_db",
    "scalar_canceller",
    "saturation_diagnostics",
    "signal_bias",
    "tone_waveform",
    "total_field",
    "windowed_ridge_fir_canceller",
    "windowed_joint_signal_reference_canceller",
]
