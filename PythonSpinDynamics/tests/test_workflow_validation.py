from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.optimization import validate_inverse_excitation_pair


def test_inverse_excitation_validation_recognizes_exact_broadband_cancellation():
    offsets = np.linspace(-2.0, 2.0, 101)
    target = np.exp(-offsets**2) * np.exp(0.3j * offsets)

    evidence = validate_inverse_excitation_pair(
        target, -target, offsets, target_snr=4.0, inverse_snr=4.0
    )

    assert evidence.residual_ratio == pytest.approx(0.0, abs=1e-15)
    assert evidence.peak_residual_ratio == pytest.approx(0.0, abs=1e-15)
    assert evidence.inverse_coherence == pytest.approx(1.0)
    assert evidence.snr_relative_error == pytest.approx(0.0)


def test_inverse_excitation_validation_exposes_amplitude_and_phase_residuals():
    offsets = np.linspace(-1.0, 1.0, 51)
    target = np.ones(offsets.size, dtype=np.complex128)

    evidence = validate_inverse_excitation_pair(
        target, -0.8j * target, offsets, target_snr=2.0, inverse_snr=1.5
    )

    assert evidence.residual_ratio == pytest.approx(abs(1.0 - 0.8j))
    assert evidence.peak_residual_ratio == pytest.approx(abs(1.0 - 0.8j))
    assert evidence.inverse_coherence == pytest.approx(0.0, abs=1e-15)
    assert evidence.snr_relative_error == pytest.approx(0.25)


def test_inverse_excitation_validation_rejects_zero_or_misaligned_targets():
    offsets = np.linspace(-1.0, 1.0, 5)
    with pytest.raises(ValueError, match="non-zero"):
        validate_inverse_excitation_pair(np.zeros(5), np.zeros(5), offsets)
    with pytest.raises(ValueError, match="matching shapes"):
        validate_inverse_excitation_pair(np.ones(5), np.ones(4), offsets)


def test_inverse_coherence_distinguishes_reinforcement_from_cancellation():
    offsets = np.linspace(-1.0, 1.0, 5)
    evidence = validate_inverse_excitation_pair(np.ones(5), np.ones(5), offsets)
    assert evidence.inverse_coherence == pytest.approx(-1.0)
    assert evidence.residual_ratio == pytest.approx(2.0)
