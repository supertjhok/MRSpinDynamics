"""Receiver-array forward-model, noise, and combination tests."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.workflows import (
    add_receiver_array_noise,
    make_imaging_field_maps,
    roemer_combine,
    run_ideal_phase_encoded_cpmg_imaging,
    run_ideal_receiver_array_cpmg_imaging,
    sensitivity_weighted_combine,
    sum_of_squares,
)
from spin_dynamics.workflows import imaging as imaging_module


def _uniform_fields(shape: tuple[int, int] = (2, 2)):
    rho = np.ones(shape, dtype=np.float64)
    return make_imaging_field_maps(
        rho,
        b1_tx_map=np.ones_like(rho),
        b1_rx_map=np.ones_like(rho),
    )


def test_single_channel_forward_model_matches_legacy_ideal_path() -> None:
    fields = _uniform_fields()
    legacy = run_ideal_phase_encoded_cpmg_imaging(
        fields,
        num_echoes=1,
        ny=1,
        maxoffs=0.1,
    )
    array = run_ideal_receiver_array_cpmg_imaging(
        fields,
        receiver_sensitivities=np.ones((1, 2, 2)),
        num_echoes=1,
        ny=1,
        maxoffs=0.1,
    )

    np.testing.assert_allclose(array.channel_kspace[0], legacy.kspace)
    np.testing.assert_allclose(array.kspace, legacy.kspace)
    np.testing.assert_allclose(array.image, legacy.image, atol=1e-14)


def test_complex_channel_phase_is_preserved() -> None:
    fields = _uniform_fields()
    sensitivities = np.stack([np.ones((2, 2)), 1j * np.ones((2, 2))], axis=0)
    result = run_ideal_receiver_array_cpmg_imaging(
        fields,
        receiver_sensitivities=sensitivities,
        channel_labels=("in_phase", "quadrature"),
        num_echoes=1,
        ny=1,
        maxoffs=0.1,
    )

    assert result.channel_labels == ("in_phase", "quadrature")
    np.testing.assert_allclose(result.channel_kspace[1], 1j * result.channel_kspace[0])
    np.testing.assert_allclose(result.channel_image[1], 1j * result.channel_image[0])


def test_spin_propagation_count_does_not_scale_with_channels(monkeypatch) -> None:
    fields = _uniform_fields((1, 1))
    sensitivities = np.ones((3, 1, 1), dtype=np.complex128)
    calls = 0
    original = imaging_module.calc_macq_ideal_probe_relax4

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(imaging_module, "calc_macq_ideal_probe_relax4", counted)
    run_ideal_receiver_array_cpmg_imaging(
        fields,
        receiver_sensitivities=sensitivities,
        num_echoes=1,
        ny=1,
        maxoffs=0.1,
    )

    # Four phase-cycle propagations for one phase-encode point, not 4*n_channels.
    assert calls == 4


def test_combination_recovers_complex_object_and_rss_is_nonnegative() -> None:
    object_image = np.array([[[1.0 + 2.0j], [0.5 - 0.25j]], [[-0.2j], [2.0 + 0.1j]]])
    sensitivities = np.array(
        [
            [[1.0, 0.5], [0.8j, 1.2]],
            [[0.5j, 1.0], [1.1, -0.7j]],
        ],
        dtype=np.complex128,
    )
    channels = sensitivities[..., np.newaxis] * object_image[np.newaxis, ...]
    covariance = np.array([[1.0, 0.25j], [-0.25j, 1.5]])

    np.testing.assert_allclose(
        sensitivity_weighted_combine(channels, sensitivities), object_image
    )
    np.testing.assert_allclose(
        roemer_combine(channels, sensitivities, covariance), object_image
    )
    rss = sum_of_squares(channels)
    assert rss.shape == object_image.shape
    assert np.all(rss >= 0.0)


def test_correlated_noise_matches_supplied_covariance() -> None:
    covariance = np.array([[1.0, 0.3 + 0.2j], [0.3 - 0.2j, 2.0]], dtype=np.complex128)
    samples = add_receiver_array_noise(
        np.zeros((2, 60000), dtype=np.complex128),
        covariance,
        seed=1234,
    )
    empirical = samples @ samples.conj().T / samples.shape[1]
    np.testing.assert_allclose(empirical, covariance, rtol=0.025, atol=0.025)


def test_identical_independent_channels_give_sqrt_n_noise_gain() -> None:
    n_channels = 4
    n_samples = 50000
    covariance = np.eye(n_channels, dtype=np.complex128)
    noise = add_receiver_array_noise(
        np.zeros((n_channels, 1, 1, n_samples), dtype=np.complex128),
        covariance,
        seed=11,
    )
    sensitivities = np.ones((n_channels, 1, 1), dtype=np.complex128)
    combined = roemer_combine(noise, sensitivities, covariance).reshape(-1)

    channel_rms = np.sqrt(np.mean(np.abs(noise[0].reshape(-1)) ** 2))
    combined_rms = np.sqrt(np.mean(np.abs(combined) ** 2))
    assert channel_rms / combined_rms == pytest.approx(np.sqrt(n_channels), rel=0.025)


def test_non_psd_noise_covariance_is_rejected() -> None:
    data = np.zeros((2, 4), dtype=np.complex128)
    with pytest.raises(ValueError, match="positive semidefinite"):
        add_receiver_array_noise(data, np.array([[1.0, 2.0], [2.0, 1.0]]))
