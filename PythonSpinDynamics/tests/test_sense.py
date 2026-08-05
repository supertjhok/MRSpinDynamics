"""Cartesian sensitivity-encoding and SENSE reconstruction tests."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.workflows import (
    CartesianSENSEEncoding,
    add_receiver_array_noise,
    make_imaging_field_maps,
    reconstruct_cartesian_sense,
    run_ideal_receiver_array_cpmg_imaging,
    uniform_cartesian_mask,
    whiten_receiver_channels,
)
from spin_dynamics.workflows import imaging as imaging_module


def _r2_sensitivities(shape: tuple[int, int] = (4, 2)) -> np.ndarray:
    first = np.ones(shape, dtype=np.complex128)
    second = np.ones(shape, dtype=np.complex128)
    second[shape[0] // 2 :, :] = -1.0
    return np.stack([first, second])


def test_uniform_mask_is_centered_and_validates_divisibility() -> None:
    mask = uniform_cartesian_mask((8, 3), 2, axis="x")
    assert mask.shape == (8, 3)
    assert np.all(mask[4])
    assert np.count_nonzero(mask[:, 0]) == 4
    np.testing.assert_array_equal(mask[:, 0], mask[:, 2])

    offset = uniform_cartesian_mask((8, 3), 2, axis="x", offset=1)
    assert not np.any(mask & offset)
    assert np.all(mask | offset)
    with pytest.raises(ValueError, match="divisible"):
        uniform_cartesian_mask((7, 3), 2)


def test_encoding_forward_and_adjoint_are_conjugate_operators() -> None:
    rng = np.random.default_rng(2)
    sensitivities = rng.normal(size=(3, 4, 2)) + 1j * rng.normal(
        size=(3, 4, 2)
    )
    mask = uniform_cartesian_mask((4, 2), 2)
    encoding = CartesianSENSEEncoding(sensitivities, mask)
    image = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
    data = rng.normal(size=(3, 4, 2, 2)) + 1j * rng.normal(
        size=(3, 4, 2, 2)
    )

    lhs = np.vdot(encoding.forward(image), data)
    rhs = np.vdot(image, encoding.adjoint(data))
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_full_sampling_sense_matches_complex_object() -> None:
    rng = np.random.default_rng(5)
    image = rng.normal(size=(4, 3, 2)) + 1j * rng.normal(size=(4, 3, 2))
    sensitivities = rng.normal(size=(3, 4, 3)) + 1j * rng.normal(
        size=(3, 4, 3)
    )
    covariance = np.array(
        [[1.0, 0.1j, 0.0], [-0.1j, 1.3, 0.2], [0.0, 0.2, 0.8]],
        dtype=np.complex128,
    )
    mask = uniform_cartesian_mask((4, 3), 1)
    encoding = CartesianSENSEEncoding(
        sensitivities,
        mask,
        noise_covariance=covariance,
    )
    result = reconstruct_cartesian_sense(
        encoding.forward(image),
        sensitivities,
        mask,
        noise_covariance=covariance,
    )

    np.testing.assert_allclose(result.image, image, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.g_factor, 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result.rank, 1)


def test_analytic_r2_alias_set_unfolds_exactly() -> None:
    image = np.arange(8, dtype=np.float64).reshape(4, 2, 1)
    sensitivities = _r2_sensitivities()
    mask = uniform_cartesian_mask((4, 2), 2)
    encoding = CartesianSENSEEncoding(sensitivities, mask)
    result = reconstruct_cartesian_sense(
        encoding.forward(image),
        sensitivities,
        mask,
        raise_on_rank_deficiency=True,
    )

    np.testing.assert_allclose(result.image, image, atol=1e-12)
    np.testing.assert_allclose(result.g_factor, 1.0, atol=1e-12)
    np.testing.assert_array_equal(result.rank, 2)
    np.testing.assert_allclose(result.condition_number, 1.0, atol=1e-12)


def test_rank_deficiency_is_reported_and_can_be_rejected() -> None:
    image = np.ones((4, 2, 1), dtype=np.complex128)
    sensitivities = np.ones((2, 4, 2), dtype=np.complex128)
    mask = uniform_cartesian_mask((4, 2), 2)
    encoding = CartesianSENSEEncoding(sensitivities, mask)

    result = reconstruct_cartesian_sense(
        encoding.forward(image),
        sensitivities,
        mask,
        regularization=1e-3,
    )
    np.testing.assert_array_equal(result.rank, 1)
    assert np.all(np.isinf(result.condition_number))
    assert np.all(np.isinf(result.g_factor))
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        reconstruct_cartesian_sense(
            encoding.forward(image),
            sensitivities,
            mask,
            raise_on_rank_deficiency=True,
        )


def test_covariance_whitening_produces_unit_channel_covariance() -> None:
    covariance = np.array(
        [[1.0, 0.25 + 0.1j], [0.25 - 0.1j, 1.6]],
        dtype=np.complex128,
    )
    samples = add_receiver_array_noise(
        np.zeros((2, 60000), dtype=np.complex128),
        covariance,
        seed=42,
    )
    whitened = whiten_receiver_channels(samples, covariance)
    empirical = whitened @ whitened.conj().T / whitened.shape[1]
    np.testing.assert_allclose(empirical, np.eye(2), rtol=0.025, atol=0.025)


def test_monte_carlo_noise_matches_predicted_g_factor() -> None:
    sensitivities = np.array(
        [
            [[[1.0]], [[1.0]], [[1.0]], [[1.0]]],
            [[[0.2]], [[0.8]], [[1.0]], [[-0.4]]],
        ],
        dtype=np.complex128,
    ).reshape(2, 4, 1)
    covariance = np.array(
        [[1.0, 0.2j], [-0.2j, 1.4]],
        dtype=np.complex128,
    )
    noise = add_receiver_array_noise(
        np.zeros((2, 4, 1, 30000), dtype=np.complex128),
        covariance,
        seed=4,
    )
    full = reconstruct_cartesian_sense(
        noise,
        sensitivities,
        uniform_cartesian_mask((4, 1), 1),
        noise_covariance=covariance,
    )
    accelerated = reconstruct_cartesian_sense(
        noise,
        sensitivities,
        uniform_cartesian_mask((4, 1), 2),
        noise_covariance=covariance,
    )
    full_variance = np.mean(np.abs(full.image) ** 2, axis=-1)
    accelerated_variance = np.mean(np.abs(accelerated.image) ** 2, axis=-1)
    empirical_g = np.sqrt(accelerated_variance / (2.0 * full_variance))

    np.testing.assert_allclose(
        empirical_g,
        accelerated.g_factor,
        rtol=0.025,
        atol=0.025,
    )


def test_receiver_array_workflow_exposes_exact_pfs_reference(monkeypatch) -> None:
    rho = np.ones((4, 2), dtype=np.float64)
    fields = make_imaging_field_maps(
        rho,
        b1_tx_map=np.ones_like(rho),
        b1_rx_map=np.ones_like(rho),
    )
    calls = 0
    original = imaging_module.calc_macq_ideal_probe_relax4

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(imaging_module, "calc_macq_ideal_probe_relax4", counted)
    result = run_ideal_receiver_array_cpmg_imaging(
        fields,
        receiver_sensitivities=_r2_sensitivities(),
        num_echoes=1,
        ny=1,
        maxoffs=0.1,
        sense_acceleration=2,
    )

    # The Bloch model retains all eight points as a ground-truth reference;
    # SENSE undersampling is then applied by the explicit PFS operator.
    assert calls == 32
    assert result.sense_acceleration == 2
    assert result.sense_image is not None
    assert result.sense_reference_image is not None
    np.testing.assert_allclose(
        result.sense_image, result.sense_reference_image, atol=1e-12
    )
    assert result.sampling_mask is not None
    assert np.count_nonzero(result.sampling_mask) == 4
    np.testing.assert_array_equal(result.sense_rank, 2)
