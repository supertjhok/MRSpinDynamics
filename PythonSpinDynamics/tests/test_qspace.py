from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.workflows import (
    acquire_pgse_qspace_walkers,
    phase_retrieve_qspace_magnitude,
    pore_form_factor_from_density,
    qspace_axes_from_real_space,
    reconstruct_qspace_image,
)
from spin_dynamics.motion import make_elliptical_reflector


def _ellipse(n: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = (np.arange(n, dtype=np.float64) - n // 2) * 1.0e-6
    z = x.copy()
    xx, zz = np.meshgrid(x, z, indexing="ij")
    rho = (((xx / 6.0e-6) ** 2 + (zz / 4.0e-6) ** 2) <= 1.0).astype(float)
    return rho, x, z


def _best_aligned_iou(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Compare magnitude-only reconstructions modulo shift and reflection."""

    n0, n1 = reference.shape
    best = 0.0
    for candidate in (
        estimate,
        estimate[::-1, :],
        estimate[:, ::-1],
        estimate[::-1, ::-1],
    ):
        corr = np.fft.ifft2(
            np.fft.fft2(candidate.astype(float))
            * np.conj(np.fft.fft2(reference.astype(float)))
        ).real
        peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
        shift = (
            -peak[0] if peak[0] <= n0 // 2 else n0 - peak[0],
            -peak[1] if peak[1] <= n1 // 2 else n1 - peak[1],
        )
        aligned = np.roll(candidate, shift, axis=(0, 1))
        union = np.logical_or(aligned, reference).sum()
        if union:
            best = max(best, float(np.logical_and(aligned, reference).sum() / union))
    return best


class QSpaceImagingTests(unittest.TestCase):
    def test_finite_pulse_walkers_reconstruct_elliptical_pore_autocorrelation(self) -> None:
        n = 12
        axis = (np.arange(n, dtype=np.float64) - n // 2) * 1.0e-6
        xx, zz = np.meshgrid(axis, axis, indexing="ij")
        semi_axes = (3.2e-6, 2.1e-6)
        rho = (
            (xx / semi_axes[0]) ** 2 + (zz / semi_axes[1]) ** 2 <= 1.0
        ).astype(float)
        qx, qz = qspace_axes_from_real_space(axis, axis)

        acquired = acquire_pgse_qspace_walkers(
            rho,
            axis,
            axis,
            qx,
            qz,
            gradient_duration=0.3e-3,
            diffusion_time=24.0e-3,
            diffusion_coefficient=1.0e-9,
            walkers_per_cell=32,
            seed=8,
            boundary=make_elliptical_reflector((0.0, 0.0), semi_axes),
            substeps_per_interval=5,
        )
        reconstructed = reconstruct_qspace_image(
            acquired.intensity,
            qx,
            qz,
            data_kind="intensity",
            clip_negative=True,
        )
        ideal = reconstruct_qspace_image(
            np.abs(pore_form_factor_from_density(rho)) ** 2,
            qx,
            qz,
            data_kind="intensity",
            clip_negative=True,
        )

        self.assertAlmostEqual(float(acquired.intensity[n // 2, n // 2]), 1.0)
        correlation = np.corrcoef(
            reconstructed.image.reshape(-1), ideal.image.reshape(-1)
        )[0, 1]
        self.assertGreater(float(correlation), 0.9)
        center = n // 2
        x_width = int(np.count_nonzero(reconstructed.image[:, center] > 0.2))
        z_width = int(np.count_nonzero(reconstructed.image[center, :] > 0.2))
        self.assertGreater(x_width, z_width)
        support = (xx / 4.5e-6) ** 2 + (zz / 3.2e-6) ** 2 <= 1.0
        retrieved = phase_retrieve_qspace_magnitude(
            acquired.intensity,
            qx,
            qz,
            support=support,
            input_is_intensity=True,
            iterations=220,
            er_iterations=60,
            seed=3,
        )
        estimate = retrieved.density / retrieved.density.max() > 0.2
        self.assertGreater(_best_aligned_iou(estimate, rho > 0.0), 0.6)

    def test_complex_form_factor_reconstructs_density(self) -> None:
        rho, x_axis, z_axis = _ellipse()
        qx, qz = qspace_axes_from_real_space(x_axis, z_axis)
        form = pore_form_factor_from_density(rho)

        result = reconstruct_qspace_image(form, qx, qz, data_kind="complex")
        image = result.image.real

        np.testing.assert_allclose(image / image.max(), rho, atol=1e-12)
        np.testing.assert_allclose(result.x_axis, x_axis)
        np.testing.assert_allclose(result.z_axis, z_axis)

    def test_intensity_reconstructs_pore_autocorrelation(self) -> None:
        rho, x_axis, z_axis = _ellipse()
        qx, qz = qspace_axes_from_real_space(x_axis, z_axis)
        form = pore_form_factor_from_density(rho)

        result = reconstruct_qspace_image(np.abs(form) ** 2, qx, qz, data_kind="intensity")
        autocorrelation = result.image

        center = tuple(size // 2 for size in rho.shape)
        self.assertAlmostEqual(float(autocorrelation[center]), 1.0, delta=1e-12)
        self.assertLess(float(np.min(autocorrelation)), 1e-12)
        # The autocorrelation support is wider than the pore itself; this is the
        # expected Patterson image from magnitude-squared diffraction data.
        central_line = autocorrelation[:, center[1]] > 1e-6
        self.assertGreater(int(central_line.sum()), int((rho[:, center[1]] > 0).sum()))

    def test_phase_retrieval_estimates_pore_shape_from_magnitude(self) -> None:
        rho, x_axis, z_axis = _ellipse()
        qx, qz = qspace_axes_from_real_space(x_axis, z_axis)
        form = pore_form_factor_from_density(rho)

        xx, zz = np.meshgrid(x_axis, z_axis, indexing="ij")
        loose_support = ((xx / 9.0e-6) ** 2 + (zz / 7.0e-6) ** 2) <= 1.0
        result = phase_retrieve_qspace_magnitude(
            np.abs(form),
            qx,
            qz,
            support=loose_support,
            iterations=220,
            er_iterations=60,
            seed=4,
        )

        estimate = result.density / result.density.max() > 0.25
        reference = rho > 0
        self.assertLess(result.residual, 1e-6)
        self.assertGreater(_best_aligned_iou(estimate, reference), 0.95)

    def test_rejects_nonuniform_q_axes(self) -> None:
        rho, x_axis, z_axis = _ellipse()
        qx, qz = qspace_axes_from_real_space(x_axis, z_axis)
        qx_bad = qx.copy()
        qx_bad[-1] += 0.2 * (qx[1] - qx[0])

        with self.assertRaises(ValueError):
            reconstruct_qspace_image(rho, qx_bad, qz)


if __name__ == "__main__":
    unittest.main()
