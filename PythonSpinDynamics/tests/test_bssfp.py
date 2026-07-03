from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.workflows import (
    bssfp_band_spacing_hz,
    bssfp_optimal_flip_deg,
    bssfp_steady_state_signal,
    bssfp_transient,
    run_bssfp_imaging,
    run_spin_warp_imaging,
)


def _null_offsets(profile: np.ndarray, df: np.ndarray, threshold: float) -> np.ndarray:
    interior = (profile[1:-1] < profile[:-2]) & (profile[1:-1] < profile[2:])
    return df[1:-1][interior & (profile[1:-1] < threshold)]


class BSSFPSteadyStateTests(unittest.TestCase):
    def test_optimal_flip_gives_half_sqrt_t2_over_t1(self) -> None:
        # Short TR relative to T2 so the small intra-TR (TE) T2 decay is
        # negligible and the ideal 0.5*sqrt(T2/T1) peak is recovered.
        t1, t2, tr = 1.0, 0.3, 2.0e-3
        alpha = bssfp_optimal_flip_deg(t1, t2)
        signal = np.abs(
            bssfp_steady_state_signal(
                0.0, flip_angle_deg=alpha, tr=tr, t1=t1, t2=t2
            )
        ).ravel()[0]
        self.assertAlmostEqual(signal, 0.5 * np.sqrt(t2 / t1), delta=0.005)

    def test_alpha_90_equal_relaxation_gives_half_m0(self) -> None:
        signal = np.abs(
            bssfp_steady_state_signal(
                0.0, flip_angle_deg=90.0, tr=2.0e-3, t1=0.5, t2=0.5
            )
        ).ravel()[0]
        self.assertAlmostEqual(signal, 0.5, delta=0.01)

    def test_band_nulls_are_spaced_one_over_tr(self) -> None:
        tr = 5.0e-3
        df = np.linspace(-260.0, 260.0, 521)
        prof_180 = np.abs(
            bssfp_steady_state_signal(
                df, flip_angle_deg=40.0, tr=tr, t1=1.0, t2=0.1, phase_increment_deg=180.0
            )
        )
        prof_0 = np.abs(
            bssfp_steady_state_signal(
                df, flip_angle_deg=40.0, tr=tr, t1=1.0, t2=0.1, phase_increment_deg=0.0
            )
        )
        # Alternating RF: passband centred, nulls at +/- 1/(2 TR).
        np.testing.assert_allclose(
            np.sort(_null_offsets(prof_180, df, 0.03)), [-1 / (2 * tr), 1 / (2 * tr)], atol=2.0
        )
        # Constant RF: the passband shifts by half a band, nulls at 0, +/- 1/TR.
        np.testing.assert_allclose(
            np.sort(_null_offsets(prof_0, df, 0.03)), [-1 / tr, 0.0, 1 / tr], atol=2.0
        )

    def test_band_spacing_helper(self) -> None:
        self.assertAlmostEqual(bssfp_band_spacing_hz(4.0e-3), 250.0)

    def test_transient_converges_to_steady_state(self) -> None:
        kw = dict(flip_angle_deg=40.0, tr=3.0e-3, t1=0.2, t2=0.08)
        final = np.abs(bssfp_transient(30.0, num_tr=3000, **kw)).ravel()[-1]
        steady = np.abs(bssfp_steady_state_signal(30.0, **kw)).ravel()[0]
        self.assertAlmostEqual(final, steady, delta=1e-3)

    def test_catalyzation_reaches_steady_state_sooner(self) -> None:
        # Off-resonance transients oscillate; the alpha/2 preparation lands the
        # magnetization near the steady-state cone, so the train settles to
        # within 5% of steady state in fewer TRs.
        kw = dict(flip_angle_deg=40.0, tr=3.0e-3, t1=0.2, t2=0.08)
        steady = np.abs(bssfp_steady_state_signal(60.0, **kw)).ravel()[0]

        def settle_tr(catalyze: bool) -> int:
            trace = np.abs(bssfp_transient(60.0, num_tr=400, catalyze=catalyze, **kw)).ravel()
            within = np.abs(trace - steady) < 0.05 * steady
            return int(np.argmax(within)) if within.any() else 400

        self.assertLessEqual(settle_tr(True), settle_tr(False))


class BSSFPImagingTests(unittest.TestCase):
    def _phantom(self, n: int = 12) -> np.ndarray:
        rho = np.zeros((n, n), dtype=np.float64)
        rho[3:9, 4:8] = 1.0
        return rho

    def test_localizes_a_point_without_half_fov_shift(self) -> None:
        # Regression guard: the alternating (phase-cycled) RF phase must be
        # demodulated, or the image shifts by half the FOV along phase encode.
        # Short T1/T2 with enough dummy TRs so the steady state (and thus clean
        # localization) is reached; the readout (x) is exact and the phase-encode
        # (z) is within one pixel -- a broken demod would shift z by pz/2 = 4.
        n = 8
        rho = np.zeros((n, n))
        rho[5, 2] = 1.0
        result = run_bssfp_imaging(
            rho, t1_map=np.full((n, n), 0.08), t2_map=np.full((n, n), 0.03),
            num_dummy_tr=250, readout_time=1.2e-3, phase_time=0.4e-3,
        )
        magnitude = result.magnitude[:, :, 0]
        px_peak, pz_peak = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
        self.assertEqual(px_peak, 5)
        self.assertLessEqual(abs(pz_peak - 2), 1)

    def test_recovers_phantom_shape_in_uniform_fields(self) -> None:
        rho = self._phantom()
        result = run_bssfp_imaging(
            rho, t1_map=np.full(rho.shape, 0.15), t2_map=np.full(rho.shape, 0.05),
            num_dummy_tr=60, readout_time=1.2e-3, phase_time=0.4e-3,
        )
        image = np.abs(result.image[:, :, 0])
        image = image / image.max()
        # The object is recovered where rho is non-zero and dark outside it.
        self.assertGreater(image[rho > 0].mean(), 4.0 * image[rho == 0].mean())
        self.assertEqual(result.method, "bssfp")
        self.assertAlmostEqual(result.band_spacing_hz, 1.0 / result.tr)

    def test_off_resonance_null_darkens_the_object(self) -> None:
        # Short T1/T2 so the steady state (and its null) develops within the
        # dummy train. A uniform offset at the first null (1/2TR) collapses the
        # signal; the object of the same phantom nearly vanishes.
        rho = np.zeros((12, 12))
        rho[3:9, 4:8] = 1.0
        t1 = np.full(rho.shape, 0.15)
        t2 = np.full(rho.shape, 0.05)
        kw = dict(t1_map=t1, t2_map=t2, num_dummy_tr=150, readout_time=1.2e-3,
                  phase_time=0.4e-3, flip_angle_deg=40.0)
        on_res = run_bssfp_imaging(rho, **kw)
        tr = on_res.tr
        null = 2.0 * np.pi * (0.5 / tr) * np.ones(rho.shape)
        banded = run_bssfp_imaging(rho, b0_map=null, **kw)
        obj = rho > 0
        self.assertLess(
            np.abs(banded.magnitude[:, :, 0])[obj].mean(),
            0.3 * np.abs(on_res.magnitude[:, :, 0])[obj].mean(),
        )

    def test_transmit_b1_shades_the_image(self) -> None:
        rho = np.ones((8, 12))
        t1 = np.full(rho.shape, 0.3)
        t2 = np.full(rho.shape, 0.1)
        b1 = np.linspace(0.5, 1.0, rho.shape[0])[:, np.newaxis] * np.ones((1, rho.shape[1]))
        kw = dict(t1_map=t1, t2_map=t2, num_dummy_tr=40, readout_time=1.2e-3,
                  phase_time=0.4e-3, flip_angle_deg=30.0)
        flat = np.abs(run_bssfp_imaging(rho, **kw).magnitude[:, :, 0])
        shaded = np.abs(run_bssfp_imaging(rho, b1_tx_map=b1, **kw).magnitude[:, :, 0])
        # The flat image is roughly uniform along x; the B1 ramp makes it vary.
        self.assertGreater(shaded.std() / shaded.mean(), flat.std() / flat.mean() + 0.05)

    def test_accepts_field_maps_container(self) -> None:
        from spin_dynamics.workflows import make_imaging_field_maps

        rho = self._phantom()
        # Match the array-path defaults (uniform B1, zero B0) so the two calls
        # describe the same experiment; make_imaging_field_maps otherwise injects
        # a synthetic Gaussian transmit-B1 map.
        fields = make_imaging_field_maps(
            rho, t1_map=np.full(rho.shape, 0.3), t2_map=np.full(rho.shape, 0.1),
            b0_map=np.zeros(rho.shape), b1_tx_map=np.ones(rho.shape),
            b1_rx_map=np.ones(rho.shape),
        )
        from_fields = run_bssfp_imaging(fields, num_dummy_tr=20, readout_time=1.2e-3,
                                        phase_time=0.4e-3)
        from_arrays = run_bssfp_imaging(
            rho, t1_map=np.full(rho.shape, 0.3), t2_map=np.full(rho.shape, 0.1),
            num_dummy_tr=20, readout_time=1.2e-3, phase_time=0.4e-3,
        )
        np.testing.assert_allclose(from_fields.kspace, from_arrays.kspace, atol=1e-12)

    def test_matches_spin_warp_geometry(self) -> None:
        # bSSFP and the spin-echo imager share the k-space geometry, so once the
        # bSSFP steady state is reached a point lands at the same pixel in both.
        n = 8
        rho = np.zeros((n, n))
        rho[2, 6] = 1.0
        common = dict(t1_map=np.full((n, n), 0.08), t2_map=np.full((n, n), 0.03))
        b = run_bssfp_imaging(rho, num_dummy_tr=250, readout_time=1.2e-3,
                              phase_time=0.4e-3, **common)
        s = run_spin_warp_imaging(rho, readout_time=1.2e-3, **common)
        bp = np.unravel_index(int(np.argmax(np.abs(b.magnitude[:, :, 0]))), (n, n))
        sp = np.unravel_index(int(np.argmax(np.abs(s.magnitude[:, :, 0]))), (n, n))
        self.assertEqual(bp[0], sp[0])
        self.assertLessEqual(abs(int(bp[1]) - int(sp[1])), 1)

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            run_bssfp_imaging(np.ones(8))  # not 2D
        with self.assertRaises(ValueError):
            bssfp_optimal_flip_deg(-1.0, 0.1)


if __name__ == "__main__":
    unittest.main()
