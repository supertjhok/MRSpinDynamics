from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.workflows.slice_selective import (
    make_slice_selective_excitation,
    resolve_time_bandwidth,
    simulate_slice_profile,
    slice_profile_table,
)


def _fwhm(positions, profile):
    half = 0.5 * profile.max()
    inside = positions[profile >= half]
    return float(inside.max() - inside.min())


class MakeSliceExcitationTests(unittest.TestCase):
    def test_on_resonance_flip_matches_request(self):
        steps = make_slice_selective_excitation(
            duration=1.0e-3, slice_gradient=2.0e4, flip_angle=np.pi / 2,
            num_substeps=64, rephase=True,
        )
        rf_steps = [s for s in steps if s.rf_amplitude != 0.0 or s.label.startswith("slice_rf")]
        flip = sum(s.rf_amplitude * s.duration for s in steps if s.label.startswith("slice_rf"))
        self.assertAlmostEqual(flip, np.pi / 2, places=9)
        # 64 RF substeps + 1 rephase lobe
        self.assertEqual(len(rf_steps), 64)
        self.assertTrue(steps[-1].label == "slice_rephase")
        self.assertEqual(steps[-1].rf_amplitude, 0.0)
        self.assertEqual(steps[-1].gradient, (-2.0e4, 0.0))

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            make_slice_selective_excitation(duration=0.0, slice_gradient=1.0)
        with self.assertRaises(ValueError):
            make_slice_selective_excitation(duration=1e-3, slice_gradient=0.0)
        with self.assertRaises(ValueError):
            make_slice_selective_excitation(
                duration=1e-3, slice_gradient=1.0, slice_axis=2, ndim=2
            )


class SliceProfileTests(unittest.TestCase):
    def test_profile_is_selective(self):
        res = simulate_slice_profile(
            duration=1.0e-3, slice_gradient=2.0e4, flip_angle=np.pi / 2,
            num_substeps=64, num_positions=161,
        )
        s = res.slice_positions
        center = res.profile[np.argmin(np.abs(s))]
        edge = np.mean(res.profile[:8])
        self.assertGreater(center, 0.95)  # fully excited in-slice
        self.assertLess(edge, 0.05)  # suppressed far off-slice
        # a selective 90 leaves ~no longitudinal magnetization in the slice
        self.assertLess(abs(res.longitudinal[np.argmin(np.abs(s))]), 0.05)

    def test_flip_angle_controls_tip(self):
        res = simulate_slice_profile(
            duration=1.0e-3, slice_gradient=2.0e4, flip_angle=np.pi / 3,
            num_substeps=64, num_positions=161,
        )
        i0 = np.argmin(np.abs(res.slice_positions))
        # in-slice transverse ~ sin(flip), residual Mz ~ cos(flip)
        self.assertAlmostEqual(res.profile[i0], np.sin(np.pi / 3), delta=0.05)
        self.assertAlmostEqual(res.longitudinal[i0], np.cos(np.pi / 3), delta=0.05)

    def test_stronger_gradient_narrows_slice(self):
        common = dict(duration=1.0e-3, flip_angle=np.pi / 2, num_substeps=64)
        narrow = simulate_slice_profile(slice_gradient=4.0e4, num_positions=241, **common)
        wide = simulate_slice_profile(slice_gradient=2.0e4, num_positions=241, **common)
        ratio = _fwhm(narrow.slice_positions, narrow.profile) / _fwhm(
            wide.slice_positions, wide.profile
        )
        # doubling the gradient halves the slice width
        self.assertAlmostEqual(ratio, 0.5, delta=0.1)



class TimeBandwidthAliasTests(unittest.TestCase):
    def test_time_bandwidth_product_is_the_conventional_tbw(self) -> None:
        # L-14: ``time_bandwidth`` is half the conventional TBW (unchanged);
        # ``time_bandwidth_product`` is the conventional one. Reference: the
        # measured small-flip slice FWHM (Hz) times the pulse duration.
        duration = 1.0e-3
        kw = dict(slice_gradient=1.0, duration=duration, flip_angle=np.deg2rad(10),
                  num=2001, num_substeps=400)
        off, aliased = slice_profile_table(
            off_resonance_max=2 * np.pi * 12.0 / duration, time_bandwidth_product=4.0, **kw
        )
        _off, legacy = slice_profile_table(
            off_resonance_max=2 * np.pi * 12.0 / duration, time_bandwidth=2.0, **kw
        )
        np.testing.assert_array_equal(aliased, legacy)
        hz = off / (2 * np.pi)
        fwhm = _fwhm(hz, np.abs(aliased))
        self.assertAlmostEqual(fwhm * duration, 4.0, delta=0.08)
        self.assertEqual(resolve_time_bandwidth(4.0, None), 4.0)
        self.assertEqual(resolve_time_bandwidth(4.0, 6.0), 3.0)
        with self.assertRaises(ValueError):
            resolve_time_bandwidth(4.0, -1.0)


if __name__ == "__main__":
    unittest.main()
