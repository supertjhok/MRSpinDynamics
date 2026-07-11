from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    b0_b1_powder_average_grid,
    fit_spectral_overlap_relaxation,
    simulate_weak_b0_spectrum,
    spectral_overlap_factors,
    transition_rms_linewidth_hz,
)


DATA_ROOT = ROOT / "validation" / "experimental"


def _load_chen2020() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with (DATA_ROOT / "chen2020_naclo3_slse.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    with (DATA_ROOT / "chen2020_naclo3_slse.json").open(
        encoding="utf-8"
    ) as stream:
        metadata = json.load(stream)
    fields = np.array([float(row["b0_gauss"]) for row in rows])
    lifetimes = np.array([float(row["value_seconds"]) for row in rows])
    return fields, lifetimes, metadata


def _linewidths_hz(
    fields_gauss: np.ndarray,
    metadata: dict[str, object],
    *,
    n_theta: int,
    n_phi: int,
) -> np.ndarray:
    site = QuadrupolarSite(
        spin=1.5,
        isotope="35Cl",
        quadrupole_frequency_hz=float(metadata["nqr_frequency_hz"]),
        eta=float(metadata["eta"]),
        gamma_hz_per_t=float(metadata["gamma_hz_per_t"]),
    )
    t2star = float(metadata["t2_star_zero_field_seconds"])
    intrinsic_sigma = 1.0 / (np.pi * t2star) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    orientations = b0_b1_powder_average_grid(
        n_theta,
        n_phi,
        n_chi=1,
        b1_b0_angle=0.0,
    )
    widths = []
    for field_gauss in fields_gauss:
        spectrum = simulate_weak_b0_spectrum(
            site,
            float(field_gauss) * 1.0e-4,
            orientations=orientations,
            broadening_hz=10.0,
            points=16,
            weak_ratio_action="ignore",
        )
        offsets = np.array(
            [
                transition.frequency_hz - spectrum.reference_frequency_hz
                for transition in spectrum.transitions
            ]
        )
        intensities = np.array(
            [transition.intensity for transition in spectrum.transitions]
        )
        widths.append(
            transition_rms_linewidth_hz(
                offsets,
                intensities,
                intrinsic_sigma_hz=intrinsic_sigma,
            )
        )
    return np.asarray(widths)


class SpectralOverlapRelaxationTests(unittest.TestCase):
    def test_transition_linewidth_is_shift_invariant(self) -> None:
        offsets = np.array([-2.0, 1.0, 5.0])
        intensities = np.array([1.0, 2.0, 1.0])

        base = transition_rms_linewidth_hz(offsets, intensities)
        shifted = transition_rms_linewidth_hz(offsets + 2.0e6, intensities)

        self.assertAlmostEqual(base, shifted)

    def test_spectral_overlap_fit_recovers_exact_coefficients(self) -> None:
        overlap = np.array([1.0, 0.5, 0.1])
        rates = 20.0 + 700.0 * overlap

        fit = fit_spectral_overlap_relaxation(1.0 / rates, overlap)

        self.assertAlmostEqual(fit.floor_rate_per_second, 20.0)
        self.assertAlmostEqual(fit.cross_relaxation_rate_per_second, 700.0)
        np.testing.assert_allclose(fit.predicted_rates_per_second, rates)

    def test_chen2020_naclo3_weak_field_validation(self) -> None:
        fields, measured_t2, metadata = _load_chen2020()
        linewidths = _linewidths_hz(
            fields,
            metadata,
            n_theta=8,
            n_phi=16,
        )
        overlap = spectral_overlap_factors(linewidths)
        fit = fit_spectral_overlap_relaxation(measured_t2, overlap)
        ratios = fit.predicted_t2_seconds / measured_t2

        self.assertLessEqual(fit.rms_residual_per_second, 10.0)
        self.assertTrue(np.all(np.diff(fit.predicted_t2_seconds) > 0.0))
        self.assertTrue(np.all(np.abs(ratios[1:] - 1.0) <= 0.15))
        self.assertAlmostEqual(ratios[0], 1.0, delta=0.01)
        self.assertGreater(fit.floor_rate_per_second, 0.0)
        self.assertGreater(fit.cross_relaxation_rate_per_second, 0.0)

    def test_chen2020_powder_linewidth_is_converged(self) -> None:
        _, _, metadata = _load_chen2020()
        field = np.array([41.0])
        coarse = _linewidths_hz(field, metadata, n_theta=6, n_phi=12)
        refined = _linewidths_hz(field, metadata, n_theta=12, n_phi=24)

        np.testing.assert_allclose(coarse, refined, rtol=2.0e-3)


class BismuthRelaxationFixtureTests(unittest.TestCase):
    def test_bi209_fixture_has_complete_transition_groups(self) -> None:
        with (DATA_ROOT / "goesweiner2020_bi209_r2.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 24)
        groups: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            key = (row["compound"], row["temperature_k"])
            groups.setdefault(key, set()).add(row["transition"])
        self.assertEqual(len(groups), 6)
        for transitions in groups.values():
            self.assertEqual(transitions, {"t1", "t2", "t3", "t4"})

    def test_published_vibrational_model_improves_aggregate_deviation(self) -> None:
        with (DATA_ROOT / "goesweiner2020_bi209_r2.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        measured = np.array([float(row["r2_exp_per_s"]) for row in rows])
        fa = np.array([float(row["r2_fa_per_s"]) for row in rows])
        fva = np.array([float(row["r2_fva_per_s"]) for row in rows])

        fa_mean = float(np.mean(np.abs(fa / measured - 1.0)))
        fva_mean = float(np.mean(np.abs(fva / measured - 1.0)))

        self.assertGreater(fa_mean, 0.15)
        self.assertLess(fva_mean, 0.10)
        self.assertLess(fva_mean, fa_mean)


if __name__ == "__main__":
    unittest.main()
