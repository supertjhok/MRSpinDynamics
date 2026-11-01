"""Edge-case regressions for the interference cancellers and recording I/O."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.interference.cancellers import (
    windowed_joint_signal_reference_canceller,
    windowed_ridge_fir_canceller,
)
from spin_dynamics.interference.masks import blank_mask
from spin_dynamics.interference.recordings import (
    RFIRecording,
    load_rfi_recording,
    save_rfi_recording,
)

COUPLING = 0.7 - 0.2j
NOISE = 1e-3


def _record(n=4096, seed=1):
    rng = np.random.default_rng(seed)
    references = rng.normal(size=(1, n)) + 1j * rng.normal(size=(1, n))
    noise = NOISE * (rng.normal(size=n) + 1j * rng.normal(size=n))
    primary = COUPLING * references[0] + noise
    # Baseline (fit) samples only in the first and last of four windows, as
    # when the middle of a record spans signal gaps.
    fit = np.zeros(n, dtype=bool)
    fit[:1024] = True
    fit[3072:] = True
    return primary, references, fit, noise


class WindowedCancellerEmptyWindowTests(unittest.TestCase):
    """Windows with no fit samples must still cancel a stationary coupling."""

    def _check(self, result, noise):
        # Independent reference: the true coupling is known, so a correct
        # canceller leaves only the additive noise in every window.
        np.testing.assert_allclose(
            result.coefficients.reshape(-1)[: 4], COUPLING, atol=5 * NOISE
        )
        residual = result.cleaned - noise
        self.assertLess(np.sqrt(np.mean(np.abs(residual) ** 2)), 5 * NOISE)

    def test_windowed_ridge_default_parameters(self) -> None:
        primary, references, fit, noise = _record()
        result = windowed_ridge_fir_canceller(primary, references, fit, window_samples=1024)
        self._check(result, noise)

    def test_windowed_ridge_with_ridge(self) -> None:
        primary, references, fit, noise = _record()
        result = windowed_ridge_fir_canceller(
            primary, references, fit, window_samples=1024, ridge=1e-6
        )
        self._check(result, noise)

    def test_windowed_joint_default_parameters(self) -> None:
        primary, references, fit, noise = _record()
        basis = np.exp(-np.arange(primary.size) / 500.0)[np.newaxis, :]
        result = windowed_joint_signal_reference_canceller(
            primary, references, fit, basis, window_samples=1024
        )
        self._check(result, noise)


class RecordingPathTests(unittest.TestCase):
    def test_save_and_load_use_the_same_name_without_npz_suffix(self) -> None:
        mask = blank_mask(1000.0, 8)
        recording = RFIRecording(
            primary=np.arange(8.0),
            references=np.ones((1, 8)),
            sample_rate_hz=1000.0,
            mask=mask,
            metadata={"note": "suffix"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shot.dat"
            written = save_rfi_recording(target, recording)
            self.assertEqual(written, Path(tmp) / "shot.dat.npz")
            self.assertTrue(written.exists())
            loaded = load_rfi_recording(target)
            np.testing.assert_array_equal(loaded.primary, recording.primary)
            self.assertEqual(loaded.metadata, {"note": "suffix"})
            # The explicit .npz spelling still works.
            np.testing.assert_array_equal(
                load_rfi_recording(written).primary, recording.primary
            )


if __name__ == "__main__":
    unittest.main()
