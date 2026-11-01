"""Documented validation gaps, reported as expected failures.

Each test below names a capability that does *not* yet have authoritative
reference parity. They used to be ``@unittest.skip`` placeholders, which read as
"skipped for an environmental reason" and could never fail. They are now
``expectedFailure`` tests whose body fails with the reason, so every run lists
them explicitly as known gaps (unittest ``expected failures``, pytest ``xfail``).

To close a gap, replace the ``self.fail(...)`` body with the real reference
comparison and remove the ``@unittest.expectedFailure`` decorator. A gap test
that starts passing while still decorated is an "unexpected success", which
fails the run, so a closed gap cannot go unnoticed.
"""

from __future__ import annotations

import unittest


class ValidationGapTests(unittest.TestCase):
    def _gap(self, reason: str) -> None:
        self.fail(f"documented validation gap: {reason}")

    @unittest.expectedFailure
    def test_broad_matched_diffusion_q_sweeps_have_reference_parity(self) -> None:
        """Validate broad diffusion sweeps and high-Q behavior against references."""

        self._gap("needs broader matched-diffusion references beyond compact Q<=2000 cases")

    @unittest.expectedFailure
    def test_probe_shaped_t1_prepared_imaging_has_reference_parity(self) -> None:
        """Validate probe-shaped inversion pulses in T1-prepared imaging."""

        self._gap("needs authoritative tuned/matched T1-prepared imaging references")

    @unittest.expectedFailure
    def test_moving_isochromat_imaging_workflows_have_reference_parity(self) -> None:
        """Validate moving-isochromat phase/frequency-encoded imaging outputs."""

        self._gap("needs full moving-isochromat imaging reference outputs")

    @unittest.expectedFailure
    def test_wurst_workflows_have_exact_matlab_fixture_parity(self) -> None:
        """Validate WURST inversion and WURST-CPMG against authoritative fixtures."""

        self._gap("needs exact MATLAB WURST fixture set beyond finite-output checks")

    @unittest.expectedFailure
    def test_historical_optimization_archives_have_full_parity(self) -> None:
        """Validate broad MATLAB result-file and fmincon optimizer parity."""

        self._gap("needs representative historical .mat archives and fmincon results")


if __name__ == "__main__":
    unittest.main()
