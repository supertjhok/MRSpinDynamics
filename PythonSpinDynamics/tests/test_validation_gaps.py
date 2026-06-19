"""Tracked validation gaps that should become real tests as references mature."""

from __future__ import annotations

import unittest


class ValidationGapTests(unittest.TestCase):
    @unittest.skip("needs broader matched-diffusion references beyond compact Q<=2000 cases")
    def test_broad_matched_diffusion_q_sweeps_have_reference_parity(self) -> None:
        """Validate broad diffusion sweeps and high-Q behavior against references."""

    @unittest.skip("needs authoritative tuned/matched T1-prepared imaging references")
    def test_probe_shaped_t1_prepared_imaging_has_reference_parity(self) -> None:
        """Validate probe-shaped inversion pulses in T1-prepared imaging."""

    @unittest.skip("needs full moving-isochromat imaging reference outputs")
    def test_moving_isochromat_imaging_workflows_have_reference_parity(self) -> None:
        """Validate moving-isochromat phase/frequency-encoded imaging outputs."""

    @unittest.skip("needs exact MATLAB WURST fixture set beyond finite-output checks")
    def test_wurst_workflows_have_exact_matlab_fixture_parity(self) -> None:
        """Validate WURST inversion and WURST-CPMG against authoritative fixtures."""

    @unittest.skip("needs representative historical .mat archives and fmincon results")
    def test_historical_optimization_archives_have_full_parity(self) -> None:
        """Validate broad MATLAB result-file and fmincon optimizer parity."""


if __name__ == "__main__":
    unittest.main()
