"""Golden-output parity gate for performance-backend work (Phase 0).

This locks the current NumPy reference output of the canonical hot-path
scenarios (``tests/_perf_scenarios.py``). It serves two purposes:

* today, a plain regression guard against accidental numerical drift in the
  reference kernels;
* once Numba/JAX backends land (see ``docs/performance.md``), the same
  scenarios are run through each backend and compared to this fixture, proving
  the accelerated path reproduces the reference within tolerance.

Regenerate the fixture after an *intended* reference change with::

    UPDATE_PERF_GOLDEN=1 python -m unittest tests.test_perf_golden

or directly::

    python tests/test_perf_golden.py --update
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import _perf_scenarios as scenarios  # noqa: E402


FIXTURE = ROOT / "tests" / "perf_fixtures" / "golden.npz"

# Tolerances are tight enough to catch real regressions but loose enough that a
# compiled backend's float reassociation (FMA, different reduction order) still
# passes. Backends may relax these locally if a scenario warrants it.
RTOL = 1e-10
ATOL = 1e-12


def write_golden() -> Path:
    """Compute every scenario and write the golden ``.npz`` fixture."""

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **scenarios.compute_all())
    return FIXTURE


class PerfGoldenParityTests(unittest.TestCase):
    """Each scenario must reproduce the stored golden output."""

    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("UPDATE_PERF_GOLDEN") == "1":
            write_golden()
        if not FIXTURE.exists():
            raise unittest.SkipTest(
                "golden fixture missing; regenerate with UPDATE_PERF_GOLDEN=1"
            )
        with np.load(FIXTURE) as data:
            cls.golden = {key: data[key] for key in data.files}
        cls.current = scenarios.compute_all()

    def test_no_extra_or_missing_keys(self) -> None:
        self.assertEqual(set(self.current), set(self.golden))

    def test_scenarios_match_golden(self) -> None:
        for key in sorted(self.golden):
            with self.subTest(array=key):
                self.assertIn(key, self.current)
                expected = self.golden[key]
                actual = self.current[key]
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_allclose(
                    actual, expected, rtol=RTOL, atol=ATOL, err_msg=key
                )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update", action="store_true", help="(re)write the golden fixture and exit"
    )
    args = parser.parse_args()
    if args.update:
        path = write_golden()
        print(f"Wrote {path}")
        return
    unittest.main(argv=[sys.argv[0]])


if __name__ == "__main__":
    _main()
