"""MATLAB/Octave fixture validation tier.

Run with:
    python -m unittest tests.fixture_tests
"""

from __future__ import annotations

import unittest

from tests.test_basic_octave_fixtures import OctaveFixtureTests


def load_tests(
    loader: unittest.TestLoader,
    _standard_tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    return loader.loadTestsFromTestCase(OctaveFixtureTests)


if __name__ == "__main__":
    unittest.main()
