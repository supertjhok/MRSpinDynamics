"""Example-script validation tier.

Run with:
    python -m unittest tests.example_tests
"""

from __future__ import annotations

import unittest

from tests.test_examples import ExampleSmokeTests


def load_tests(
    loader: unittest.TestLoader,
    _standard_tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    return loader.loadTestsFromTestCase(ExampleSmokeTests)


if __name__ == "__main__":
    unittest.main()
