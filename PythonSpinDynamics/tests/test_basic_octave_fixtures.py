"""Compatibility entry point for MATLAB/Octave fixture validation.

The fixture suite is split into grouped modules under ``tests.octave_fixtures``.
This module re-exports the grouped ``unittest.TestCase`` classes so historical
commands such as ``python -m unittest tests.test_basic_octave_fixtures`` and
``python -m unittest discover -s tests`` continue to exercise the same checks.
"""

import unittest

from tests.octave_fixtures import core as _core
from tests.octave_fixtures import optimization as _optimization
from tests.octave_fixtures import workflows as _workflows


class OctaveFixtureTests(
    _core.OctaveCoreFixtureTests,
    _optimization.OctaveOptimizationFixtureTests,
    _workflows.OctaveWorkflowFixtureTests,
):
    """Combined fixture test case preserving the historical entry point."""


__all__ = ["OctaveFixtureTests"]


if __name__ == "__main__":
    unittest.main()
