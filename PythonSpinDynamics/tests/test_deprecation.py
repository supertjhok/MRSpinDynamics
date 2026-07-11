from __future__ import annotations

import unittest
import warnings

from spin_dynamics.deprecation import (
    DeprecationInfo,
    SpinDynamicsDeprecationWarning,
    deprecated,
    warn_deprecated,
)


class DeprecationTests(unittest.TestCase):
    def test_warn_deprecated_uses_standard_visible_message(self) -> None:
        message = (
            r"old_api is deprecated since 0\.2 and will be removed in 0\.4; "
            r"use new_api instead\."
        )
        with self.assertWarnsRegex(SpinDynamicsDeprecationWarning, message):
            warn_deprecated(
                "old_api",
                since="0.2",
                removal="0.4",
                alternative="new_api",
            )

    def test_deprecated_decorator_preserves_callable_and_metadata(self) -> None:
        @deprecated(since="0.2", removal="0.4", alternative="replacement")
        def old_scale(value: int, factor: int = 2) -> int:
            """Scale a value."""

            return value * factor

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(old_scale(3, factor=4), 12)

        self.assertEqual(old_scale.__name__, "old_scale")
        self.assertEqual(old_scale.__doc__, "Scale a value.")
        self.assertEqual(
            old_scale.__deprecated__,
            DeprecationInfo("0.2", "0.4", "replacement"),
        )
        self.assertEqual(len(caught), 1)
        self.assertTrue(
            issubclass(caught[0].category, SpinDynamicsDeprecationWarning)
        )

    def test_deprecation_metadata_requires_nonempty_fields(self) -> None:
        for field in ("since", "removal", "alternative"):
            with self.subTest(field=field):
                values = {
                    "since": "0.2",
                    "removal": "0.4",
                    "alternative": "new_api",
                }
                values[field] = "  "
                with self.assertRaisesRegex(ValueError, field):
                    deprecated(**values)


if __name__ == "__main__":
    unittest.main()
