from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_impacted_tests import (  # noqa: E402
    CONFIG_PATH,
    changed_examples,
    select_groups,
    select_test_modules,
    module_paths,
)


class TestSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with CONFIG_PATH.open(encoding="utf-8") as handle:
            cls.config = json.load(handle)

    def test_nqr_source_change_selects_nqr_group_and_smoke(self) -> None:
        paths = ["src/spin_dynamics/nqr/crossover.py"]

        groups = select_groups(paths, self.config)
        modules = select_test_modules(paths, groups, self.config)

        self.assertEqual(groups, ("nqr",))
        self.assertEqual(modules[0], "tests.smoke_tests")
        self.assertIn("tests.test_nqr_crossover", modules)
        self.assertIn("tests.test_nqr", modules)

    def test_changed_test_module_is_always_selected_directly(self) -> None:
        paths = ["tests/test_motion.py"]

        modules = select_test_modules(paths, (), self.config, include_smoke=False)

        self.assertEqual(modules, ("tests.test_motion",))

    def test_only_changed_examples_receive_cli_checks(self) -> None:
        examples = changed_examples(
            [
                "examples/plot_nqr_nmr_crossover.py",
                "examples/plot_cpmg.py",
                "docs/python_api/nqr.md",
            ]
        )

        self.assertEqual(
            [item.name for item in examples],
            ["plot_nqr_nmr_crossover.py", "plot_cpmg.py"],
        )

    def test_module_names_are_converted_to_pytest_paths(self) -> None:
        self.assertEqual(
            module_paths(("tests.smoke_tests", "tests.test_experiment")),
            ("tests/smoke_tests.py", "tests/test_experiment.py"),
        )


if __name__ == "__main__":
    unittest.main()
