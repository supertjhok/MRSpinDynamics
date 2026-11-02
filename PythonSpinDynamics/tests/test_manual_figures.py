"""Manual figure registry and palette-PNG optimizer."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

DOCS = Path(__file__).resolve().parents[1] / "docs"
HAS_PIL = importlib.util.find_spec("PIL") is not None


def _load(name):
    spec = importlib.util.spec_from_file_location(name, DOCS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_PIL, "Pillow not installed")
class ManualFigureRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = _load("optimize_figures")
        cls.generator = _load("generate_manual_figures")

    def test_every_manual_figure_has_a_generator(self):
        generated = {
            (self.generator.IMAGE_DIR / parts[-1]).resolve()
            for parts in self.generator.FIGURES.values()
        }
        missing = [
            str(path.relative_to(DOCS))
            for path in self.optimizer.manual_figures(DOCS)
            if path.resolve() not in generated
        ]
        self.assertEqual(missing, [], "add these figures to generate_manual_figures.FIGURES")

    def test_every_generator_script_exists(self):
        examples = DOCS.parent / "examples"
        for name, parts in self.generator.FIGURES.items():
            with self.subTest(name=name):
                self.assertTrue((examples / parts[0]).is_file())


@unittest.skipUnless(HAS_PIL, "Pillow not installed")
class PaletteOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = _load("optimize_figures")

    @staticmethod
    def _rgba(rgb):
        from PIL import Image

        alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
        return Image.fromarray(np.concatenate([rgb, alpha], axis=2), "RGBA")

    def _plot_like_image(self):
        # Like an opaque matplotlib RGBA figure: white background, a dark axis,
        # and a colour-map strip whose ~600 shades lie along one curve.
        y, x = np.mgrid[20:220, 40:300]
        level = (3 * x + 7 * y) % 260
        red = level * 255 // 259
        rgb = np.full((240, 320, 3), 255, dtype=np.uint8)
        rgb[20:220, 40:300, 0] = red
        rgb[20:220, 40:300, 1] = 255 - red
        rgb[20:220, 40:300, 2] = 120 + 4 * (level % 2)
        rgb[220:224, 30:310] = (42, 42, 42)
        return self._rgba(rgb), rgb

    def test_palette_png_keeps_dominant_colours_and_is_idempotent(self):
        from PIL import Image

        image, rgb = self._plot_like_image()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.png"
            image.save(path)
            report = self.optimizer.optimize_png(path)
            self.assertIn("PSNR", report)
            with Image.open(path) as result:
                self.assertEqual(result.mode, "P")
                out = np.asarray(result.convert("RGB"))
            np.testing.assert_array_equal(out[0, 0], (255, 255, 255))
            np.testing.assert_array_equal(out[221, 100], (42, 42, 42))
            self.assertLessEqual(int(np.abs(out.astype(int) - rgb).max()), 32)
            before = path.read_bytes()
            self.assertIn("kept", self.optimizer.optimize_png(path))
            self.assertEqual(path.read_bytes(), before)

    def test_images_beyond_the_error_budget_are_left_alone(self):
        # A full 2-D colour ramp (~65,000 colours) cannot fit 256 entries.
        y, x = np.mgrid[0:256, 0:256]
        rgb = np.stack([x, y, np.full_like(x, 128)], axis=2).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.png"
            self._rgba(rgb).save(path)
            before = path.read_bytes()
            self.assertIn("error budget", self.optimizer.optimize_png(path))
            self.assertEqual(path.read_bytes(), before)

    def test_transparent_figures_are_left_alone(self):
        image, _ = self._plot_like_image()
        image.putpixel((0, 0), (255, 255, 255, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.png"
            image.save(path)
            before = path.read_bytes()
            self.assertIn("transparent", self.optimizer.optimize_png(path))
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
