"""Shrink the user manual's PNG figures by storing them as 256-colour palettes.

Matplotlib writes full-colour RGBA PNGs even when every pixel is opaque and the
plot uses only a few thousand distinct colours, mostly anti-aliasing shades.
This script rewrites each figure as an 8-bit palette PNG, about 3x smaller,
which is what pdfTeX embeds in ``user_manual.pdf``.

The palette is built so the conversion is visually lossless: the most frequent
colours (background, panel shading, line and text colours) are kept exactly,
and the rest of the palette is fitted to the remaining shades. A figure is only
rewritten when the result passes the error budget below and is smaller than the
original; otherwise it is left untouched. Figures with real transparency are
skipped. Running the script again on optimized figures changes nothing.

Usage::

    python docs/optimize_figures.py            # every PNG the manual includes
    python docs/optimize_figures.py --dry-run  # report only
    python docs/optimize_figures.py docs/images/example_probe_cpmg.png
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import re

import numpy as np
from PIL import Image


DOCS_DIR = Path(__file__).resolve().parent
PALETTE_SIZE = 256
EXACT_COLOURS = 64
"""Most frequent colours that are kept exactly in the palette."""
MIN_PSNR_DB = 40.0
LARGE_ERROR = 32
"""Per-channel error (out of 255) counted as a visible deviation."""
MAX_LARGE_ERROR_FRACTION = 0.005


def manual_figures(docs_dir: Path = DOCS_DIR) -> list[Path]:
    """Return the PNG files included by the LaTeX manual sources."""
    pattern = re.compile(r"includegraphics(?:\[[^]]*\])?\{([^}]*\.png)\}")
    names = {
        name
        for tex in docs_dir.glob("*.tex")
        for name in pattern.findall(tex.read_text(encoding="utf-8"))
    }
    return sorted(docs_dir / name for name in names)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def _palette_candidates(pixels: np.ndarray) -> list[Image.Image]:
    wide = pixels.astype(np.int32)
    packed = (wide[..., 0] << 16) | (wide[..., 1] << 8) | wide[..., 2]
    colours, inverse, counts = np.unique(packed.ravel(), return_inverse=True, return_counts=True)
    unique_rgb = np.stack([(colours >> 16) & 255, (colours >> 8) & 255, colours & 255], axis=1)
    if len(colours) <= PALETTE_SIZE:
        return [_indexed(pixels.shape, inverse, unique_rgb)]
    anchors = unique_rgb[np.argsort(-counts, kind="stable")[:EXACT_COLOURS]]
    rgb = Image.fromarray(pixels.astype(np.uint8), "RGB")
    candidates = []
    for method in (Image.Quantize.MEDIANCUT, Image.Quantize.FASTOCTREE):
        size = PALETTE_SIZE - len(anchors)
        fitted = rgb.quantize(size, method=method, kmeans=2).getpalette()[: 3 * size]
        palette = np.concatenate([anchors, np.asarray(fitted).reshape(-1, 3)])
        # Pillow's remap onto a given palette is approximate, so map each
        # distinct colour to its nearest palette entry here.
        nearest = np.empty(len(colours), dtype=np.int64)
        for start in range(0, len(colours), 4096):
            block = unique_rgb[start : start + 4096, None, :] - palette[None, :, :]
            nearest[start : start + 4096] = np.argmin(np.sum(block * block, axis=2), axis=1)
        candidates.append(_indexed(pixels.shape, nearest[inverse], palette))
    return candidates


def _indexed(shape: tuple[int, ...], indices: np.ndarray, palette: np.ndarray) -> Image.Image:
    image = Image.fromarray(indices.reshape(shape[:2]).astype(np.uint8), "P")
    image.putpalette(palette.astype(np.uint8).ravel().tolist())
    return image


def _errors(original: np.ndarray, candidate: Image.Image) -> tuple[float, float]:
    """Return (PSNR in dB, fraction of pixels with a large channel error)."""
    diff = np.abs(original - np.asarray(candidate.convert("RGB"), dtype=np.int16))
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = float("inf") if mse == 0.0 else 10.0 * np.log10(255.0**2 / mse)
    return psnr, float(np.mean(diff.max(axis=2) > LARGE_ERROR))


def optimize_png(path: Path, *, dry_run: bool = False) -> str:
    """Rewrite ``path`` as a palette PNG when that is smaller and within budget."""
    path = Path(path)
    original_size = path.stat().st_size
    with Image.open(path) as image:
        image.load()
    if image.mode == "P":
        return f"kept (already a palette PNG): {path.name}"
    if image.mode in ("RGBA", "LA") and image.getchannel("A").getextrema() != (255, 255):
        return f"skipped (transparent): {path.name}"
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.int16)

    best: tuple[float, float, bytes] | None = None
    for candidate in _palette_candidates(pixels):
        psnr, large = _errors(pixels, candidate)
        if psnr < MIN_PSNR_DB or large > MAX_LARGE_ERROR_FRACTION:
            continue
        if best is None or psnr > best[0]:
            best = (psnr, large, _png_bytes(candidate))
    if best is None:
        return f"kept (palette over error budget): {path.name}"
    psnr, large, data = best
    if len(data) >= original_size:
        return f"kept (already small): {path.name}"
    if not dry_run:
        path.write_bytes(data)
    return (
        f"{original_size / 1e3:7.0f} kB -> {len(data) / 1e3:6.0f} kB  "
        f"PSNR {psnr:5.1f} dB  large-error pixels {100 * large:.3f}%  {path.name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path, help="PNG files; default: the manual's figures.")
    parser.add_argument("--dry-run", action="store_true", help="Report without rewriting files.")
    args = parser.parse_args()
    for path in args.paths or manual_figures():
        print(optimize_png(path, dry_run=args.dry_run), flush=True)


if __name__ == "__main__":
    main()
