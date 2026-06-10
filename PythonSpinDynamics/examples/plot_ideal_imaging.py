"""Plot a compact CPMG image reconstruction from the flower phantom."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import (  # noqa: E402
    run_ideal_cpmg_imaging,
    run_matched_cpmg_imaging,
    run_tuned_cpmg_imaging,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT.parent / "SpinDynamicsUpdated" / "Version_2" / "code" / "Images" / "flower.png"
DEFAULT_OUTPUT = ROOT / "results" / "ideal_imaging.png"
CONTRAST_NOTE = (
    "The default flower bitmap is inverted before simulation so the phantom is "
    "a bright object on a dark background, reducing bright-background aliasing "
    "when the field of view is small."
)


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        return None
    return plt


def _resize_nearest(image: np.ndarray, pixels: int) -> np.ndarray:
    # Nearest-neighbor resizing preserves a small discrete phantom grid, which
    # keeps the example fast and makes individual pixels easy to inspect.
    rows = np.linspace(0, image.shape[0] - 1, int(pixels)).round().astype(np.int64)
    cols = np.linspace(0, image.shape[1] - 1, int(pixels)).round().astype(np.int64)
    return image[rows[:, np.newaxis], cols[np.newaxis, :]]


def _load_phantom(path: Path, pixels: int, *, invert: bool = True) -> np.ndarray:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Pillow is required to load the phantom image when matplotlib is unavailable."
        ) from exc
    with Image.open(path) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.float64)
    image = image.astype(np.float64)
    if np.max(image) > 0:
        image = image / np.max(image)
    if invert:
        image = 1.0 - image
    # The simulation accepts a spin-density map. The default inversion turns
    # bright-background artwork into a compact bright object in a dark FOV.
    return _resize_nearest(image, pixels)


def _normalize_panel(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr - np.min(arr)
    vmax = np.max(arr)
    if vmax > 0:
        arr = arr / vmax
    return (255 * arr).clip(0, 255).astype(np.uint8)


def _save_with_pillow(output: Path, panels: list[np.ndarray]) -> None:
    from PIL import Image, ImageDraw

    # Pillow fallback keeps the plotting example usable in minimal environments
    # where Matplotlib is not installed.
    scale = 36
    labels = ["Spin-density phantom", "log |k-space|", "Reconstruction"]
    images = []
    for panel, label in zip(panels, labels):
        img = Image.fromarray(_normalize_panel(panel), mode="L").resize(
            (panel.shape[1] * scale, panel.shape[0] * scale),
            Image.Resampling.NEAREST,
        )
        canvas = Image.new("L", (img.width, img.height + 24), color=255)
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), label, fill=0)
        canvas.paste(img, (0, 24))
        images.append(canvas.convert("RGB"))
    out = Image.new("RGB", (sum(img.width for img in images), max(img.height for img in images)), color="white")
    x = 0
    for img in images:
        out.paste(img, (x, 0))
        x += img.width
    out.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=6, help="Output phantom width and height.")
    parser.add_argument("--ny", type=int, default=7, help="Number of off-plane offset samples.")
    parser.add_argument("--num-echoes", type=int, default=2, help="Number of CPMG echoes.")
    parser.add_argument(
        "--probe",
        choices=["ideal", "tuned", "matched"],
        default="ideal",
        help="Probe model used for image acquisition.",
    )
    parser.add_argument("--phase-workers", type=int, default=1, help="Parallel phase-encode workers.")
    parser.add_argument("--workers", type=int, default=1, help="Isochromat workers per phase point.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input phantom image.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path.")
    parser.add_argument(
        "--raw-image",
        action="store_true",
        help="Use the source bitmap without inverting its grayscale contrast.",
    )
    args = parser.parse_args()

    plt = _load_matplotlib()
    rho = _load_phantom(args.image, args.pixels, invert=not args.raw_image)

    runners = {
        "ideal": run_ideal_cpmg_imaging,
        "tuned": run_tuned_cpmg_imaging,
        "matched": run_matched_cpmg_imaging,
    }
    # The imaging runner returns k-space and reconstructed image arrays for
    # every echo. Small pixel counts are deliberate because this scales steeply.
    # Bright backgrounds can dominate the FOV and alias in compact demos, so the
    # default loader inverts the source bitmap before using it as spin density.
    result = runners[args.probe](
        rho,
        num_echoes=args.num_echoes,
        ny=args.ny,
        num_workers=args.workers,
        phase_workers=args.phase_workers,
    )
    # Echo 2 is the MATLAB demo's default display target; fall back to echo 1
    # when the user asks for a single-echo run.
    echo_index = min(1, args.num_echoes - 1)
    kspace = result.kspace[:, :, echo_index]
    recon = result.magnitude[:, :, echo_index]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        # Minimal dependency path: create a simple three-panel PNG with Pillow.
        _save_with_pillow(args.output, [result.rho, np.log1p(np.abs(kspace)), recon])
    else:
        # Matplotlib path: show the same three panels with scientific colormaps.
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
        axes[0].imshow(result.rho, cmap="gray")
        axes[0].set_title(f"{result.probe.capitalize()} spin-density phantom")
        axes[1].imshow(np.log1p(np.abs(kspace)), cmap="magma")
        axes[1].set_title("log |k-space|")
        axes[2].imshow(recon, cmap="gray")
        axes[2].set_title("Reconstruction")
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        if not args.raw_image:
            fig.text(0.5, 0.01, CONTRAST_NOTE, ha="center", va="bottom", fontsize=8)
        fig.savefig(args.output, dpi=150)
    print(f"saved: {args.output}")
    print(f"probe: {result.probe}")
    print(f"kspace shape: {result.kspace.shape}")
    print(f"image shape: {result.image.shape}")
    print(f"echo plotted: {echo_index + 1}")
    if args.raw_image:
        print("note: using raw source-image contrast; bright backgrounds may alias in small FOVs.")
    else:
        print(f"note: {CONTRAST_NOTE}")


if __name__ == "__main__":
    main()
