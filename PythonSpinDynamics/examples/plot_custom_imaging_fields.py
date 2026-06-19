"""Plot CPMG imaging with custom B0, transmit-B1, and receive-B1 maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.workflows import (  # noqa: E402
    form_imaging_image,
    make_imaging_field_maps,
    run_ideal_phase_encoded_cpmg_imaging,
    run_matched_phase_encoded_cpmg_imaging,
    run_t1_encoded_phase_encoded_cpmg_imaging,
    run_tuned_phase_encoded_cpmg_imaging,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "custom_imaging_fields.png"




def _normalize_panel(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr - np.min(arr)
    vmax = np.max(arr)
    if vmax > 0:
        arr = arr / vmax
    return (255 * arr).clip(0, 255).astype(np.uint8)


def _save_with_pillow(output: Path, panels: list[tuple[np.ndarray, str]]) -> None:
    from PIL import Image, ImageDraw

    scale = 34
    tile_width = max(panel.shape[1] for panel, _ in panels) * scale
    tile_height = max(panel.shape[0] for panel, _ in panels) * scale + 24
    images = []
    for panel, label in panels:
        img = Image.fromarray(_normalize_panel(panel), mode="L").resize(
            (panel.shape[1] * scale, panel.shape[0] * scale),
            Image.Resampling.NEAREST,
        )
        canvas = Image.new("L", (tile_width, tile_height), color=255)
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), label, fill=0)
        canvas.paste(img, (0, 24))
        images.append(canvas.convert("RGB"))

    out = Image.new("RGB", (3 * tile_width, 2 * tile_height), color="white")
    for idx, img in enumerate(images):
        x = (idx % 3) * tile_width
        y = (idx // 3) * tile_height
        out.paste(img, (x, y))
    out.save(output)


def _synthetic_maps(pixels: int):
    axis = np.linspace(-1.0, 1.0, int(pixels), dtype=np.float64)
    x, z = np.meshgrid(axis, axis, indexing="ij")

    # Two compact density features make field-map effects easier to see than a
    # single centered disk in this tiny demonstration grid.
    rho = np.exp(-10.0 * ((x + 0.35) ** 2 + (z - 0.2) ** 2))
    rho += 0.7 * np.exp(-18.0 * ((x - 0.35) ** 2 + (z + 0.25) ** 2))
    rho = rho / np.max(rho)

    # The imaging workflow treats B0 as a normalized offset added to each
    # isochromat's off-plane offset. Keep the scale modest for a compact demo.
    b0_map = 0.35 * x + 0.15 * np.sin(np.pi * z)

    # Separate transmit and receive maps let probe sensitivity and reception
    # weighting be inspected independently.
    b1_tx_map = 0.55 + 0.45 * np.exp(-1.2 * ((x + 0.45) ** 2 + z**2))
    b1_rx_map = 0.50 + 0.50 * np.exp(-1.4 * ((x - 0.45) ** 2 + (z + 0.15) ** 2))
    return rho, b0_map, b1_tx_map, b1_rx_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=8, help="Phantom width and height.")
    parser.add_argument("--ny", type=int, default=7, help="Number of off-plane offset samples.")
    parser.add_argument("--num-echoes", type=int, default=1, help="Number of CPMG echoes.")
    parser.add_argument(
        "--probe",
        choices=["ideal", "tuned", "matched"],
        default="ideal",
        help="Probe model used for image acquisition.",
    )
    parser.add_argument("--phase-workers", type=int, default=1, help="Parallel phase-encode workers.")
    parser.add_argument("--workers", type=int, default=1, help="Isochromat workers per phase point.")
    parser.add_argument(
        "--image-mode",
        choices=["single", "echo-sum", "fit-rho", "fit-t2"],
        default="single",
        help="Image formation mode: selected echo, echo-summed magnitude, fitted rho, or fitted T2.",
    )
    parser.add_argument(
        "--echo-index",
        type=int,
        default=1,
        help="One-based echo index for --image-mode single and the k-space panel.",
    )
    parser.add_argument(
        "--t1-encoded",
        action="store_true",
        help="Use an ideal inversion-recovery preparation before phase encoding and CPMG.",
    )
    parser.add_argument(
        "--inversion-time",
        type=float,
        default=0.5e-3,
        help="Inversion delay in seconds for --t1-encoded.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path.")
    args = parser.parse_args()

    plt = load_matplotlib(required=False)
    rho, b0_map, b1_tx_map, b1_rx_map = _synthetic_maps(args.pixels)
    field_maps = make_imaging_field_maps(
        rho,
        b0_map=b0_map,
        b1_tx_map=b1_tx_map,
        b1_rx_map=b1_rx_map,
    )

    runners = {
        "ideal": run_ideal_phase_encoded_cpmg_imaging,
        "tuned": run_tuned_phase_encoded_cpmg_imaging,
        "matched": run_matched_phase_encoded_cpmg_imaging,
    }
    if args.t1_encoded and args.probe != "ideal":
        raise SystemExit("--t1-encoded is currently available only with --probe ideal")
    runner = run_t1_encoded_phase_encoded_cpmg_imaging if args.t1_encoded else runners[args.probe]
    runner_kwargs = {"inversion_time_seconds": args.inversion_time} if args.t1_encoded else {}
    result = runner(
        field_maps,
        num_echoes=args.num_echoes,
        ny=args.ny,
        num_workers=args.workers,
        phase_workers=args.phase_workers,
        **runner_kwargs,
    )

    echo_index = args.echo_index - 1
    if echo_index < 0 or echo_index >= args.num_echoes:
        raise SystemExit("--echo-index must be between 1 and --num-echoes")
    kspace = result.kspace[:, :, echo_index]
    recon = form_imaging_image(result, mode=args.image_mode, echo_index=echo_index)
    image_label = args.image_mode.replace("-", " ").title()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        (result.rho, "Spin density", "gray"),
        (result.b0_map, "B0 offset map", "coolwarm"),
        (result.b1_tx_map, "Transmit B1", "viridis"),
        (result.b1_rx_map, "Receive B1", "viridis"),
        (np.log1p(np.abs(kspace)), "log |k-space|", "magma"),
        (recon, image_label, "gray"),
    ]
    if plt is None:
        _save_with_pillow(args.output, [(data, title) for data, title, _ in panels])
    else:
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), constrained_layout=True)
        for axis, (data, title, cmap) in zip(axes.flat, panels):
            im = axis.imshow(data, cmap=cmap)
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
        fig.suptitle(f"{result.probe.capitalize()} CPMG imaging with custom fields")
        fig.savefig(args.output, dpi=150)

    print(f"saved: {args.output}")
    print(f"probe: {result.probe}")
    if args.t1_encoded:
        print(f"T1 encoded: inversion time {args.inversion_time:.6g} s")
    print(f"kspace shape: {result.kspace.shape}")
    print(f"image mode: {args.image_mode}")
    print(f"k-space echo plotted: {echo_index + 1}")
    print(f"B0 range: {np.min(result.b0_map):.3g} to {np.max(result.b0_map):.3g}")
    print(f"transmit B1 range: {np.min(result.b1_tx_map):.3g} to {np.max(result.b1_tx_map):.3g}")
    print(f"receive B1 range: {np.min(result.b1_rx_map):.3g} to {np.max(result.b1_rx_map):.3g}")


if __name__ == "__main__":
    main()
