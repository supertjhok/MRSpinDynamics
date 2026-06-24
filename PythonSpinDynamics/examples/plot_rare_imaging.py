"""Frequency-encoded imaging: spin-warp vs RARE / fast spin echo.

The existing CPMG imaging fills k-space one point per phase-encode step. These
workflows add a readout (frequency-encode) gradient so each spin echo samples a
whole k-space line:

* ``run_spin_warp_imaging`` -- one spin echo per phase-encode line (``pz``
  excitations). All lines share the same echo time, so there is no relaxation
  blurring: this is the reference image.
* ``run_rare_imaging`` -- a CPMG echo train where each echo reads a different
  line, so ``echo_train_length`` lines are acquired per excitation. The image
  needs only ``ceil(pz / echo_train_length)`` shots, but the T2 decay across the
  train weights the phase-encode lines and broadens the point-spread function
  (the characteristic RARE blurring).

This example builds a synthetic phantom with two T2 compartments and compares
the spin-warp reference with a single-shot RARE acquisition, then shows the
k-space T2 weighting that causes the blurring.

Run with ``--output figure.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare spin-warp and RARE (fast spin echo) frequency-encoded "
            "imaging, showing the RARE speed/blurring trade-off."
        )
    )
    parser.add_argument("--pixels", type=int, default=32,
                        help="Image size (pixels per side).")
    parser.add_argument("--echo-train-length", type=int, default=None,
                        help="RARE echoes per excitation (default: full single shot).")
    parser.add_argument("--readout-time", type=float, default=2.0e-3,
                        help="Readout (frequency-encode) duration per echo (s).")
    parser.add_argument("--t2-long", type=float, default=120.0e-3,
                        help="T2 of the bright compartment (s).")
    parser.add_argument("--t2-short", type=float, default=25.0e-3,
                        help="T2 of the dim compartment (s).")
    parser.add_argument("--fov", type=float, default=0.02,
                        help="Field of view (m), square.")
    parser.add_argument("--output", type=Path,
                        help="Optional output PNG path. If omitted, show the plot.")
    return parser.parse_args()


def _phantom(n: int, t2_long: float, t2_short: float):
    """A two-compartment phantom: disk + bars (long T2), small disk (short T2)."""

    yy, xx = np.mgrid[0:n, 0:n]
    cx = (n - 1) / 2.0
    rho = np.zeros((n, n), dtype=np.float64)
    t2 = np.full((n, n), t2_long, dtype=np.float64)

    disk = (xx - cx) ** 2 + (yy - n * 0.32) ** 2 <= (n * 0.18) ** 2
    rho[disk] = 1.0

    small = (xx - n * 0.66) ** 2 + (yy - n * 0.66) ** 2 <= (n * 0.12) ** 2
    rho[small] = 0.7
    t2[small] = t2_short

    # A set of vertical bars (resolution feature) in the lower-left quadrant.
    for col in range(int(n * 0.18), int(n * 0.45), 3):
        bars = (xx >= col) & (xx < col + 1) & (yy >= n * 0.6) & (yy < n * 0.85)
        rho[bars] = 0.9
    return rho, t2


def _normalize(image: np.ndarray, reference_max: float) -> np.ndarray:
    image = np.abs(image)
    peak = image.max()
    return image / peak * reference_max if peak > 0 else image


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.workflows import run_rare_imaging, run_spin_warp_imaging

    n = int(args.pixels)
    etl = int(args.echo_train_length) if args.echo_train_length else n
    rho, t2_map = _phantom(n, args.t2_long, args.t2_short)
    fov = (args.fov, args.fov)
    common = dict(fov=fov, t2_map=t2_map, readout_time=args.readout_time)

    spin_warp = run_spin_warp_imaging(rho, **common)
    rare = run_rare_imaging(rho, echo_train_length=etl,
                            phase_encode_order="linear", **common)

    sw_image = _normalize(spin_warp.magnitude[:, :, 0], rho.max())
    rare_image = _normalize(rare.magnitude[:, :, 0], rho.max())
    sw_err = np.linalg.norm(sw_image - rho) / np.linalg.norm(rho)
    rare_err = np.linalg.norm(rare_image - rho) / np.linalg.norm(rho)

    print(f"image {n}x{n}, FOV {args.fov*1e3:.0f} mm, readout {args.readout_time*1e3:.1f} ms")
    print(f"spin-warp: {spin_warp.num_shots} shots, relative error {sw_err:.3f}")
    print(f"RARE ETL={etl}: {rare.num_shots} shot(s) "
          f"({spin_warp.num_shots / rare.num_shots:.0f}x fewer excitations), "
          f"relative error {rare_err:.3f}")
    print(f"RARE echo-time range across k-space: "
          f"{rare.line_echo_time.min()*1e3:.0f}-{rare.line_echo_time.max()*1e3:.0f} ms")

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.6))

    axes[0, 0].imshow(rho.T, origin="lower", cmap="gray")
    axes[0, 0].set_title("phantom (spin density)")

    axes[0, 1].imshow(sw_image.T, origin="lower", cmap="gray")
    axes[0, 1].set_title(f"spin-warp: {spin_warp.num_shots} shots, err {sw_err:.2f}")

    axes[1, 0].imshow(rare_image.T, origin="lower", cmap="gray")
    axes[1, 0].set_title(
        f"RARE ETL={etl}: {rare.num_shots} shot(s), err {rare_err:.2f}"
    )

    # k-space T2 weighting that drives RARE blurring: order lines by k_z and show
    # the relative signal each line retains at its echo time (short T2 = stronger
    # low-pass filter along the phase-encode axis).
    k_line = np.arange(n) - n // 2
    weight_short = np.exp(-rare.line_echo_time / args.t2_short)
    weight_long = np.exp(-rare.line_echo_time / args.t2_long)
    order = np.argsort(k_line)
    axes[1, 1].plot(k_line[order], weight_long[order], "o-", color="#1f77b4",
                    markersize=3, label=f"T2={args.t2_long*1e3:.0f} ms")
    axes[1, 1].plot(k_line[order], weight_short[order], "s-", color="#d62728",
                    markersize=3, label=f"T2={args.t2_short*1e3:.0f} ms")
    axes[1, 1].set_xlabel("phase-encode line (k_z index)")
    axes[1, 1].set_ylabel("retained signal at echo time")
    axes[1, 1].set_title("RARE k-space T2 weighting")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(fontsize="small")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
