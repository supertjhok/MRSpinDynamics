"""Quantify q-space pore-image robustness to realistic acquisition limits.

The study repeats magnitude-only phase retrieval for an ellipse, a narrow slit,
and a connected dumbbell pore. It separates finite intensity SNR, radial q-range
truncation, and randomly missing q samples, then reports shift/reflection-
invariant correlation and intersection-over-union across independent trials.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


@dataclass(frozen=True)
class Condition:
    name: str
    snr: float
    qmax_fraction: float
    missing_fraction: float
    threshold_sigma: float = 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=48)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--er-iterations", type=int, default=60)
    parser.add_argument("--snr", type=float, default=30.0)
    parser.add_argument("--qmax-fraction", type=float, default=0.7)
    parser.add_argument("--missing-fraction", type=float, default=0.25)
    parser.add_argument("--threshold-sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


def _geometries(
    pixels: int,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    axis = (np.arange(pixels, dtype=np.float64) - pixels // 2) * 1.0e-6
    xx, zz = np.meshgrid(axis, axis, indexing="ij")

    ellipse = (xx / 8.0e-6) ** 2 + (zz / 5.0e-6) ** 2 <= 1.0
    ellipse_support = (xx / 10.0e-6) ** 2 + (zz / 7.0e-6) ** 2 <= 1.0

    slit = (np.abs(xx) <= 10.0e-6) & (np.abs(zz) <= 2.5e-6)
    slit_support = (np.abs(xx) <= 12.0e-6) & (np.abs(zz) <= 4.5e-6)

    left = (xx + 6.0e-6) ** 2 + zz**2 <= (4.5e-6) ** 2
    right = (xx - 6.0e-6) ** 2 + zz**2 <= (4.5e-6) ** 2
    bridge = (np.abs(xx) <= 6.0e-6) & (np.abs(zz) <= 1.5e-6)
    connected = left | right | bridge
    connected_support = (np.abs(xx) <= 12.0e-6) & (np.abs(zz) <= 7.0e-6)

    return axis, {
        "ellipse": (ellipse.astype(float), ellipse_support),
        "slit": (slit.astype(float), slit_support),
        "connected": (connected.astype(float), connected_support),
    }


def _conditions(args: argparse.Namespace) -> tuple[Condition, ...]:
    return (
        Condition("fully sampled", np.inf, 1.0, 0.0),
        Condition("finite SNR raw", float(args.snr), 1.0, 0.0),
        Condition(
            "finite SNR gated",
            float(args.snr),
            1.0,
            0.0,
            float(args.threshold_sigma),
        ),
        Condition("limited q range", np.inf, float(args.qmax_fraction), 0.0),
        Condition(
            "combined gated",
            float(args.snr),
            float(args.qmax_fraction),
            float(args.missing_fraction),
            float(args.threshold_sigma),
        ),
    )


def _run_study(args: argparse.Namespace):
    from spin_dynamics.workflows import (
        add_qspace_intensity_noise,
        phase_retrieve_qspace_magnitude,
        pore_form_factor_from_density,
        qspace_axes_from_real_space,
        qspace_sampling_mask,
        qspace_shape_metrics,
        threshold_qspace_intensity,
    )

    axis, geometries = _geometries(int(args.pixels))
    qx, qz = qspace_axes_from_real_space(axis, axis)
    rows: list[dict[str, object]] = []
    representative: dict[tuple[str, str], np.ndarray] = {}

    for geometry_index, (geometry, (rho, support)) in enumerate(geometries.items()):
        intensity = np.abs(pore_form_factor_from_density(rho)) ** 2
        for condition_index, condition in enumerate(_conditions(args)):
            for trial in range(int(args.trials)):
                trial_seed = (
                    int(args.seed)
                    + 1000 * geometry_index
                    + 100 * condition_index
                    + trial
                )
                sample_mask = qspace_sampling_mask(
                    qx,
                    qz,
                    qmax_fraction=condition.qmax_fraction,
                    missing_fraction=condition.missing_fraction,
                    seed=trial_seed,
                )
                measured, sigma = add_qspace_intensity_noise(
                    intensity,
                    snr=condition.snr,
                    seed=trial_seed + 10_000,
                    sample_mask=sample_mask,
                )
                if condition.threshold_sigma > 0.0:
                    measured = threshold_qspace_intensity(
                        measured,
                        noise_sigma=sigma,
                        threshold_sigma=condition.threshold_sigma,
                        sample_mask=sample_mask,
                    )
                retrieved = phase_retrieve_qspace_magnitude(
                    measured,
                    qx,
                    qz,
                    support=support,
                    sample_mask=sample_mask,
                    input_is_intensity=True,
                    iterations=int(args.iterations),
                    er_iterations=int(args.er_iterations),
                    seed=trial_seed + 20_000,
                )
                metrics = qspace_shape_metrics(retrieved.density, rho, threshold=0.2)
                rows.append(
                    {
                        "geometry": geometry,
                        "condition": condition.name,
                        "trial": trial,
                        "snr": condition.snr,
                        "qmax_fraction": condition.qmax_fraction,
                        "missing_fraction": condition.missing_fraction,
                        "threshold_sigma": condition.threshold_sigma,
                        "coverage_fraction": retrieved.coverage_fraction,
                        "noise_sigma": sigma,
                        "correlation": metrics.correlation,
                        "iou": metrics.intersection_over_union,
                        "area_ratio": metrics.area_ratio,
                        "residual": retrieved.residual,
                    }
                )
                if trial == 0:
                    representative[(geometry, condition.name)] = retrieved.density
    return axis, geometries, rows, representative


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    keys = sorted({(str(row["geometry"]), str(row["condition"])) for row in rows})
    for geometry, condition in keys:
        selected = [
            row
            for row in rows
            if row["geometry"] == geometry and row["condition"] == condition
        ]
        correlations = np.asarray([row["correlation"] for row in selected], dtype=float)
        ious = np.asarray([row["iou"] for row in selected], dtype=float)
        summaries.append(
            {
                "geometry": geometry,
                "condition": condition,
                "coverage": float(np.mean([row["coverage_fraction"] for row in selected])),
                "correlation_mean": float(np.mean(correlations)),
                "correlation_std": float(np.std(correlations, ddof=0)),
                "iou_mean": float(np.mean(ious)),
                "iou_std": float(np.std(ious, ddof=0)),
                "success_fraction": float(np.mean(ious >= 0.5)),
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(plt, args, axis, geometries, summaries, representative):
    conditions = _conditions(args)
    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(
        len(geometries),
        len(conditions) + 1,
        figsize=(20.0, 10.0),
        constrained_layout=True,
    )
    plot_labels = {
        "fully sampled": "Full q grid",
        "finite SNR raw": f"SNR {args.snr:g}, raw",
        "finite SNR gated": f"SNR {args.snr:g}, gated",
        "limited q range": f"qmax {args.qmax_fraction:g}",
        "combined gated": "Limited + missing + noise",
    }
    extent = [axis[0] * 1e6, axis[-1] * 1e6, axis[0] * 1e6, axis[-1] * 1e6]
    for row_index, (geometry, (rho, _)) in enumerate(geometries.items()):
        axes[row_index, 0].imshow(rho, origin="lower", extent=extent, cmap="gray_r")
        axes[row_index, 0].set_title(f"{geometry.title()}\nTruth", fontsize=10)
        for condition_index, condition in enumerate(conditions, start=1):
            estimate = representative[(geometry, condition.name)]
            estimate = estimate / max(float(np.max(estimate)), 1e-12)
            axes[row_index, condition_index].imshow(
                estimate, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1
            )
            summary = next(
                item
                for item in summaries
                if item["geometry"] == geometry and item["condition"] == condition.name
            )
            axes[row_index, condition_index].set_title(
                f"{plot_labels[condition.name]}\n"
                f"IoU {summary['iou_mean']:.2f} +/- {summary['iou_std']:.2f}\n"
                f"success {summary['success_fraction']:.0%}",
                fontsize=9,
            )
        for ax in axes[row_index]:
            ax.set_aspect("equal")
            if row_index == len(geometries) - 1:
                ax.set_xlabel("z (um)")
            if ax is axes[row_index, 0]:
                ax.set_ylabel("x (um)")
            else:
                ax.set_yticklabels([])
    return fig


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    args = _parse_args()
    if args.pixels < 24:
        raise SystemExit("--pixels must be at least 24")
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")
    if args.iterations < 0 or args.er_iterations < 0:
        raise SystemExit("iteration counts must be non-negative")
    if args.snr <= 0.0:
        raise SystemExit("--snr must be positive")
    if not (0.0 < args.qmax_fraction <= 1.0):
        raise SystemExit("--qmax-fraction must be in (0, 1]")
    if not (0.0 <= args.missing_fraction < 1.0):
        raise SystemExit("--missing-fraction must be in [0, 1)")
    if args.threshold_sigma < 0.0:
        raise SystemExit("--threshold-sigma must be non-negative")

    axis, geometries, rows, representative = _run_study(args)
    summaries = _summaries(rows)
    print("q-space imaging robustness study")
    print("geometry   condition          coverage   correlation       IoU       success")
    for row in summaries:
        print(
            f"{row['geometry']:<10} {row['condition']:<18} "
            f"{row['coverage']:>7.1%}   "
            f"{row['correlation_mean']:.3f}+/-{row['correlation_std']:.3f}   "
            f"{row['iou_mean']:.3f}+/-{row['iou_std']:.3f}   "
            f"{row['success_fraction']:.0%}"
        )
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"saved trials: {args.csv}")
    if args.output:
        # Load Matplotlib only after choosing interactive or headless output mode.
        plt = load_matplotlib(headless=True)
        fig = _plot(plt, args, axis, geometries, summaries, representative)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved figure: {args.output}")


if __name__ == "__main__":
    main()
