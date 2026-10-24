"""Compare full and reduced nitroxide relaxation with published TEMPOL/water data.

python examples/plot_tempol_published_benchmarks.py --output .tmp/tempol_comparison.png
Also writes a sibling <stem>_anchors.png and, with --report, a JSON benchmark.
All inputs are frozen locally; no network or parameter fitting is performed.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from _source_path import load_matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))

from validate_tempol_water import (  # noqa: E402
    compare,
    digitized_points,
    load_data,
    profile_model,
    saturation_product_curves,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    data = load_data()
    report = compare(data)
    profile = data["sezer_profile"]
    e = digitized_points(profile["figure5"])
    s = digitized_points(profile["figure6c"])
    frequencies = np.linspace(-95, 95, 601)
    prediction_s, prediction_e = profile_model(data, frequencies)
    without_exchange, _ = profile_model(data, frequencies, exchange_scale=0)
    low = profile_model(data, frequencies, b1_scale=0.8 / 0.9)[1]
    high = profile_model(data, frequencies, b1_scale=1 / 0.9)[1]
    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(frequencies, prediction_s, label="Full nitrogen relaxation")
    axes[0, 0].plot(
        frequencies,
        without_exchange,
        ":",
        color="purple",
        label="Full matrix, exchange off",
    )
    for mode, style, color, label in (
        ("reduced", ":", "0.5", "Legacy common T2"),
        ("line_dependent", "--", "tab:orange", "Line-dependent T2 only"),
    ):
        ms, me = profile_model(data, frequencies, relaxation=mode)
        axes[0, 0].plot(frequencies, ms, style, color=color, label=label)
        axes[0, 1].plot(frequencies, me, style, color=color, label=label)
    axes[0, 0].errorbar(
        s[:, 0],
        s[:, 1],
        xerr=1,
        yerr=0.006,
        fmt="o",
        ms=3,
        label="Published saturation THEORY, Fig. 6c",
    )
    axes[0, 0].set(
        xlabel="Microwave offset (MHz)", ylabel="Effective saturation", ylim=(0, 0.6)
    )
    axes[0, 1].fill_between(
        frequencies,
        np.minimum(low, high),
        np.maximum(low, high),
        alpha=0.15,
        color="tab:blue",
        label="B1 = 0.8–1.0 G sensitivity; not CI",
    )
    axes[0, 1].plot(
        frequencies, prediction_e, color="tab:blue", label="Full matrix; no refitting"
    )
    axes[0, 1].errorbar(
        e[:, 0],
        e[:, 1],
        xerr=1,
        yerr=0.05,
        fmt="s",
        ms=3,
        label="Measured DNP, digitized Fig. 5",
    )
    axes[0, 1].set(
        xlabel="Microwave offset (MHz)", ylabel="ε = E − 1", ylim=(-5.5, 0.1)
    )
    residual = profile_model(data, e[:, 0])[1] - e[:, 1]
    legacy_residual = profile_model(data, e[:, 0], relaxation="reduced")[1] - e[:, 1]
    axes[1, 0].axhline(0, color="0.5", lw=1)
    axes[1, 0].axhspan(
        -0.05, 0.05, color="0.8", alpha=0.5, label="Digitization half-width only"
    )
    axes[1, 0].plot(
        e[:, 0],
        residual,
        "o-",
        ms=3,
        label=f"Full matrix; RMSE {report['enhancement_profile']['rmse']:.3f}",
    )
    axes[1, 0].plot(
        e[:, 0],
        legacy_residual,
        ":",
        color="0.5",
        label=f"Legacy; RMSE {report['model_comparison']['reduced']['enhancement']['rmse']:.3f}",
    )
    axes[1, 0].set(xlabel="Microwave offset (MHz)", ylabel="Enhancement residual")
    b1 = np.linspace(0, 2, 301)
    measured_fit, other_parameters = saturation_product_curves(data, b1)
    axes[1, 1].plot(b1, measured_fit, label="Reconstructed EPR fit product (2009 JACS)")
    axes[1, 1].plot(
        b1, other_parameters, "--", label="Other study's RT central-line T1 × T2"
    )
    axes[1, 1].set(
        xlabel="Co-rotating B1 (G)", ylabel="Homogeneous-line saturation", ylim=(0, 1)
    )
    axes[1, 1].text(
        0.04,
        0.95,
        f"Products differ by {report['epr_product_comparison']['ratio']:.2f}×\nCurves reconstructed from parameters; no raw EPR points",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=8,
    )
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right" if ax is axes[1, 1] else "best")
    fig.suptitle(
        "Published TEMPOL/water benchmarks: full nitrogen relaxation versus reduced models\nSezer et al., PCCP 2009; Prandolini et al., JACS 2009 — no DNP refitting",
        fontsize=12,
    )
    fig.text(
        0.5,
        -0.015,
        "Figure-point bars show extraction precision; experimental measurement errors were not reported. No statistical CI is asserted.",
        ha="center",
        fontsize=8,
    )

    anchor_fig, anchor_axes = plt.subplots(
        1, 2, figsize=(10, 4.5), constrained_layout=True
    )
    anchors = data["hofer_anchors"]["points"]
    locations = np.arange(len(anchors))
    anchor_axes[0].bar(
        locations,
        [p["saturation_factor"] for p in anchors],
        yerr=[p["saturation_reported_plus_minus"] for p in anchors],
        capsize=4,
    )
    anchor_axes[0].axhline(
        1 / 3, color="0.4", ls="--", label="Isolated 14N line, selective-drive limit"
    )
    anchor_axes[0].set(
        xticks=locations,
        xticklabels=[f"{p['field_tesla']:g} T" for p in anchors],
        ylabel="DNP-derived effective saturation",
        ylim=(0, 1.15),
    )
    anchor_axes[0].legend(fontsize=8)
    anchor_axes[1].plot(
        locations,
        [p["approximate_epsilon_max"] for p in anchors],
        "s",
        label="Reported maximum; baseline unverified",
    )
    anchor_axes[1].plot(
        locations,
        [p["epsilon_factor_closure"] for p in report["hofer_factor_closure"]],
        "o",
        label="NMRD + reported s factor closure",
    )
    anchor_axes[1].set(
        xticks=locations,
        xticklabels=[f"{p['field_tesla']:g} T" for p in anchors],
        ylabel="ε = E − 1",
    )
    anchor_axes[1].legend(fontsize=8)
    for ax in anchor_axes:
        ax.grid(axis="y", alpha=0.25)
    anchor_fig.suptitle(
        "Höfer et al., JACS 2008: consistency anchors\nSaturation inferred from DNP; not independent saturation validation",
        fontsize=12,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160, bbox_inches="tight")
        anchor_fig.savefig(
            args.output.with_name(args.output.stem + "_anchors" + args.output.suffix),
            dpi=160,
        )
        plt.close(fig)
        plt.close(anchor_fig)
    else:
        plt.show()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
