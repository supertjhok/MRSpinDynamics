"""Reproduce key ENDOR-QDyne predictions from Meinel et al. (2023).

The example checks the phase-cycled carrier, the weak-measurement signal, and
the initialization-infidelity linewidth model reported in Communications
Physics 6, 302. Run with ``--output`` to save the three-panel comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nano_mr import (  # noqa: E402
    initialization_infidelity_decay_rate,
    meinel_2023_endor_qdyne_protocol,
    simulate_endor_qdyne,
)


def parse_args() -> argparse.Namespace:
    """Return command-line controls for the paper comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shots",
        type=int,
        default=512,
        help="number of sequential ENDOR-QDyne measurements",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional PNG output path",
    )
    return parser.parse_args()


def main() -> None:
    """Simulate the reported sequence and plot its validation targets."""

    args = parse_args()
    plt = load_matplotlib(headless=args.output is not None)

    if args.shots < 16:
        raise ValueError("--shots must be at least 16")

    # The paper phase-cycles the RF mapping pulses by pi/2 per measurement.
    # At resonance this deliberately places the signal at fs/4 = 2.369 kHz.
    reference_hz = 2.5e6
    fitted = meinel_2023_endor_qdyne_protocol(
        rf_reference_frequency_hz=reference_hz,
        sensor_initialization_fidelity=1.0,
        intrinsic_nuclear_decay_rate_per_second=0.6e3,
    )
    result = simulate_endor_qdyne(
        fitted,
        target_frequency_hz=reference_hz,
        shot_count=args.shots,
        include_measurement_backaction=False,
    )

    # A second trace retains the paper's estimated initialization fidelity and
    # the exact weak-measurement back-action map from Supplementary Eq. 14.
    physical = meinel_2023_endor_qdyne_protocol(
        rf_reference_frequency_hz=reference_hz,
        sensor_initialization_fidelity=0.9,
    )
    with_backaction = simulate_endor_qdyne(
        physical,
        target_frequency_hz=reference_hz,
        shot_count=args.shots,
        include_measurement_backaction=True,
    )

    # Main-text Eq. 3 predicts an oscillatory coupling dependence because a
    # failed sensor initialization gives the nucleus a coherent phase kick.
    couplings_hz = np.linspace(0.0, 10.0e3, 1001)
    linewidths = np.array(
        [
            initialization_infidelity_decay_rate(
                coupling,
                105.0e-6,
                0.9,
                leading_order=True,
            )
            for coupling in couplings_hz
        ]
    )
    paper_rate = initialization_infidelity_decay_rate(
        6.0e3,
        105.0e-6,
        0.9,
        leading_order=True,
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    display = result.nominal_times_seconds <= 4.0e-3
    axes[0].plot(
        1.0e3 * result.nominal_times_seconds[display],
        result.sensor_z_expectation[display],
        label=r"ideal mapping, $\Gamma=0.6$ kHz",
    )
    axes[0].plot(
        1.0e3 * with_backaction.nominal_times_seconds[display],
        with_backaction.sensor_z_expectation[display],
        label=r"$f=0.9$ + back-action",
        alpha=0.85,
    )
    axes[0].set(
        xlabel="wall-clock time (ms)",
        ylabel=r"$\langle S_z\rangle$",
        title="Sequential ENDOR-QDyne record",
    )
    axes[0].legend(fontsize=8)

    spectrum = result.spectrum / max(np.max(result.spectrum), np.finfo(float).tiny)
    axes[1].plot(result.baseband_frequencies_hz / 1.0e3, spectrum)
    axes[1].axvline(
        2.368,
        color="tab:orange",
        linestyle="--",
        label="paper: 2.368 kHz",
    )
    axes[1].set_xlim(0.0, 4.5)
    axes[1].set(
        xlabel="baseband frequency (kHz)",
        ylabel="normalized amplitude",
        title="Quarter-sampling phase-cycle carrier",
    )
    axes[1].legend(fontsize=8)

    axes[2].plot(couplings_hz / 1.0e3, linewidths / 1.0e3)
    axes[2].scatter([6.0], [paper_rate / 1.0e3], color="tab:red", zorder=3)
    axes[2].annotate(
        f"6 kHz: {paper_rate / 1.0e3:.2f} kHz",
        (6.0, paper_rate / 1.0e3),
        xytext=(6.4, 1.45),
        arrowprops={"arrowstyle": "->"},
        fontsize=8,
    )
    axes[2].set(
        xlabel=r"$A_{zz}$ (kHz)",
        ylabel=r"$\Gamma_\mathrm{init}$ (kHz)",
        title="Initialization-infidelity linewidth",
    )

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()

    print(
        "phase-cycle carrier: "
        f"{result.expected_beat_frequency_hz / 1.0e3:.6f} kHz "
        "(paper target 2.368 kHz)"
    )
    print(
        "Eq. 3 initialization rate: "
        f"{paper_rate / 1.0e3:.6f} kHz "
        "(paper: approximately 1 kHz)"
    )
    print(
        "measurement strength: "
        f"{result.measurement_strength_rad:.6f} rad"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180)
        print(f"wrote {args.output}")
    else:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
