"""Pulsed dipolar/hyperfine ESR: ESEEM, HYSCORE, and ENDOR for an S=1/2,I=1/2 pair.

Shows the three-pulse ESEEM trace and its spectrum, a 2D HYSCORE spectrum with
its cross-peaks, and Davies vs Mims ENDOR (including a Mims blind spot) for a
single anisotropic hyperfine coupling.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    HyperfineCoupling,
    cross_peak_positions,
    nuclear_frequencies,
)
from spin_dynamics.experiment import (  # noqa: E402
    ESRDaviesENDOR,
    ESRHYSCORE,
    ESRMimsENDOR,
    ESRThreePulseESEEM,
    Experiment,
    Sample,
)


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--larmor-mhz", type=float, default=14.5)
    parser.add_argument("--secular-mhz", type=float, default=3.0)
    parser.add_argument("--pseudosecular-mhz", type=float, default=2.5)
    parser.add_argument("--tau-ns", type=float, default=136.0)
    parser.add_argument("--eseem-points", type=int, default=400)
    parser.add_argument("--hyscore-points", type=int, default=72)
    parser.add_argument("--hyscore-step-ns", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    coupling = HyperfineCoupling(
        larmor_hz=args.larmor_mhz * 1e6,
        secular_hz=args.secular_mhz * 1e6,
        pseudosecular_hz=args.pseudosecular_mhz * 1e6,
    )
    nu_alpha, nu_beta = nuclear_frequencies(coupling)
    print(
        f"nu_alpha={nu_alpha / 1e6:.3f} MHz, nu_beta={nu_beta / 1e6:.3f} MHz, "
        f"cross-peaks at {[(round(a / 1e6, 2), round(b / 1e6, 2)) for a, b in cross_peak_positions(coupling)]}"
    )

    sample = Sample(hyperfine_coupling=coupling)

    # Three-pulse ESEEM (analytic vs density matrix) and its spectrum.
    tau = args.tau_ns * 1e-9
    eseem_spec = ESRThreePulseESEEM(
        acquisition_seconds=8.0e-6,
        tau_seconds=tau,
        num_points=args.eseem_points,
        zero_fill=8,
    )
    analytic = Experiment(sequence=eseem_spec, sample=sample).run().result
    quantum = Experiment(
        sequence=dataclasses.replace(eseem_spec, model="quantum"), sample=sample
    ).run().result

    # HYSCORE 2D.
    evolution = (args.hyscore_points - 1) * args.hyscore_step_ns * 1e-9
    hys = Experiment(
        sequence=ESRHYSCORE(
            evolution1_seconds=evolution,
            evolution2_seconds=evolution,
            tau_seconds=tau,
            num_points1=args.hyscore_points,
            num_points2=args.hyscore_points,
            zero_fill=4,
        ),
        sample=sample,
    ).run().result

    # ENDOR: Davies and a Mims trace at a non-blind tau.
    endor_common = dict(
        num_points=2000,
        linewidth_hz=1.5e5,
        frequency_min_hz=8.0e6,
        frequency_max_hz=22.0e6,
    )
    davies = Experiment(
        sequence=ESRDaviesENDOR(**endor_common), sample=sample
    ).run().result
    mims = Experiment(
        sequence=ESRMimsENDOR(tau_seconds=tau, **endor_common), sample=sample
    ).run().result

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    axes[0, 0].plot(
        1e6 * analytic.times_seconds,
        analytic.signal,
        label="analytic",
        color="tab:blue",
    )
    axes[0, 0].plot(
        1e6 * quantum.times_seconds,
        quantum.signal,
        "--",
        label="density matrix",
        color="tab:orange",
    )
    axes[0, 0].set_xlabel("T (us)")
    axes[0, 0].set_ylabel("V(T)")
    axes[0, 0].set_title("Three-Pulse ESEEM")
    axes[0, 0].legend()

    axes[0, 1].plot(
        analytic.frequencies_hz / 1e6, analytic.spectrum, color="tab:green"
    )
    for nu in (nu_alpha, nu_beta):
        axes[0, 1].axvline(nu / 1e6, color="tab:red", linestyle=":")
    axes[0, 1].set_xlim(0, 1.4 * max(nu_alpha, nu_beta) / 1e6)
    axes[0, 1].set_xlabel("Frequency (MHz)")
    axes[0, 1].set_title("ESEEM Spectrum")

    extent = [
        hys.frequencies2_hz[0] / 1e6,
        hys.frequencies2_hz[-1] / 1e6,
        hys.frequencies1_hz[0] / 1e6,
        hys.frequencies1_hz[-1] / 1e6,
    ]
    axes[1, 0].imshow(
        hys.spectrum, origin="lower", extent=extent, aspect="auto", cmap="inferno"
    )
    axes[1, 0].set_xlim(0, 1.4 * max(nu_alpha, nu_beta) / 1e6)
    axes[1, 0].set_ylim(0, 1.4 * max(nu_alpha, nu_beta) / 1e6)
    axes[1, 0].set_xlabel("nu2 (MHz)")
    axes[1, 0].set_ylabel("nu1 (MHz)")
    axes[1, 0].set_title("HYSCORE (cross-peaks)")

    axes[1, 1].plot(
        davies.frequencies_hz / 1e6,
        davies.spectrum,
        label="Davies",
        color="tab:blue",
    )
    axes[1, 1].plot(
        mims.frequencies_hz / 1e6,
        mims.spectrum,
        label="Mims",
        color="tab:orange",
    )
    axes[1, 1].set_xlabel("RF frequency (MHz)")
    axes[1, 1].set_ylabel("ENDOR intensity")
    axes[1, 1].set_title("Davies vs Mims ENDOR")
    axes[1, 1].legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
