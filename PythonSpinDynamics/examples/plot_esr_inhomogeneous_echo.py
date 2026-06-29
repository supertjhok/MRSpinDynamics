"""ESR in inhomogeneous (B0, B1) fields: T2* dephasing, echo refocusing, B1 loss.

This study highlights the package's inhomogeneous-field focus for ESR. An
ensemble of electron spins sees a distribution of static-field offsets (B0
inhomogeneity / g-strain) and a distribution of microwave-field amplitudes (B1
inhomogeneity, so a nominal pi/2 or pi pulse is mistuned across the sample):

1. B0 inhomogeneity dephases the free-induction decay (the inhomogeneous T2*).
2. A two-pulse Hahn echo refocuses that static B0 spread, recovering the signal
   at 2*tau -- the classic motivation for the echo.
3. B1 inhomogeneity is NOT refocused: a spread of flip angles reduces the echo
   amplitude, limiting how much the echo can recover.

Panels: (A) FID decay vs B0 spread; (B) Hahn echo with homogeneous vs
inhomogeneous B1; (C) echo amplitude vs B1 spread; (D) single-spin echo amplitude
over the (B0 offset, B1 scaling) plane -- the refocusing bandwidth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    BOHR_MAGNETON_HZ_PER_T,
    ESRSpinSystem,
    flip_angle_duration,
    resonance_field_tesla,
    resonance_frequency_hz,
    simulate_fid,
    simulate_hahn_echo,
)


def _gaussian_grid(half_width: float, sigma: float, points: int, center: float = 0.0):
    """Return (values, normalized weights) for a truncated Gaussian distribution."""

    if sigma <= 0 or points <= 1:
        return np.array([center]), np.array([1.0])
    values = np.linspace(center - half_width, center + half_width, points)
    weights = np.exp(-0.5 * ((values - center) / sigma) ** 2)
    return values, weights / weights.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g", type=float, default=2.0023)
    parser.add_argument("--microwave-ghz", type=float, default=9.5)
    parser.add_argument("--nutation-mhz", type=float, default=20.0, help="Nominal Rabi rate.")
    parser.add_argument("--tau-ns", type=float, default=200.0)
    parser.add_argument(
        "--b0-spread-mhz",
        type=float,
        default=8.0,
        help="Std of the resonance-offset (B0 inhomogeneity) distribution.",
    )
    parser.add_argument(
        "--b1-spread",
        type=float,
        default=0.25,
        help="Relative std of the B1 (nutation) distribution for the inhomogeneous case.",
    )
    parser.add_argument("--n-b0", type=int, default=41)
    parser.add_argument("--n-b1", type=int, default=15)
    parser.add_argument("--points", type=int, default=241)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    system = ESRSpinSystem(g_tensor=args.g)
    microwave_hz = args.microwave_ghz * 1e9
    nominal_nutation = args.nutation_mhz * 1e6
    tau = args.tau_ns * 1e-9
    gamma_eff = BOHR_MAGNETON_HZ_PER_T * args.g

    b0_resonant = resonance_field_tesla(system, microwave_hz)
    carrier = resonance_frequency_hz(system, [0.0, 0.0, b0_resonant])
    t90 = flip_angle_duration(np.pi / 2.0, nominal_nutation)
    t180 = flip_angle_duration(np.pi, nominal_nutation)

    def field_for_offset(offset_hz: float) -> list[float]:
        return [0.0, 0.0, (carrier + offset_hz) / gamma_eff]

    # B0-offset and B1-scale ensembles.
    b0_sigma = args.b0_spread_mhz * 1e6
    offsets, b0_weights = _gaussian_grid(3.0 * b0_sigma, b0_sigma, args.n_b0)
    scales, b1_weights = _gaussian_grid(
        3.0 * args.b1_spread, args.b1_spread, args.n_b1, center=1.0
    )
    scales = np.clip(scales, 0.0, None)

    # --- Panel A: FID dephasing for several B0 spreads (homogeneous B1) ---------
    fid_times = np.linspace(0.0, 6.0 / (2.0 * np.pi * b0_sigma), args.points)
    axes_data_a = []
    for factor in (0.5, 1.0, 2.0):
        sigma = factor * b0_sigma
        off, wts = _gaussian_grid(3.0 * sigma, sigma, args.n_b0)
        signal = np.zeros(fid_times.size, dtype=np.complex128)
        for offset, weight in zip(off, wts):
            fid = simulate_fid(
                system,
                field_for_offset(offset),
                nutation_hz=nominal_nutation,
                pulse_duration_seconds=t90,
                times_seconds=fid_times,
                rf_frequency_hz=carrier,
            )
            signal += weight * fid.signal
        axes_data_a.append((sigma, np.abs(signal)))

    # --- Panel B: Hahn echo, homogeneous vs inhomogeneous B1 --------------------
    echo_times = np.linspace(0.0, 2.0 * tau, args.points)

    def ensemble_echo(scale_values, scale_weights, times):
        total = np.zeros(times.size, dtype=np.complex128)
        for offset, w_b0 in zip(offsets, b0_weights):
            field = field_for_offset(offset)
            for scale, w_b1 in zip(scale_values, scale_weights):
                result = simulate_hahn_echo(
                    system,
                    field,
                    nutation_hz=nominal_nutation * float(scale),
                    excitation_duration_seconds=t90,
                    refocus_duration_seconds=t180,
                    tau_seconds=tau,
                    times_seconds=times,
                    rf_frequency_hz=carrier,
                )
                total += w_b0 * w_b1 * result.signal
        return total

    echo_homog = ensemble_echo(np.array([1.0]), np.array([1.0]), echo_times)
    echo_inhomog = ensemble_echo(scales, b1_weights, echo_times)
    reference = float(np.max(np.abs(echo_homog)))

    # --- Panel C: echo amplitude vs B1 spread -----------------------------------
    b1_spreads = np.linspace(0.0, 0.6, 13)
    echo_vs_b1 = []
    for spread in b1_spreads:
        sc, wt = _gaussian_grid(3.0 * spread, spread, args.n_b1, center=1.0)
        sc = np.clip(sc, 0.0, None)
        peak = float(np.max(np.abs(ensemble_echo(sc, wt, np.array([tau])))))
        echo_vs_b1.append(peak)
    echo_vs_b1 = np.array(echo_vs_b1) / reference

    # --- Panel D: single-spin echo amplitude over (B0 offset, B1 scale) ---------
    grid_offsets = np.linspace(-3.0 * b0_sigma, 3.0 * b0_sigma, 41)
    grid_scales = np.linspace(0.3, 1.7, 31)
    echo_map = np.empty((grid_scales.size, grid_offsets.size))
    for i, scale in enumerate(grid_scales):
        for j, offset in enumerate(grid_offsets):
            result = simulate_hahn_echo(
                system,
                field_for_offset(offset),
                nutation_hz=nominal_nutation * float(scale),
                excitation_duration_seconds=t90,
                refocus_duration_seconds=t180,
                tau_seconds=tau,
                times_seconds=[tau],
                rf_frequency_hz=carrier,
            )
            echo_map[i, j] = float(np.abs(result.signal[0]))

    peak_inhomog = float(np.max(np.abs(echo_inhomog)))
    print(
        f"B0 spread {args.b0_spread_mhz:.1f} MHz, B1 spread {args.b1_spread:.2f}: "
        f"echo amplitude {peak_inhomog / reference:.3f} of the homogeneous-B1 echo"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)

    for sigma, magnitude in axes_data_a:
        axes[0, 0].plot(
            1e9 * fid_times,
            magnitude / magnitude[0],
            label=f"{sigma / 1e6:.0f} MHz",
        )
    axes[0, 0].set_xlabel("Time after pulse (ns)")
    axes[0, 0].set_ylabel("|FID| (normalized)")
    axes[0, 0].set_title("(A) FID dephasing vs B0 spread (T2*)")
    axes[0, 0].legend(title="B0 spread")

    axes[0, 1].plot(
        1e9 * echo_times, np.abs(echo_homog) / reference, label="homogeneous B1"
    )
    axes[0, 1].plot(
        1e9 * echo_times, np.abs(echo_inhomog) / reference, label="inhomogeneous B1"
    )
    axes[0, 1].axvline(1e9 * tau, color="0.6", linestyle="--", label="2*tau echo")
    axes[0, 1].set_xlabel("Time after refocusing pulse (ns)")
    axes[0, 1].set_ylabel("|echo| (rel. homog.)")
    axes[0, 1].set_title("(B) Hahn echo refocuses B0 inhomogeneity")
    axes[0, 1].legend()

    axes[1, 0].plot(100.0 * b1_spreads, echo_vs_b1, "o-", color="tab:red")
    axes[1, 0].set_xlabel("B1 spread (% of nominal)")
    axes[1, 0].set_ylabel("Echo amplitude (rel.)")
    axes[1, 0].set_title("(C) B1 inhomogeneity reduces the echo")

    extent = [
        grid_offsets[0] / 1e6,
        grid_offsets[-1] / 1e6,
        grid_scales[0],
        grid_scales[-1],
    ]
    image = axes[1, 1].imshow(
        echo_map / reference,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    axes[1, 1].set_xlabel("B0 offset (MHz)")
    axes[1, 1].set_ylabel("B1 scaling (x nominal)")
    axes[1, 1].set_title("(D) Single-spin echo over (B0, B1)")
    fig.colorbar(image, ax=axes[1, 1], label="|echo|")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
