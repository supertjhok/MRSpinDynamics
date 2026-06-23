"""Run a compact matched-probe diffusion CPMG Q sweep."""

from __future__ import annotations

import argparse

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import run_matched_diffusion_q_sweep  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=21, help="Number of offset points.")
    parser.add_argument("--num-echoes", type=int, default=3, help="Number of echoes.")
    parser.add_argument("--workers", type=int, default=1, help="Isochromat workers.")
    parser.add_argument("--sweep-workers", type=int, default=1, help="Parallel Q-value workers.")
    parser.add_argument("--dz-um", type=float, default=50.0, help="Slice thickness in micrometers.")
    parser.add_argument(
        "--diffusion-time-us",
        type=float,
        default=1000.0,
        help="Diffusion encoding block duration in microseconds.",
    )
    parser.add_argument("--t90-us", type=float, default=100.0, help="90-degree pulse length.")
    parser.add_argument(
        "--phase-step",
        type=float,
        default=None,
        help="Optional absolute RF phase advance per CPMG echo, in cycles.",
    )
    parser.add_argument(
        "--phase-bins",
        type=int,
        default=None,
        help="Optional absolute-phase bins for pulse-shape reuse.",
    )
    parser.add_argument(
        "--no-auto-refine-grid",
        dest="auto_refine_grid",
        action="store_false",
        help="Keep the requested numpts even when the offset grid may rephase.",
    )
    parser.add_argument(
        "--rephase-action",
        choices=["warn", "ignore", "raise"],
        default="raise",
        help="Action if auto-refinement is disabled and the grid may rephase.",
    )
    parser.set_defaults(auto_refine_grid=True)
    args = parser.parse_args()
    if args.dz_um <= 0:
        raise SystemExit("--dz-um must be positive")

    # Keep the default Q list modest. The diffusion path is most useful here as
    # a compact sanity check that echo attenuation, probe response, and Q
    # dependence stay finite. Very high-Q diffusion cases need extra transient
    # solver validation before they are good teaching examples.
    absolute_phase = None
    if args.phase_step is not None:
        echo_spacing_seconds = 1000e-6
        phase_step = 1.0 if args.phase_step == 0.0 else float(args.phase_step)
        absolute_phase = {
            "rf_frequency_hz": phase_step / echo_spacing_seconds,
            "phase_bins": args.phase_bins,
        }
    result = run_matched_diffusion_q_sweep(
        q_values=[20, 50],
        num_echoes=args.num_echoes,
        numpts=args.numpts,
        diffusion_time=args.diffusion_time_us * 1.0e-6,
        dz=args.dz_um * 1.0e-6,
        t90_seconds=args.t90_us * 1.0e-6,
        num_workers=args.workers,
        sweep_workers=args.sweep_workers,
        auto_refine_grid=args.auto_refine_grid,
        rephase_action=args.rephase_action,
        absolute_phase=absolute_phase,
    )
    # Results are stacked as (Q value, echo, time) for `echo` and
    # (Q value, echo) for the integrated scalar summaries.
    print("Matched diffusion CPMG Q sweep")
    print(f"q values: {result.values.size}")
    print(f"num offsets: {result.del_w.size}")
    print(f"num echoes: {result.sequence_time.size}")
    print(f"echo shape: {result.echo.shape}")
    print(f"echo integral shape: {result.echo_integrals.shape}")
    print(f"max |integral|: {abs(result.echo_integrals).max():.6g}")
    if result.absolute_phase:
        metadata = result.absolute_phase[0]
        if metadata is not None:
            print(f"absolute phase step: {metadata.delta_refocus_phase_cycles:.6g} cycles")
            print(f"pulse matrices: {metadata.pulse_matrix_count}")


if __name__ == "__main__":
    main()
