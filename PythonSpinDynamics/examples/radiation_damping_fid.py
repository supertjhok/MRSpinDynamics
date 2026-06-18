"""Run a radiation-damping FID example coupled to tuned/matched probes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import run_radiation_damping_fid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["tuned", "matched"], default="matched")
    parser.add_argument("--fill-factor", type=float, default=0.7)
    parser.add_argument("--field-tesla", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--proton-concentration", type=float, default=111.0)
    parser.add_argument(
        "--polarization-scale",
        type=float,
        default=250.0,
        help="Multiplier for thermal Mth; useful for visible compact examples.",
    )
    parser.add_argument("--flip-angle", type=float, default=np.pi / 3)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--detuning", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--points", type=int, default=401)
    parser.add_argument("--model", choices=["instant", "circuit"], default="instant")
    parser.add_argument("--save-npz", type=Path, default=None)
    args = parser.parse_args()

    result = run_radiation_damping_fid(
        probe=args.probe,
        fill_factor=args.fill_factor,
        field_tesla=args.field_tesla,
        proton_concentration_mol_per_liter=args.proton_concentration,
        temperature_kelvin=args.temperature,
        polarization_scale=args.polarization_scale,
        flip_angle=args.flip_angle,
        phase=args.phase,
        detuning=args.detuning,
        duration_seconds=args.duration,
        num_points=args.points,
        model=args.model,
    )

    print("Radiation damping FID")
    print(f"probe: {result.probe.name}")
    print(f"Q: {result.probe.q:.6g}")
    print(f"fill factor: {result.probe.fill_factor:.6g}")
    print(f"phase rad: {result.probe.phase:.12g}")
    print(f"detuning rad/s: {result.probe.detuning:.12g}")
    print(f"equilibrium magnetization A/m: {result.probe.equilibrium_magnetization:.12g}")
    print(f"Trd seconds: {result.probe.trd:.12g}")
    print(f"probe ringdown seconds: {result.probe.resonator_time_constant:.12g}")
    print(f"initial |mxy|: {result.envelope[0]:.12g}")
    print(f"final |mxy|: {result.envelope[-1]:.12g}")
    print(f"final mz: {result.mz[-1]:.12g}")
    print(f"analytic final |mxy|: {result.analytic_envelope[-1]:.12g}")

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            time=result.time_seconds,
            normalized_time=result.normalized_time,
            mxy=result.mxy,
            mz=result.mz,
            feedback=result.feedback,
            analytic_envelope=result.analytic_envelope,
        )
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
