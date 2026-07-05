"""Imaging with an automatically solved transmit-coil B1 map.

Describe a phantom and a physical transmit coil in SI meters; the facade
solves the coil's Biot-Savart field on the phantom grid at plan time,
projects it transverse to B0, normalizes it (transmit calibration at the
sample), and hands the resulting B1 map to the imaging workflow -- replacing
its synthetic default. ``plan()`` reports a transmit-efficiency diagnostic
and warns if the coil is poorly oriented relative to B0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.experiment import (
    CPMGImaging,
    Experiment,
    Hardware,
    ImagingPlane,
    Phantom,
    Sample,
    SolenoidCoil,
    TxCoil,
    UniformB0,
)


def disc_phantom(n: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n)
    return ((x[:, None] ** 2 + x[None, :] ** 2) < 0.7).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=8, help="Phantom grid size.")
    parser.add_argument("--ny", type=int, default=5, help="Phase-encode steps.")
    parser.add_argument(
        "--coil-axis",
        default="x",
        choices=["x", "y", "z"],
        help="Solenoid axis; 'z' is parallel to B0 and warns (low efficiency).",
    )
    parser.add_argument("--save-npz", type=Path, default=None, help="Optional .npz path.")
    args = parser.parse_args()

    phantom = Phantom(rho=disc_phantom(args.grid))
    study = Experiment(
        sequence=CPMGImaging(ny=args.ny),
        sample=Sample(phantom=phantom, t1_seconds=5e-3, t2_seconds=5e-3),
        hardware=Hardware(
            b0=UniformB0(direction=(0.0, 0.0, 1.0)),
            tx_coil=TxCoil(
                geometry=SolenoidCoil(
                    radius_m=0.015, length_m=0.03, turns=10, axis=args.coil_axis
                )
            ),
            plane=ImagingPlane(extent_m=(0.02, 0.02)),
        ),
    )

    plan = study.plan()
    print("== plan ==")
    print(plan.report())
    for finding in plan.findings:
        if finding.rule == "hardware_wiring" and finding.details:
            for key, value in finding.details.items():
                print(f"  {key}: {value:.3f}")

    record = study.run()
    result = record.result
    b1 = result.b1_tx_map
    weights = np.abs(phantom.rho)
    mean_b1 = float(np.sum(b1 * weights) / np.sum(weights))
    print("\n== result ==")
    print(f"image shape: {result.image.shape}")
    print(f"solved B1 range: {b1.min():.3f} to {b1.max():.3f}")
    print(f"rho-weighted mean B1 (calibration target = 1.0): {mean_b1:.4f}")

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        record.save(str(args.save_npz))
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
