"""NQR SLSE with automatic reduced-vs-full engine selection.

The facade routes an ``NQRSLSE`` spec through ``select_nqr_model`` at plan
time: an isolated spin-1 line under a soft selective pulse resolves to the
reduced two-level engine, while a spin-3/2 site resolves to the full
``(2I+1)`` density-matrix engine. ``plan()`` reports which engine was chosen
and why, and ``run()`` dispatches accordingly -- the choice follows the
physics, not the spin quantum number alone.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.experiment import Experiment, NQRSLSE, Sample
from spin_dynamics.nqr import QuadrupolarSite


def describe(name: str, site: QuadrupolarSite, sequence: NQRSLSE) -> None:
    study = Experiment(sequence=sequence, sample=Sample(site=site))
    plan = study.plan()
    selection = next(f for f in plan.findings if f.rule == "nqr_model")
    print(f"== {name} (spin {site.spin}) ==")
    print(f"resolved engine: {plan.workflow}")
    print(f"recommendation: {selection.details['recommended_model']}")
    print(f"reasons: {'; '.join(selection.details['reasons']) or '(isolated line)'}")
    record = study.run()
    echoes = np.abs(record.result.echo_amplitudes)
    print(f"echo magnitudes: {np.array2string(echoes, precision=5, separator=', ')}\n")


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-echoes", type=int, default=4, help="SLSE echoes.")
    args = parser.parse_args()

    # Soft 90-degree selective pulse (nutation * duration = 0.25).
    soft = dict(
        pulse_duration_seconds=100e-6,
        nutation_hz=2.5e3,
        echo_spacing_seconds=1e-3,
        num_echoes=args.num_echoes,
        orientations="single",
    )
    describe(
        "isolated 14N line",
        QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3),
        NQRSLSE(**soft),
    )
    # Hard pulse on a spin-3/2 site: the full engine handles the Kramers
    # doublet the reduced two-level model cannot.
    describe(
        "35Cl spin-3/2 line",
        QuadrupolarSite(spin=1.5, isotope="35Cl", quadrupole_frequency_hz=30e6, eta=0.1),
        NQRSLSE(
            pulse_duration_seconds=10e-6,
            nutation_hz=25e3,
            echo_spacing_seconds=0.5e-3,
            num_echoes=args.num_echoes,
            orientations="single",
        ),
    )


if __name__ == "__main__":
    main()
