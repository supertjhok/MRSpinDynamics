"""Inspect singlet order versus parahydrogen enrichment."""
# Follow the example from physical inputs through simulation to reported observables.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.hyperpolarization import (  # noqa: E402
    parahydrogen_state,
    singlet_order_amplitude,
    triplet_population,
)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--para-fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 1.00],
    )
    args = parser.parse_args()

    print("para fraction  singlet population  triplet population  order amplitude")
    for fraction in args.para_fractions:
        state = parahydrogen_state(fraction)
        amplitude = singlet_order_amplitude(state.deviation_density)
        triplet = triplet_population(state.density_matrix)
        print(
            f"{fraction:13.3f}  {state.singlet_population:18.3f}  "
            f"{triplet:18.3f}  {amplitude:15.3f}"
        )


if __name__ == "__main__":
    main()
