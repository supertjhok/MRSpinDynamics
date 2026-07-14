"""Regenerate the example-result figures embedded in the user manual.

Run from any directory with a Python environment containing the package's
``plot`` and ``opt`` extras. The examples remain the single source of truth;
this script only supplies stable documentation output paths and, where useful,
the fast deterministic plotting mode.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PACKAGE_ROOT / "docs" / "images"

FIGURES: dict[str, tuple[str, ...]] = {
    "probe-cpmg": ("plot_probe_cpmg.py", "--output", "example_probe_cpmg.png"),
    "bpp-temperature": (
        "plot_bpp_water_t1t2_temperature.py",
        "--output",
        "example_bpp_water_t1t2_temperature.png",
    ),
    "esr-powder": (
        "plot_esr_powder_spectrum.py",
        "--output",
        "example_esr_powder_spectrum.png",
    ),
    "nqr-nmr-crossover": (
        "plot_nqr_nmr_crossover.py",
        "--output",
        "example_nqr_nmr_crossover.png",
    ),
    "sequence-timeline": (
        "plot_sequence_timeline.py",
        "--output",
        "example_sequence_timeline.png",
    ),
    "solenoid-properties": (
        "plot_solenoid_coil_properties.py",
        "--save",
        "example_solenoid_coil_properties.png",
    ),
    "pore-diffraction": (
        "plot_pgse_circular_pore_diffraction.py",
        "--method",
        "sgp-propagator",
        "--output",
        "example_pgse_circular_pore_diffraction.png",
    ),
    "zulf-quadrupolar-jcoupling": (
        "plot_zulf_quadrupolar_jcoupling.py",
        "--output",
        "example_zulf_quadrupolar_jcoupling.png",
    ),
    "bssfp-field-inhomogeneity": (
        "plot_bssfp_field_inhomogeneity.py",
        "--output",
        "example_bssfp_field_inhomogeneity.png",
    ),
    "radiation-damping": (
        "plot_radiation_damping.py",
        "--output",
        "example_radiation_damping.png",
    ),
    "t2-t2-exchange": (
        "plot_t2_t2_exchange.py",
        "--output",
        "example_t2_t2_exchange.png",
    ),
    "nmr-mouse-fields": (
        "plot_nmr_mouse_fields.py",
        "--output",
        "example_nmr_mouse_fields.png",
    ),
    "gradient-coil-design": (
        "plot_stream_function_gradient_coil.py",
        "--output",
        "example_gradient_coil_design.png",
    ),
    "squid-ulf-crossover": (
        "plot_squid_ulf_crossover.py",
        "--save",
        "example_squid_ulf_crossover.png",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="*",
        choices=tuple(FIGURES),
        help="Figure names to regenerate; omit to regenerate all figures.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available figure names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list:
        print("\n".join(FIGURES))
        return

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    for name in args.names or FIGURES:
        parts = list(FIGURES[name])
        script = PACKAGE_ROOT / "examples" / parts.pop(0)
        output_name = parts[-1]
        parts[-1] = str(IMAGE_DIR / output_name)
        command = [sys.executable, str(script), *parts]
        print(f"[{name}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=PACKAGE_ROOT, env=env, check=True)

        output = IMAGE_DIR / output_name
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"figure was not created: {output}")


if __name__ == "__main__":
    main()
