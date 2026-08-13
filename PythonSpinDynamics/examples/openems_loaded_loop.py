"""Generate or run the Phase 3 openEMS loaded-loop reference problem."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spin_dynamics.fields import (
    check_openems_python,
    loaded_loop_openems_reference,
    run_openems,
    write_openems_project,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".tmp/openems_loaded_loop"),
        help="project and solver-output directory",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="external Python executable containing openEMS and CSXCAD",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute openEMS after generating the standalone project",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="ask openEMS to prepare the model without time stepping",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="external solver timeout in seconds",
    )
    args = parser.parse_args()

    project = loaded_loop_openems_reference()
    project_path, driver_path = write_openems_project(project, args.output)
    print(f"Wrote {project_path}")
    print(f"Wrote {driver_path}")
    print(f"Model SHA-256: {project.model_hash}")
    if not args.run:
        print("Generation complete. Pass --run to execute the optional backend.")
        return

    availability = check_openems_python(args.python)
    if not availability.available:
        raise SystemExit(
            f"openEMS is unavailable in {args.python}: {availability.detail}"
        )
    print(f"Using openEMS {availability.version} from {args.python}")
    result = run_openems(
        project,
        directory=args.output,
        python_executable=args.python,
        timeout=args.timeout,
        setup_only=args.setup_only,
    )
    if result.solution is None:
        print(f"Setup completed in {result.project_directory}")
    else:
        print(f"Imported normalized fields with shape {result.solution.field_shape}")
        print(f"Saved {result.project_directory / 'harmonic_solution.npz'}")


if __name__ == "__main__":
    main()
