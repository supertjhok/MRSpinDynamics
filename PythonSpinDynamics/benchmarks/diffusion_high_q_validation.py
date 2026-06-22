"""Solver-stability sweep for matched diffusion CPMG across coil Q values.

This benchmark probes the *numerical* stability of the matched-probe transient
solver as coil Q grows; it records whether the outputs stay finite, not whether
the diffusion attenuation is physically accurate. Physical correctness of the
constant-gradient CPMG diffusion law is covered by the analytic regression
tests in ``tests/test_diffusion_physics.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.workflows import run_matched_diffusion_cpmg  # noqa: E402


def _parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def _case(q_value: float, args: argparse.Namespace) -> dict[str, str | float | int | bool]:
    start = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        try:
            result = run_matched_diffusion_cpmg(
                num_echoes=args.num_echoes,
                echo_spacing_seconds=args.echo_spacing,
                t1_seconds=args.t1,
                t2_seconds=args.t2,
                dz=args.dz,
                diffusion_time=args.diffusion_time,
                t90_seconds=args.t90,
                q_value=q_value,
                numpts=args.numpts,
                apply_receiver=args.apply_receiver,
                num_workers=args.workers,
                q_stability_action="ignore",
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "q": q_value,
                "finite": False,
                "runtime_seconds": perf_counter() - start,
                "num_warnings": len(caught),
                "peak_abs_integral": np.nan,
                "num_offsets": args.numpts,
                "num_echoes": args.num_echoes,
                "first_bad_echo_index": "",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
    elapsed = perf_counter() - start
    finite = bool(np.all(np.isfinite(result.echo)) and np.all(np.isfinite(result.echo_integrals)))
    peak_integral = float(np.max(np.abs(result.echo_integrals)))
    first_bad = ""
    if not finite:
        bad = np.argwhere(~np.isfinite(result.echo))
        if bad.size:
            first_bad = ",".join(str(int(item)) for item in bad[0])
    return {
        "q": q_value,
        "finite": finite,
        "runtime_seconds": elapsed,
        "num_warnings": len(caught),
        "peak_abs_integral": peak_integral,
        "num_offsets": int(result.del_w.size),
        "num_echoes": int(result.sequence_time.size),
        "first_bad_echo_index": first_bad,
        "error_type": "",
        "error_message": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-values", default="20,50,80,100,200,500,1000,2000,2500,5000")
    parser.add_argument("--numpts", type=int, default=17)
    parser.add_argument("--num-echoes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--echo-spacing", type=float, default=1000e-6)
    parser.add_argument("--t1", type=float, default=100e-3)
    parser.add_argument("--t2", type=float, default=100e-3)
    parser.add_argument("--dz", type=float, default=0.001)
    parser.add_argument("--diffusion-time", type=float, default=1000e-6)
    parser.add_argument("--t90", type=float, default=100e-6)
    parser.add_argument("--apply-receiver", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = [_case(q_value, args) for q_value in _parse_float_list(args.q_values)]
    fields = [
        "q",
        "finite",
        "runtime_seconds",
        "num_warnings",
        "peak_abs_integral",
        "num_offsets",
        "num_echoes",
        "first_bad_echo_index",
        "error_type",
        "error_message",
    ]
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {args.output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
